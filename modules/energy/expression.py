"""Expression-based energy module."""

from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from geometry.entities import Mesh, _fast_cross
from runtime.expr_eval import eval_expr


def _expression_from_options(options: dict | None) -> str | None:
    if not options:
        return None
    return (
        options.get("expression")
        or options.get("energy_expression")
        or options.get("expr")
    )


def _expression_measure(entity_type: str, options: dict | None) -> str:
    if options and options.get("expression_measure"):
        return str(options.get("expression_measure"))
    if entity_type == "edge":
        return "length"
    if entity_type == "facet":
        return "area"
    if entity_type == "body":
        return "volume"
    return "point"


def _build_names(mesh: Mesh, options: dict | None, pos: np.ndarray) -> Dict[str, float]:
    names: Dict[str, float] = {
        "x": float(pos[0]),
        "y": float(pos[1]),
        "z": float(pos[2]),
        "x1": float(pos[0]),
        "x2": float(pos[1]),
        "x3": float(pos[2]),
    }
    gp = getattr(mesh, "global_parameters", None)
    if gp is not None:
        for key, val in gp.to_dict().items():
            if isinstance(val, (int, float)):
                names[key] = float(val)
    if options:
        params = options.get("expr_params") or {}
        for key, val in params.items():
            if isinstance(val, (int, float)):
                names[key] = float(val)
    return names


def _edge_length(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(b - a))


def _facet_area(positions: np.ndarray) -> float:
    v0 = positions[0]
    v1 = positions[1:-1] - v0
    v2 = positions[2:] - v0
    cross = _fast_cross(v1, v2)
    return 0.5 * float(np.linalg.norm(cross, axis=1).sum())


def _body_volume(mesh: Mesh, body, positions: np.ndarray, index_map: Dict[int, int]):
    return body.compute_volume(mesh, positions=positions, index_map=index_map)


def _entity_positions(
    mesh: Mesh,
    vertex_ids: Iterable[int],
    positions: np.ndarray,
    index_map: Dict[int, int],
    overrides: dict[int, np.ndarray] | None,
) -> np.ndarray:
    coords = []
    for vidx in vertex_ids:
        if overrides and vidx in overrides:
            coords.append(overrides[vidx])
        else:
            coords.append(positions[index_map[vidx]])
    return np.asarray(coords, dtype=float)


def _entity_energy(
    mesh: Mesh,
    entity_type: str,
    entity,
    options: dict | None,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    overrides: dict[int, np.ndarray] | None = None,
) -> float:
    expr = _expression_from_options(options)
    if expr is None:
        return 0.0

    measure = _expression_measure(entity_type, options)
    scale = float(options.get("expression_scale", 1.0)) if options else 1.0

    if entity_type == "vertex":
        pos = (
            overrides[entity.index]
            if overrides and entity.index in overrides
            else positions[index_map[entity.index]]
        )
        names = _build_names(mesh, options, pos)
        return scale * eval_expr(expr, names)

    if entity_type == "edge":
        a = (
            overrides[entity.tail_index]
            if overrides and entity.tail_index in overrides
            else positions[index_map[entity.tail_index]]
        )
        b = (
            overrides[entity.head_index]
            if overrides and entity.head_index in overrides
            else positions[index_map[entity.head_index]]
        )
        mid = 0.5 * (a + b)
        names = _build_names(mesh, options, mid)
        val = eval_expr(expr, names)
        if measure == "length":
            val *= _edge_length(a, b)
        return scale * float(val)

    if entity_type == "facet":
        if (
            getattr(mesh, "facet_vertex_loops", None)
            and entity.index in mesh.facet_vertex_loops
        ):
            v_ids = mesh.facet_vertex_loops[entity.index].tolist()
        else:
            v_ids = []
            for signed_ei in entity.edge_indices:
                edge = mesh.edges[abs(signed_ei)]
                if signed_ei > 0:
                    tail, head = edge.tail_index, edge.head_index
                else:
                    tail, head = edge.head_index, edge.tail_index
                if not v_ids:
                    v_ids.append(tail)
                v_ids.append(head)
            if len(v_ids) > 1:
                v_ids = v_ids[:-1]

        coords = _entity_positions(mesh, v_ids, positions, index_map, overrides)
        centroid = coords.mean(axis=0)
        names = _build_names(mesh, options, centroid)
        val = eval_expr(expr, names)
        if measure == "area":
            val *= _facet_area(coords)
        return scale * float(val)

    if entity_type == "body":
        body_vertices = set()
        for fid in entity.facet_indices:
            facet = mesh.facets[fid]
            if (
                getattr(mesh, "facet_vertex_loops", None)
                and facet.index in mesh.facet_vertex_loops
            ):
                v_ids = mesh.facet_vertex_loops[facet.index].tolist()
            else:
                v_ids = []
                for signed_ei in facet.edge_indices:
                    edge = mesh.edges[abs(signed_ei)]
                    if signed_ei > 0:
                        tail, head = edge.tail_index, edge.head_index
                    else:
                        tail, head = edge.head_index, edge.tail_index
                    if not v_ids:
                        v_ids.append(tail)
                    v_ids.append(head)
                if len(v_ids) > 1:
                    v_ids = v_ids[:-1]
            body_vertices.update(v_ids)

        coords = _entity_positions(mesh, body_vertices, positions, index_map, overrides)
        centroid = coords.mean(axis=0)
        names = _build_names(mesh, options, centroid)
        val = eval_expr(expr, names)
        if measure == "volume":
            val *= _body_volume(mesh, entity, positions, index_map)
        return scale * float(val)

    return 0.0


def _entity_gradient(
    mesh: Mesh,
    entity_type: str,
    entity,
    options: dict | None,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    eps: float,
) -> Dict[int, np.ndarray]:
    grad: Dict[int, np.ndarray] = {}
    if entity_type == "vertex":
        vertices = [entity.index]
    elif entity_type == "edge":
        vertices = [entity.tail_index, entity.head_index]
    elif entity_type == "facet":
        if (
            getattr(mesh, "facet_vertex_loops", None)
            and entity.index in mesh.facet_vertex_loops
        ):
            vertices = mesh.facet_vertex_loops[entity.index].tolist()
        else:
            vertices = []
            for signed_ei in entity.edge_indices:
                edge = mesh.edges[abs(signed_ei)]
                if signed_ei > 0:
                    tail, head = edge.tail_index, edge.head_index
                else:
                    tail, head = edge.head_index, edge.tail_index
                if not vertices:
                    vertices.append(tail)
                vertices.append(head)
            if len(vertices) > 1:
                vertices = vertices[:-1]
    elif entity_type == "body":
        vertices = []
        for fid in entity.facet_indices:
            facet = mesh.facets[fid]
            if (
                getattr(mesh, "facet_vertex_loops", None)
                and facet.index in mesh.facet_vertex_loops
            ):
                vertices.extend(mesh.facet_vertex_loops[facet.index].tolist())
            else:
                v_ids = []
                for signed_ei in facet.edge_indices:
                    edge = mesh.edges[abs(signed_ei)]
                    if signed_ei > 0:
                        tail, head = edge.tail_index, edge.head_index
                    else:
                        tail, head = edge.head_index, edge.tail_index
                    if not v_ids:
                        v_ids.append(tail)
                    v_ids.append(head)
                if len(v_ids) > 1:
                    v_ids = v_ids[:-1]
                vertices.extend(v_ids)
        vertices = list(sorted(set(vertices)))
    else:
        return grad

    for vidx in vertices:
        base = positions[index_map[vidx]]
        deriv = np.zeros(3)
        for dim in range(3):
            plus = base.copy()
            minus = base.copy()
            plus[dim] += eps
            minus[dim] -= eps
            overrides_plus = {vidx: plus}
            overrides_minus = {vidx: minus}
            e_plus = _entity_energy(
                mesh,
                entity_type,
                entity,
                options,
                positions=positions,
                index_map=index_map,
                overrides=overrides_plus,
            )
            e_minus = _entity_energy(
                mesh,
                entity_type,
                entity,
                options,
                positions=positions,
                index_map=index_map,
                overrides=overrides_minus,
            )
            deriv[dim] = (e_plus - e_minus) / (2.0 * eps)
        grad[vidx] = deriv
    return grad


def compute_energy_and_gradient(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    compute_gradient: bool = True,
) -> tuple[float, Dict[int, np.ndarray]]:
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    eps = float(global_params.get("expression_eps", 1e-6))
    energy = 0.0
    grad: Dict[int, np.ndarray] = {}

    entities = [
        ("vertex", mesh.vertices.values()),
        ("edge", mesh.edges.values()),
        ("facet", mesh.facets.values()),
        ("body", mesh.bodies.values()),
    ]
    for entity_type, items in entities:
        for entity in items:
            options = getattr(entity, "options", None)
            expr = _expression_from_options(options)
            if expr is None:
                continue
            energy += _entity_energy(
                mesh,
                entity_type,
                entity,
                options,
                positions=positions,
                index_map=index_map,
            )
            if not compute_gradient:
                continue
            g_entity = _entity_gradient(
                mesh,
                entity_type,
                entity,
                options,
                positions=positions,
                index_map=index_map,
                eps=eps,
            )
            for vidx, gvec in g_entity.items():
                if vidx not in grad:
                    grad[vidx] = gvec.copy()
                else:
                    grad[vidx] += gvec

    return float(energy), grad if compute_gradient else {}


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray,
) -> float:
    eps = float(global_params.get("expression_eps", 1e-6))
    energy = 0.0

    entities = [
        ("vertex", mesh.vertices.values()),
        ("edge", mesh.edges.values()),
        ("facet", mesh.facets.values()),
        ("body", mesh.bodies.values()),
    ]
    for entity_type, items in entities:
        for entity in items:
            options = getattr(entity, "options", None)
            expr = _expression_from_options(options)
            if expr is None:
                continue
            energy += _entity_energy(
                mesh,
                entity_type,
                entity,
                options,
                positions=positions,
                index_map=index_map,
            )
            g_entity = _entity_gradient(
                mesh,
                entity_type,
                entity,
                options,
                positions=positions,
                index_map=index_map,
                eps=eps,
            )
            for vidx, gvec in g_entity.items():
                row = index_map.get(vidx)
                if row is not None:
                    grad_arr[row] += gvec

    return float(energy)


__all__ = ["compute_energy_and_gradient", "compute_energy_and_gradient_array"]

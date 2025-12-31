"""Expression-based hard constraints."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from geometry.entities import Mesh
from modules.energy.expression import (
    _entity_energy,
    _entity_gradient,
)


def _constraint_spec(options: dict | None):
    if not options:
        return None, None
    expr = options.get("constraint_expression") or options.get("expression_constraint")
    target = options.get("constraint_target") or options.get("expression_target")
    if expr is None or target is None:
        return None, None
    return str(expr), float(target)


def _constraint_options(options: dict | None, expr: str) -> dict:
    merged = dict(options or {})
    merged["expression"] = expr
    if "constraint_measure" in merged:
        merged["expression_measure"] = merged.get("constraint_measure")
    if "constraint_scale" in merged:
        merged["expression_scale"] = merged.get("constraint_scale")
    return merged


def constraint_gradients(
    mesh: Mesh, global_params
) -> list[Dict[int, np.ndarray]] | None:
    positions = None
    index_map = None
    if getattr(mesh, "facet_vertex_loops", None):
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row
    if positions is None:
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row

    eps = float(global_params.get("expression_eps", 1e-6))
    gradients: list[Dict[int, np.ndarray]] = []
    entities = [
        ("vertex", mesh.vertices.values()),
        ("edge", mesh.edges.values()),
        ("facet", mesh.facets.values()),
        ("body", mesh.bodies.values()),
    ]
    for entity_type, items in entities:
        for entity in items:
            options = getattr(entity, "options", None)
            expr, target = _constraint_spec(options)
            if expr is None:
                continue
            expr_options = _constraint_options(options, expr)
            g_entity = _entity_gradient(
                mesh,
                entity_type,
                entity,
                expr_options,
                positions=positions,
                index_map=index_map,
                eps=eps,
            )
            gradients.append(g_entity)
    return gradients or None


def constraint_gradients_array(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: dict[int, int],
) -> list[np.ndarray] | None:
    """Return dense constraint gradients for expression constraints."""
    eps = float(global_params.get("expression_eps", 1e-6))
    gradients: list[np.ndarray] = []
    entities = [
        ("vertex", mesh.vertices.values()),
        ("edge", mesh.edges.values()),
        ("facet", mesh.facets.values()),
        ("body", mesh.bodies.values()),
    ]
    for entity_type, items in entities:
        for entity in items:
            options = getattr(entity, "options", None)
            expr, _target = _constraint_spec(options)
            if expr is None:
                continue
            expr_options = _constraint_options(options, expr)
            g_entity = _entity_gradient(
                mesh,
                entity_type,
                entity,
                expr_options,
                positions=positions,
                index_map=index_map,
                eps=eps,
            )
            g_arr = np.zeros_like(positions)
            for vidx, gvec in g_entity.items():
                row = index_map.get(vidx)
                if row is None:
                    continue
                g_arr[row] += gvec
            gradients.append(g_arr)
    return gradients or None


def enforce_constraint(
    mesh: Mesh,
    tol: float = 1e-12,
    max_iter: int = 5,
    global_params: Any | None = None,
    **_kwargs,
) -> None:
    if global_params is None:
        return

    positions = None
    index_map = None
    if getattr(mesh, "facet_vertex_loops", None):
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row
    if positions is None:
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row

    eps = float(global_params.get("expression_eps", 1e-6))

    entities = [
        ("vertex", mesh.vertices.values()),
        ("edge", mesh.edges.values()),
        ("facet", mesh.facets.values()),
        ("body", mesh.bodies.values()),
    ]
    for entity_type, items in entities:
        for entity in items:
            options = getattr(entity, "options", None)
            expr, target = _constraint_spec(options)
            if expr is None:
                continue
            expr_options = _constraint_options(options, expr)

            for _ in range(max_iter):
                current = _entity_energy(
                    mesh,
                    entity_type,
                    entity,
                    expr_options,
                    positions=positions,
                    index_map=index_map,
                )
                delta = current - target
                if abs(delta) < tol:
                    break

                grad = _entity_gradient(
                    mesh,
                    entity_type,
                    entity,
                    expr_options,
                    positions=positions,
                    index_map=index_map,
                    eps=eps,
                )
                norm_sq = sum(np.dot(vec, vec) for vec in grad.values()) + 1e-18
                lam = delta / norm_sq

                for vidx, gvec in grad.items():
                    vertex = mesh.vertices[vidx]
                    if getattr(vertex, "fixed", False):
                        continue
                    vertex.position -= lam * gvec

                mesh.increment_version()
                positions = mesh.positions_view()


__all__ = ["enforce_constraint", "constraint_gradients", "constraint_gradients_array"]

"""Outer-leaflet caveolin rim source energy (soft boundary driving term).

See `modules/energy/tilt_rim_source_in.py` for the full description. This
variant drives the outer leaflet tilt field.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh

USES_TILT_LEAFLETS = True


def _pin_to_circle_group(options: dict | None) -> str | None:
    if not options:
        return None
    group = options.get("pin_to_circle_group")
    return "default" if group is None else str(group)


def _selected_boundary_edges(mesh: Mesh, group: str) -> list[int]:
    mesh.build_connectivity_maps()
    boundary_edges = [eid for eid, fs in mesh.edge_to_facets.items() if len(fs) < 2]
    selected: list[int] = []
    for eid in boundary_edges:
        edge = mesh.edges.get(int(eid))
        if edge is None:
            continue
        v0 = mesh.vertices[int(edge.tail_index)]
        v1 = mesh.vertices[int(edge.head_index)]
        if _pin_to_circle_group(v0.options) != group:
            continue
        if _pin_to_circle_group(v1.options) != group:
            continue
        selected.append(int(eid))
    return selected


def _resolve_group(param_resolver) -> str | None:
    raw = param_resolver.get(None, "tilt_rim_source_group_out")
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _resolve_strength(param_resolver, edge) -> float:
    val = param_resolver.get(edge, "tilt_rim_source_strength_out")
    if val is None:
        val = param_resolver.get(None, "tilt_rim_source_strength_out")
    return float(val or 0.0)


def _resolve_center(param_resolver) -> np.ndarray:
    center = param_resolver.get(None, "tilt_rim_source_center")
    if center is None:
        center = [0.0, 0.0, 0.0]
    arr = np.asarray(center, dtype=float).reshape(3)
    return arr


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver, *, compute_gradient: bool = True
) -> (
    Tuple[float, Dict[int, np.ndarray]]
    | Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Dict-based wrapper returning (E, shape_grad[, tilt_out_grad])."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_out_grad_arr = np.zeros_like(positions) if compute_gradient else None
    energy = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts_out=None,
        tilt_out_grad_arr=tilt_out_grad_arr,
    )
    if not compute_gradient:
        return float(energy), {}

    tilt_grad = {
        int(vid): tilt_out_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if tilt_out_grad_arr is not None and np.any(tilt_out_grad_arr[row])
    }
    return float(energy), {}, tilt_grad


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray,
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
    tilt_in_grad_arr: np.ndarray | None = None,
    tilt_out_grad_arr: np.ndarray | None = None,
) -> float:
    """Dense-array outer-leaflet rim source energy accumulation."""
    _ = global_params, index_map, grad_arr, tilts_in, tilt_in_grad_arr
    group = _resolve_group(param_resolver)
    if group is None:
        return 0.0

    selected = _selected_boundary_edges(mesh, group)
    if not selected:
        return 0.0

    if tilts_out is None:
        tilts_out = mesh.tilts_out_view()
    else:
        tilts_out = np.asarray(tilts_out, dtype=float)
        if tilts_out.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_out must have shape (N_vertices, 3)")

    center = _resolve_center(param_resolver)

    tail_rows: list[int] = []
    head_rows: list[int] = []
    strengths: list[float] = []
    for eid in selected:
        edge = mesh.edges[int(eid)]
        t_row = mesh.vertex_index_to_row.get(int(edge.tail_index))
        h_row = mesh.vertex_index_to_row.get(int(edge.head_index))
        if t_row is None or h_row is None:
            continue
        tail_rows.append(int(t_row))
        head_rows.append(int(h_row))
        strengths.append(_resolve_strength(param_resolver, edge))

    if not tail_rows:
        return 0.0

    tails = np.asarray(tail_rows, dtype=int)
    heads = np.asarray(head_rows, dtype=int)
    gamma = np.asarray(strengths, dtype=float)
    if not np.any(gamma):
        return 0.0

    p0 = positions[tails]
    p1 = positions[heads]
    mid = 0.5 * (p0 + p1)
    r = mid - center[None, :]
    r[:, 2] = 0.0
    rn = np.linalg.norm(r, axis=1)
    good = rn > 1e-12
    if not np.any(good):
        return 0.0

    r_hat = np.zeros_like(r)
    r_hat[good] = r[good] / rn[good][:, None]

    edge_vec = p1 - p0
    lengths = np.linalg.norm(edge_vec, axis=1)

    t_avg = 0.5 * (tilts_out[tails] + tilts_out[heads])
    dots = np.einsum("ij,ij->i", t_avg, r_hat)
    energy = float(-np.sum(gamma * lengths * dots))

    if tilt_out_grad_arr is not None:
        tilt_out_grad_arr = np.asarray(tilt_out_grad_arr, dtype=float)
        if tilt_out_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_out_grad_arr must have shape (N_vertices, 3)")
        contrib = (-0.5 * (gamma * lengths))[:, None] * r_hat
        np.add.at(tilt_out_grad_arr, tails, contrib)
        np.add.at(tilt_out_grad_arr, heads, contrib)

    return energy


__all__ = ["compute_energy_and_gradient", "compute_energy_and_gradient_array"]

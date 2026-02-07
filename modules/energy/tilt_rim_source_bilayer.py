"""Bilayer caveolin rim source energy (soft boundary driving term).

This module implements a Kozlov-style *driving* term that couples to the
bilayer boundary mode

    θ_B ≈ θ^d(R) + θ^p(R),

as used in the analytic derivation in `docs/tex/1_disk_3d.pdf` (tensionless
case). In the discrete solver, the axisymmetric radial tilt angles correspond
to the radial components of the stored 3D tangent tilt vectors. We therefore
model the contact contribution as a line integral along a tagged circular rim:

    E_source = - ∮ γ_B [(t_in + t_out) · r_hat] dl

where `r_hat` is the in-plane radial unit vector from `tilt_rim_source_center`
to the edge midpoint (defined in the circle plane), and `t_in/t_out` are the
inner/outer leaflet tilt vectors.

This is equivalent to loading both `tilt_rim_source_in` and
`tilt_rim_source_out` with equal parameters, but packaged as a single module
for clarity when matching θ_B-based continuum formulas.

Parameters
----------
- `tilt_rim_source_group`: group name (string); when unset, this module is inactive.
- `tilt_rim_source_strength`: γ_B (float; default 0).
- Contact-parameter alternative (maps to `tilt_rim_source_strength`):
  - `tilt_rim_source_contact_h`
  - `tilt_rim_source_contact_delta_epsilon_over_a` or
    (`tilt_rim_source_contact_delta_epsilon` and `tilt_rim_source_contact_a`)
  - Optional unit conversion: `tilt_rim_source_contact_units` in
    `{solver,physical}` with `tilt_rim_source_contact_length_unit_m` and
    `tilt_rim_source_contact_kappa_ref_J`.
- `tilt_rim_source_center`: 3D center point (default [0,0,0]).
- `tilt_rim_source_edge_mode`: edge selection mode: `boundary` (default) or
  `all` (includes internal rims where the tagged edges are not boundary).

Notes
-----
This energy contributes only to the leaflet tilt gradients (no shape gradient).
It is a signed *work* term, so its energy can be negative.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh
from modules.energy.contact_mapping import resolve_contact_line_strength

USES_TILT_LEAFLETS = True
IS_EXTERNAL_WORK = True


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


def _resolve_edge_mode(param_resolver) -> str:
    raw = param_resolver.get(None, "tilt_rim_source_edge_mode")
    mode = str(raw or "boundary").strip().lower()
    return "all" if mode == "all" else "boundary"


def _selected_rim_edges(mesh: Mesh, group: str, *, mode: str) -> list[int]:
    """Return rim edges for `group` based on edge selection mode.

    Modes:
    - "boundary": only true boundary edges (backwards compatible).
    - "all": any edge whose endpoints are tagged with the group (allows internal rims).
    """
    if mode == "all":
        selected: list[int] = []
        for eid, edge in mesh.edges.items():
            v0 = mesh.vertices[int(edge.tail_index)]
            v1 = mesh.vertices[int(edge.head_index)]
            if _pin_to_circle_group(v0.options) != group:
                continue
            if _pin_to_circle_group(v1.options) != group:
                continue
            selected.append(int(eid))
        return selected
    return _selected_boundary_edges(mesh, group)


def _resolve_group(param_resolver) -> str | None:
    raw = param_resolver.get(None, "tilt_rim_source_group")
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _resolve_strength(param_resolver, edge) -> float:
    resolved = resolve_contact_line_strength(
        param_resolver,
        edge,
        strength_key="tilt_rim_source_strength",
        contact_suffix="",
    )
    return float(resolved.gamma or 0.0)


def _resolve_center(param_resolver) -> np.ndarray:
    center = param_resolver.get(None, "tilt_rim_source_center")
    if center is None:
        center = [0.0, 0.0, 0.0]
    return np.asarray(center, dtype=float).reshape(3)


def _normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-15:
        return None
    return vec / norm


def _fit_plane_normal(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3:
        return None
    centroid = np.mean(points, axis=0)
    X = points - centroid
    try:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    return _normalize(vh[-1, :])


def _pin_to_circle_mode(mesh: Mesh, options: dict | None) -> str:
    gp = getattr(mesh, "global_parameters", None)
    raw = None
    if options and options.get("pin_to_circle_mode") is not None:
        raw = options.get("pin_to_circle_mode")
    elif gp is not None and gp.get("pin_to_circle_mode") is not None:
        raw = gp.get("pin_to_circle_mode")
    mode = str(raw or "fixed").strip().lower()
    return "fit" if mode == "fit" else "fixed"


def _pin_to_circle_normal(mesh: Mesh, options: dict | None) -> np.ndarray | None:
    gp = getattr(mesh, "global_parameters", None)
    raw = None
    if options and options.get("pin_to_circle_normal") is not None:
        raw = options.get("pin_to_circle_normal")
    elif gp is not None and gp.get("pin_to_circle_normal") is not None:
        raw = gp.get("pin_to_circle_normal")
    if raw is None:
        return None
    return _normalize(np.asarray(raw, dtype=float).reshape(3))


def _resolve_followed_circle_frame(
    mesh: Mesh, *, group: str, rows: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, normal) for the circle associated with a rim group."""
    pts = mesh.positions_view()[rows]
    center = np.mean(pts, axis=0)

    normal = None
    for row in rows:
        vid = int(mesh.vertex_ids[int(row)])
        v = mesh.vertices.get(vid)
        if v is None:
            continue
        normal = _pin_to_circle_normal(mesh, getattr(v, "options", None))
        if normal is not None:
            break
    if normal is None:
        normal = _fit_plane_normal(pts)
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    return center, normal


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver, *, compute_gradient: bool = True
) -> (
    Tuple[float, Dict[int, np.ndarray]]
    | Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Dict-based wrapper returning (E, shape_grad[, tilt_in_grad, tilt_out_grad])."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_in_grad_arr = np.zeros_like(positions) if compute_gradient else None
    tilt_out_grad_arr = np.zeros_like(positions) if compute_gradient else None
    energy = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts_in=None,
        tilts_out=None,
        tilt_in_grad_arr=tilt_in_grad_arr,
        tilt_out_grad_arr=tilt_out_grad_arr,
    )
    if not compute_gradient:
        return float(energy), {}

    tilt_in_grad = {
        int(vid): tilt_in_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if tilt_in_grad_arr is not None and np.any(tilt_in_grad_arr[row])
    }
    tilt_out_grad = {
        int(vid): tilt_out_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if tilt_out_grad_arr is not None and np.any(tilt_out_grad_arr[row])
    }
    return float(energy), {}, tilt_in_grad, tilt_out_grad


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
    """Dense-array bilayer rim source energy accumulation."""
    _ = global_params, index_map, grad_arr
    group = _resolve_group(param_resolver)
    if group is None:
        return 0.0

    mode = _resolve_edge_mode(param_resolver)
    selected = _selected_rim_edges(mesh, group, mode=mode)
    if not selected:
        return 0.0

    if tilts_in is None:
        tilts_in = mesh.tilts_in_view()
    else:
        tilts_in = np.asarray(tilts_in, dtype=float)
        if tilts_in.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

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

    follow = False
    for row in np.unique(np.concatenate([tails, heads])):
        vid = int(mesh.vertex_ids[int(row)])
        v = mesh.vertices.get(vid)
        if v is None:
            continue
        if _pin_to_circle_group(getattr(v, "options", None)) != group:
            continue
        if _pin_to_circle_mode(mesh, getattr(v, "options", None)) == "fit":
            follow = True
            break

    if follow:
        rim_rows = np.unique(np.concatenate([tails, heads]))
        center, normal = _resolve_followed_circle_frame(
            mesh, group=group, rows=rim_rows
        )
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

    r = mid - center[None, :]
    r = r - np.einsum("ij,j->i", r, normal)[:, None] * normal[None, :]
    rn = np.linalg.norm(r, axis=1)
    good = rn > 1e-12
    if not np.any(good):
        return 0.0

    r_hat = np.zeros_like(r)
    r_hat[good] = r[good] / rn[good][:, None]

    edge_vec = p1 - p0
    lengths = np.linalg.norm(edge_vec, axis=1)

    t_in_avg = 0.5 * (tilts_in[tails] + tilts_in[heads])
    t_out_avg = 0.5 * (tilts_out[tails] + tilts_out[heads])
    dots = np.einsum("ij,ij->i", t_in_avg + t_out_avg, r_hat)
    energy = float(-np.sum(gamma * lengths * dots))

    if tilt_in_grad_arr is not None:
        tilt_in_grad_arr = np.asarray(tilt_in_grad_arr, dtype=float)
        if tilt_in_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_in_grad_arr must have shape (N_vertices, 3)")
        contrib = (-0.5 * (gamma * lengths))[:, None] * r_hat
        np.add.at(tilt_in_grad_arr, tails, contrib)
        np.add.at(tilt_in_grad_arr, heads, contrib)

    if tilt_out_grad_arr is not None:
        tilt_out_grad_arr = np.asarray(tilt_out_grad_arr, dtype=float)
        if tilt_out_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_out_grad_arr must have shape (N_vertices, 3)")
        contrib = (-0.5 * (gamma * lengths))[:, None] * r_hat
        np.add.at(tilt_out_grad_arr, tails, contrib)
        np.add.at(tilt_out_grad_arr, heads, contrib)

    return energy


def compute_energy_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
) -> float:
    """Dense-array bilayer rim source energy (energy only)."""
    _ = global_params, index_map
    group = _resolve_group(param_resolver)
    if group is None:
        return 0.0

    mode = _resolve_edge_mode(param_resolver)
    selected = _selected_rim_edges(mesh, group, mode=mode)
    if not selected:
        return 0.0

    if tilts_in is None:
        tilts_in = mesh.tilts_in_view()
    else:
        tilts_in = np.asarray(tilts_in, dtype=float)
        if tilts_in.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

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

    follow = False
    for row in np.unique(np.concatenate([tails, heads])):
        vid = int(mesh.vertex_ids[int(row)])
        v = mesh.vertices.get(vid)
        if v is None:
            continue
        if _pin_to_circle_group(getattr(v, "options", None)) != group:
            continue
        if _pin_to_circle_mode(mesh, getattr(v, "options", None)) == "fit":
            follow = True
            break

    if follow:
        rim_rows = np.unique(np.concatenate([tails, heads]))
        center, normal = _resolve_followed_circle_frame(
            mesh, group=group, rows=rim_rows
        )
    else:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

    r = mid - center[None, :]
    r = r - np.einsum("ij,j->i", r, normal)[:, None] * normal[None, :]
    rn = np.linalg.norm(r, axis=1)
    good = rn > 1e-12
    if not np.any(good):
        return 0.0

    r_hat = np.zeros_like(r)
    r_hat[good] = r[good] / rn[good][:, None]

    edge_vec = p1 - p0
    lengths = np.linalg.norm(edge_vec, axis=1)

    t_in_avg = 0.5 * (tilts_in[tails] + tilts_in[heads])
    t_out_avg = 0.5 * (tilts_out[tails] + tilts_out[heads])
    dots = np.einsum("ij,ij->i", t_in_avg + t_out_avg, r_hat)
    return float(-np.sum(gamma * lengths * dots))


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]

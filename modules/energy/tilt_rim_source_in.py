"""Inner-leaflet caveolin rim source energy (soft boundary driving term).

This module implements a Kozlov-style *driving* term on a circular inclusion rim
without clamping tilt values. It approximates the 1D "differential contact
energy" effect as a line integral along a tagged boundary:

    E_source = - ∮ γ_in (t_in · r_hat) dl

where ``t_in`` is the inner leaflet tilt vector and ``r_hat`` is the in-plane
radial unit vector from ``tilt_rim_source_center`` to the edge midpoint.

The boundary is selected by matching boundary edges whose *endpoints* are
tagged with ``pin_to_circle_group == tilt_rim_source_group_in``.

Parameters
----------
- ``tilt_rim_source_group_in``: group name (string); when unset, this module is inactive.
- ``tilt_rim_source_strength_in``: γ_in (float; default 0).
- Contact-parameter alternative (maps to ``tilt_rim_source_strength_in``):
  - ``tilt_rim_source_contact_h[_in]``
  - ``tilt_rim_source_contact_delta_epsilon_over_a[_in]`` or
    (``tilt_rim_source_contact_delta_epsilon[_in]`` and
    ``tilt_rim_source_contact_a[_in]``)
  - Optional unit conversion: ``tilt_rim_source_contact_units`` in
    ``{solver,physical}`` with ``tilt_rim_source_contact_length_unit_m`` and
    ``tilt_rim_source_contact_kappa_ref_J``.
- ``tilt_rim_source_center``: 3D center point (default [0,0,0]).
- ``tilt_rim_source_edge_mode``: edge selection mode: ``boundary`` (default) or
  ``all`` (includes internal rims where the tagged edges are not boundary).

Notes
-----
This energy contributes only to the leaflet tilt gradient (no shape gradient).
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
    raw = param_resolver.get(None, "tilt_rim_source_group_in")
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _resolve_strength(param_resolver, edge) -> float:
    resolved = resolve_contact_line_strength(
        param_resolver,
        edge,
        strength_key="tilt_rim_source_strength_in",
        contact_suffix="_in",
    )
    return float(resolved.gamma or 0.0)


def _resolve_center(param_resolver) -> np.ndarray:
    center = param_resolver.get(None, "tilt_rim_source_center")
    if center is None:
        center = [0.0, 0.0, 0.0]
    arr = np.asarray(center, dtype=float).reshape(3)
    return arr


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


def _rim_selection_payload(mesh: Mesh, *, group: str, mode: str):
    """Return cached rim edge/row data for a `(group, mode)` selection."""
    mesh.build_position_cache()
    cache_key = (
        int(getattr(mesh, "_facet_loops_version", 0)),
        int(getattr(mesh, "_vertex_ids_version", 0)),
        str(group),
        str(mode),
    )
    cache_attr = "_tilt_rim_source_in_selection_cache"
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached.get("value")

    selected = _selected_rim_edges(mesh, group, mode=mode)
    if not selected:
        setattr(mesh, cache_attr, {"key": cache_key, "value": None})
        return None

    edge_ids: list[int] = []
    tail_rows: list[int] = []
    head_rows: list[int] = []
    for eid in selected:
        edge = mesh.edges[int(eid)]
        t_row = mesh.vertex_index_to_row.get(int(edge.tail_index))
        h_row = mesh.vertex_index_to_row.get(int(edge.head_index))
        if t_row is None or h_row is None:
            continue
        edge_ids.append(int(eid))
        tail_rows.append(int(t_row))
        head_rows.append(int(h_row))

    if not tail_rows:
        setattr(mesh, cache_attr, {"key": cache_key, "value": None})
        return None

    tails = np.asarray(tail_rows, dtype=int)
    heads = np.asarray(head_rows, dtype=int)
    rim_rows = np.unique(np.concatenate([tails, heads]))

    follow = False
    normal_row: int | None = None
    for row in rim_rows:
        vid = int(mesh.vertex_ids[int(row)])
        vertex = mesh.vertices.get(vid)
        if vertex is None:
            continue
        options = getattr(vertex, "options", None)
        if _pin_to_circle_mode(mesh, options) == "fit":
            follow = True
        if normal_row is None and _pin_to_circle_normal(mesh, options) is not None:
            normal_row = int(row)
        if follow and normal_row is not None:
            break

    value = {
        "edge_ids": np.asarray(edge_ids, dtype=int),
        "tails": tails,
        "heads": heads,
        "rim_rows": rim_rows,
        "follow": bool(follow),
        "normal_row": normal_row,
    }
    setattr(mesh, cache_attr, {"key": cache_key, "value": value})
    return value


def _gamma_cache_key(param_resolver, edge_ids: np.ndarray) -> tuple:
    """Return a conservative cache key for resolved rim-source strengths."""
    strength = param_resolver.get(None, "tilt_rim_source_strength_in")
    gamma_direct = param_resolver.get(None, "tilt_rim_source_contact_gamma_in")
    h = param_resolver.get(None, "tilt_rim_source_contact_h_in")
    delta_over_a = param_resolver.get(
        None, "tilt_rim_source_contact_delta_epsilon_over_a_in"
    )
    delta = param_resolver.get(None, "tilt_rim_source_contact_delta_epsilon_in")
    a = param_resolver.get(None, "tilt_rim_source_contact_a_in")
    units = param_resolver.get(None, "tilt_rim_source_contact_units")
    length_unit = param_resolver.get(None, "tilt_rim_source_contact_length_unit_m")
    kappa_ref = param_resolver.get(None, "tilt_rim_source_contact_kappa_ref_J")
    return (
        tuple(int(x) for x in np.asarray(edge_ids, dtype=int)),
        None if strength is None else float(strength),
        None if gamma_direct is None else float(gamma_direct),
        None if h is None else float(h),
        None if delta_over_a is None else float(delta_over_a),
        None if delta is None else float(delta),
        None if a is None else float(a),
        None if units is None else str(units),
        None if length_unit is None else float(length_unit),
        None if kappa_ref is None else float(kappa_ref),
    )


def _resolved_gamma(mesh: Mesh, param_resolver, *, edge_ids: np.ndarray) -> np.ndarray:
    """Return cached resolved line strengths for the selected rim edges."""
    cache_key = _gamma_cache_key(param_resolver, edge_ids)
    cache_attr = "_tilt_rim_source_in_gamma_cache"
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["gamma"]

    gamma = np.asarray(
        [
            _resolve_strength(param_resolver, mesh.edges[int(eid)])
            for eid in np.asarray(edge_ids, dtype=int)
        ],
        dtype=float,
    )
    setattr(mesh, cache_attr, {"key": cache_key, "gamma": gamma})
    return gamma


def _fixed_circle_frame(mesh: Mesh, param_resolver, *, normal_row: int | None):
    """Return cached fixed-center/fixed-normal frame for non-follow rims."""
    center = _resolve_center(param_resolver)
    normal = np.array([0.0, 0.0, 1.0], dtype=float)
    if normal_row is not None:
        vid = int(mesh.vertex_ids[int(normal_row)])
        vertex = mesh.vertices.get(vid)
        if vertex is not None:
            normal_candidate = _pin_to_circle_normal(
                mesh, getattr(vertex, "options", None)
            )
            if normal_candidate is not None:
                normal = normal_candidate

    cache_key = (
        int(getattr(mesh, "_vertex_ids_version", 0)),
        None if normal_row is None else int(normal_row),
        float(center[0]),
        float(center[1]),
        float(center[2]),
        float(normal[0]),
        float(normal[1]),
        float(normal[2]),
    )
    cache_attr = "_tilt_rim_source_in_fixed_frame_cache"
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["center"], cached["normal"]

    setattr(mesh, cache_attr, {"key": cache_key, "center": center, "normal": normal})
    return center, normal


def _resolve_followed_circle_frame(
    mesh: Mesh, *, rows: np.ndarray, normal_row: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Return (center, normal) for the circle associated with a rim group.

    When the rim uses `pin_to_circle_mode: fit`, the pinning circle can translate
    (and optionally rotate). In that case we treat the circle center as the mean
    of its tagged vertices, and the normal as either the user-specified
    `pin_to_circle_normal` or a fitted plane normal.
    """
    pts = mesh.positions_view()[rows]
    center = np.mean(pts, axis=0)

    normal = None
    if normal_row is not None:
        vid = int(mesh.vertex_ids[int(normal_row)])
        vertex = mesh.vertices.get(vid)
        if vertex is not None:
            normal = _pin_to_circle_normal(mesh, getattr(vertex, "options", None))
    elif rows.size > 0:
        vid = int(mesh.vertex_ids[int(rows[0])])
        vertex = mesh.vertices.get(vid)
        if vertex is not None:
            normal = _pin_to_circle_normal(mesh, getattr(vertex, "options", None))
    if normal is None:
        normal = _fit_plane_normal(pts)
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    return center, normal


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver, *, compute_gradient: bool = True
) -> (
    Tuple[float, Dict[int, np.ndarray]]
    | Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Dict-based wrapper returning (E, shape_grad[, tilt_in_grad])."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_in_grad_arr = np.zeros_like(positions) if compute_gradient else None
    energy = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts_in=None,
        tilt_in_grad_arr=tilt_in_grad_arr,
    )
    if not compute_gradient:
        return float(energy), {}

    tilt_grad = {
        int(vid): tilt_in_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if tilt_in_grad_arr is not None and np.any(tilt_in_grad_arr[row])
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
    """Dense-array inner-leaflet rim source energy accumulation."""
    _ = global_params, index_map, grad_arr, tilts_out, tilt_out_grad_arr
    group = _resolve_group(param_resolver)
    if group is None:
        return 0.0

    mode = _resolve_edge_mode(param_resolver)
    payload = _rim_selection_payload(mesh, group=group, mode=mode)
    if payload is None:
        return 0.0

    if tilts_in is None:
        tilts_in = mesh.tilts_in_view()
    else:
        tilts_in = np.asarray(tilts_in, dtype=float)
        if tilts_in.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

    gamma = _resolved_gamma(mesh, param_resolver, edge_ids=payload["edge_ids"])
    if not np.any(gamma):
        return 0.0

    tails = payload["tails"]
    heads = payload["heads"]

    p0 = positions[tails]
    p1 = positions[heads]
    mid = 0.5 * (p0 + p1)

    # Default: legacy fixed center. When the rim is in `pin_to_circle_mode: fit`,
    # follow the fitted circle as it translates/rotates with the mesh.
    if payload["follow"]:
        center, normal = _resolve_followed_circle_frame(
            mesh,
            rows=payload["rim_rows"],
            normal_row=payload["normal_row"],
        )
    else:
        center, normal = _fixed_circle_frame(
            mesh, param_resolver, normal_row=payload["normal_row"]
        )

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

    t_avg = 0.5 * (tilts_in[tails] + tilts_in[heads])
    dots = np.einsum("ij,ij->i", t_avg, r_hat)
    energy = float(-np.sum(gamma * lengths * dots))

    if tilt_in_grad_arr is not None:
        tilt_in_grad_arr = np.asarray(tilt_in_grad_arr, dtype=float)
        if tilt_in_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_in_grad_arr must have shape (N_vertices, 3)")
        contrib = (-0.5 * (gamma * lengths))[:, None] * r_hat
        np.add.at(tilt_in_grad_arr, tails, contrib)
        np.add.at(tilt_in_grad_arr, heads, contrib)

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
    """Dense-array inner-leaflet rim source energy (energy only)."""
    _ = global_params, index_map, tilts_out
    group = _resolve_group(param_resolver)
    if group is None:
        return 0.0

    mode = _resolve_edge_mode(param_resolver)
    payload = _rim_selection_payload(mesh, group=group, mode=mode)
    if payload is None:
        return 0.0

    if tilts_in is None:
        tilts_in = mesh.tilts_in_view()
    else:
        tilts_in = np.asarray(tilts_in, dtype=float)
        if tilts_in.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

    gamma = _resolved_gamma(mesh, param_resolver, edge_ids=payload["edge_ids"])
    if not np.any(gamma):
        return 0.0

    tails = payload["tails"]
    heads = payload["heads"]

    p0 = positions[tails]
    p1 = positions[heads]
    mid = 0.5 * (p0 + p1)

    if payload["follow"]:
        center, normal = _resolve_followed_circle_frame(
            mesh,
            rows=payload["rim_rows"],
            normal_row=payload["normal_row"],
        )
    else:
        center, normal = _fixed_circle_frame(
            mesh, param_resolver, normal_row=payload["normal_row"]
        )

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

    t_avg = 0.5 * (tilts_in[tails] + tilts_in[heads])
    dots = np.einsum("ij,ij->i", t_avg, r_hat)
    return float(-np.sum(gamma * lengths * dots))


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]

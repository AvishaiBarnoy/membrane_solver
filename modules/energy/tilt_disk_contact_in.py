"""Inner-leaflet disk contact driving term (Kozlov/Barnoy F_cont).

Implements the linear contact energy from `docs/tex/1_disk_3d.tex`:

    F_cont = -2π R h (Δε/a) θ_B

where θ_B is the boundary tilt value on the disk-covered patch. In the
discrete mesh we approximate θ_B as the arc-length-weighted mean of the
radial tilt component on a tagged disk boundary ring.

Parameters
----------
- `tilt_disk_contact_group_in`: group name for disk boundary vertices.
  (If unset, falls back to `rim_slope_match_disk_group`.)
- `tilt_disk_contact_strength_in`: γ = h(Δε/a) in solver units (float).
- Contact mapping alternative:
  - `tilt_disk_contact_h_in`
  - `tilt_disk_contact_delta_epsilon_over_a_in` or
    (`tilt_disk_contact_delta_epsilon_in` and `tilt_disk_contact_a_in`)
  - Optional unit conversion: `tilt_disk_contact_units` in
    `{solver,physical}` with `tilt_disk_contact_length_unit_m` and
    `tilt_disk_contact_kappa_ref_J`.
- `tilt_disk_contact_center`: 3D center point (default [0,0,0]).
- `tilt_disk_contact_normal`: disk plane normal (optional; fitted if absent).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh

USES_TILT_LEAFLETS = True
IS_EXTERNAL_WORK = True


def _resolve_group(param_resolver) -> str | None:
    raw = param_resolver.get(None, "tilt_disk_contact_group_in")
    if raw is None:
        raw = param_resolver.get(None, "rim_slope_match_disk_group")
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _resolve_center(param_resolver) -> np.ndarray:
    center = param_resolver.get(None, "tilt_disk_contact_center")
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


def _resolve_normal(param_resolver, points: np.ndarray) -> np.ndarray:
    raw = param_resolver.get(None, "tilt_disk_contact_normal")
    if raw is not None:
        normal = _normalize(np.asarray(raw, dtype=float).reshape(3))
        if normal is not None:
            return normal
    fitted = _fit_plane_normal(points)
    if fitted is not None:
        return fitted
    return np.array([0.0, 0.0, 1.0], dtype=float)


def _order_by_angle(
    positions: np.ndarray, *, center: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    trial = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(trial, normal))) > 0.9:
        trial = np.array([0.0, 1.0, 0.0], dtype=float)
    u = trial - float(np.dot(trial, normal)) * normal
    nrm = float(np.linalg.norm(u))
    if nrm < 1e-15:
        u = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        u = u / nrm
    v = np.cross(normal, u)
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-15:
        v = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        v = v / v_norm

    rel = positions - center[None, :]
    rel_plane = rel - np.einsum("ij,j->i", rel, normal)[:, None] * normal[None, :]
    x = rel_plane @ u
    y = rel_plane @ v
    angles = np.arctan2(y, x)
    return np.argsort(angles)


def _arc_length_weights(positions: np.ndarray, order: np.ndarray) -> np.ndarray:
    n = len(order)
    if n == 0:
        return np.zeros(0, dtype=float)
    pos = positions[order]
    diffs_next = pos[(np.arange(n) + 1) % n] - pos
    diffs_prev = pos - pos[(np.arange(n) - 1) % n]
    l_next = np.linalg.norm(diffs_next, axis=1)
    l_prev = np.linalg.norm(diffs_prev, axis=1)
    return 0.5 * (l_next + l_prev)


def _resolve_strength(param_resolver, obj) -> float:
    val = param_resolver.get(obj, "tilt_disk_contact_strength_in")
    if val is None:
        val = param_resolver.get(None, "tilt_disk_contact_strength_in")
    if val is not None:
        return float(val or 0.0)

    def get_key(base: str):
        key = f"{base}_in"
        got = param_resolver.get(obj, key)
        if got is None:
            got = param_resolver.get(None, key)
        if got is None:
            got = param_resolver.get(obj, base)
            if got is None:
                got = param_resolver.get(None, base)
        return got

    h = get_key("tilt_disk_contact_h")
    if h is None:
        return 0.0

    delta_over_a = get_key("tilt_disk_contact_delta_epsilon_over_a")
    if delta_over_a is None:
        delta = get_key("tilt_disk_contact_delta_epsilon")
        a = get_key("tilt_disk_contact_a")
        if delta is None or a is None:
            return 0.0
        delta_over_a = float(delta) / float(a)

    gamma_raw = float(h) * float(delta_over_a)
    units = param_resolver.get(None, "tilt_disk_contact_units")
    units = str(units or "solver").strip().lower()
    if units in {"si", "physical", "physical_si"}:
        length_unit_m = param_resolver.get(None, "tilt_disk_contact_length_unit_m")
        kappa_ref_j = param_resolver.get(None, "tilt_disk_contact_kappa_ref_J")
        if length_unit_m is not None and kappa_ref_j is not None:
            length_unit_m = float(length_unit_m)
            kappa_ref_j = float(kappa_ref_j)
            if abs(length_unit_m) > 1e-30 and abs(kappa_ref_j) > 1e-30:
                return gamma_raw * length_unit_m / kappa_ref_j
    return gamma_raw


def _collect_group_rows(mesh: Mesh, group: str) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if (
            opts.get("rim_slope_match_group") == group
            or opts.get("tilt_disk_contact_group") == group
        ):
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    return np.asarray(rows, dtype=int)


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver, *, compute_gradient: bool = True
) -> (
    Tuple[float, Dict[int, np.ndarray]]
    | Tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
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
    """Dense-array inner-leaflet disk contact energy."""
    _ = global_params, index_map, grad_arr, tilts_out, tilt_out_grad_arr
    group = _resolve_group(param_resolver)
    if group is None:
        return 0.0

    if tilts_in is None:
        tilts_in = mesh.tilts_in_view()
    else:
        tilts_in = np.asarray(tilts_in, dtype=float)
        if tilts_in.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

    rows = _collect_group_rows(mesh, group)
    if rows.size == 0:
        return 0.0

    pts = positions[rows]
    center = _resolve_center(param_resolver)
    normal = _resolve_normal(param_resolver, pts)

    order = _order_by_angle(pts, center=center, normal=normal)
    rows = rows[order]
    pts = pts[order]

    weights = _arc_length_weights(pts, np.arange(len(rows)))
    wsum = float(np.sum(weights))
    if wsum <= 1e-12:
        return 0.0

    r_vec = pts - center[None, :]
    r_vec = r_vec - np.einsum("ij,j->i", r_vec, normal)[:, None] * normal[None, :]
    r_len = np.linalg.norm(r_vec, axis=1)
    good = r_len > 1e-12
    if not np.any(good):
        return 0.0

    r_hat = np.zeros_like(r_vec)
    r_hat[good] = r_vec[good] / r_len[good][:, None]

    theta_vals = np.einsum("ij,ij->i", tilts_in[rows], r_hat)
    theta_B = float(np.sum(weights * theta_vals) / wsum)
    R_eff = float(np.sum(weights * r_len) / wsum)

    gamma = _resolve_strength(param_resolver, None)
    if gamma == 0.0:
        return 0.0

    energy = float(-2.0 * np.pi * R_eff * gamma * theta_B)

    if tilt_in_grad_arr is not None:
        tilt_in_grad_arr = np.asarray(tilt_in_grad_arr, dtype=float)
        if tilt_in_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_in_grad_arr must have shape (N_vertices, 3)")
        coeff = (-2.0 * np.pi * R_eff * gamma) * (weights / wsum)
        contrib = coeff[:, None] * r_hat
        np.add.at(tilt_in_grad_arr, rows, contrib)

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
    """Dense-array inner-leaflet disk contact energy (energy only)."""
    _ = global_params, index_map, tilts_out
    group = _resolve_group(param_resolver)
    if group is None:
        return 0.0

    if tilts_in is None:
        tilts_in = mesh.tilts_in_view()
    else:
        tilts_in = np.asarray(tilts_in, dtype=float)
        if tilts_in.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

    rows = _collect_group_rows(mesh, group)
    if rows.size == 0:
        return 0.0

    pts = positions[rows]
    center = _resolve_center(param_resolver)
    normal = _resolve_normal(param_resolver, pts)

    order = _order_by_angle(pts, center=center, normal=normal)
    rows = rows[order]
    pts = pts[order]

    weights = _arc_length_weights(pts, np.arange(len(rows)))
    wsum = float(np.sum(weights))
    if wsum <= 1e-12:
        return 0.0

    r_vec = pts - center[None, :]
    r_vec = r_vec - np.einsum("ij,j->i", r_vec, normal)[:, None] * normal[None, :]
    r_len = np.linalg.norm(r_vec, axis=1)
    good = r_len > 1e-12
    if not np.any(good):
        return 0.0

    r_hat = np.zeros_like(r_vec)
    r_hat[good] = r_vec[good] / r_len[good][:, None]

    theta_vals = np.einsum("ij,ij->i", tilts_in[rows], r_hat)
    theta_B = float(np.sum(weights * theta_vals) / wsum)
    R_eff = float(np.sum(weights * r_len) / wsum)

    gamma = _resolve_strength(param_resolver, None)
    if gamma == 0.0:
        return 0.0

    return float(-2.0 * np.pi * R_eff * gamma * theta_B)


__all__ = [
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]

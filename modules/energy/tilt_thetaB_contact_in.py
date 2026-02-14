"""Inner-leaflet θ_B contact work term (Kozlov-style scalar boundary mode).

Continuum theory (`docs/tex/1_disk_3d.tex`) treats the boundary condition

    t_in · r_hat (r=R) = θ_B

as a Dirichlet constraint rather than an energy penalty. Accordingly, the
default behavior of this module is to report *only* the linear contact work:

    E_contact = -2π R_eff γ θ_B

where R_eff is the arc-length-weighted effective boundary radius and γ is the
contact strength.

Legacy penalty mode
-------------------
Historically this module also included a large quadratic penalty

    E_penalty = 1/2 * k_B * Σ w_i (θ_i - θ_B)^2

to enforce the boundary condition approximately. This penalty is not part of
the continuum energy breakdown; it is kept behind an explicit flag for
backwards compatibility:

    global_parameters.tilt_thetaB_contact_penalty_mode: "legacy"

When penalty mode is "legacy", the module also returns the corresponding tilt
gradient contribution.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from geometry.entities import Mesh

USES_TILT_LEAFLETS = True
IS_EXTERNAL_WORK = True


def _resolve_group(param_resolver) -> str | None:
    raw = param_resolver.get(None, "tilt_thetaB_group_in")
    if raw is None:
        raw = param_resolver.get(None, "rim_slope_match_disk_group")
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _resolve_center(param_resolver) -> np.ndarray:
    center = param_resolver.get(None, "tilt_thetaB_center")
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
    raw = param_resolver.get(None, "tilt_thetaB_normal")
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


def _resolve_k(param_resolver) -> float:
    val = param_resolver.get(None, "tilt_thetaB_strength_in")
    return float(val or 0.0)


def _resolve_gamma(param_resolver) -> float:
    val = param_resolver.get(None, "tilt_thetaB_contact_strength_in")
    if val is not None:
        return float(val or 0.0)
    return 0.0


def _resolve_thetaB(global_params) -> float:
    val = global_params.get("tilt_thetaB_value")
    return float(val or 0.0)


def _penalty_mode(global_params) -> str:
    raw = (
        None
        if global_params is None
        else global_params.get("tilt_thetaB_contact_penalty_mode")
    )
    if raw is None:
        return "off"
    mode = str(raw).strip().lower()
    if mode in {"legacy", "on", "true", "1"}:
        return "legacy"
    return "off"


def _collect_group_rows(mesh: Mesh, group: str) -> np.ndarray:
    cache_key = (mesh._vertex_ids_version, str(group))
    cache_attr = "_tilt_thetaB_contact_group_rows_cache"
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["rows"]

    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if (
            opts.get("rim_slope_match_group") == group
            or opts.get("tilt_thetaB_group") == group
        ):
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    out = np.asarray(rows, dtype=int)
    setattr(mesh, cache_attr, {"key": cache_key, "rows": out})
    return out


def update_scalar_params(mesh: Mesh, global_params, param_resolver) -> None:
    """Update global θ_B by minimizing the local quadratic energy."""
    if _penalty_mode(global_params) != "legacy":
        # In the theory-aligned formulation θ_B is handled either as an explicit
        # global DOF (reduced-energy sampling) or by the calling script. This
        # legacy closed-form update only applies when using the penalty energy.
        return

    group = _resolve_group(param_resolver)
    if group is None:
        return

    rows = _collect_group_rows(mesh, group)
    if rows.size == 0:
        return

    positions = mesh.positions_view()
    pts = positions[rows]
    center = _resolve_center(param_resolver)
    normal = _resolve_normal(param_resolver, pts)
    order = _order_by_angle(pts, center=center, normal=normal)
    rows = rows[order]
    pts = pts[order]

    weights = _arc_length_weights(pts, np.arange(len(rows)))
    wsum = float(np.sum(weights))
    if wsum <= 1e-12:
        return

    r_vec = pts - center[None, :]
    r_vec = r_vec - np.einsum("ij,j->i", r_vec, normal)[:, None] * normal[None, :]
    r_len = np.linalg.norm(r_vec, axis=1)
    good = r_len > 1e-12
    if not np.any(good):
        return

    r_hat = np.zeros_like(r_vec)
    r_hat[good] = r_vec[good] / r_len[good][:, None]

    tilts_in = mesh.tilts_in_view()
    theta_vals = np.einsum("ij,ij->i", tilts_in[rows], r_hat)
    theta_mean = float(np.sum(weights * theta_vals) / wsum)
    R_eff = float(np.sum(weights * r_len) / wsum)

    k = _resolve_k(param_resolver)
    gamma = _resolve_gamma(param_resolver)
    if k <= 0.0:
        return

    theta_B = theta_mean + (2.0 * np.pi * R_eff * gamma) / (k * wsum)
    global_params.set("tilt_thetaB_value", float(theta_B))


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
    """Dense-array inner-leaflet θ_B contact energy."""
    _ = index_map, grad_arr, tilts_out, tilt_out_grad_arr
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
    theta_B = _resolve_thetaB(global_params)
    k = _resolve_k(param_resolver)
    gamma = _resolve_gamma(param_resolver)
    if k == 0.0 and gamma == 0.0:
        return 0.0

    R_eff = float(np.sum(weights * r_len) / wsum)
    energy = 0.0
    if gamma != 0.0:
        # Pure contact work (theory-aligned default).
        energy += float(-2.0 * np.pi * R_eff * gamma * theta_B)

    if _penalty_mode(global_params) == "legacy" and k != 0.0:
        diff = theta_vals - theta_B
        energy += float(0.5 * k * np.sum(weights * diff * diff))

        if tilt_in_grad_arr is not None:
            tilt_in_grad_arr = np.asarray(tilt_in_grad_arr, dtype=float)
            if tilt_in_grad_arr.shape != (len(mesh.vertex_ids), 3):
                raise ValueError("tilt_in_grad_arr must have shape (N_vertices, 3)")
            coeff = (k * weights * diff)[:, None]
            np.add.at(tilt_in_grad_arr, rows, coeff * r_hat)

    return energy


__all__ = [
    "update_scalar_params",
    "compute_energy_and_gradient",
    "compute_energy_and_gradient_array",
    "compute_energy_array",
]


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
    """Dense-array inner-leaflet θ_B contact energy (energy only)."""
    _ = index_map, tilts_out
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
    theta_B = _resolve_thetaB(global_params)
    k = _resolve_k(param_resolver)
    gamma = _resolve_gamma(param_resolver)
    if k == 0.0 and gamma == 0.0:
        return 0.0

    R_eff = float(np.sum(weights * r_len) / wsum)
    energy = 0.0
    if gamma != 0.0:
        energy += float(-2.0 * np.pi * R_eff * gamma * theta_B)

    if _penalty_mode(global_params) == "legacy" and k != 0.0:
        diff = theta_vals - theta_B
        energy += float(0.5 * k * np.sum(weights * diff * diff))

    return float(energy)

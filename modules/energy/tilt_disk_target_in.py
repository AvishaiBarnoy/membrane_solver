"""Inner-leaflet disk tilt target energy (soft profile enforcement).

This module applies a soft penalty that drives the inner-leaflet tilt field
to a prescribed radial profile over a tagged disk:

    E = 1/2 * k * ∫ |t_in - theta(r) r_hat|^2 dA

The target profile is based on the tensionless analytic disk solution in
`docs/tex/1_disk_3d.tex`:

    theta(r) = theta_B * I1(lambda r) / I1(lambda R)

where R is the disk radius and lambda = sqrt(kappa_t / kappa).
When lambda -> 0, the profile falls back to theta(r) = theta_B * r / R.

Parameters
----------
- tilt_disk_target_group_in: group name (string; required).
- tilt_disk_target_strength_in: k (float; default 0).
- tilt_disk_target_theta_B: boundary tilt at r=R (float; default 0).
- tilt_disk_target_lambda: lambda (float; optional; computed from moduli if absent).
- tilt_disk_target_center: 3D center point (default [0,0,0]).
- tilt_disk_target_normal: plane normal (optional; fit from disk if absent).
- tilt_disk_target_radius: R (float; optional; uses max disk radius if absent).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.entities import Mesh, _fast_cross

USES_TILT_LEAFLETS = True


def _resolve_group(param_resolver) -> str | None:
    raw = param_resolver.get(None, "tilt_disk_target_group_in")
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _resolve_strength(param_resolver) -> float:
    val = param_resolver.get(None, "tilt_disk_target_strength_in")
    return float(val or 0.0)


def _resolve_theta_b(param_resolver) -> float:
    val = param_resolver.get(None, "tilt_disk_target_theta_B_in")
    if val is None:
        val = param_resolver.get(None, "tilt_disk_target_theta_B")
    return float(val or 0.0)


def _resolve_center(param_resolver) -> np.ndarray:
    center = param_resolver.get(None, "tilt_disk_target_center_in")
    if center is None:
        center = param_resolver.get(None, "tilt_disk_target_center")
    if center is None:
        center = [0.0, 0.0, 0.0]
    return np.asarray(center, dtype=float).reshape(3)


def _resolve_normal(param_resolver) -> np.ndarray | None:
    raw = param_resolver.get(None, "tilt_disk_target_normal_in")
    if raw is None:
        raw = param_resolver.get(None, "tilt_disk_target_normal")
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-15:
        return None
    return arr / norm


def _fit_plane_normal(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3:
        return None
    centroid = np.mean(points, axis=0)
    X = points - centroid
    try:
        _, _, vh = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1, :]
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-15:
        return None
    return normal / nrm


def _resolve_radius(param_resolver) -> float | None:
    raw = param_resolver.get(None, "tilt_disk_target_radius_in")
    if raw is None:
        raw = param_resolver.get(None, "tilt_disk_target_radius")
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    return val if val > 0.0 else None


def _resolve_lambda(param_resolver) -> float:
    raw = param_resolver.get(None, "tilt_disk_target_lambda_in")
    if raw is None:
        raw = param_resolver.get(None, "tilt_disk_target_lambda")
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    k_tilt = param_resolver.get(None, "tilt_modulus_in")
    if k_tilt is None:
        k_tilt = param_resolver.get(None, "tilt_modolus_in")
    kappa = param_resolver.get(None, "bending_modulus_in")
    if kappa is None:
        kappa = param_resolver.get(None, "bending_modulus")
    if k_tilt is None or kappa is None:
        return 0.0
    try:
        k_tilt = float(k_tilt)
        kappa = float(kappa)
    except (TypeError, ValueError):
        return 0.0
    if k_tilt <= 0.0 or kappa <= 0.0:
        return 0.0
    return float(np.sqrt(k_tilt / kappa))


def _collect_group_rows(mesh: Mesh, group: str) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("tilt_disk_target_group_in") == group:
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    return np.asarray(rows, dtype=int)


def _bessel_i1_series(x: np.ndarray, n_terms: int = 30) -> np.ndarray:
    """Vectorized series approximation to I1(x)."""
    t = 0.5 * x
    t2 = t * t
    term = t.copy()
    out = term.copy()
    for k in range(1, int(n_terms)):
        term *= t2 / (k * (k + 1))
        out += term
    return out


def compute_energy_and_gradient_array(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray | None,
    tilts_in: np.ndarray | None = None,
    tilts_out: np.ndarray | None = None,
    tilt_in_grad_arr: np.ndarray | None = None,
    tilt_out_grad_arr: np.ndarray | None = None,
) -> float:
    """Dense-array inner-leaflet disk target energy accumulation."""
    _ = global_params, index_map, tilts_out, tilt_out_grad_arr
    group = _resolve_group(param_resolver)
    if group is None:
        return 0.0

    k_target = _resolve_strength(param_resolver)
    theta_b = _resolve_theta_b(param_resolver)
    if k_target == 0.0 or theta_b == 0.0:
        return 0.0

    disk_rows = _collect_group_rows(mesh, group)
    if disk_rows.size == 0:
        return 0.0

    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return 0.0

    if tilts_in is None:
        tilts_in = mesh.tilts_in_view()
    else:
        tilts_in = np.asarray(tilts_in, dtype=float)
        if tilts_in.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilts_in must have shape (N_vertices, 3)")

    center = _resolve_center(param_resolver)
    normal = _resolve_normal(param_resolver)
    if normal is None:
        normal = _fit_plane_normal(positions[disk_rows])
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)

    disk_pos = positions[disk_rows]
    r_vec = disk_pos - center[None, :]
    r_vec = r_vec - np.einsum("ij,j->i", r_vec, normal)[:, None] * normal[None, :]
    r_len = np.linalg.norm(r_vec, axis=1)
    good = r_len > 1e-12
    if not np.any(good):
        return 0.0

    r_hat = np.zeros_like(r_vec)
    r_hat[good] = r_vec[good] / r_len[good][:, None]

    radius = _resolve_radius(param_resolver)
    if radius is None:
        radius = float(np.max(r_len))
    if radius <= 0.0:
        return 0.0

    lam = _resolve_lambda(param_resolver)
    if abs(lam) < 1e-12:
        theta = theta_b * r_len / radius
    else:
        num = _bessel_i1_series(lam * r_len)
        den = _bessel_i1_series(np.array([lam * radius], dtype=float))[0]
        if abs(den) < 1e-15:
            return 0.0
        theta = theta_b * num / den

    target = np.zeros_like(tilts_in)
    target[disk_rows] = theta[:, None] * r_hat

    diff = tilts_in - target
    if disk_rows.size != len(mesh.vertex_ids):
        diff_mask = np.ones(len(mesh.vertex_ids), dtype=bool)
        diff_mask[disk_rows] = False
        diff[diff_mask] = 0.0
    diff_sq = np.einsum("ij,ij->i", diff, diff)

    tri_pos = positions[tri_rows]
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]

    n = _fast_cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm >= 1e-12
    if not np.any(mask):
        return 0.0

    areas = 0.5 * n_norm[mask]
    tri_diff_sq_sum = diff_sq[tri_rows[mask]].sum(axis=1)
    coeff = 0.5 * k_target * (tri_diff_sq_sum / 3.0)

    energy = float(np.dot(coeff, areas))

    n_hat = n[mask] / n_norm[mask][:, None]
    g0 = 0.5 * _fast_cross(n_hat, (v2[mask] - v1[mask]))
    g1 = 0.5 * _fast_cross(n_hat, (v0[mask] - v2[mask]))
    g2 = 0.5 * _fast_cross(n_hat, (v1[mask] - v0[mask]))

    c = coeff[:, None]
    if grad_arr is not None:
        np.add.at(grad_arr, tri_rows[mask, 0], c * g0)
        np.add.at(grad_arr, tri_rows[mask, 1], c * g1)
        np.add.at(grad_arr, tri_rows[mask, 2], c * g2)

    if tilt_in_grad_arr is not None:
        tilt_in_grad_arr = np.asarray(tilt_in_grad_arr, dtype=float)
        if tilt_in_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_in_grad_arr must have shape (N_vertices, 3)")

        vertex_areas = np.zeros(len(mesh.vertex_ids), dtype=float)
        area_thirds = areas / 3.0
        np.add.at(vertex_areas, tri_rows[mask, 0], area_thirds)
        np.add.at(vertex_areas, tri_rows[mask, 1], area_thirds)
        np.add.at(vertex_areas, tri_rows[mask, 2], area_thirds)

        tilt_in_grad_arr += k_target * diff * vertex_areas[:, None]

    return energy


def compute_energy_and_gradient(
    mesh: Mesh, global_params, param_resolver
) -> tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """Return energy and gradients for the inner-leaflet disk target."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad_arr = np.zeros_like(positions)
    energy = compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts_in=None,
        tilt_in_grad_arr=tilt_grad_arr,
    )
    shape_grad = {
        int(vid): grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(grad_arr[row])
    }
    tilt_grad = {
        int(vid): tilt_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(tilt_grad_arr[row])
    }
    return float(energy), shape_grad, tilt_grad


__all__ = ["compute_energy_and_gradient", "compute_energy_and_gradient_array"]

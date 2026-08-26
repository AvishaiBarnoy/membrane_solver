"""Shared implementation for inner/outer disk tilt target energies."""

from __future__ import annotations

from typing import Literal

import numpy as np

from geometry.entities import Mesh, _fast_cross

Leaflet = Literal["in", "out"]


def _get(param_resolver, name: str, leaflet: Leaflet):
    value = param_resolver.get(None, f"{name}_{leaflet}")
    return param_resolver.get(None, name) if value is None else value


def _resolve_group(param_resolver, leaflet: Leaflet) -> str | None:
    raw = param_resolver.get(None, f"tilt_disk_target_group_{leaflet}")
    if raw is None:
        return None
    group = str(raw).strip()
    return group if group else None


def _resolve_strength(param_resolver, leaflet: Leaflet) -> float:
    value = param_resolver.get(None, f"tilt_disk_target_strength_{leaflet}")
    return float(value or 0.0)


def _resolve_theta_b(param_resolver, leaflet: Leaflet) -> float:
    return float(_get(param_resolver, "tilt_disk_target_theta_B", leaflet) or 0.0)


def _resolve_center(param_resolver, leaflet: Leaflet) -> np.ndarray:
    center = _get(param_resolver, "tilt_disk_target_center", leaflet)
    if center is None:
        center = [0.0, 0.0, 0.0]
    return np.asarray(center, dtype=float).reshape(3)


def _resolve_normal(param_resolver, leaflet: Leaflet) -> np.ndarray | None:
    raw = _get(param_resolver, "tilt_disk_target_normal", leaflet)
    if raw is None:
        return None
    normal = np.asarray(raw, dtype=float).reshape(3)
    norm = float(np.linalg.norm(normal))
    return None if norm < 1e-15 else normal / norm


def _fit_plane_normal(points: np.ndarray) -> np.ndarray | None:
    if points.shape[0] < 3:
        return None
    centered = points - np.mean(points, axis=0)
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = vh[-1, :]
    norm = float(np.linalg.norm(normal))
    return None if norm < 1e-15 else normal / norm


def _resolve_radius(param_resolver, leaflet: Leaflet) -> float | None:
    raw = _get(param_resolver, "tilt_disk_target_radius", leaflet)
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if value > 0.0 else None


def _resolve_lambda(param_resolver, leaflet: Leaflet) -> float:
    raw = _get(param_resolver, "tilt_disk_target_lambda", leaflet)
    if raw is not None:
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0
    tilt_modulus = param_resolver.get(None, f"tilt_modulus_{leaflet}")
    if tilt_modulus is None and leaflet == "in":
        tilt_modulus = param_resolver.get(None, "tilt_modolus_in")
    bending_modulus = param_resolver.get(None, f"bending_modulus_{leaflet}")
    if bending_modulus is None:
        bending_modulus = param_resolver.get(None, "bending_modulus")
    if tilt_modulus is None or bending_modulus is None:
        return 0.0
    try:
        tilt_modulus = float(tilt_modulus)
        bending_modulus = float(bending_modulus)
    except (TypeError, ValueError):
        return 0.0
    if tilt_modulus <= 0.0 or bending_modulus <= 0.0:
        return 0.0
    return float(np.sqrt(tilt_modulus / bending_modulus))


def _collect_group_rows(mesh: Mesh, group: str, leaflet: Leaflet) -> np.ndarray:
    option_key = f"tilt_disk_target_group_{leaflet}"
    rows = []
    for vertex_id in mesh.vertex_ids:
        options = getattr(mesh.vertices[int(vertex_id)], "options", None) or {}
        if options.get(option_key) == group:
            row = mesh.vertex_index_to_row.get(int(vertex_id))
            if row is not None:
                rows.append(int(row))
    return np.asarray(rows, dtype=int)


def _bessel_i1_series(x: np.ndarray, n_terms: int = 30) -> np.ndarray:
    half_x = 0.5 * x
    half_x_sq = half_x * half_x
    term = half_x.copy()
    result = term.copy()
    for k in range(1, int(n_terms)):
        term *= half_x_sq / (k * (k + 1))
        result += term
    return result


def _target_difference(
    mesh: Mesh,
    param_resolver,
    *,
    leaflet: Leaflet,
    positions: np.ndarray,
    tilts: np.ndarray | None,
    legacy_inner_energy: bool = False,
) -> tuple[float, np.ndarray, np.ndarray] | None:
    if legacy_inner_energy:
        strength = _resolve_strength(param_resolver, leaflet)
        if strength == 0.0:
            return None
        group = _resolve_group(param_resolver, leaflet)
        if group is None:
            return None
        if tilts is None:
            tilts = mesh.tilts_in_view()
        else:
            tilts = np.asarray(tilts, dtype=float)
            if tilts.shape != (len(mesh.vertex_ids), 3):
                raise ValueError("tilts_in must have shape (N_vertices, 3)")
        tri_rows, _ = mesh.triangle_row_cache()
        if tri_rows is None or len(tri_rows) == 0:
            return None
        disk_rows = _collect_group_rows(mesh, group, leaflet)
        if disk_rows.size == 0:
            return None
        theta_b = _resolve_theta_b(param_resolver, leaflet)
    else:
        group = _resolve_group(param_resolver, leaflet)
        if group is None:
            return None
        strength = _resolve_strength(param_resolver, leaflet)
        theta_b = _resolve_theta_b(param_resolver, leaflet)
        if strength == 0.0 or theta_b == 0.0:
            return None
        disk_rows = _collect_group_rows(mesh, group, leaflet)
        if disk_rows.size == 0:
            return None
        tri_rows, _ = mesh.triangle_row_cache()
        if tri_rows is None or len(tri_rows) == 0:
            return None
        if tilts is None:
            tilts = mesh.tilts_in_view() if leaflet == "in" else mesh.tilts_out_view()
        else:
            tilts = np.asarray(tilts, dtype=float)
            if tilts.shape != (len(mesh.vertex_ids), 3):
                raise ValueError(f"tilts_{leaflet} must have shape (N_vertices, 3)")

    center = _resolve_center(param_resolver, leaflet)
    normal = _resolve_normal(param_resolver, leaflet)
    if normal is None:
        normal = _fit_plane_normal(positions[disk_rows])
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    radial = positions[disk_rows] - center[None, :]
    radial -= np.einsum("ij,j->i", radial, normal)[:, None] * normal[None, :]
    radii = np.linalg.norm(radial, axis=1)
    good = radii > 1e-12
    if not np.any(good):
        return None
    radial_unit = np.zeros_like(radial)
    radial_unit[good] = radial[good] / radii[good][:, None]

    radius = None if legacy_inner_energy else _resolve_radius(param_resolver, leaflet)
    if radius is None:
        radius = float(np.max(radii))
    if radius <= 0.0:
        return None
    decay = _resolve_lambda(param_resolver, leaflet)
    zero_decay = decay <= 1e-12 if legacy_inner_energy else abs(decay) < 1e-12
    if zero_decay:
        theta = theta_b * radii / radius
    else:
        numerator = _bessel_i1_series(decay * radii)
        denominator = _bessel_i1_series(np.array([decay * radius], dtype=float))[0]
        if abs(denominator) < 1e-15 and not legacy_inner_energy:
            return None
        theta = theta_b * numerator / denominator

    target = np.zeros_like(tilts)
    target[disk_rows] = theta[:, None] * radial_unit
    difference = tilts - target
    if disk_rows.size != len(mesh.vertex_ids):
        outside_disk = np.ones(len(mesh.vertex_ids), dtype=bool)
        outside_disk[disk_rows] = False
        difference[outside_disk] = 0.0
    return strength, difference, tri_rows


def compute_target_energy(
    mesh: Mesh,
    param_resolver,
    *,
    leaflet: Leaflet,
    positions: np.ndarray,
    tilts: np.ndarray | None,
    grad_arr: np.ndarray | None,
    tilt_grad_arr: np.ndarray | None,
    compute_gradients: bool = True,
    legacy_inner_energy: bool = False,
) -> float:
    payload = _target_difference(
        mesh,
        param_resolver,
        leaflet=leaflet,
        positions=positions,
        tilts=tilts,
        legacy_inner_energy=legacy_inner_energy,
    )
    if payload is None:
        return 0.0
    strength, difference, tri_rows = payload
    difference_sq = np.einsum("ij,ij->i", difference, difference)
    tri_pos = positions[tri_rows]
    v0, v1, v2 = tri_pos[:, 0, :], tri_pos[:, 1, :], tri_pos[:, 2, :]
    normals = _fast_cross(v1 - v0, v2 - v0)
    normal_norms = np.linalg.norm(normals, axis=1)
    mask = normal_norms >= 1e-12
    if not np.any(mask):
        return 0.0
    areas = 0.5 * normal_norms[mask]
    coefficients = 0.5 * strength * (difference_sq[tri_rows[mask]].sum(axis=1) / 3.0)
    energy = float(np.dot(coefficients, areas))
    if not compute_gradients:
        return energy

    normal_unit = normals[mask] / normal_norms[mask][:, None]
    weighted = coefficients[:, None]
    if grad_arr is not None:
        g0 = 0.5 * _fast_cross(normal_unit, v2[mask] - v1[mask])
        g1 = 0.5 * _fast_cross(normal_unit, v0[mask] - v2[mask])
        g2 = 0.5 * _fast_cross(normal_unit, v1[mask] - v0[mask])
        np.add.at(grad_arr, tri_rows[mask, 0], weighted * g0)
        np.add.at(grad_arr, tri_rows[mask, 1], weighted * g1)
        np.add.at(grad_arr, tri_rows[mask, 2], weighted * g2)
    if tilt_grad_arr is not None:
        tilt_grad_arr = np.asarray(tilt_grad_arr, dtype=float)
        if tilt_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError(f"tilt_{leaflet}_grad_arr must have shape (N_vertices, 3)")
        vertex_areas = mesh.barycentric_vertex_areas(
            positions, tri_rows=tri_rows, areas=areas, mask=mask, cache=True
        )
        tilt_grad_arr += strength * difference * vertex_areas[:, None]
    return energy

"""Tilt projection helpers used by the minimizer."""

from __future__ import annotations

import numpy as np


def project_tilts_to_tangent_array(
    tilts: np.ndarray, normals: np.ndarray
) -> np.ndarray:
    """Project a dense tilt array into vertex tangent planes."""
    dot = np.einsum("ij,ij->i", tilts, normals)
    return tilts - dot[:, None] * normals


def project_tilts_axisymmetric_about_center(
    *,
    positions: np.ndarray,
    tilts: np.ndarray,
    normals: np.ndarray,
    center: np.ndarray,
    axis: np.ndarray,
    fixed_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Project tilt vectors to the axisymmetric radial tangent subspace."""
    center = np.asarray(center, dtype=float).reshape(3)
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-15:
        axis = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis = axis / axis_norm

    r_vec = positions - center[None, :]
    r_vec = r_vec - np.einsum("ij,j->i", r_vec, axis)[:, None] * axis[None, :]
    r_len = np.linalg.norm(r_vec, axis=1)
    good = r_len > 1e-12

    r_hat = np.zeros_like(r_vec)
    r_hat[good] = r_vec[good] / r_len[good][:, None]

    r_dir = r_hat - np.einsum("ij,ij->i", r_hat, normals)[:, None] * normals
    r_norm = np.linalg.norm(r_dir, axis=1)
    good &= r_norm > 1e-12

    r_dir_unit = np.zeros_like(r_dir)
    r_dir_unit[good] = r_dir[good] / r_norm[good][:, None]

    proj = np.zeros_like(tilts)
    amp = np.einsum("ij,ij->i", tilts, r_dir_unit)
    proj[good] = amp[good][:, None] * r_dir_unit[good]

    if fixed_mask is not None and np.any(fixed_mask):
        proj[fixed_mask] = tilts[fixed_mask]
    return proj


def project_leaflet_tilts_with_optional_axisymmetry(
    *,
    global_params,
    positions: np.ndarray,
    normals: np.ndarray,
    tilts_in: np.ndarray,
    tilts_out: np.ndarray,
    fixed_mask_in: np.ndarray | None = None,
    fixed_mask_out: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Optionally project leaflet tilts into the axisymmetric radial subspace."""
    if not bool(global_params.get("tilt_axisymmetric_about_thetaB_center", False)):
        return tilts_in, tilts_out

    center = np.asarray(
        global_params.get("tilt_thetaB_center") or [0.0, 0.0, 0.0],
        dtype=float,
    ).reshape(3)
    axis = np.asarray(
        global_params.get("tilt_thetaB_normal") or [0.0, 0.0, 1.0],
        dtype=float,
    ).reshape(3)
    proj_in = project_tilts_axisymmetric_about_center(
        positions=positions,
        tilts=tilts_in,
        normals=normals,
        center=center,
        axis=axis,
        fixed_mask=fixed_mask_in,
    )
    proj_out = project_tilts_axisymmetric_about_center(
        positions=positions,
        tilts=tilts_out,
        normals=normals,
        center=center,
        axis=axis,
        fixed_mask=fixed_mask_out,
    )
    return proj_in, proj_out

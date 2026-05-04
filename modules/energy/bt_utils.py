"""Numerical utilities for leaflet-specific bending-tilt coupling."""

from __future__ import annotations

import numpy as np


def _accumulate_leaflet_tilt_gradient(
    tilt_grad_arr: np.ndarray,
    tri_rows: np.ndarray,
    factor: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    g2: np.ndarray,
    *,
    ctx=None,
    scratch_tag: str,
) -> None:
    """Accumulate tilt gradients while reusing scratch buffers when available."""
    if ctx is None:
        np.add.at(tilt_grad_arr, tri_rows[:, 0], factor * g0)
        np.add.at(tilt_grad_arr, tri_rows[:, 1], factor * g1)
        np.add.at(tilt_grad_arr, tri_rows[:, 2], factor * g2)
        return

    scaled = ctx.scratch_array(scratch_tag, shape=g0.shape, dtype=g0.dtype)
    np.multiply(factor, g0, out=scaled)
    np.add.at(tilt_grad_arr, tri_rows[:, 0], scaled)
    np.multiply(factor, g1, out=scaled)
    np.add.at(tilt_grad_arr, tri_rows[:, 1], scaled)
    np.multiply(factor, g2, out=scaled)
    np.add.at(tilt_grad_arr, tri_rows[:, 2], scaled)


def _mean_reconstructed_field(
    field: np.ndarray,
    endpoint_rows: np.ndarray,
    recon_idx: np.ndarray,
    recon_count: np.ndarray,
) -> np.ndarray:
    """Return reconstructed edge-side field, falling back to endpoint values."""
    out = field[endpoint_rows].copy()
    mask1 = recon_count == 1
    if np.any(mask1):
        out[mask1] = field[recon_idx[..., 0][mask1]]
    mask2 = recon_count >= 2
    if np.any(mask2):
        out[mask2] = 0.5 * (
            field[recon_idx[..., 0][mask2]] + field[recon_idx[..., 1][mask2]]
        )
    return out

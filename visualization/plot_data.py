import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import warnings

import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np

from geometry.entities import Mesh

logger = logging.getLogger("membrane_solver")

_TILT_COLOR_BY = {
    "tilt_mag",
    "tilt_div",
    "tilt_in",
    "tilt_out",
    "tilt_div_in",
    "tilt_div_out",
    "tilt_bilayer",
}


def _tilt_field_for_color_by(mesh: Mesh, color_by: str | None) -> np.ndarray:
    if color_by in {"tilt_in", "tilt_div_in"}:
        return mesh.tilts_in_view()
    if color_by in {"tilt_out", "tilt_div_out"}:
        return mesh.tilts_out_view()
    return mesh.tilts_view()


def _bilayer_offset_scale(positions: np.ndarray) -> float:
    if positions.size == 0:
        return 0.0
    span = positions.max(axis=0) - positions.min(axis=0)
    max_range = float(np.max(span)) if span.size else 0.0
    return 0.01 * max_range if max_range > 0 else 0.0


def _triangle_unit_normals(tri_data: np.ndarray) -> np.ndarray:
    normals = np.cross(tri_data[:, 1] - tri_data[:, 0], tri_data[:, 2] - tri_data[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    unit = np.zeros_like(normals)
    good = norms > 1e-12
    unit[good] = normals[good] / norms[good][:, None]
    return unit


def _loop_unit_normal(loop_positions: np.ndarray) -> np.ndarray:
    if loop_positions.shape[0] < 3:
        return np.zeros(3, dtype=float)
    normal = np.cross(
        loop_positions[1] - loop_positions[0],
        loop_positions[2] - loop_positions[0],
    )
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return np.zeros(3, dtype=float)
    return normal / norm


def _safe_pause(interval: float) -> None:
    """Pause briefly to let interactive backends process events.

    Matplotlib emits a UserWarning on non-interactive backends (e.g. Agg); we
    silence that warning so headless/test runs remain clean even if warnings are
    treated as errors.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*non-interactive.*cannot be shown.*",
            category=UserWarning,
        )
        plt.pause(interval)


def triangle_tilt_magnitudes(
    mesh: Mesh, *, tilts: np.ndarray | None = None
) -> tuple[np.ndarray, list[int]]:
    """Return per-triangle mean tilt magnitudes for facets in triangle_row_cache."""
    tri_rows, tri_facets = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return np.array([], dtype=float), []
    if tilts is None:
        tilts = mesh.tilts_view()
    mags = np.linalg.norm(tilts, axis=1)
    values = mags[tri_rows].mean(axis=1)
    return values, tri_facets


def _triangle_divergence_from_arrays(
    positions: np.ndarray, tilts: np.ndarray, tri_rows: np.ndarray
) -> np.ndarray:
    """Compute per-triangle divergence values for a nodal vector field.

    The nodal field is assumed piecewise-linear on each triangle. Divergence is
    constant per triangle and returned as a dense array with one value per row
    in ``tri_rows``.
    """
    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]

    t0 = tilts[tri_rows[:, 0]]
    t1 = tilts[tri_rows[:, 1]]
    t2 = tilts[tri_rows[:, 2]]

    n = np.cross(v1 - v0, v2 - v0)
    denom = np.einsum("ij,ij->i", n, n)
    good = denom > 1e-24

    g0 = np.zeros_like(n)
    g1 = np.zeros_like(n)
    g2 = np.zeros_like(n)

    g0[good] = np.cross(n[good], v2[good] - v1[good]) / denom[good][:, None]
    g1[good] = np.cross(n[good], v0[good] - v2[good]) / denom[good][:, None]
    g2[good] = np.cross(n[good], v1[good] - v0[good]) / denom[good][:, None]

    div = (
        np.einsum("ij,ij->i", t0, g0)
        + np.einsum("ij,ij->i", t1, g1)
        + np.einsum("ij,ij->i", t2, g2)
    )
    div[~good] = 0.0
    return div


def triangle_tilt_divergence(
    mesh: Mesh, *, tilts: np.ndarray | None = None
) -> tuple[np.ndarray, list[int]]:
    """Return per-triangle divergence of the tilt field for facets in triangle_row_cache."""
    tri_rows, tri_facets = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return np.array([], dtype=float), []
    positions = mesh.positions_view()
    if tilts is None:
        tilts = mesh.tilts_view()
    values = _triangle_divergence_from_arrays(positions, tilts, tri_rows)
    return values, tri_facets


def _colormap_norm_for_scalars(
    color_by: str, values: np.ndarray
) -> tuple[mpl_colors.Colormap, mpl_colors.Normalize]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        values = np.zeros_like(values)
        finite = np.ones_like(values, dtype=bool)

    if color_by in {"tilt_mag", "tilt_in", "tilt_out"}:
        vmax = float(values[finite].max()) if finite.any() else 1.0
        vmax = vmax if vmax > 0 else 1.0
        norm = mpl_colors.Normalize(vmin=0.0, vmax=vmax)
        cmap = plt.get_cmap("viridis")
    elif color_by in {"tilt_div", "tilt_div_in", "tilt_div_out"}:
        vlim = float(np.abs(values[finite]).max()) if finite.any() else 1.0
        vlim = vlim if vlim > 0 else 1.0
        norm = mpl_colors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
        cmap = plt.get_cmap("coolwarm")
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported color_by={color_by!r}")
    return cmap, norm


def _colors_from_scalars(color_by: str, values: np.ndarray) -> np.ndarray:
    cmap, norm = _colormap_norm_for_scalars(color_by, values)
    colors = cmap(norm(np.nan_to_num(values, nan=0.0)))
    colors = np.asarray(colors, dtype=float)
    if colors.ndim == 2 and colors.shape[1] == 4:
        colors[:, 3] = 1.0
    return colors

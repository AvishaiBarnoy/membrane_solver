"""Common utilities for diagnostic and audit scripts."""

from __future__ import annotations

import numpy as np


def apply_global_parameter_overrides(mesh, overrides: dict[str, object] | None) -> None:
    """Apply optional global-parameter overrides to ``mesh`` in place."""
    if not overrides:
        return
    gp = mesh.global_parameters
    for key, value in overrides.items():
        gp.set(str(key), value)


def energy_total(breakdown: dict[str, float]) -> float:
    """Return total energy from an energy-breakdown mapping."""
    return float(sum(float(v) for v in breakdown.values()))


def positions_radii(mesh, positions: np.ndarray | None = None) -> np.ndarray:
    """Return radial distances for all vertices."""
    if positions is None:
        positions = mesh.positions_view()
    return np.linalg.norm(positions[:, :2], axis=1)


def triangle_region_masks(
    mesh,
    tri_rows_eff: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return standard free-disk triangle region masks."""
    n_vertices = len(mesh.vertex_ids)
    disk_mask = np.zeros(n_vertices, dtype=bool)
    rim_mask = np.zeros(n_vertices, dtype=bool)
    outer_mask = np.zeros(n_vertices, dtype=bool)

    for vid in mesh.vertex_ids:
        row = mesh.vertex_index_to_row[int(vid)]
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if str(opts.get("preset") or "") == "disk":
            disk_mask[row] = True
        if str(opts.get("rim_slope_match_group") or "") == "rim":
            rim_mask[row] = True
        if str(opts.get("rim_slope_match_group") or "") == "outer":
            outer_mask[row] = True

    has_disk = np.any(disk_mask[tri_rows_eff], axis=1)
    has_rim = np.any(rim_mask[tri_rows_eff], axis=1)
    has_outer = np.any(outer_mask[tri_rows_eff], axis=1)

    return {
        "disk_core": has_disk & (~has_rim) & (~has_outer),
        "disk_rim": has_disk & has_rim & (~has_outer),
        "rim_outer": has_rim & has_outer & (~has_disk),
        "outer_support_band": has_outer & (~has_rim) & (~has_disk),
        "outer_far": (~has_disk) & (~has_rim) & (~has_outer),
        "outer_membrane": (~has_disk) & (~has_rim),
    }


def mean_abs(values: list[float] | np.ndarray) -> float:
    """Return mean of absolute values."""
    if len(values) == 0:
        return 0.0
    return float(np.mean(np.abs(values)))


def row_region(mesh, row: int) -> str:
    """Return the region label for a given vertex row."""
    vid = int(mesh.vertex_ids[row])
    vertex = mesh.vertices[vid]
    opts = getattr(vertex, "options", None) or {}
    if str(opts.get("preset") or "") == "disk":
        if str(opts.get("rim_slope_match_group") or "") == "rim":
            return "rim"
        return "disk"
    if str(opts.get("rim_slope_match_group") or "") == "outer":
        return "outer"
    return "far"

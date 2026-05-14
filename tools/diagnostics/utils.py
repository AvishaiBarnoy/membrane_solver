"""Common utilities for diagnostic and audit scripts."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("membrane_solver")


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


def radius_labels(mesh, decimals: int = 8) -> np.ndarray:
    """Return rounded radial labels for all vertices."""
    return np.round(positions_radii(mesh), decimals=decimals)


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
            return "shared_rim"
        return "disk"
    if str(opts.get("rim_slope_match_group") or "") == "outer":
        return "outer_support"
    return "outer_free"


def row_labels(mesh) -> list[str]:
    """Return a list of region labels for all vertex rows."""
    masks = row_region_mask_dict(mesh)
    n_vertices = len(mesh.vertex_ids)
    labels = ["unknown"] * n_vertices

    # Order of assignment matters if masks overlap (they shouldn't here)
    for row in np.flatnonzero(masks["disk"]):
        labels[row] = "disk"
    for row in np.flatnonzero(masks["shared_rim"]):
        labels[row] = "shared_rim"
    for row in np.flatnonzero(masks["outer_support"]):
        labels[row] = "outer_support"
    for row in np.flatnonzero(masks["outer_free"]):
        labels[row] = "outer_free"

    return labels


def row_region_mask_dict(mesh) -> dict[str, np.ndarray]:
    """Return boolean masks for standard regions."""
    n_vertices = len(mesh.vertex_ids)
    disk = np.zeros(n_vertices, dtype=bool)
    rim = np.zeros(n_vertices, dtype=bool)
    outer = np.zeros(n_vertices, dtype=bool)

    for row, vid in enumerate(mesh.vertex_ids):
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if str(opts.get("preset") or "") == "disk":
            disk[row] = True
        if str(opts.get("rim_slope_match_group") or "") == "rim":
            rim[row] = True
        if str(opts.get("rim_slope_match_group") or "") == "outer":
            outer[row] = True

    return {
        "disk": disk & (~rim),
        "shared_rim": rim,
        "outer_support": outer,
        "outer_free": (~disk) & (~rim) & (~outer),
    }


def radial_projection(mesh, vectors: np.ndarray) -> np.ndarray:
    """Project 3D vectors onto local radial directions in XY plane."""
    positions = mesh.positions_view()
    radii = positions_radii(mesh, positions)
    r_hat = np.zeros_like(positions)
    good = radii > 1.0e-12
    r_hat[good, :2] = positions[good, :2] / radii[good, None]
    return np.einsum("ij,ij->i", vectors, r_hat)


def radial_thetas(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return radii, theta_in, theta_out, and common theta mode."""
    radii = positions_radii(mesh)
    theta_in = radial_projection(mesh, mesh.tilts_in_view())
    theta_out = radial_projection(mesh, mesh.tilts_out_view())
    return radii, theta_in, theta_out, 0.5 * (theta_in + theta_out)


def abs_by_region(mesh, values: np.ndarray) -> dict[str, float]:
    """Return sum of absolute values grouped by region labels."""
    masks = row_region_mask_dict(mesh)
    vals = np.abs(np.asarray(values, dtype=float))
    return {
        "disk": float(np.sum(vals[masks["disk"]])),
        "shared_rim": float(np.sum(vals[masks["shared_rim"]])),
        "outer_support": float(np.sum(vals[masks["outer_support"]])),
        "outer_free": float(np.sum(vals[masks["outer_free"]])),
    }


def capture_state(mesh) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a deep copy of mesh positions and all tilt fields."""
    return (
        mesh.positions_view().copy(order="F"),
        mesh.tilts_view().copy(order="F"),
        mesh.tilts_in_view().copy(order="F"),
        mesh.tilts_out_view().copy(order="F"),
    )


def restore_state(
    mesh,
    positions: np.ndarray,
    tilts: np.ndarray,
    tilts_in: np.ndarray,
    tilts_out: np.ndarray,
) -> None:
    """Restore mesh positions and tilt fields from captured state."""
    for row, vid in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vid)].position[:] = positions[row]
    mesh.set_tilts_from_array(tilts)
    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)
    mesh.increment_version()
    mesh.build_position_cache()

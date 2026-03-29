"""Shared local shell construction for curved disk-boundary interface checks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from geometry.entities import Mesh


@dataclass(frozen=True)
class LocalInterfaceShellData:
    """Ordered local shell family adjacent to the disk boundary."""

    disk_rows: np.ndarray
    rim_rows: np.ndarray
    outer_rows: np.ndarray
    disk_rows_matched: np.ndarray
    rim_rows_matched: np.ndarray
    rim_rows_for_disk: np.ndarray
    outer_rows_for_rim: np.ndarray
    outer_rows_for_disk: np.ndarray
    disk_radius: float
    rim_radius: float
    outer_radius: float
    disk_r_hat: np.ndarray
    rim_r_hat: np.ndarray
    matching_strategy: str = "nearest_azimuth"
    shell_source: str = "disk_boundary_local_shells"


def collect_disk_boundary_rows(mesh: Mesh, *, group: str = "disk") -> np.ndarray:
    """Return rows tagged as the disk boundary for rim/tilt matching."""
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if (
            opts.get("rim_slope_match_group") == group
            or opts.get("tilt_thetaB_group") == group
            or opts.get("tilt_thetaB_group_in") == group
        ):
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    out = np.asarray(rows, dtype=int)
    if out.size == 0:
        raise AssertionError(f"Missing or empty disk boundary group: {group!r}")
    return out


def order_rows_by_angle(positions: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Return row indices sorted by azimuthal angle."""
    phi = np.mod(np.arctan2(positions[rows, 1], positions[rows, 0]), 2.0 * np.pi)
    return np.asarray(rows[np.argsort(phi)], dtype=int)


def radial_unit_vectors(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cylindrical radii and in-plane radial unit vectors."""
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    return r, r_hat


def _wrapped_angle_delta(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return wrapped absolute angular differences."""
    diff = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return np.minimum(diff, 2.0 * np.pi - diff)


def _match_rows_by_azimuth(
    source_phi: np.ndarray, target_rows: np.ndarray, target_phi: np.ndarray
) -> np.ndarray:
    """Match ordered rows by azimuth, preserving cyclic order when counts agree."""
    source_phi = np.asarray(source_phi, dtype=float)
    target_rows = np.asarray(target_rows, dtype=int)
    target_phi = np.asarray(target_phi, dtype=float)

    if source_phi.size == target_rows.size and source_phi.size > 0:
        best_shift = 0
        best_cost = float("inf")
        for shift in range(source_phi.size):
            cand_phi = np.roll(target_phi, -shift)
            cost = float(np.mean(_wrapped_angle_delta(source_phi, cand_phi)))
            if cost < best_cost:
                best_cost = cost
                best_shift = shift
        return np.asarray(np.roll(target_rows, -best_shift), dtype=int)

    dphi = _wrapped_angle_delta(source_phi[:, None], target_phi[None, :])
    return np.asarray(target_rows[np.argmin(dphi, axis=1)], dtype=int)


def extrapolate_trace_to_radius(
    *,
    target_radius: float,
    first_radius: float,
    first_values: np.ndarray,
    second_radius: float | None = None,
    second_values: np.ndarray | None = None,
) -> np.ndarray:
    """Extrapolate aligned shell values back to ``target_radius``.

    Uses a two-shell linear extrapolation when the second shell is available.
    Falls back to the first-shell values when only one usable outer shell exists.
    """
    out = np.asarray(first_values, dtype=float)
    if second_radius is None or second_values is None:
        return np.array(out, copy=True)

    dr = float(second_radius) - float(first_radius)
    if abs(dr) <= 1.0e-12:
        return np.array(out, copy=True)

    second = np.asarray(second_values, dtype=float)
    slope = (second - out) / dr
    return out + (float(target_radius) - float(first_radius)) * slope


def build_local_interface_shell_data(
    mesh: Mesh,
    *,
    positions: np.ndarray,
    group: str = "disk",
) -> LocalInterfaceShellData:
    """Build the disk/rim/outer local shell family adjacent to the disk boundary."""
    disk_rows = order_rows_by_angle(
        positions, collect_disk_boundary_rows(mesh, group=group)
    )
    radii = np.linalg.norm(positions[:, :2], axis=1)
    disk_radius = float(np.max(radii[disk_rows]))
    disk_mask = np.zeros(radii.shape[0], dtype=bool)
    disk_mask[disk_rows] = True
    rim_candidates = (~disk_mask) & (radii > (disk_radius + 1.0e-9))
    if not np.any(rim_candidates):
        raise AssertionError("Missing outer candidates beyond disk boundary radius.")
    rim_radius = float(np.min(radii[rim_candidates]))
    rim_tol = max(1.0e-9, 1.0e-5 * max(1.0, abs(rim_radius)))
    rim_rows = order_rows_by_angle(
        positions,
        np.flatnonzero((~disk_mask) & (np.abs(radii - rim_radius) <= rim_tol)),
    )
    outer_mask = (~disk_mask) & (~np.isin(np.arange(radii.shape[0]), rim_rows))
    outer_candidates = outer_mask & (radii > (rim_radius + rim_tol))
    if not np.any(outer_candidates):
        raise AssertionError("Missing second outer shell for local interface data.")
    outer_radius = float(np.min(radii[outer_candidates]))
    outer_tol = max(1.0e-9, 1.0e-5 * max(1.0, abs(outer_radius)))
    outer_rows = order_rows_by_angle(
        positions,
        np.flatnonzero(outer_mask & (np.abs(radii - outer_radius) <= outer_tol)),
    )

    phi_rim = np.mod(
        np.arctan2(positions[rim_rows, 1], positions[rim_rows, 0]), 2.0 * np.pi
    )
    phi_out = np.mod(
        np.arctan2(positions[outer_rows, 1], positions[outer_rows, 0]), 2.0 * np.pi
    )
    phi_disk = np.mod(
        np.arctan2(positions[disk_rows, 1], positions[disk_rows, 0]), 2.0 * np.pi
    )

    rim_rows_matched = _match_rows_by_azimuth(phi_out, rim_rows, phi_rim)
    disk_rows_matched = _match_rows_by_azimuth(phi_rim, disk_rows, phi_disk)
    rim_rows_for_disk = _match_rows_by_azimuth(phi_disk, rim_rows, phi_rim)
    outer_rows_for_rim = _match_rows_by_azimuth(phi_rim, outer_rows, phi_out)
    outer_rows_for_disk = _match_rows_by_azimuth(phi_disk, outer_rows, phi_out)

    _, rim_r_hat = radial_unit_vectors(positions[rim_rows_matched])
    _, disk_r_hat = radial_unit_vectors(positions[disk_rows_matched])
    return LocalInterfaceShellData(
        disk_rows=disk_rows,
        rim_rows=rim_rows,
        outer_rows=outer_rows,
        disk_rows_matched=disk_rows_matched,
        rim_rows_matched=rim_rows_matched,
        rim_rows_for_disk=rim_rows_for_disk,
        outer_rows_for_rim=outer_rows_for_rim,
        outer_rows_for_disk=outer_rows_for_disk,
        disk_radius=float(disk_radius),
        rim_radius=float(rim_radius),
        outer_radius=float(outer_radius),
        disk_r_hat=disk_r_hat,
        rim_r_hat=rim_r_hat,
    )


def local_interface_constraint_diagnostics(
    mesh: Mesh,
    *,
    positions: np.ndarray,
    mode: str,
    active: bool,
) -> dict[str, object]:
    """Describe the shared local shell family used by the curved-local path."""
    try:
        data = build_local_interface_shell_data(mesh, positions=positions)
    except AssertionError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "mode": str(mode),
            "active": bool(active),
            "uses_shared_shell_builder": True,
            "matching_strategy": "nearest_azimuth",
            "shell_source": "disk_boundary_local_shells",
        }
    return {
        "available": True,
        "reason": "ok",
        "mode": str(mode),
        "active": bool(active),
        "disk_count": int(data.disk_rows.size),
        "rim_count": int(data.rim_rows.size),
        "outer_count": int(data.outer_rows.size),
        "disk_radius": float(data.disk_radius),
        "rim_radius": float(data.rim_radius),
        "outer_radius": float(data.outer_radius),
        "uses_shared_shell_builder": True,
        "matching_strategy": str(data.matching_strategy),
        "shell_source": str(data.shell_source),
    }


__all__ = [
    "LocalInterfaceShellData",
    "build_local_interface_shell_data",
    "collect_disk_boundary_rows",
    "extrapolate_trace_to_radius",
    "local_interface_constraint_diagnostics",
    "order_rows_by_angle",
    "radial_unit_vectors",
]

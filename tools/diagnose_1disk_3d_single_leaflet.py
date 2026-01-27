"""Diagnostics for the single-leaflet 1_disk_3d tensionless setup."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _mesh_path(path: str | None) -> Path:
    if path:
        return Path(path)
    return (
        _repo_root()
        / "meshes"
        / "caveolin"
        / ("kozlov_1disk_3d_tensionless_single_leaflet_source.yaml")
    )


def _collect_group_rows(mesh, key: str, value: str) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get(key) == value:
            rows.append(mesh.vertex_index_to_row[int(vid)])
    return np.asarray(rows, dtype=int)


def _order_by_angle(positions: np.ndarray) -> np.ndarray:
    angles = np.arctan2(positions[:, 1], positions[:, 0])
    return np.argsort(angles)


def _radial_unit_vectors(positions: np.ndarray) -> np.ndarray:
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    return r_hat


def _outer_free_ring_rows(mesh, positions: np.ndarray) -> np.ndarray:
    rows: list[int] = []
    radii: list[float] = []
    for vid in mesh.vertex_ids:
        v = mesh.vertices[int(vid)]
        opts = getattr(v, "options", None) or {}
        if opts.get("pin_to_circle_group") == "outer":
            continue
        row = mesh.vertex_index_to_row[int(vid)]
        rows.append(row)
        radii.append(float(np.linalg.norm(positions[row, :2])))
    if not rows:
        return np.zeros(0, dtype=int)
    radii_arr = np.asarray(radii, dtype=float)
    r_max = float(np.max(radii_arr))
    tol = 1e-6
    rows_arr = np.asarray(rows, dtype=int)
    return rows_arr[np.abs(radii_arr - r_max) <= tol]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", default=None, help="Optional YAML/JSON mesh path.")
    parser.add_argument("--steps", type=int, default=50, help="Minimization steps.")
    args = parser.parse_args()

    repo_root = _repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from geometry.geom_io import load_data, parse_geometry
    from runtime.constraint_manager import ConstraintModuleManager
    from runtime.energy_manager import EnergyModuleManager
    from runtime.minimizer import Minimizer
    from runtime.steppers.gradient_descent import GradientDescent

    mesh = parse_geometry(load_data(_mesh_path(args.mesh)))
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    minim.minimize(n_steps=int(args.steps))

    positions = mesh.positions_view()
    z_span = float(np.ptp(positions[:, 2]))

    rim_rows = _collect_group_rows(mesh, "rim_slope_match_group", "rim")
    outer_rows = _collect_group_rows(mesh, "rim_slope_match_group", "outer")
    disk_rows = _collect_group_rows(mesh, "rim_slope_match_group", "disk")
    disk_profile_rows = _collect_group_rows(mesh, "tilt_disk_target_group_in", "disk")

    rim_rows = rim_rows[_order_by_angle(positions[rim_rows])]
    outer_rows = outer_rows[_order_by_angle(positions[outer_rows])]

    rim_pos = positions[rim_rows]
    outer_pos = positions[outer_rows]
    r_rim = np.linalg.norm(rim_pos[:, :2], axis=1)
    r_outer = np.linalg.norm(outer_pos[:, :2], axis=1)
    dr = np.maximum(r_outer - r_rim, 1e-6)
    phi = float(np.mean((outer_pos[:, 2] - rim_pos[:, 2]) / dr))

    rim_r_hat = _radial_unit_vectors(rim_pos)
    theta_in_rim = float(
        np.mean(np.einsum("ij,ij->i", mesh.tilts_in_view()[rim_rows], rim_r_hat))
    )
    theta_out_rim = float(
        np.mean(np.einsum("ij,ij->i", mesh.tilts_out_view()[rim_rows], rim_r_hat))
    )

    free_rows = _outer_free_ring_rows(mesh, positions)
    free_r_hat = _radial_unit_vectors(positions[free_rows])
    theta_out_far = float(
        np.mean(
            np.abs(
                np.einsum(
                    "ij,ij->i",
                    mesh.tilts_out_view()[free_rows],
                    free_r_hat,
                )
            )
        )
    )

    print("=== 1_disk_3d single-leaflet diagnostics ===")
    print(f"z-span (outer membrane): {z_span:.4e}")
    print(f"phi (rim→outer slope):   {phi:.4e}")
    print(f"theta_in(rim):           {theta_in_rim:.4e}")
    print(f"theta_out(rim):          {theta_out_rim:.4e}")
    print(f"theta_out(far):          {theta_out_far:.4e}")
    print(f"phi*theta_in(rim):       {phi * theta_in_rim:.4e}")
    print(f"disk-ring count:         {len(disk_rows)}")
    print(f"rim-ring count:          {len(rim_rows)}")
    print(f"outer-ring count:        {len(outer_rows)}")
    if disk_profile_rows.size:
        r_disk = np.linalg.norm(positions[disk_profile_rows, :2], axis=1)
        r_max = float(np.max(r_disk))
        r_hat_disk = _radial_unit_vectors(positions[disk_profile_rows])
        theta_disk = np.einsum(
            "ij,ij->i", mesh.tilts_in_view()[disk_profile_rows], r_hat_disk
        )
        inner_band = theta_disk[r_disk < 0.4 * r_max]
        outer_band = theta_disk[r_disk > 0.8 * r_max]
        print(f"disk tilt (inner band):  {float(np.mean(inner_band)):.4e}")
        print(f"disk tilt (outer band):  {float(np.mean(outer_band)):.4e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

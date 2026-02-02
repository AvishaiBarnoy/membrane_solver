#!/usr/bin/env python3
"""Diagnose disk↔membrane interface topology and curvature baselines.

Usage:
  python scripts/diagnose_disk_interface.py -i meshes/caveolin/kozlov_toy_1disk_coarse.yaml --refine 1 --avg 5

This script is intended as a lightweight, reproducible diagnostic to compare
"toy" vs "real" Kozlov-style meshes and pinpoint interface/topology issues that
lead to tangential curvature artifacts and large flat-state baselines.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.curvature import (  # noqa: E402
    compute_angle_defects,
    compute_curvature_data,
)
from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.interface_validation import validate_disk_interface_topology  # noqa: E402
from runtime.refinement import refine_triangle_mesh  # noqa: E402
from runtime.vertex_average import vertex_average  # noqa: E402


def _rows_for_group(mesh, group: str) -> np.ndarray:
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
    return np.asarray(rows, dtype=int)


def _summarize_kvec(mesh) -> dict[str, float]:
    mesh.build_position_cache()
    pos = mesh.positions_view()
    idx = mesh.vertex_index_to_row
    k_vecs, areas, _weights, _tri_rows = compute_curvature_data(mesh, pos, idx)
    k_mag = np.linalg.norm(k_vecs, axis=1)

    boundary = sorted(mesh.boundary_vertex_ids)
    b_rows = np.asarray(
        [idx[int(vid)] for vid in boundary if int(vid) in idx], dtype=int
    )
    share = float(np.sum(k_mag[b_rows]) / np.sum(k_mag)) if np.sum(k_mag) > 0 else 0.0

    return {
        "k_mag_sum": float(np.sum(k_mag)),
        "k_mag_max": float(np.max(k_mag)) if k_mag.size else 0.0,
        "k_mag_boundary_share": share,
        "boundary_vertex_count": float(len(boundary)),
    }


def _summarize_angle_defects(mesh) -> dict[str, float]:
    mesh.build_position_cache()
    pos = mesh.positions_view()
    idx = mesh.vertex_index_to_row
    defects = compute_angle_defects(mesh, pos, idx)
    return {
        "angle_defect_max_abs": float(np.max(np.abs(defects))) if defects.size else 0.0,
        "angle_defect_sum": float(np.sum(defects)) if defects.size else 0.0,
    }


def _summarize_ring(mesh, group: str) -> dict[str, float]:
    rows = _rows_for_group(mesh, group)
    if rows.size == 0:
        return {"ring_rows": 0.0}
    pos = mesh.positions_view()[rows]
    r = np.linalg.norm(pos[:, :2], axis=1)
    return {
        "ring_rows": float(rows.size),
        "ring_r_min": float(np.min(r)),
        "ring_r_max": float(np.max(r)),
        "ring_r_mean": float(np.mean(r)),
    }


def _force_flat(mesh) -> None:
    for v in mesh.vertices.values():
        v.position[2] = 0.0
    mesh.increment_version()


def _iter_refine(mesh, n: int) -> object:
    m = mesh
    for _ in range(int(n)):
        m = refine_triangle_mesh(m)
    return m


def _iter_avg(mesh, n: int) -> None:
    for _ in range(int(n)):
        vertex_average(mesh)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="YAML/JSON mesh path")
    ap.add_argument("--refine", type=int, default=0, help="Number of refinement passes")
    ap.add_argument(
        "--avg", type=int, default=0, help="Number of vertex-averaging passes"
    )
    ap.add_argument(
        "--flat",
        action="store_true",
        help="Force z=0 for all vertices before diagnostics (reference state)",
    )
    ap.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip disk interface validation (still prints diagnostics).",
    )
    ap.add_argument(
        "--group",
        default="disk",
        help="Disk boundary group name (default: disk)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    mesh = parse_geometry(load_data(args.input))
    mesh.global_parameters.set("disk_interface_validate", True)

    if args.flat:
        _force_flat(mesh)

    if args.refine:
        mesh = _iter_refine(mesh, args.refine)
    if args.avg:
        _iter_avg(mesh, args.avg)

    # Validation (optional)
    if not args.no_validate:
        validate_disk_interface_topology(mesh, mesh.global_parameters)

    out: dict[str, float] = {}
    out.update(_summarize_ring(mesh, str(args.group)))
    out.update(_summarize_kvec(mesh))
    out.update(_summarize_angle_defects(mesh))

    # Print a stable key=value report for copy/paste into issues/PRs.
    keys = sorted(out)
    for k in keys:
        print(f"{k}={out[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

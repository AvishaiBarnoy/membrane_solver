#!/usr/bin/env python3
# ruff: noqa: E402
"""Read-only audit of tagged inclusion-boundary components and local frames."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from geometry.geom_io import load_data, parse_geometry
from modules.constraints.inclusion_components import collect_group_components


def _configured_center(mesh) -> np.ndarray:
    params = mesh.global_parameters
    raw = params.get("rim_slope_match_center")
    if raw is None:
        raw = params.get("tilt_thetaB_center")
    if raw is None:
        raw = [0.0, 0.0, 0.0]
    return np.asarray(raw, dtype=float).reshape(3)


def run_inclusion_boundary_audit(
    *,
    mesh_path: str | Path,
    group: str,
) -> dict[str, Any]:
    """Return component-local geometry without mutating or minimizing the mesh."""
    path = Path(mesh_path)
    mesh = parse_geometry(load_data(str(path)))
    mesh.build_position_cache()
    positions = mesh.positions_view()
    configured_center = _configured_center(mesh)
    components = collect_group_components(mesh, group=str(group))

    component_rows: list[dict[str, Any]] = []
    for index, component in enumerate(components):
        points = positions[component.rows]
        local_radii = np.linalg.norm(points - component.center[None, :], axis=1)
        configured_radii = np.linalg.norm(points - configured_center[None, :], axis=1)
        component_rows.append(
            {
                "index": int(index),
                "vertex_ids": [int(vid) for vid in component.vertex_ids],
                "rows": [int(row) for row in component.rows],
                "center": [float(value) for value in component.center],
                "configured_center_offset": float(
                    np.linalg.norm(component.center - configured_center)
                ),
                "local_radius_mean": float(np.mean(local_radii)),
                "local_radius_std": float(np.std(local_radii)),
                "configured_radius_mean": float(np.mean(configured_radii)),
                "configured_radius_std": float(np.std(configured_radii)),
            }
        )

    return {
        "mesh": str(path),
        "group": str(group),
        "configured_center": [float(value) for value in configured_center],
        "component_count": len(component_rows),
        "legacy_single_frame_unsafe": len(component_rows) > 1,
        "components": component_rows,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--group", required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    audit = run_inclusion_boundary_audit(mesh_path=args.mesh, group=args.group)
    text = yaml.safe_dump(audit, sort_keys=False)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

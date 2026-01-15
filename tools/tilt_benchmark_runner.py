"""Run tilt benchmark meshes and emit summary diagnostics.

Usage:
    python tools/tilt_benchmark_runner.py \
        --glob "meshes/tilt_benchmarks/*.yaml" \
        --output-json outputs/tilt_benchmarks/summary.json \
        --plots-dir outputs/tilt_benchmarks/plots \
        --color-by tilt_mag
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from geometry.geom_io import load_data, parse_geometry
from geometry.tilt_operators import p1_vertex_divergence
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _stat_summary(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    vals = np.asarray(values, dtype=float)
    return {
        "min": float(np.min(vals)),
        "mean": float(np.mean(vals)),
        "p90": float(np.quantile(vals, 0.9)),
        "max": float(np.max(vals)),
    }


def _compute_metrics(path: Path) -> dict[str, Any]:
    mesh = parse_geometry(load_data(path))
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    energy = float(minim.compute_energy())

    mesh.build_position_cache()
    tilts = mesh.tilts_view()
    tri_rows, _ = mesh.triangle_row_cache()
    if tilts is None or tri_rows is None or tilts.size == 0 or tri_rows.size == 0:
        mags = np.zeros(0, dtype=float)
        div_v = np.zeros(0, dtype=float)
    else:
        mags = np.linalg.norm(tilts, axis=1)
        div_v, _ = p1_vertex_divergence(
            n_vertices=len(mesh.vertex_ids),
            positions=mesh.positions_view(),
            tilts=tilts,
            tri_rows=tri_rows,
        )

    boundary_vids = getattr(mesh, "boundary_vertex_ids", None) or []
    boundary_rows = np.array(
        [
            mesh.vertex_index_to_row[vid]
            for vid in boundary_vids
            if vid in mesh.vertex_index_to_row
        ],
        dtype=int,
    )
    mask_interior = np.ones(len(mesh.vertex_ids), dtype=bool)
    if boundary_rows.size:
        mask_interior[boundary_rows] = False

    metrics = {
        "path": str(path),
        "vertices": len(mesh.vertices),
        "facets": len(mesh.facets),
        "energy": energy,
        "tilt_mag": _stat_summary(mags),
        "tilt_mag_interior": _stat_summary(mags[mask_interior]) if mags.size else None,
        "tilt_div": _stat_summary(div_v),
        "tilt_div_interior": _stat_summary(div_v[mask_interior])
        if div_v.size
        else None,
    }
    return metrics


def _print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["mesh", "energy", "|t|_mean", "|t|_max", "div_mean", "div_max"]
    print(" ".join(h.ljust(20) for h in headers))
    for row in rows:
        tilt_mag = row.get("tilt_mag") or {}
        tilt_div = row.get("tilt_div") or {}
        print(
            f"{Path(row['path']).name:<20}"
            f"{row['energy']:>20.6e}"
            f"{tilt_mag.get('mean', 0.0):>20.6e}"
            f"{tilt_mag.get('max', 0.0):>20.6e}"
            f"{tilt_div.get('mean', 0.0):>20.6e}"
            f"{tilt_div.get('max', 0.0):>20.6e}"
        )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "path",
                "vertices",
                "facets",
                "energy",
                "tilt_mag_min",
                "tilt_mag_mean",
                "tilt_mag_p90",
                "tilt_mag_max",
                "tilt_div_min",
                "tilt_div_mean",
                "tilt_div_p90",
                "tilt_div_max",
            ]
        )
        for row in rows:
            mag = row.get("tilt_mag") or {}
            div = row.get("tilt_div") or {}
            writer.writerow(
                [
                    row["path"],
                    row["vertices"],
                    row["facets"],
                    row["energy"],
                    mag.get("min"),
                    mag.get("mean"),
                    mag.get("p90"),
                    mag.get("max"),
                    div.get("min"),
                    div.get("mean"),
                    div.get("p90"),
                    div.get("max"),
                ]
            )


def _save_plots(rows: list[dict[str, Any]], *, color_by: str, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from visualization.plotting import plot_geometry

    out_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = Path(row["path"])
        mesh = parse_geometry(load_data(path))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_geometry(
            mesh,
            ax=ax,
            show=False,
            color_by=color_by,
            show_colorbar=True,
            show_tilt_arrows=True,
            draw_edges=False,
        )
        fig.savefig(out_dir / f"{path.stem}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--glob",
        default="meshes/tilt_benchmarks/*.yaml",
        help="Glob for benchmark meshes (default: meshes/tilt_benchmarks/*.yaml).",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write a JSON summary.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional path to write a CSV summary.",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help="Optional directory to write PNG plots.",
    )
    parser.add_argument(
        "--color-by",
        default="tilt_mag",
        choices=[
            "tilt_mag",
            "tilt_div",
            "tilt_in",
            "tilt_out",
            "tilt_div_in",
            "tilt_div_out",
        ],
        help="Color scheme for plots (default: tilt_mag).",
    )
    args = parser.parse_args()

    import glob

    paths = sorted(Path(p).resolve() for p in glob.glob(args.glob))
    if not paths:
        print(f"No meshes matched glob: {args.glob}")
        return 1

    rows = [_compute_metrics(path) for path in paths]
    _print_table(rows)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if args.output_csv:
        _write_csv(Path(args.output_csv), rows)

    if args.plots_dir:
        _save_plots(rows, color_by=args.color_by, out_dir=Path(args.plots_dir))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

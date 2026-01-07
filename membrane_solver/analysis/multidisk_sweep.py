"""Analyze multi-disk sweep outputs and generate standard plots/reports.

This tool is aimed at “multi-disk” (inclusion) studies where facets are tagged
with a patch label (default: ``disk_patch``). Given a set of output meshes, it
computes:

- Separation ``L`` between two patches (chord length by default).
- Total energy + per-module energy breakdown.
- Shape observables (area, volume, surface radius of gyration, min edge length).
- Optional boundary-loop diagnostics (Gauss–Bonnet boundary geodesic sums).

It then writes a machine-readable table (`results.csv` and `results.json`) and
generates summary plots under the output directory.

Examples
--------
Analyze a set of output meshes and write plots to `outputs/report/`:

    python -m membrane_solver.analysis.multidisk_sweep outputs/sweep --outdir outputs/report

Use explicit patch labels and compute arc-length separation on a sphere:

    python -m membrane_solver.analysis.multidisk_sweep outputs/sweep \\
      --pair top,bottom --separation arc --sphere-radius 3.0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.diagnostics.gauss_bonnet import (
    boundary_geodesic_sum,
    extract_boundary_loops,
    find_boundary_edges,
)
from runtime.diagnostics.patches import patch_boundary_lengths
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


@dataclass(frozen=True)
class CaseResult:
    """Computed metrics for a single mesh case."""

    path: Path
    metrics: dict[str, Any]


def _collect_mesh_files(paths: Sequence[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            for suffix in ("*.json", "*.yaml", "*.yml"):
                files.extend(sorted(path.glob(suffix)))
        else:
            files.append(path)
    # Deduplicate while preserving order.
    seen: set[Path] = set()
    unique: list[Path] = []
    for p in files:
        rp = p.resolve()
        if rp in seen:
            continue
        seen.add(rp)
        unique.append(p)
    return unique


def _triangle_facet_vertices(mesh, facet_id: int) -> np.ndarray:
    if not getattr(mesh, "facet_vertex_loops", None):
        mesh.build_facet_vertex_loops()
    loop = mesh.facet_vertex_loops.get(int(facet_id))
    if loop is None or len(loop) < 3:
        return np.zeros((0, 3), dtype=float)
    return mesh.positions_view()[[mesh.vertex_index_to_row[int(v)] for v in loop]]


def _facet_centroid(mesh, facet_id: int) -> np.ndarray:
    verts = _triangle_facet_vertices(mesh, facet_id)
    if len(verts) == 0:
        return np.zeros(3, dtype=float)
    return verts.mean(axis=0)


def _patch_centroid(mesh, *, patch_key: str, label: str) -> np.ndarray:
    if not getattr(mesh, "facet_vertex_loops", None):
        mesh.build_facet_vertex_loops()
    mesh.build_position_cache()

    total_area = 0.0
    centroid_sum = np.zeros(3, dtype=float)
    for fid, facet in mesh.facets.items():
        if facet.options.get(patch_key) != label:
            continue
        area = float(facet.compute_area(mesh))
        if area <= 0.0:
            continue
        centroid_sum += area * _facet_centroid(mesh, fid)
        total_area += area
    if total_area <= 0.0:
        raise ValueError(f"No area found for patch {label!r} using key {patch_key!r}.")
    return centroid_sum / total_area


def _parse_pair(value: str) -> tuple[str, str]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("--pair must be 'labelA,labelB'.")
    return parts[0], parts[1]


def _compute_separation(
    c0: np.ndarray,
    c1: np.ndarray,
    *,
    mode: str,
    sphere_center: np.ndarray,
    sphere_radius: float | None,
) -> float:
    if mode == "chord":
        return float(np.linalg.norm(c1 - c0))

    u = c0 - sphere_center
    v = c1 - sphere_center
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu <= 0.0 or nv <= 0.0:
        raise ValueError(
            "Patch centroids coincide with sphere center; cannot compute angle."
        )

    cosang = float(np.dot(u, v) / (nu * nv))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    angle = float(math.acos(cosang))
    if mode == "angle":
        return angle
    if mode == "arc":
        r = float(sphere_radius) if sphere_radius is not None else 0.5 * (nu + nv)
        return float(r * angle)
    raise ValueError(f"Unsupported separation mode {mode!r}.")


def _min_edge_length(mesh) -> float:
    if not mesh.edges:
        return 0.0
    mesh.build_position_cache()
    pos = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    tail = np.array([idx_map[e.tail_index] for e in mesh.edges.values()], dtype=int)
    head = np.array([idx_map[e.head_index] for e in mesh.edges.values()], dtype=int)
    seg = pos[head] - pos[tail]
    lengths = np.linalg.norm(seg, axis=1)
    if lengths.size == 0:
        return 0.0
    return float(lengths.min())


def _energy_metrics(mesh) -> tuple[float, dict[str, float]]:
    energy_manager = EnergyModuleManager(mesh.energy_modules)
    constraint_manager = ConstraintModuleManager(mesh.constraint_modules)
    minimizer = Minimizer(
        mesh=mesh,
        global_params=mesh.global_parameters,
        stepper=GradientDescent(max_iter=1),
        energy_manager=energy_manager,
        constraint_manager=constraint_manager,
        quiet=True,
    )
    total = minimizer.compute_energy()
    breakdown = minimizer.compute_energy_breakdown()
    return float(total), breakdown


def analyze_mesh(
    path: Path,
    *,
    patch_key: str,
    pair: tuple[str, str] | None,
    separation: str,
    sphere_center: np.ndarray,
    sphere_radius: float | None,
    include_boundary_diagnostics: bool,
) -> CaseResult:
    """Compute sweep metrics for a single output mesh file."""
    data = load_data(str(path))
    mesh = parse_geometry(data)

    metrics: dict[str, Any] = {
        "case": path.stem,
        "path": str(path),
        "n_vertices": int(len(mesh.vertices)),
        "n_edges": int(len(mesh.edges)),
        "n_facets": int(len(mesh.facets)),
    }

    if pair is None:
        labels = sorted(
            {
                str(facet.options.get(patch_key))
                for facet in mesh.facets.values()
                if isinstance(facet.options.get(patch_key), str)
            }
        )
        metrics["patch_labels"] = ",".join(labels)
        if len(labels) == 2:
            pair = (labels[0], labels[1])

    if pair is not None:
        c0 = _patch_centroid(mesh, patch_key=patch_key, label=pair[0])
        c1 = _patch_centroid(mesh, patch_key=patch_key, label=pair[1])
        metrics["patch0"] = pair[0]
        metrics["patch1"] = pair[1]
        metrics["patch0_centroid_x"] = float(c0[0])
        metrics["patch0_centroid_y"] = float(c0[1])
        metrics["patch0_centroid_z"] = float(c0[2])
        metrics["patch1_centroid_x"] = float(c1[0])
        metrics["patch1_centroid_y"] = float(c1[1])
        metrics["patch1_centroid_z"] = float(c1[2])
        metrics["L"] = _compute_separation(
            c0,
            c1,
            mode=separation,
            sphere_center=sphere_center,
            sphere_radius=sphere_radius,
        )
    else:
        metrics["L"] = float("nan")

    e_total, e_break = _energy_metrics(mesh)
    metrics["E_total"] = float(e_total)
    for name, val in e_break.items():
        metrics[f"E_{name}"] = float(val)

    metrics["area"] = float(mesh.compute_total_surface_area())
    metrics["volume"] = float(mesh.compute_total_volume())
    metrics["rg_surface"] = float(mesh.compute_surface_radius_of_gyration())
    metrics["min_edge_length"] = _min_edge_length(mesh)

    patch_lengths = patch_boundary_lengths(mesh, patch_key=patch_key)
    for label, length in patch_lengths.items():
        metrics[f"patch_boundary_length_{label}"] = float(length)

    if include_boundary_diagnostics:
        boundary_edges = find_boundary_edges(mesh)
        loops = extract_boundary_loops(mesh, boundary_edges)
        per_loop = boundary_geodesic_sum(mesh, loops)
        metrics["n_boundary_loops"] = int(len(loops))
        metrics["boundary_geodesic_sum_total"] = float(sum(per_loop.values()))
        for idx, val in per_loop.items():
            metrics[f"boundary_geodesic_sum_{idx}"] = float(val)

    return CaseResult(path=path, metrics=metrics)


def _write_results_csv(results: list[CaseResult], path: Path) -> None:
    rows = [r.metrics for r in results]
    keys: list[str] = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_results_json(results: list[CaseResult], path: Path) -> None:
    payload = [r.metrics for r in results]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _plot_series(
    outdir: Path,
    *,
    xs: np.ndarray,
    ys: dict[str, np.ndarray],
    xlabel: str,
    title: str,
    filename: str,
    ylabel: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for plotting (install the dev requirements)."
        ) from exc

    fig = plt.figure(figsize=(7, 4.2))
    ax = fig.add_subplot(111)
    for label, arr in ys.items():
        ax.plot(xs, arr, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(ys) > 1:
        ax.legend()
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / filename, dpi=200)


def _results_sorted_by_L(results: list[CaseResult]) -> list[CaseResult]:
    def _key(item: CaseResult) -> float:
        return float(item.metrics.get("L", float("nan")))

    return sorted(results, key=_key)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Output mesh files or directories containing them.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs/multidisk_report"),
        help="Directory to write CSV/JSON + plots (default: outputs/multidisk_report).",
    )
    parser.add_argument(
        "--patch-key",
        default="disk_patch",
        help="Facet option key used to label disk patches (default: disk_patch).",
    )
    parser.add_argument(
        "--pair",
        type=_parse_pair,
        default=None,
        help="Two patch labels to compare, as 'labelA,labelB'.",
    )
    parser.add_argument(
        "--separation",
        choices=["chord", "angle", "arc"],
        default="chord",
        help="Separation measure between patch centroids.",
    )
    parser.add_argument(
        "--sphere-center",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Sphere center for angle/arc separation (default: 0 0 0).",
    )
    parser.add_argument(
        "--sphere-radius",
        type=float,
        default=None,
        help="Sphere radius for arc-length separation (default: infer from centroids).",
    )
    parser.add_argument(
        "--no-boundary-diagnostics",
        action="store_true",
        help="Skip boundary-loop geodesic-curvature diagnostics.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    mesh_files = _collect_mesh_files(args.paths)
    if not mesh_files:
        raise SystemExit("No mesh files found.")

    sphere_center = np.array(args.sphere_center, dtype=float)
    include_boundary = not bool(args.no_boundary_diagnostics)

    results: list[CaseResult] = []
    for path in mesh_files:
        results.append(
            analyze_mesh(
                path,
                patch_key=args.patch_key,
                pair=args.pair,
                separation=args.separation,
                sphere_center=sphere_center,
                sphere_radius=args.sphere_radius,
                include_boundary_diagnostics=include_boundary,
            )
        )

    results = _results_sorted_by_L(results)
    outdir: Path = args.outdir
    _write_results_csv(results, outdir / "results.csv")
    _write_results_json(results, outdir / "results.json")

    L = np.array(
        [float(r.metrics.get("L", float("nan"))) for r in results], dtype=float
    )
    E = np.array(
        [float(r.metrics.get("E_total", float("nan"))) for r in results], dtype=float
    )
    if np.isfinite(L).any() and np.isfinite(E).any():
        _plot_series(
            outdir,
            xs=L,
            ys={"E_total": E},
            xlabel="L",
            ylabel="Energy",
            title="Energy vs separation",
            filename="energy_vs_L.png",
        )
        ref = float(E[np.nanargmax(L)])
        E_int = E - ref
        _plot_series(
            outdir,
            xs=L,
            ys={"E_int": E_int},
            xlabel="L",
            ylabel="Energy difference",
            title="Interaction energy (relative to max-L case)",
            filename="interaction_energy_vs_L.png",
        )

        area = np.array([float(r.metrics.get("area", float("nan"))) for r in results])
        volume = np.array(
            [float(r.metrics.get("volume", float("nan"))) for r in results]
        )
        rg = np.array(
            [float(r.metrics.get("rg_surface", float("nan"))) for r in results]
        )
        _plot_series(
            outdir,
            xs=L,
            ys={"area": area, "volume": volume, "rg_surface": rg},
            xlabel="L",
            ylabel="Observable",
            title="Shape observables vs separation",
            filename="observables_vs_L.png",
        )

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

#!/usr/bin/env python3
"""Sweep named near-edge interface profiles for theory parity."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.reproduce_theory_parity import (
    DEFAULT_MESH,
    DEFAULT_PROTOCOL,
    ROOT,
    _collect_report,
)
from tools.theory_parity_interface_profiles import build_profiled_fixture


def _report_fixture_path(mesh_path: Path) -> str:
    """Return a stable report-friendly fixture path."""
    try:
        return str(Path(mesh_path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(mesh_path).resolve())


DEFAULT_OUT = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "diagnostics"
    / "theory_parity_interface_sweep.yaml"
)
DEFAULT_CANDIDATES = (
    "coarse:base",
    "default_lo:profile",
    "default:profile",
    "default_hi:profile",
)


def parse_candidate(spec: str) -> dict[str, Any]:
    """Parse `label:base` or `label:profile` candidate specs."""
    text = str(spec).strip()
    parts = [part.strip() for part in text.split(":")]
    if len(parts) != 2 or not parts[0]:
        raise ValueError(f"invalid candidate spec: {spec}")
    mode = parts[1].lower()
    if mode not in {"base", "profile"}:
        raise ValueError(f"invalid candidate spec: {spec}")
    return {"label": parts[0], "mode": mode}


def _write_temp_fixture(doc: dict[str, Any], *, label: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{label}.yaml",
        prefix="theory_parity_interface_",
        delete=False,
        dir=str(ROOT / "tests" / "fixtures"),
        encoding="utf-8",
    )
    tmp_path = Path(handle.name)
    try:
        handle.write(yaml.safe_dump(doc, sort_keys=False))
    finally:
        handle.close()
    return tmp_path


def _build_profile_doc(
    *, base_doc: dict[str, Any], profile: str, lane: str
) -> dict[str, Any]:
    doc = build_profiled_fixture(base_doc=base_doc, profile=profile, lane=lane)
    gp = dict(doc.get("global_parameters") or {})
    gp["rim_slope_match_mode"] = "physical_edge_staggered_v1"
    gp["tilt_solver"] = "cg"
    gp["tilt_cg_max_iters"] = 120
    gp["tilt_mass_mode_in"] = "consistent"
    doc["global_parameters"] = gp
    constraints = [str(x) for x in (doc.get("constraint_modules") or [])]
    doc["constraint_modules"] = [
        x for x in constraints if x != "tilt_thetaB_boundary_in"
    ]
    return doc


def _score(report: dict[str, Any], *, target_ratio: float) -> float:
    ratios = report["metrics"]["tex_benchmark"]["ratios"]
    return float(sum(abs(float(ratios[key]) - float(target_ratio)) for key in ratios))


def run_candidate(
    *,
    base_doc: dict[str, Any],
    base_mesh: Path,
    candidate: dict[str, Any],
    protocol: tuple[str, ...],
    target_ratio: float,
) -> dict[str, Any]:
    """Run one candidate and return a compact scored summary."""
    label = str(candidate["label"])
    cleanup_path: Path | None = None
    mesh_path = Path(base_mesh)
    if candidate["mode"] == "profile":
        doc = _build_profile_doc(base_doc=base_doc, profile=label, lane=label)
        mesh_path = _write_temp_fixture(doc, label=label)
        cleanup_path = mesh_path
    try:
        t0 = time.perf_counter()
        report = _collect_report(mesh_path=mesh_path, protocol=protocol)
        runtime_s = time.perf_counter() - t0
    finally:
        if cleanup_path is not None and cleanup_path.exists():
            cleanup_path.unlink()

    metrics = report["metrics"]
    geom = metrics["diagnostics"]["outer_shell_geometry"]
    rim_radius = float(geom["rim_radius"])
    outer_radius = float(geom["outer_radius"])
    return {
        "label": label,
        "candidate": candidate,
        "fixture": report["meta"]["fixture"],
        "lane": report["meta"]["lane"],
        "runtime_s": float(runtime_s),
        "score": float(_score(report, target_ratio=target_ratio)),
        "thetaB_value": float(metrics["thetaB_value"]),
        "final_energy": float(metrics["final_energy"]),
        "legacy_ratios": {
            key: float(val) for key, val in metrics["legacy_anchor"]["ratios"].items()
        },
        "tex_ratios": {
            key: float(val) for key, val in metrics["tex_benchmark"]["ratios"].items()
        },
        "outer_split": {
            key: float(val)
            for key, val in metrics["diagnostics"]["outer_split"].items()
            if key
            in {
                "phi_mean",
                "t_in_mean",
                "t_out_mean",
                "theta_disk_mean",
                "phi_over_half_theta",
            }
        },
        "outer_shell_geometry": {
            "rim_radius": rim_radius,
            "outer_radius": outer_radius,
            "delta_r": float(outer_radius - rim_radius),
        },
    }


def rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank by closeness to TeX parity, then runtime, then label."""
    return sorted(
        rows,
        key=lambda row: (
            float(row["score"]),
            float(row["runtime_s"]),
            str(row["label"]),
        ),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--candidate",
        action="append",
        default=None,
        help="Candidate spec: `label:base` or `label:profile`.",
    )
    parser.add_argument(
        "--protocol",
        nargs="+",
        default=list(DEFAULT_PROTOCOL),
        help="Command sequence used to compute fixed-lane parity report.",
    )
    parser.add_argument("--target-ratio", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    mesh_path = Path(args.mesh)
    base_doc = yaml.safe_load(mesh_path.read_text(encoding="utf-8")) or {}
    protocol = tuple(str(cmd) for cmd in args.protocol)
    specs = list(args.candidate) if args.candidate else list(DEFAULT_CANDIDATES)
    candidates = [parse_candidate(spec) for spec in specs]
    rows = [
        run_candidate(
            base_doc=base_doc,
            base_mesh=mesh_path,
            candidate=candidate,
            protocol=protocol,
            target_ratio=float(args.target_ratio),
        )
        for candidate in candidates
    ]
    ranked = rank_rows(rows)
    out = {
        "meta": {
            "fixture": _report_fixture_path(mesh_path),
            "protocol": list(protocol),
            "format": "yaml",
            "target_ratio": float(args.target_ratio),
        },
        "rows": ranked,
        "summary": {
            "candidate_count": len(ranked),
            "best_label": ranked[0]["label"] if ranked else None,
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

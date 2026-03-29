#!/usr/bin/env python3
"""Compare theory-parity profiles at a fixed thetaB."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.executor import execute_command_line
from tools.reproduce_theory_parity import (
    DEFAULT_MESH,
    DEFAULT_PROTOCOL,
    ROOT,
    _activate_local_outer_shell_for_parity,
    _build_context,
    _collect_report_from_context,
    _stabilize_rim_radius_for_parity,
)
from tools.theory_parity_interface_profiles import build_profiled_fixture

DEFAULT_OUT = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "diagnostics"
    / "theory_parity_fixed_theta_compare.yaml"
)


def _report_fixture_path(mesh_path: Path) -> str:
    """Return a stable report-friendly fixture path."""
    try:
        return str(Path(mesh_path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(mesh_path).resolve())


def _write_temp_fixture(doc: dict[str, Any], *, label: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=f"_{label}.yaml",
        prefix="theory_parity_fixed_theta_",
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


def _run_fixed_theta_report(
    *,
    mesh_path: Path,
    protocol: tuple[str, ...],
    theta_value: float,
) -> dict[str, Any]:
    ctx = _build_context(mesh_path)
    gp = ctx.mesh.global_parameters
    gp.set("tilt_thetaB_optimize", False)
    gp.set("tilt_thetaB_value", float(theta_value))
    _stabilize_rim_radius_for_parity(ctx.mesh)
    _activate_local_outer_shell_for_parity(ctx.mesh)
    for cmd in protocol:
        execute_command_line(ctx, cmd)
        _stabilize_rim_radius_for_parity(ctx.mesh)
        _activate_local_outer_shell_for_parity(ctx.mesh)
    return _collect_report_from_context(ctx=ctx, mesh_path=mesh_path, protocol=protocol)


def _summarize_row(
    *, label: str, fixture: str, report: dict[str, Any]
) -> dict[str, Any]:
    metrics = report["metrics"]
    geom = metrics["diagnostics"]["outer_shell_geometry"]
    rim_radius = float(geom["rim_radius"])
    outer_radius = float(geom["outer_radius"])
    return {
        "label": str(label),
        "fixture": str(fixture),
        "lane": report["meta"]["lane"],
        "thetaB_value": float(metrics["thetaB_value"]),
        "final_energy": float(metrics["final_energy"]),
        "breakdown": {k: float(v) for k, v in metrics["breakdown"].items()},
        "reduced_terms": {k: float(v) for k, v in metrics["reduced_terms"].items()},
        "tex_ratios": {
            k: float(v) for k, v in metrics["tex_benchmark"]["ratios"].items()
        },
        "outer_split": {
            k: float(v)
            for k, v in metrics["diagnostics"]["outer_split"].items()
            if k
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, default=DEFAULT_MESH)
    parser.add_argument("--theta", type=float, default=7.0 / 38.0)
    parser.add_argument(
        "--profile",
        action="append",
        default=None,
        help="Named interface profile to compare (repeat to add more).",
    )
    parser.add_argument(
        "--protocol",
        nargs="+",
        default=list(DEFAULT_PROTOCOL),
        help="Command sequence used before collecting the fixed-theta report.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    mesh_path = Path(args.mesh)
    base_doc = yaml.safe_load(mesh_path.read_text(encoding="utf-8")) or {}
    protocol = tuple(str(cmd) for cmd in args.protocol)
    profiles = (
        list(args.profile)
        if args.profile
        else [
            "coarse",
            "default_lo",
            "default",
            "default_hi",
        ]
    )

    rows: list[dict[str, Any]] = []
    for profile in profiles:
        label = str(profile)
        if label == "coarse":
            current_mesh = mesh_path
            cleanup_path = None
            fixture_label = _report_fixture_path(mesh_path)
        else:
            doc = _build_profile_doc(
                base_doc=base_doc,
                profile=label,
                lane=f"fixed_theta_{label}",
            )
            cleanup_path = _write_temp_fixture(doc, label=label)
            current_mesh = cleanup_path
            fixture_label = _report_fixture_path(current_mesh)
        try:
            report = _run_fixed_theta_report(
                mesh_path=current_mesh,
                protocol=protocol,
                theta_value=float(args.theta),
            )
        finally:
            if cleanup_path is not None and cleanup_path.exists():
                cleanup_path.unlink()
        rows.append(_summarize_row(label=label, fixture=fixture_label, report=report))

    out = {
        "meta": {
            "base_fixture": _report_fixture_path(mesh_path),
            "theta_value": float(args.theta),
            "protocol": list(protocol),
            "format": "yaml",
        },
        "rows": rows,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    print(f"wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

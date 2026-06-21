#!/usr/bin/env python3
"""Broad parity diagnostics for interface elastic-response investigations."""

from __future__ import annotations

import argparse
import copy
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.constraints.local_interface_shells import (  # noqa: E402
    build_local_interface_shell_data,
)
from modules.energy.leaflet_presence import (  # noqa: E402
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)
from tools.diagnostics.free_disk_profile_protocol import (  # noqa: E402
    _bending_tilt_leaflet_region_split,
    _tilt_in_region_split,
    _tilt_out_region_split,
)
from tools.diagnostics.parity_acceptance_triage import (  # noqa: E402
    DEFAULT_FIXTURE,
    DEFAULT_PROTOCOL,
    FIXED_THETA_SWEEP_VALUES,
    GHOST_FIXTURE,
    _as_float,
    _interface_summary,
    _reduced_terms_summary,
    _run_fixed_theta_sweep,
    _runtime_breakdown_summary,
)
from tools.diagnostics.scaffold_energy_imbalance_audit import (  # noqa: E402
    _base_term_summary_for_fixture,
)
from tools.diagnostics.thetaB_normalization_audit import (  # noqa: E402
    summarize_fixed_theta_sweep,
)
from tools.diagnostics.utils import (  # noqa: E402
    radial_projection,
    row_region_mask_dict,
    triangle_region_masks,
)
from tools.reproduce_theory_parity import (  # noqa: E402
    _build_context,
    _collect_report_from_context,
    _run_protocol_with_parity_activation,
)

ROOT = Path(__file__).resolve().parents[2]
FULL_COUPLING_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_physical_edge_full_coupling_v1.yaml"
)
FULL_COUPLING_TRACE_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_physical_edge_full_coupling_trace_eps005_v1.yaml"
)


def _write_temp_fixture(doc: dict[str, Any], directory: Path, label: str) -> Path:
    path = directory / f"{label}.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def _variant_specs() -> list[dict[str, Any]]:
    return [
        {
            "label": "ghost",
            "base_fixture": GHOST_FIXTURE,
            "overrides": {},
            "family": "ghost",
        },
        {
            "label": "default_current",
            "base_fixture": DEFAULT_FIXTURE,
            "overrides": {},
            "family": "default",
        },
        {
            "label": "full_coupling_trace",
            "base_fixture": FULL_COUPLING_TRACE_FIXTURE,
            "overrides": {},
            "family": "full_physics",
        },
        {
            "label": "full_coupling",
            "base_fixture": FULL_COUPLING_FIXTURE,
            "overrides": {},
            "family": "full_physics",
        },
        {
            "label": "default_no_outer_absence",
            "base_fixture": DEFAULT_FIXTURE,
            "overrides": {"leaflet_out_absent_presets": []},
            "family": "default",
        },
    ]


def _build_variant_doc(base_fixture: Path, overrides: dict[str, Any]) -> dict[str, Any]:
    doc = yaml.safe_load(base_fixture.read_text(encoding="utf-8")) or {}
    gp = dict(doc.get("global_parameters") or {})
    for key, value in (overrides or {}).items():
        gp[key] = copy.deepcopy(value)
    doc["global_parameters"] = gp
    return doc


def _run_context_report(
    *, mesh_path: Path, protocol: tuple[str, ...]
) -> tuple[Any, dict[str, Any]]:
    ctx = _build_context(mesh_path)
    _run_protocol_with_parity_activation(ctx, protocol=protocol)
    report = _collect_report_from_context(
        ctx=ctx,
        mesh_path=mesh_path,
        protocol=protocol,
    )
    return ctx, report


def _mean_and_max(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"mean": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _field_stats_by_region(mesh) -> dict[str, Any]:
    masks = row_region_mask_dict(mesh)
    tin = np.asarray(mesh.tilts_in_view(), dtype=float)
    tout = np.asarray(mesh.tilts_out_view(), dtype=float)
    tin_norm = np.linalg.norm(tin, axis=1)
    tout_norm = np.linalg.norm(tout, axis=1)
    tin_rad = radial_projection(mesh, tin)
    tout_rad = radial_projection(mesh, tout)
    out: dict[str, Any] = {}
    for region, mask in masks.items():
        out[region] = {
            "count": int(np.sum(mask)),
            "tilt_in_norm": _mean_and_max(tin_norm[mask]),
            "tilt_out_norm": _mean_and_max(tout_norm[mask]),
            "tilt_in_radial": _mean_and_max(np.abs(tin_rad[mask])),
            "tilt_out_radial": _mean_and_max(np.abs(tout_rad[mask])),
        }
    return out


def _triangle_count_by_region(mesh, tri_rows: np.ndarray) -> dict[str, int]:
    if tri_rows.size == 0:
        return {
            "disk_core": 0,
            "disk_rim": 0,
            "rim_outer": 0,
            "outer_support_band": 0,
            "outer_far": 0,
            "outer_membrane": 0,
        }
    masks = triangle_region_masks(mesh, tri_rows)
    return {key: int(np.sum(mask)) for key, mask in masks.items()}


def _leaflet_participation(mesh, *, leaflet: str) -> dict[str, Any]:
    mesh.build_position_cache()
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return {
            "absent_vertex_count": 0,
            "triangle_counts": {
                "total": 0,
                "kept": 0,
                "mixed_absent_present": 0,
                "fully_absent": 0,
                "fully_present": 0,
            },
            "triangle_regions_all": {},
            "triangle_regions_kept": {},
        }

    tri_rows = np.asarray(tri_rows, dtype=np.int32)
    absent = leaflet_absent_vertex_mask(mesh, mesh.global_parameters, leaflet=leaflet)
    tri_keep = leaflet_present_triangle_mask(mesh, tri_rows, absent_vertex_mask=absent)
    tri_abs = absent[tri_rows]
    kept_rows = tri_rows[tri_keep] if tri_keep.size else tri_rows

    out = {
        "absent_vertex_count": int(np.sum(absent)),
        "absent_vertex_rows_by_region": {
            region: int(np.sum(absent & mask))
            for region, mask in row_region_mask_dict(mesh).items()
        },
        "triangle_counts": {
            "total": int(len(tri_rows)),
            "kept": int(np.sum(tri_keep)) if tri_keep.size else int(len(tri_rows)),
            "mixed_absent_present": int(
                np.sum(np.any(tri_abs, axis=1) & np.any(~tri_abs, axis=1))
            ),
            "fully_absent": int(np.sum(np.all(tri_abs, axis=1))),
            "fully_present": int(np.sum(np.all(~tri_abs, axis=1))),
        },
        "triangle_regions_all": _triangle_count_by_region(mesh, tri_rows),
        "triangle_regions_kept": _triangle_count_by_region(mesh, kept_rows),
    }

    try:
        shell_data = build_local_interface_shell_data(
            mesh, positions=mesh.positions_view()
        )
    except AssertionError:
        return out

    out["shell_rows"] = {
        "disk": {
            "count": int(len(shell_data.disk_rows)),
            "absent": int(np.sum(absent[np.asarray(shell_data.disk_rows, dtype=int)])),
        },
        "rim": {
            "count": int(len(shell_data.rim_rows)),
            "absent": int(np.sum(absent[np.asarray(shell_data.rim_rows, dtype=int)])),
        },
        "outer": {
            "count": int(len(shell_data.outer_rows)),
            "absent": int(np.sum(absent[np.asarray(shell_data.outer_rows, dtype=int)])),
        },
    }
    return out


def _geometry_summary(mesh) -> dict[str, Any]:
    mesh.build_position_cache()
    tri_rows, _ = mesh.triangle_row_cache()
    summary = {
        "n_vertices": int(len(mesh.vertex_ids)),
        "n_triangles": int(0 if tri_rows is None else len(tri_rows)),
        "row_region_counts": {
            key: int(np.sum(mask)) for key, mask in row_region_mask_dict(mesh).items()
        },
    }
    try:
        shell = build_local_interface_shell_data(mesh, positions=mesh.positions_view())
    except AssertionError:
        return summary
    summary["interface_shell"] = {
        "disk_radius": float(shell.disk_radius),
        "rim_radius": float(shell.rim_radius),
        "outer_radius": float(shell.outer_radius),
        "delta_r": float(shell.outer_radius - shell.rim_radius),
        "disk_rows": int(len(shell.disk_rows)),
        "rim_rows": int(len(shell.rim_rows)),
        "outer_rows": int(len(shell.outer_rows)),
    }
    return summary


def _region_energy_splits(mesh) -> dict[str, Any]:
    return {
        "tilt_in": _tilt_in_region_split(mesh),
        "tilt_out": _tilt_out_region_split(mesh),
        "bending_tilt_in": _bending_tilt_leaflet_region_split(mesh, leaflet="in"),
        "bending_tilt_out": _bending_tilt_leaflet_region_split(mesh, leaflet="out"),
    }


def _scan_tail(report: dict[str, Any], n: int = 5) -> list[dict[str, Any]]:
    scans = (
        report.get("metrics", {}).get("diagnostics", {}).get("thetaB_scan_trace", [])
    )
    if not isinstance(scans, list):
        return []
    return scans[-n:]


def _optimized_case_summary(
    *, label: str, mesh_path: Path, protocol: tuple[str, ...]
) -> dict[str, Any]:
    ctx, report = _run_context_report(mesh_path=mesh_path, protocol=protocol)
    mesh = ctx.mesh
    breakdown = _runtime_breakdown_summary(report)
    reduced = _reduced_terms_summary(report)
    tex_ratio = _as_float(
        report.get("metrics", {})
        .get("tex_benchmark", {})
        .get("ratios", {})
        .get("total_ratio")
    )
    return {
        "label": label,
        "thetaB_value": _as_float(report.get("metrics", {}).get("thetaB_value")),
        "tex_total_ratio": tex_ratio,
        "tex_ratio_summary": {
            "elastic_ratio": _as_float(
                report.get("metrics", {})
                .get("tex_benchmark", {})
                .get("ratios", {})
                .get("elastic_ratio")
            ),
            "contact_ratio": _as_float(
                report.get("metrics", {})
                .get("tex_benchmark", {})
                .get("ratios", {})
                .get("contact_ratio")
            ),
        },
        "model_intent": str(report.get("metrics", {}).get("model_intent") or ""),
        "reference_mode": str(report.get("metrics", {}).get("reference_mode") or ""),
        "thetaB_scan_count": len(
            report.get("metrics", {})
            .get("diagnostics", {})
            .get("thetaB_scan_trace", [])
            or []
        ),
        "thetaB_scan_tail": _scan_tail(report),
        "energy_breakdown": breakdown,
        "reduced_terms": reduced,
        "interface_summary": _interface_summary(report),
        "base_term_summary": _base_term_summary_for_fixture(mesh_path, label),
        "geometry_summary": _geometry_summary(mesh),
        "field_stats_by_region": _field_stats_by_region(mesh),
        "region_energy_splits": _region_energy_splits(mesh),
        "outer_leaflet_participation": _leaflet_participation(mesh, leaflet="out"),
        "inner_leaflet_participation": _leaflet_participation(mesh, leaflet="in"),
    }


def _fixed_theta_focus_rows(
    sweep: dict[str, Any], labels: tuple[float, ...]
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for theta in labels:
        key = f"{theta:.2f}"
        row = sweep.get(key, {})
        out[key] = {
            "thetaB_value": _as_float(row.get("thetaB_value")),
            "energy_breakdown": row.get("energy_breakdown", {}),
            "reduced_terms": row.get("reduced_terms", {}),
            "interface_summary": row.get("interface_summary", {}),
            "base_term_summary": row.get("base_term_summary", {}),
        }
    return out


def _comparison_rows(
    optimized_cases: list[dict[str, Any]],
    fixed_theta_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    fixed_map = {row["label"]: row for row in fixed_theta_cases}
    out: list[dict[str, Any]] = []
    for opt in optimized_cases:
        fixed = fixed_map.get(opt["label"], {})
        summary = fixed.get("summary", {})
        module_fits = summary.get("module_fits", {})
        out.append(
            {
                "label": opt["label"],
                "optimized_thetaB": _as_float(opt.get("thetaB_value")),
                "optimized_tex_total_ratio": _as_float(opt.get("tex_total_ratio")),
                "fixed_elastic_A_ratio": _as_float(
                    summary.get("ratios", {}).get("elastic_A")
                ),
                "fixed_contact_B_ratio": _as_float(
                    summary.get("ratios", {}).get("contact_B")
                ),
                "fixed_theta_min_ratio": _as_float(
                    summary.get("ratios", {}).get("theta_min")
                ),
                "tilt_out_quadratic": _as_float(
                    module_fits.get("tilt_out", {}).get("quadratic")
                ),
                "bending_tilt_out_quadratic": _as_float(
                    module_fits.get("bending_tilt_out", {}).get("quadratic")
                ),
            }
        )
    return out


def _top_observations(report: dict[str, Any]) -> list[str]:
    rows = report.get("comparison_matrix", [])
    if not rows:
        return []
    by_label = {str(row["label"]): row for row in rows}
    observations: list[str] = []
    default = by_label.get("default_current")
    no_absent = by_label.get("default_no_outer_absence")
    ghost = by_label.get("ghost")
    if default and no_absent:
        observations.append(
            "Removing outer absence entirely increases fixed-theta outer elastic terms relative to the current default lane, which keeps the absence mask path a live contributor."
        )
        if _as_float(default.get("fixed_elastic_A_ratio")) < _as_float(
            no_absent.get("fixed_elastic_A_ratio")
        ):
            observations.append(
                "The current default lane still underperforms the no-absence control in total elastic_A, so restored outer participation is not yet sufficient."
            )
    if default and ghost:
        if (
            _as_float(default.get("tilt_out_quadratic")) > 0.0
            and _as_float(ghost.get("tilt_out_quadratic")) > 0.0
        ):
            observations.append(
                "Both ghost and default lanes now show nonzero outer fixed-theta tilt response, which distinguishes residual parity loss from the earlier total outer-dropout failure."
            )
    full = by_label.get("full_coupling")
    full_trace = by_label.get("full_coupling_trace")
    if default and (full or full_trace):
        observations.append(
            "The full-coupling lane is reported separately from the analytical default lane so outer-shell response can be compared without redefining Tex-parity acceptance."
        )
    if full_trace and full:
        observations.append(
            "The explicit-trace current-geometry lane restores the ghost-like outer branch, while the no-trace current-geometry lane remains a control for geometry-limited response."
        )
    return observations


def _full_physics_lane_matrix(
    optimized_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in optimized_cases:
        if str(row.get("label")) not in {
            "default_current",
            "full_coupling",
            "full_coupling_trace",
            "ghost",
        }:
            continue
        base_out = row.get("base_term_summary", {}).get("leaflets", {}).get("out", {})
        rows.append(
            {
                "label": str(row.get("label")),
                "model_intent": str(row.get("model_intent") or ""),
                "reference_mode": str(row.get("reference_mode") or ""),
                "thetaB_value": _as_float(row.get("thetaB_value")),
                "tex_total_ratio": _as_float(row.get("tex_total_ratio")),
                "tex_elastic_ratio": _as_float(
                    row.get("tex_ratio_summary", {}).get("elastic_ratio")
                ),
                "tex_contact_ratio": _as_float(
                    row.get("tex_ratio_summary", {}).get("contact_ratio")
                ),
                "base_energy_out": _as_float(base_out.get("base_energy")),
                "div_energy_out": _as_float(base_out.get("div_energy")),
                "cross_energy_out": _as_float(base_out.get("cross_energy")),
                "direct_t_out": _as_float(
                    row.get("interface_summary", {}).get("direct_t_out")
                ),
                "direct_phi": _as_float(
                    row.get("interface_summary", {}).get("direct_phi")
                ),
            }
        )
    return rows


def run_diagnostic(*, protocol: tuple[str, ...], mode: str = "run") -> dict[str, Any]:
    specs = _variant_specs()
    if mode == "schema":
        return {
            "meta": {"mode": "schema", "protocol": list(protocol)},
            "variants": [spec["label"] for spec in specs],
            "sections": [
                "optimized_cases",
                "fixed_theta_cases",
                "comparison_matrix",
                "full_physics_lane_matrix",
                "observations",
            ],
        }

    optimized_cases: list[dict[str, Any]] = []
    fixed_theta_cases: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="parity_broad_diag_") as tmp:
        tmpdir = Path(tmp)
        for spec in specs:
            doc = _build_variant_doc(spec["base_fixture"], spec["overrides"])
            fixture_path = _write_temp_fixture(doc, tmpdir, spec["label"])
            optimized_cases.append(
                _optimized_case_summary(
                    label=spec["label"], mesh_path=fixture_path, protocol=protocol
                )
            )
            sweep = _run_fixed_theta_sweep(
                base_fixture=fixture_path,
                label=spec["label"],
                tmpdir=tmpdir,
            )
            fixed_theta_cases.append(
                {
                    "label": spec["label"],
                    "summary": summarize_fixed_theta_sweep(sweep, fixture=fixture_path),
                    "focus_rows": _fixed_theta_focus_rows(sweep, labels=(0.21, 0.30)),
                }
            )

    report = {
        "meta": {
            "mode": "run",
            "protocol": list(protocol),
            "theta_values": [float(theta) for theta in FIXED_THETA_SWEEP_VALUES],
        },
        "optimized_cases": optimized_cases,
        "fixed_theta_cases": fixed_theta_cases,
        "comparison_matrix": _comparison_rows(optimized_cases, fixed_theta_cases),
        "full_physics_lane_matrix": _full_physics_lane_matrix(optimized_cases),
    }
    report["observations"] = _top_observations(report)
    return report


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def render_markdown_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    meta = report.get("meta", {})
    lines.append("# Parity Broad Diagnostic")
    lines.append("")
    lines.append(f"- Mode: `{meta.get('mode', 'unknown')}`")
    lines.append(f"- Protocol: `{' '.join(str(x) for x in meta.get('protocol', []))}`")
    lines.append(
        f"- Fixed theta values: `{', '.join(_fmt(x, digits=2) for x in meta.get('theta_values', []))}`"
    )
    lines.append("")
    lines.append("## Comparison Matrix")
    lines.append("")
    lines.append(
        "| Variant | thetaB | tex ratio | elastic_A ratio | contact_B ratio | tilt_out quad | bending_tilt_out quad |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report.get("comparison_matrix", []):
        lines.append(
            f"| `{row['label']}` | {_fmt(row.get('optimized_thetaB'))} | {_fmt(row.get('optimized_tex_total_ratio'))} | {_fmt(row.get('fixed_elastic_A_ratio'))} | {_fmt(row.get('fixed_contact_B_ratio'))} | {_fmt(row.get('tilt_out_quadratic'))} | {_fmt(row.get('bending_tilt_out_quadratic'))} |"
        )
    full_rows = report.get("full_physics_lane_matrix", [])
    if full_rows:
        lines.append("")
        lines.append("## Full-Physics Lane Matrix")
        lines.append("")
        lines.append(
            "| Lane | intent | ref mode | thetaB | tex ratio | elastic ratio | contact ratio | base_out | div_out | cross_out | direct_t_out | direct_phi |"
        )
        lines.append(
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for row in full_rows:
            lines.append(
                f"| `{row['label']}` | `{row.get('model_intent')}` | `{row.get('reference_mode')}` | {_fmt(row.get('thetaB_value'))} | {_fmt(row.get('tex_total_ratio'))} | {_fmt(row.get('tex_elastic_ratio'))} | {_fmt(row.get('tex_contact_ratio'))} | {_fmt(row.get('base_energy_out'))} | {_fmt(row.get('div_energy_out'))} | {_fmt(row.get('cross_energy_out'))} | {_fmt(row.get('direct_t_out'))} | {_fmt(row.get('direct_phi'))} |"
            )
    observations = report.get("observations", [])
    if observations:
        lines.append("")
        lines.append("## Observations")
        lines.append("")
        for row in observations:
            lines.append(f"- {row}")
    lines.append("")
    lines.append("## Variant Snapshots")
    lines.append("")
    for row in report.get("optimized_cases", []):
        breakdown = row.get("energy_breakdown", {})
        shell = row.get("outer_leaflet_participation", {}).get("shell_rows", {})
        lines.append(f"### `{row['label']}`")
        lines.append("")
        lines.append(
            f"- thetaB `{_fmt(row.get('thetaB_value'))}`, tex ratio `{_fmt(row.get('tex_total_ratio'))}`, scan count `{int(row.get('thetaB_scan_count', 0))}`"
        )
        lines.append(
            f"- outer terms: `tilt_out={_fmt(breakdown.get('tilt_out'))}`, `bending_tilt_out={_fmt(breakdown.get('bending_tilt_out'))}`"
        )
        if shell:
            lines.append(
                f"- shell absence: `disk={shell['disk']['absent']}/{shell['disk']['count']}`, `rim={shell['rim']['absent']}/{shell['rim']['count']}`, `outer={shell['outer']['absent']}/{shell['outer']['count']}`"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "schema"), default="run")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument(
        "--protocol",
        nargs="*",
        default=list(DEFAULT_PROTOCOL),
        help="Parity protocol commands passed to reproduce_theory_parity helpers.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report = run_diagnostic(
        protocol=tuple(str(x) for x in args.protocol), mode=args.mode
    )
    yaml_text = yaml.safe_dump(report, sort_keys=False)
    md_text = render_markdown_report(report)

    if args.out is None:
        print(yaml_text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(yaml_text, encoding="utf-8")
        print(f"wrote: {args.out}")

    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(md_text, encoding="utf-8")
        print(f"wrote: {args.report_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

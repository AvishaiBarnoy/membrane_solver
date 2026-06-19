from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.diagnostics.scaffold_energy_imbalance_audit import (  # noqa: E402
    _base_term_summary_for_fixture,
)
from tools.diagnostics.thetaB_cadence_relaxation_audit import (  # noqa: E402
    _collect_live_summary,
    _one_step_shell_update_summary,
    _outer_bending_tilt_gradient_components,
    _outer_shell_rows,
    _run_protocol_summary,
    _shell_vector_summary,
    _write_temp_fixture,
)
from tools.reproduce_theory_parity import (  # noqa: E402
    DEFAULT_PROTOCOL,
    _build_context,
    _collect_report_from_context,
    _run_protocol_with_parity_activation,
)
from tools.theory_parity_interface_profiles import (  # noqa: E402
    build_full_physics_trace_fixture,
    build_profiled_fixture,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE = (
    ROOT
    / "tests"
    / "fixtures"
    / "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml"
)
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


def _run_report(mesh_path: Path) -> dict:
    ctx = _build_context(mesh_path)
    _run_protocol_with_parity_activation(ctx, protocol=tuple(DEFAULT_PROTOCOL))
    return _collect_report_from_context(
        ctx=ctx, mesh_path=mesh_path, protocol=tuple(DEFAULT_PROTOCOL)
    )


def _full_coupling_shell_metrics(mesh_path: Path) -> dict[str, float]:
    ctx, _summary = _run_protocol_summary(
        mesh_path=mesh_path, protocol=tuple(DEFAULT_PROTOCOL)
    )
    shell_rows = _outer_shell_rows(ctx.mesh)
    coupling = _outer_bending_tilt_gradient_components(
        ctx=ctx, div_term_sign=1.0, pullback_sign=1.0
    )
    return {
        "combined_shell_gradient": float(
            _shell_vector_summary(ctx.mesh, coupling["combined_gradient"], shell_rows)[
                "norm"
            ]
        ),
        "first_shell_update": float(
            _one_step_shell_update_summary(
                ctx=ctx,
                theta=float(
                    _collect_live_summary(
                        ctx=ctx, mesh_path=mesh_path, protocol=tuple(DEFAULT_PROTOCOL)
                    )["thetaB_value"]
                    or 0.0
                ),
            )["norm"]
        ),
    }


def test_full_coupling_fixture_reports_lane_and_reference_mode() -> None:
    report = _run_report(FULL_COUPLING_FIXTURE)
    assert report["meta"]["lane"] == "physical_edge_full_coupling_v1"
    assert report["metrics"]["model_intent"] == "full_physics_candidate"
    assert report["metrics"]["reference_mode"] == "current_geometry"


def test_full_coupling_trace_fixture_reports_lane_and_reference_mode() -> None:
    report = _run_report(FULL_COUPLING_TRACE_FIXTURE)
    assert report["meta"]["lane"] == "physical_edge_full_coupling_trace_eps005_v1"
    assert report["metrics"]["model_intent"] == "full_physics_candidate"
    assert report["metrics"]["reference_mode"] == "current_geometry"


def test_full_coupling_keeps_existing_default_lane_unchanged() -> None:
    report = _run_report(DEFAULT_FIXTURE)
    assert float(report["metrics"]["thetaB_value"]) == pytest.approx(0.18, abs=1.0e-9)
    assert float(report["metrics"]["tex_benchmark"]["ratios"]["total_ratio"]) == (
        pytest.approx(1.0063565852776968, abs=1.0e-6)
    )
    assert report["metrics"]["model_intent"] == "analytical_parity"
    assert report["metrics"]["reference_mode"] == "flat_reference_zero_j0"


def test_full_coupling_outer_shell_base_term_is_nonzero() -> None:
    summary = _base_term_summary_for_fixture(FULL_COUPLING_FIXTURE, "full_coupling")
    out = summary["leaflets"]["out"]
    assert out["available"] is True
    assert float(out["base_energy"]) > 1.0e-6
    assert out["config"]["base_term_reference_mode"] == "current_geometry"


def test_full_coupling_trace_outer_shell_base_term_is_nonzero() -> None:
    summary = _base_term_summary_for_fixture(
        FULL_COUPLING_TRACE_FIXTURE, "full_coupling_trace"
    )
    out = summary["leaflets"]["out"]
    assert out["available"] is True
    assert float(out["base_energy"]) > 1.0e-6
    assert out["config"]["base_term_reference_mode"] == "current_geometry"


def test_full_coupling_combined_shell_gradient_is_measurable_and_distinct() -> None:
    default_metrics = _full_coupling_shell_metrics(DEFAULT_FIXTURE)
    full_metrics = _full_coupling_shell_metrics(FULL_COUPLING_FIXTURE)
    assert full_metrics["combined_shell_gradient"] > 0.0
    assert default_metrics["combined_shell_gradient"] > 0.0
    assert (
        abs(
            full_metrics["combined_shell_gradient"]
            - default_metrics["combined_shell_gradient"]
        )
        > 1.0e-6
    )


def test_full_coupling_first_shell_update_is_finite_and_nonzero() -> None:
    metrics = _full_coupling_shell_metrics(FULL_COUPLING_FIXTURE)
    assert metrics["first_shell_update"] > 0.0


def test_full_coupling_trace_shell_response_exceeds_no_trace_control() -> None:
    control_metrics = _full_coupling_shell_metrics(FULL_COUPLING_FIXTURE)
    trace_metrics = _full_coupling_shell_metrics(FULL_COUPLING_TRACE_FIXTURE)

    assert (
        trace_metrics["combined_shell_gradient"]
        > control_metrics["combined_shell_gradient"]
    )
    assert trace_metrics["first_shell_update"] > control_metrics["first_shell_update"]


def test_full_coupling_trace_reports_explicit_trace_and_direct_outer_response() -> None:
    report = _run_report(FULL_COUPLING_TRACE_FIXTURE)
    traces = report["metrics"]["diagnostics"]["interface_traces_at_R"]
    shell = report["metrics"]["diagnostics"]["interface_shell_at_R_plus_epsilon"]
    primary = report["metrics"]["diagnostics"]["interface_primary_readout"]

    assert primary["source"] == "direct_trace_layer"
    assert shell["shell_source"] == "explicit_trace_layer"
    assert float(traces["outer_t_out_trace_at_R_plus"]) > 0.1
    assert float(traces["phi_trace_at_R_plus"]) > 0.1
    assert float(report["metrics"]["thetaB_value"]) == pytest.approx(0.3, abs=1.0e-9)


def test_full_coupling_trace_builder_supports_convergence_epsilons() -> None:
    base_doc = yaml.safe_load(DEFAULT_FIXTURE.read_text(encoding="utf-8")) or {}
    for epsilon in (0.0025, 0.005, 0.01):
        doc = build_full_physics_trace_fixture(
            base_doc=base_doc,
            lane=f"physical_edge_full_coupling_trace_eps{str(epsilon).replace('.', '')}_test",
            trace_radius=(7.0 / 15.0) + float(epsilon),
            planar_geometry=False,
        )
        radii = {
            round((float(v[0]) ** 2 + float(v[1]) ** 2) ** 0.5, 12)
            for v in doc["vertices"]
        }
        assert round((7.0 / 15.0) + float(epsilon), 12) in radii
        assert (
            doc["global_parameters"]["bending_tilt_base_term_reference_mode"]
            == "current_geometry"
        )


def test_full_coupling_shell_rows_are_present_free_and_area_active() -> None:
    ctx, summary = _run_protocol_summary(
        mesh_path=FULL_COUPLING_FIXTURE, protocol=tuple(DEFAULT_PROTOCOL)
    )
    participation = summary["leaflet_relaxation_stats"]
    shell_rows = _outer_shell_rows(ctx.mesh)
    assert len(shell_rows) > 0
    assert int(participation.get("outer_shell_row_count", 0)) > 0
    assert int(participation.get("active_outer_area_rows", 0)) > 0
    assert int(participation.get("free_rows_out", 0)) > 0


def test_full_coupling_trace_direct_outer_response_exceeds_default_lane() -> None:
    default_report = _run_report(DEFAULT_FIXTURE)
    trace_report = _run_report(FULL_COUPLING_TRACE_FIXTURE)
    default_primary = default_report["metrics"]["diagnostics"][
        "interface_primary_readout"
    ]
    trace_primary = trace_report["metrics"]["diagnostics"]["interface_primary_readout"]

    assert default_primary["source"] == "extrapolated_trace"
    assert trace_primary["source"] == "direct_trace_layer"
    assert float(trace_primary["t_out"]) > float(default_primary["t_out"])
    assert float(trace_primary["phi"]) > float(default_primary["phi"])


def test_full_coupling_resolution_reports_finite_metrics() -> None:
    base_doc = (
        yaml.safe_load(FULL_COUPLING_TRACE_FIXTURE.read_text(encoding="utf-8")) or {}
    )
    rows = []
    with tempfile.TemporaryDirectory(prefix="full_coupling_resolution_") as tmp:
        tmpdir = Path(tmp)
        for profile in ("default_lo", "default", "default_hi"):
            doc = build_profiled_fixture(
                base_doc=base_doc,
                profile=profile,
                lane=f"physical_edge_full_coupling_trace_{profile}",
                base_term_reference_mode="current_geometry",
            )
            path = _write_temp_fixture(doc, tmpdir, profile)
            rows.append(_full_coupling_shell_metrics(path))
    gradients = [row["combined_shell_gradient"] for row in rows]
    updates = [row["first_shell_update"] for row in rows]
    assert all(val >= 0.0 for val in gradients)
    assert all(val >= 0.0 for val in updates)
    assert any(val > 0.0 for val in gradients)

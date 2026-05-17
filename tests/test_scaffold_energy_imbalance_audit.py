from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from geometry.geom_io import load_data, parse_geometry
from tools.diagnostics.scaffold_energy_imbalance_audit import (
    DEFAULT_FIXTURE,
    QUICK_PROTOCOL,
    _bending_tilt_base_term_audit,
    _mesh_topology_audit,
    run_audit,
)
from tools.reproduce_theory_parity import _build_context

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "diagnostics" / "scaffold_energy_imbalance_audit.py"


def test_scaffold_mesh_row_cache_consistency_for_gapfill_fixture() -> None:
    mesh = parse_geometry(load_data(str(DEFAULT_FIXTURE)))
    topo = _mesh_topology_audit(mesh)

    assert topo["n_vertices"] == len(mesh.vertex_ids)
    assert topo["n_triangle_rows"] == len(mesh.triangle_row_cache()[0])
    assert topo["negative_or_zero_area_count"] == 0
    assert topo["role_counts"]["trace_shell"] == 12
    assert topo["role_counts"]["support_shell_1"] == 12
    assert topo["role_counts"]["release_ring"] == 12


def test_scaffold_energy_imbalance_audit_emits_core_sections() -> None:
    audit = run_audit(
        mesh_path=DEFAULT_FIXTURE,
        protocol=QUICK_PROTOCOL,
        resolution_mode="none",
    )

    for section in (
        "mesh_topology",
        "refinement_trace",
        "module_energy_audit",
        "interface_target_audit",
        "constraint_audit",
        "coupled_stationarity_audit",
        "elastic_magnitude_audit",
        "bending_tilt_base_term_audit",
        "base_term_fixture_comparison",
        "constrained_gradient_audit",
        "protocol_snapshot_audit",
        "energy_normalization_audit",
        "bulk_boundary_split",
        "resolution_matrix",
        "cadence_variants",
        "advanced_flags",
        "parity_summary",
    ):
        assert section in audit

    module_audit = audit["module_energy_audit"]
    assert module_audit["boundary_rows"] == 12
    assert set(module_audit["modules"]) == {
        "bending_tilt_in",
        "bending_tilt_out",
        "tilt_in",
        "tilt_out",
        "tilt_thetaB_contact_in",
    }
    for row in module_audit["modules"].values():
        assert abs(float(row["energy_minus_breakdown"])) < 1.0e-9
        assert np.isfinite(float(row["boundary_tilt_slope_fd"]))

    constraint = audit["constraint_audit"]
    assert bool(constraint["available_before"])
    assert bool(constraint["available_after"])
    assert float(constraint["outer_residual_abs_after"]) <= (
        float(constraint["outer_residual_abs_before"]) + 1.0e-8
    )

    interface = audit["interface_target_audit"]
    assert bool(interface["available"])
    assert interface["target_source"] == "scaffold_trace_shell"
    assert interface["row_summaries"]["outer_rows"]["count"] == 12
    assert interface["row_summaries"]["tilt_rows"]["count"] == 12
    assert interface["state_delta_after_audit"]["positions_max_abs"] == 0.0
    assert interface["state_delta_after_audit"]["tilts_in_max_abs"] == 0.0
    assert interface["state_delta_after_audit"]["tilts_out_max_abs"] == 0.0


def test_scaffold_elastic_and_constrained_diagnostics_are_finite() -> None:
    audit = run_audit(
        mesh_path=DEFAULT_FIXTURE,
        protocol=QUICK_PROTOCOL,
        resolution_mode="none",
    )

    elastic = audit["elastic_magnitude_audit"]
    assert elastic["field_probes"]
    for probe in elastic["field_probes"]:
        assert np.isfinite(float(probe["elastic"]))
        assert np.isfinite(float(probe["bending_tilt_in"]))
        assert np.isfinite(float(probe["tilt_in"]))
    assert "trace_shell" in elastic["role_stats"]
    assert "bending_tilt_in" in elastic["module_gradient_norms_by_role"]

    base = audit["bending_tilt_base_term_audit"]
    for leaflet in ("in", "out"):
        row = base["leaflets"][leaflet]
        assert bool(row["available"])
        assert np.isfinite(float(row["module_energy"]))
        assert np.isfinite(float(row["decomposed_total_energy"]))
        assert np.isfinite(float(row["base_energy"]))
        assert "trace_shell" in row["role_summaries"]
        assert "zeroed_rows" in row
    assert base["state_delta_after_audit"]["positions_max_abs"] == 0.0
    assert base["state_delta_after_audit"]["tilts_in_max_abs"] == 0.0
    assert base["state_delta_after_audit"]["tilts_out_max_abs"] == 0.0

    comparison = audit["base_term_fixture_comparison"]
    assert comparison["fixtures"]
    for row in comparison["fixtures"]:
        assert "label" in row
        if "error" not in row:
            assert "in" in row["leaflets"]
            assert "out" in row["leaflets"]

    constrained = audit["constrained_gradient_audit"]
    assert constrained["probes"]
    labels = {row["label"] for row in constrained["probes"]}
    assert "boundary_radial_tilt_in" in labels
    assert "trace_shell_height" in labels
    assert "release_ring_radius" in labels
    for row in constrained["probes"]:
        assert np.isfinite(float(row["raw_fd_slope"]))
        assert np.isfinite(float(row["enforced_fd_slope"]))
        assert np.isfinite(float(row["enforced_relaxed_fd_slope"]))
        assert np.isfinite(float(row["projected_gradient_dot_direction"]))

    snapshots = audit["protocol_snapshot_audit"]
    assert snapshots[0]["label"] == "initial"
    assert any(row["command"] == "g1" for row in snapshots)
    for row in snapshots:
        assert np.isfinite(float(row["energy"]))


def test_scaffold_coupled_stationarity_audit_restores_state() -> None:
    audit = run_audit(
        mesh_path=DEFAULT_FIXTURE,
        protocol=QUICK_PROTOCOL,
        resolution_mode="none",
    )

    coupled = audit["coupled_stationarity_audit"]
    for state_name in (
        "fixed_state",
        "constrained_state",
        "constrained_tilt_relaxed",
        "line_search_trial_like",
    ):
        row = coupled["states"][state_name]
        assert row["boundary_rows"] == 12
        assert np.isfinite(float(row["total_slope"]))
        for module_row in row["modules"].values():
            assert np.isfinite(float(module_row["boundary_tilt_slope_fd"]))

    trace = coupled["cadence_trace"]
    assert len(trace["checkpoints"]) >= 6
    for row in trace["checkpoints"]:
        assert np.isfinite(float(row["energy"]))
        assert np.isfinite(float(row["theta_contact_mean"]))
        assert np.isfinite(float(row["thetaB_value"]))
    assert trace["state_delta_after_restore"]["positions_max_abs"] == 0.0
    assert trace["state_delta_after_restore"]["tilts_in_max_abs"] == 0.0
    assert trace["state_delta_after_restore"]["tilts_out_max_abs"] == 0.0
    assert coupled["state_delta_after_audit"]["positions_max_abs"] == 0.0
    assert coupled["state_delta_after_audit"]["tilts_in_max_abs"] == 0.0
    assert coupled["state_delta_after_audit"]["tilts_out_max_abs"] == 0.0


def test_scaffold_energy_normalization_audit_identities() -> None:
    audit = run_audit(
        mesh_path=DEFAULT_FIXTURE,
        protocol=QUICK_PROTOCOL,
        resolution_mode="none",
    )

    norm = audit["energy_normalization_audit"]
    measured = norm["measured"]
    identities = norm["identities"]
    matrix = norm["normalization_matrix"]

    assert abs(float(identities["elastic_minus_active_module_sum"])) < 1.0e-12
    assert abs(float(identities["total_minus_breakdown_sum"])) < 1.0e-8
    assert abs(float(identities["contact_minus_formula_R_eff"])) < 1.0e-8
    assert np.isfinite(float(identities["contact_minus_formula_R_theory"]))
    assert np.isfinite(float(identities["R_eff_over_R_theory"]))
    assert np.isfinite(float(measured["elastic_in"]))
    assert np.isfinite(float(measured["elastic_out"]))

    for label in (
        "legacy_anchor",
        "tex_benchmark",
        "in_only_elastic",
        "out_only_elastic",
        "contact_R_eff",
    ):
        row = matrix[label]
        assert np.isfinite(float(row["thetaB_star"]))
        assert np.isfinite(float(row["elastic_star"]))
        assert np.isfinite(float(row["contact_star"]))
        for ratio in row["ratios"].values():
            assert np.isfinite(float(ratio))


def test_flat_reference_mode_removes_zero_tilt_bending_tilt_base_energy() -> None:
    ctx = _build_context(DEFAULT_FIXTURE)
    mesh = ctx.mesh
    mesh.set_tilts_in_from_array(np.zeros_like(mesh.tilts_in_view()))
    mesh.set_tilts_out_from_array(np.zeros_like(mesh.tilts_out_view()))
    mesh.increment_version()

    breakdown = ctx.minimizer.compute_energy_breakdown()
    assert abs(float(breakdown.get("bending_tilt_in") or 0.0)) < 1.0e-9
    assert abs(float(breakdown.get("bending_tilt_out") or 0.0)) < 1.0e-9

    audit = _bending_tilt_base_term_audit(ctx)
    for leaflet in ("in", "out"):
        row = audit["leaflets"][leaflet]
        assert abs(float(row["base_energy"])) < 1.0e-9
        assert row["config"]["base_term_reference_mode"] == "flat_reference_zero_j0"


def test_current_geometry_base_reference_still_has_curvature_energy_when_unset() -> (
    None
):
    ctx = _build_context(DEFAULT_FIXTURE)
    mesh = ctx.mesh
    mesh.global_parameters.unset("bending_tilt_base_term_reference_mode")
    mesh.set_tilts_in_from_array(np.zeros_like(mesh.tilts_in_view()))
    mesh.set_tilts_out_from_array(np.zeros_like(mesh.tilts_out_view()))
    mesh.increment_version()

    breakdown = ctx.minimizer.compute_energy_breakdown()
    assert float(breakdown.get("bending_tilt_in") or 0.0) > 10.0
    assert float(breakdown.get("bending_tilt_out") or 0.0) > 10.0


def test_scaffold_energy_imbalance_audit_quick_resolution_matrix() -> None:
    audit = run_audit(
        mesh_path=DEFAULT_FIXTURE,
        protocol=QUICK_PROTOCOL,
        resolution_mode="quick",
    )

    matrix = audit["resolution_matrix"]
    assert matrix["mode"] == "quick"
    assert len(matrix["variants"]) == 2
    for row in matrix["variants"]:
        assert "error" not in row
        assert np.isfinite(float(row["stationarity_residual"]))
        assert np.isfinite(float(row["total_ratio"]))


def test_scaffold_energy_imbalance_audit_cli_writes_yaml(tmp_path) -> None:
    out_yaml = tmp_path / "audit.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mesh",
            str(DEFAULT_FIXTURE),
            "--protocol",
            *QUICK_PROTOCOL,
            "--resolution-mode",
            "none",
            "--out",
            str(out_yaml),
        ],
        check=True,
        cwd=str(ROOT),
    )

    audit = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert audit["meta"]["resolution_mode"] == "none"
    assert audit["resolution_matrix"]["variants"] == []

from __future__ import annotations

import numpy as np

from commands.context import CommandContext
from commands.executor import execute_command_line
from core.parameters.resolver import ParameterResolver
from geometry.geom_io import load_data, parse_geometry
from modules.energy import bending_tilt_out
from tools.build_stage_a_fixtures import BASE_FIXTURE
from tools.reproduce_flat_disk_one_leaflet import _build_minimizer


def _load_stage_a_state(commands: list[str]):
    mesh = parse_geometry(load_data(str(BASE_FIXTURE)))
    minimizer = _build_minimizer(mesh)
    ctx = CommandContext(mesh, minimizer, minimizer.stepper)
    for command in commands:
        execute_command_line(ctx, command)
    return ctx.mesh, minimizer


def test_stage_a_outer_grad_linear_transition_operator_is_enabled() -> None:
    mesh, _ = _load_stage_a_state(["g3", "r", "g3", "r"])

    grad = np.zeros_like(mesh.positions_view())
    tilt_grad = np.zeros_like(mesh.tilts_out_view())
    bending_tilt_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        ParameterResolver(mesh.global_parameters),
        positions=mesh.positions_view(),
        index_map=mesh.vertex_index_to_row,
        grad_arr=grad,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=tilt_grad,
    )

    stats = getattr(mesh, "_last_bending_tilt_out_grad_linear_transition_stats", {})
    assert stats.get("enabled") is True
    assert stats.get("mode") == "outer_grad_linear_transition_operator_v1"
    assert stats.get("cache_tag") == "out"
    assert str(stats.get("lane")) == "stage_a_emergent"
    assert stats.get("reconstructed_state_variable") == "factor_K_vec"
    assert stats.get("nontransition_uses_raw_state") is True
    assert int(stats.get("transition_edge_count", 0)) > 0
    assert int(stats.get("same_domain_edge_count", 0)) > 0

    example = stats.get("transition_example") or {}
    assert example.get("endpoint_domains") in (["partial", "full"], ["full", "partial"])
    raw = np.asarray(example.get("raw_endpoint_factor_K_vec"), dtype=float)
    recon = np.asarray(example.get("reconstructed_side_states"), dtype=float)
    assert raw.shape == (2, 3)
    assert recon.shape == (2, 3)

    shell_summary = {
        round(float(item["radius"]), 6): item for item in stats.get("shell_summary", [])
    }
    for radius in (0.965910, 0.974541, 0.999984):
        item = shell_summary[radius]
        assert int(item["vertex_count"]) > 0
        assert np.isfinite(float(item["transition_z_mean"]))
        assert np.isfinite(float(item["same_z_mean"]))
        assert np.isfinite(float(item["total_z_mean"]))

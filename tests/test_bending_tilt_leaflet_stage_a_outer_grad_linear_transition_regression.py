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
    assert stats.get("mode") == "outer_grad_linear_transition_patch_operator_v1"
    assert stats.get("cache_tag") == "out"
    assert str(stats.get("lane")) == "stage_a_emergent"
    assert stats.get("reconstructed_state_variable") == "factor_K_vec"
    assert stats.get("exterior_uses_raw_state") is True
    assert int(stats.get("transition_edge_count", 0)) > 0
    assert int(stats.get("patch_component_count", 0)) > 0
    assert int(stats.get("patch_vertex_count", 0)) > 0
    assert int(stats.get("partial_vertex_count", 0)) > 0
    assert int(stats.get("halo_full_vertex_count", 0)) > 0
    assert stats.get("halo_vertices_are_direct_transition_full_endpoints") is True
    assert int(stats.get("fallback_component_count", -1)) == 0

    example = stats.get("patch_example") or {}
    assert int(example.get("component_index", -1)) >= 0
    assert int(example.get("patch_vertex_count", 0)) > 0
    assert int(example.get("boundary_vertex_count", 0)) > 0
    field = np.asarray(example.get("boundary_factor_K_vec_sample"), dtype=float)
    assert field.ndim == 2
    assert field.shape[1] == 3

    patch_shell_summary = {
        round(float(item["radius"]), 6): item
        for item in stats.get("patch_shell_summary", [])
    }
    assert set(patch_shell_summary) == {
        0.837822,
        0.842474,
        0.853008,
        0.866643,
        0.965910,
        0.974541,
        0.999984,
    }

    shell_summary = {
        round(float(item["radius"]), 6): item for item in stats.get("shell_summary", [])
    }
    for radius in (0.965910, 0.974541, 0.999984):
        item = shell_summary[radius]
        assert int(item["vertex_count"]) > 0
        assert np.isfinite(float(item["patch_internal_z_mean"]))
        assert np.isfinite(float(item["patch_boundary_z_mean"]))
        assert np.isfinite(float(item["exterior_z_mean"]))
        assert np.isfinite(float(item["total_z_mean"]))

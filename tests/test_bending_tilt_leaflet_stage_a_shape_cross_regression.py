from __future__ import annotations

import numpy as np

from commands.context import CommandContext
from commands.executor import execute_command_line
from core.parameters.resolver import ParameterResolver
from geometry.geom_io import load_data, parse_geometry
from modules.energy import bending_tilt_in
from tools.build_stage_a_fixtures import BASE_FIXTURE
from tools.reproduce_flat_disk_one_leaflet import _build_minimizer


def _load_stage_a_state(commands: list[str]):
    mesh = parse_geometry(load_data(str(BASE_FIXTURE)))
    minimizer = _build_minimizer(mesh)
    ctx = CommandContext(mesh, minimizer, minimizer.stepper)
    for command in commands:
        execute_command_line(ctx, command)
    return ctx.mesh, minimizer


def test_stage_a_refined_inner_shape_cross_suppression_is_enabled() -> None:
    mesh, minimizer = _load_stage_a_state(["g3", "r"])

    grad = np.zeros_like(mesh.positions_view())
    tilt_grad = np.zeros_like(mesh.tilts_in_view())
    bending_tilt_in.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        ParameterResolver(mesh.global_parameters),
        positions=mesh.positions_view(),
        index_map=mesh.vertex_index_to_row,
        grad_arr=grad,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
        tilt_in_grad_arr=tilt_grad,
    )

    stats = getattr(mesh, "_last_bending_tilt_in_shape_cross_stats", {})
    assert stats.get("enabled") is True
    assert stats.get("cache_tag") == "in"
    assert str(stats.get("lane")) == "stage_a_emergent"

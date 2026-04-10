from __future__ import annotations

from pathlib import Path

import pytest

from commands.context import CommandContext
from commands.executor import execute_command_line
from core.parameters.resolver import ParameterResolver
from geometry.geom_io import load_data, parse_geometry
from modules.energy import tilt_rim_source_bilayer
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent
from tools.build_stage_a_fixtures import BASE_FIXTURE


def _build_minimizer(mesh) -> Minimizer:
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def _load_stage_a_state(commands: list[str]):
    mesh = parse_geometry(load_data(str(Path(BASE_FIXTURE))))
    minimizer = _build_minimizer(mesh)
    ctx = CommandContext(mesh, minimizer, minimizer.stepper)
    for command in commands:
        execute_command_line(ctx, command)
    return ctx.mesh


def _source_diagnostics(mesh) -> dict:
    resolver = ParameterResolver(mesh.global_parameters)
    payload = tilt_rim_source_bilayer._selection_diagnostics(mesh, resolver)
    assert payload is not None
    return payload


def test_stage_a_source_support_stays_localized_to_physical_rim_after_refine() -> None:
    coarse = _load_stage_a_state(["g3"])
    refined = _load_stage_a_state(["g3", "r"])

    coarse_diag = _source_diagnostics(coarse)
    refined_diag = _source_diagnostics(refined)

    assert coarse_diag["rim_row_radii"] == pytest.approx([0.466667], abs=1.0e-6)
    assert refined_diag["rim_row_radii"] == pytest.approx(
        [0.450765, 0.466667], abs=1.0e-6
    )


def test_stage_a_source_total_line_drive_is_refinement_invariant() -> None:
    coarse = _load_stage_a_state(["g3"])
    refined = _load_stage_a_state(["g3", "r"])

    coarse_diag = _source_diagnostics(coarse)
    refined_diag = _source_diagnostics(refined)

    assert coarse_diag["total_edge_length"] == pytest.approx(
        refined_diag["total_edge_length"], rel=1.0e-12, abs=1.0e-12
    )
    assert coarse_diag["total_source_load"] == pytest.approx(
        refined_diag["total_source_load"], rel=1.0e-12, abs=1.0e-12
    )

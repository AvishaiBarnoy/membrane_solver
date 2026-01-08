import os
import sys

sys.path.append(os.getcwd())

from commands.context import CommandContext
from commands.executor import execute_command_line
from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _make_context(data: dict) -> CommandContext:
    mesh = parse_geometry(data)
    gp = mesh.global_parameters
    stepper = GradientDescent()
    minimizer = Minimizer(
        mesh,
        gp,
        stepper,
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    minimizer.step_size = float(gp.get("step_size", 1e-3) or 1e-3)
    return CommandContext(mesh, minimizer, stepper)


def _tetra_body_input(target_volume: float = 0.2) -> dict:
    verts = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    edges = [[0, 1], [1, 2], [2, 0], [0, 3], [1, 3], [2, 3]]
    faces = [
        ["r2", "r1", "r0"],  # (0, 2, 1)
        [0, 4, "r3"],  # (0, 1, 3)
        [3, "r5", 2],  # (0, 3, 2)
        [1, 5, "r4"],  # (1, 2, 3)
    ]
    bodies = {"faces": [[0, 1, 2, 3]], "target_volume": [target_volume]}
    return {
        "vertices": verts,
        "edges": edges,
        "faces": faces,
        "bodies": bodies,
        "global_parameters": {
            "surface_tension": 1.0,
            "volume_constraint_mode": "lagrange",
            "step_size": 1e-2,
            "step_size_mode": "fixed",
        },
        "instructions": [],
    }


def test_command_executor_runs_minimization_and_records_history() -> None:
    ctx = _make_context(_tetra_body_input(target_volume=0.15))

    e0 = ctx.minimizer.compute_energy()
    execute_command_line(ctx, "g2")
    e1 = ctx.minimizer.compute_energy()

    assert ctx.history[-1] == "g2"
    assert e1 <= e0 + 1e-12


def test_refine_command_updates_mesh_and_keeps_body_orientation() -> None:
    ctx = _make_context(_tetra_body_input(target_volume=0.25))

    n0 = (len(ctx.mesh.vertices), len(ctx.mesh.edges), len(ctx.mesh.facets))
    topo0 = getattr(ctx.mesh, "_topology_version", 0)

    execute_command_line(ctx, "r1")

    n1 = (len(ctx.mesh.vertices), len(ctx.mesh.edges), len(ctx.mesh.facets))
    topo1 = getattr(ctx.mesh, "_topology_version", 0)

    assert topo1 > topo0
    assert n1[0] > n0[0]
    assert n1[1] > n0[1]
    assert n1[2] > n0[2]
    assert ctx.mesh.validate_body_orientation() is True
    assert ctx.mesh.validate_body_outwardness() is True

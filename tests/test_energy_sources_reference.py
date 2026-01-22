import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.context import CommandContext
from commands.executor import execute_command_line
from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _annulus_source_mesh(*, n: int = 10) -> dict:
    vertices: list[list] = []
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append(
            [
                float(np.cos(theta)),
                float(np.sin(theta)),
                0.0,
                {
                    "pin_to_circle_group": "inner",
                    "pin_to_circle_mode": "fit",
                    "pin_to_circle_normal": [0.0, 0.0, 1.0],
                },
            ]
        )
    for i in range(n):
        theta = 2.0 * np.pi * i / n
        vertices.append([float(2.0 * np.cos(theta)), float(2.0 * np.sin(theta)), 0.0])

    edges: list[list[int]] = []
    inner_edges = []
    outer_edges = []
    spokes = []
    for i in range(n):
        inner_edges.append(len(edges))
        edges.append([i, (i + 1) % n])
    for i in range(n):
        outer_edges.append(len(edges))
        edges.append([n + i, n + ((i + 1) % n)])
    for i in range(n):
        spokes.append(len(edges))
        edges.append([i, n + i])

    faces: list[list] = []
    for i in range(n):
        i_next = (i + 1) % n
        faces.append(
            [
                inner_edges[i],
                spokes[i_next],
                f"r{outer_edges[i]}",
                f"r{spokes[i]}",
            ]
        )

    return {
        "global_parameters": {
            "tilt_rim_source_group_in": "inner",
            "tilt_rim_source_strength_in": 1.0,
            "tilt_rim_source_center": [0.0, 0.0, 0.0],
        },
        "energy_modules": ["tilt_rim_source_in"],
        "vertices": vertices,
        "edges": edges,
        "faces": faces,
        "instructions": [],
    }


def _build_context(mesh_data: dict) -> CommandContext:
    mesh = parse_geometry(mesh_data)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    return CommandContext(mesh, minim, minim.stepper)


def _set_radial_tilt_in(mesh) -> None:
    positions = mesh.positions_view()
    r = positions.copy()
    r[:, 2] = 0.0
    rn = np.linalg.norm(r, axis=1)
    tilts = np.zeros_like(positions)
    good = rn > 1e-12
    tilts[good] = r[good] / rn[good][:, None]
    mesh.set_tilts_in_from_array(tilts)


def test_energy_command_reports_sources_and_reference(capsys):
    ctx = _build_context(_annulus_source_mesh())
    _set_radial_tilt_in(ctx.mesh)

    execute_command_line(ctx, "energy")
    out = capsys.readouterr().out
    assert "external work (sources)" in out
    assert "tilt_rim_source_in" in out

    execute_command_line(ctx, "energy ref")
    _ = capsys.readouterr().out

    ctx.mesh.set_tilts_in_from_array(np.zeros_like(ctx.mesh.positions_view()))
    execute_command_line(ctx, "energy")
    out = capsys.readouterr().out
    assert "Δtotal vs ref" in out

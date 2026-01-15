import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer


class _VertexLiftStepper:
    """Test stepper that deterministically changes the surface normal."""

    def __init__(self, *, vertex_id: int, delta: np.ndarray) -> None:
        self.vertex_id = int(vertex_id)
        self.delta = np.asarray(delta, dtype=float)

    def step(self, mesh, grad, step_size, energy_fn, constraint_enforcer=None):
        mesh.vertices[self.vertex_id].position += self.delta
        mesh.increment_version()
        return True, float(step_size)


def test_minimizer_projects_tilts_to_tangent_after_step():
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "global_parameters": {"surface_tension": 0.0},
        "instructions": [],
    }
    mesh = parse_geometry(data)

    # Start with a tangent tilt field (initial triangle normal is +z).
    for vid in mesh.vertices:
        mesh.vertices[vid].tilt = np.array([1.0, 0.0, 0.0], dtype=float)
        mesh.vertices[vid].tilt_in = np.array([0.5, 0.0, 0.5], dtype=float)
        mesh.vertices[vid].tilt_out = np.array([-0.2, 0.0, -0.3], dtype=float)
    mesh.touch_tilts()
    mesh.touch_tilts_in()
    mesh.touch_tilts_out()

    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        _VertexLiftStepper(vertex_id=1, delta=np.array([0.0, 0.0, 1.0])),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        tol=0.0,  # ensure the loop runs even with a zero gradient
        quiet=True,
    )
    minim.minimize(n_steps=1)

    normals = mesh.vertex_normals()
    max_abs_dot = 0.0
    for row, vid in enumerate(mesh.vertex_ids):
        max_abs_dot = max(
            max_abs_dot, abs(float(np.dot(mesh.vertices[int(vid)].tilt, normals[row])))
        )
    assert max_abs_dot < 1e-12

    max_abs_dot_in = 0.0
    max_abs_dot_out = 0.0
    for row, vid in enumerate(mesh.vertex_ids):
        max_abs_dot_in = max(
            max_abs_dot_in,
            abs(float(np.dot(mesh.vertices[int(vid)].tilt_in, normals[row]))),
        )
        max_abs_dot_out = max(
            max_abs_dot_out,
            abs(float(np.dot(mesh.vertices[int(vid)].tilt_out, normals[row]))),
        )
    assert max_abs_dot_in < 1e-12
    assert max_abs_dot_out < 1e-12

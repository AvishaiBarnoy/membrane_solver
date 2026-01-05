import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _build_square_mesh_with_center(*, z_offset: float) -> Mesh:
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([1.0, 1.0, 0.0])),
        3: Vertex(3, np.array([0.0, 1.0, 0.0])),
        4: Vertex(4, np.array([0.5, 0.5, float(z_offset)])),
    }
    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 3),
        4: Edge(4, 3, 0),
        5: Edge(5, 0, 4),
        6: Edge(6, 1, 4),
        7: Edge(7, 2, 4),
        8: Edge(8, 3, 4),
    }
    mesh.facets = {
        1: Facet(1, [1, 6, -5]),
        2: Facet(2, [2, 7, -6]),
        3: Facet(3, [3, 8, -7]),
        4: Facet(4, [4, 5, -8]),
    }
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def test_surface_relaxation_fixed_steps_signature() -> None:
    """Numerical signature: fixed seed + fixed N steps yields stable observables."""
    np.random.seed(0)

    mesh = _build_square_mesh_with_center(z_offset=0.2)
    for vid in range(4):
        mesh.vertices[vid].fixed = True

    mesh.energy_modules = ["surface"]
    mesh.constraint_modules = []
    mesh.global_parameters.set("surface_tension", 1.0)
    mesh.global_parameters.set("step_size_mode", "fixed")
    mesh.global_parameters.set("step_size", 2e-2)

    minimizer = Minimizer(
        mesh=mesh,
        global_params=mesh.global_parameters,
        stepper=GradientDescent(max_iter=10),
        energy_manager=EnergyModuleManager(mesh.energy_modules),
        constraint_manager=ConstraintModuleManager(mesh.constraint_modules),
        step_size=2e-2,
        tol=-1.0,
        quiet=True,
    )

    for _ in range(100):
        minimizer.minimize(n_steps=1)

    energy = minimizer.compute_energy()
    center_z = float(mesh.vertices[4].position[2])

    # Tight signature tolerances: intended to catch changes in line search,
    # surface gradients, or caching behavior.
    assert energy == pytest.approx(1.0000000050010467, rel=0.0, abs=1e-10)
    assert center_z == pytest.approx(5.0005233763034544e-05, rel=0.0, abs=1e-10)

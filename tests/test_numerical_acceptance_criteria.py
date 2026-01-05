import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

from sample_meshes import cube_soft_volume_input

from geometry.entities import Edge, Facet, Mesh, Vertex
from geometry.geom_io import parse_geometry
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


def _triangle_normals(mesh: Mesh) -> np.ndarray:
    tri_rows, _ = mesh.triangle_row_cache()
    assert tri_rows is not None and len(tri_rows) == len(mesh.facets)
    pos = mesh.positions_view()
    tri_pos = pos[tri_rows]
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]
    return np.cross(v1 - v0, v2 - v0)


def test_surface_relaxation_energy_monotone_and_no_flips() -> None:
    """Acceptance: energy decreases and triangles do not invert."""
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

    normals0 = _triangle_normals(mesh)
    energies = [minimizer.compute_energy()]
    for _ in range(20):
        minimizer.minimize(n_steps=1)
        energies.append(minimizer.compute_energy())

    assert all(b <= a + 1e-12 for a, b in zip(energies, energies[1:]))

    normals1 = _triangle_normals(mesh)
    dot = np.einsum("ij,ij->i", normals0, normals1)
    assert np.all(dot >= 0.0)
    assert mesh.validate_edge_indices()


def test_cube_penalty_minimization_acceptance_criteria() -> None:
    """Acceptance: energy decreases, volume stays near target, topology is sane."""
    mesh = parse_geometry(cube_soft_volume_input(volume_mode="penalty"))
    mesh.global_parameters.set("step_size_mode", "fixed")
    mesh.global_parameters.set("step_size", 1e-2)

    minimizer = Minimizer(
        mesh=mesh,
        global_params=mesh.global_parameters,
        stepper=GradientDescent(max_iter=10),
        energy_manager=EnergyModuleManager(mesh.energy_modules),
        constraint_manager=ConstraintModuleManager(mesh.constraint_modules),
        step_size=1e-2,
        tol=-1.0,
        quiet=True,
    )

    energies = [minimizer.compute_energy()]
    for _ in range(10):
        minimizer.minimize(n_steps=1)
        energies.append(minimizer.compute_energy())
    assert all(b <= a + 1e-10 for a, b in zip(energies, energies[1:]))

    target_volume = mesh.bodies[0].options.get("target_volume", 0.0)
    final_volume = mesh.compute_total_volume()
    assert math.isclose(final_volume, target_volume, rel_tol=5e-2, abs_tol=5e-2)

    mesh.build_connectivity_maps()
    edge_facet_counts = [len(fs) for fs in mesh.edge_to_facets.values()]
    assert min(edge_facet_counts) == 2
    assert max(edge_facet_counts) == 2
    assert mesh.validate_body_orientation()
    assert mesh.validate_edge_indices()

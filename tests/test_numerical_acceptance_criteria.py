import math

import numpy as np

from geometry.entities import Mesh
from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent
from tests.sample_meshes import (
    cube_soft_volume_input,
)
from tests.sample_meshes import (
    square_mesh_with_center as _build_square_mesh_with_center,
)


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

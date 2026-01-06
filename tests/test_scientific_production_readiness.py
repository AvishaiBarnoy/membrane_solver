import pytest

from geometry.geom_io import parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.bfgs import BFGS
from runtime.steppers.conjugate_gradient import ConjugateGradient
from runtime.steppers.gradient_descent import GradientDescent


def _tetra_mesh(target_volume=0.2):
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
    bodies = {
        "faces": [[0, 1, 2, 3]],
        "target_volume": [target_volume],
    }
    return {
        "vertices": verts,
        "edges": edges,
        "faces": faces,
        "bodies": bodies,
        "global_parameters": {
            "surface_tension": 1.0,
            "volume_constraint_mode": "lagrange",
        },
        "instructions": [],
    }


def _square_patch():
    verts = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.5, 0.5, 0.2],  # Center popped up
    ]
    # Fixed outer ring
    for i in range(4):
        verts[i].append({"fixed": True})

    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [0, 4], [1, 4], [2, 4], [3, 4]]
    faces = [
        [0, 5, "r4"],
        [1, 6, "r5"],
        [2, 7, "r6"],
        [3, 4, "r7"],
    ]
    return {
        "vertices": verts,
        "edges": edges,
        "faces": faces,
        "global_parameters": {"surface_tension": 1.0},
        "instructions": [],
    }


@pytest.mark.parametrize("stepper_cls", [GradientDescent, ConjugateGradient, BFGS])
def test_energy_monotonic_decrease(stepper_cls):
    """Verify that all steppers reduce energy monotonically on a pop-up patch."""
    mesh = parse_geometry(_square_patch())
    gp = mesh.global_parameters
    minim = Minimizer(
        mesh,
        gp,
        stepper_cls(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )

    e_history = []
    for _ in range(5):
        e_history.append(minim.compute_energy())
        minim.minimize(n_steps=1)

    # Check monotonicity
    for i in range(len(e_history) - 1):
        assert e_history[i + 1] <= e_history[i] + 1e-12


def _square_mesh(scale=1.0, target_area=None):
    verts = [
        [0.0, 0.0, 0.0],
        [scale, 0.0, 0.0],
        [scale, scale, 0.0],
        [0.0, scale, 0.0],
    ]
    edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
    faces = [[0, 1, 2, 3]]
    return {
        "vertices": verts,
        "edges": edges,
        "faces": faces,
        "global_parameters": {
            "surface_tension": 1.0,
            "target_surface_area": target_area,
            "area_stiffness": 100.0,
        },
        "constraint_modules": ["global_area"],
        "instructions": [],
    }


def test_hard_constraint_residuals():
    """Verify that volume and area constraints residuals stay within tight tolerance."""
    # Volume test
    target_vol = 0.15
    mesh_vol = parse_geometry(_tetra_mesh(target_volume=target_vol))
    gp_vol = mesh_vol.global_parameters

    minim_vol = Minimizer(
        mesh_vol,
        gp_vol,
        ConjugateGradient(),
        EnergyModuleManager(mesh_vol.energy_modules),
        ConstraintModuleManager(mesh_vol.constraint_modules),
        quiet=True,
        step_size=1e-2,
    )
    minim_vol.minimize(n_steps=100)
    final_vol = mesh_vol.compute_total_volume()
    assert abs(final_vol - target_vol) < 1e-3

    # Area test (on a separate, more stable mesh)
    target_area = 1.0
    mesh_area = parse_geometry(_square_mesh(scale=1.1, target_area=target_area))
    gp_area = mesh_area.global_parameters
    minim_area = Minimizer(
        mesh_area,
        gp_area,
        ConjugateGradient(),
        EnergyModuleManager(mesh_area.energy_modules),
        ConstraintModuleManager(mesh_area.constraint_modules),
        quiet=True,
        step_size=1e-2,
    )
    minim_area.minimize(n_steps=50)
    final_area = mesh_area.compute_total_surface_area()
    assert abs(final_area - target_area) < 1e-3


def test_deterministic_numerical_signature():
    """Verify that a fixed sequence of steps results in a deterministic energy (numerical signature)."""
    mesh = parse_geometry(_tetra_mesh(target_volume=0.25))
    gp = mesh.global_parameters
    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        step_size=1e-3,
    )

    minim.minimize(n_steps=10)
    final_energy = minim.compute_energy()

    # This value depends on the exact implementation.
    # If the math changes, this test should be updated to the new "signature".
    # For a unit surface tension tetra with V=0.25 after 10 GD steps:
    expected_energy = 2.9289410122111974
    assert final_energy == pytest.approx(expected_energy, rel=1e-6)


def test_mesh_sanity_after_minimization():
    """Check manifoldness and orientation consistency after large shape changes."""
    mesh = parse_geometry(_tetra_mesh(target_volume=0.5))  # Blow it up
    gp = mesh.global_parameters
    minim = Minimizer(
        mesh,
        gp,
        BFGS(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        step_size=1e-2,
    )

    minim.minimize(n_steps=20)

    assert mesh.validate_edge_indices() is True
    assert mesh.validate_body_orientation() is True
    # Ensure no degenerate triangles
    for fid, facet in mesh.facets.items():
        assert facet.compute_area(mesh) > 1e-8


def test_topology_invariants_stability():
    """Verify that boundary loop counts remain stable during minimization."""
    from runtime.diagnostics.gauss_bonnet import (
        extract_boundary_loops,
        find_boundary_edges,
    )

    mesh = parse_geometry(_square_patch())
    initial_boundary_edges = len(find_boundary_edges(mesh))
    initial_loops = len(extract_boundary_loops(mesh, find_boundary_edges(mesh)))

    gp = mesh.global_parameters
    minim = Minimizer(
        mesh,
        gp,
        ConjugateGradient(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )

    minim.minimize(n_steps=10)

    final_boundary_edges = len(find_boundary_edges(mesh))
    final_loops = len(extract_boundary_loops(mesh, find_boundary_edges(mesh)))

    assert final_boundary_edges == initial_boundary_edges
    assert final_loops == initial_loops

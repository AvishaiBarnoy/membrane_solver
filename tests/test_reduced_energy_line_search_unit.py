import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _build_single_triangle_leaflet_mesh() -> Mesh:
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([0.0, 1.0, 0.0])),
    }
    mesh.edges = {1: Edge(1, 0, 1), 2: Edge(2, 1, 2), 3: Edge(3, 2, 0)}
    mesh.facets = {0: Facet(0, edge_indices=[1, 2, 3])}
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def _build_minimizer(mesh: Mesh, gp: GlobalParameters) -> Minimizer:
    mesh.energy_modules = ["tilt_in", "tilt_out"]
    mesh.constraint_modules = []
    return Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def test_line_search_reduced_energy_relaxes_tilts_and_restores_param_overrides() -> (
    None
):
    mesh = _build_single_triangle_leaflet_mesh()
    rng = np.random.default_rng(0)
    tin0 = rng.normal(size=(len(mesh.vertex_ids), 3))
    tout0 = rng.normal(size=(len(mesh.vertex_ids), 3))
    # Make tilts tangent to z=0 plane for determinism.
    tin0[:, 2] = 0.0
    tout0[:, 2] = 0.0
    mesh.set_tilts_in_from_array(tin0)
    mesh.set_tilts_out_from_array(tout0)

    gp = GlobalParameters(
        {
            "tilt_modulus_in": 1.0,
            "tilt_modulus_out": 1.0,
            "tilt_solve_mode": "coupled",
            "tilt_step_size": 0.25,
            "tilt_inner_steps": 10,
            "tilt_tol": 0.0,
            "line_search_reduced_energy": True,
            "line_search_reduced_tilt_inner_steps": 5,
            "tilt_backtracking_warm_start": False,
        }
    )
    minim = _build_minimizer(mesh, gp)

    # Raw energy uses the current (non-relaxed) tilts.
    E_raw = float(minim.compute_energy())

    energy_fn = minim._line_search_energy_fn()
    E_reduced = float(energy_fn())

    # Reduced evaluation should not be higher than the raw energy.
    assert E_reduced <= E_raw + 1e-10
    # Reduced evaluation relaxes tilts in-place; line search is responsible for
    # restoring tilts on rejected trial steps.
    assert not np.allclose(mesh.tilts_in_view(), tin0)
    assert not np.allclose(mesh.tilts_out_view(), tout0)

    # Parameter overrides must be restored (no state leakage into outer loop).
    assert int(gp.get("tilt_inner_steps")) == 10
    assert bool(gp.get("tilt_backtracking_warm_start")) is False


def test_line_search_energy_fn_projects_leaflet_tilts_to_tangent() -> None:
    mesh = _build_single_triangle_leaflet_mesh()

    # Deliberately give both leaflets a non-tangent component on the z=0 plane.
    tin0 = np.zeros((len(mesh.vertex_ids), 3))
    tout0 = np.zeros((len(mesh.vertex_ids), 3))
    tin0[:, 2] = 1.0
    tout0[:, 2] = -2.0
    mesh.set_tilts_in_from_array(tin0)
    mesh.set_tilts_out_from_array(tout0)

    gp = GlobalParameters(
        {
            "tilt_modulus_in": 1.0,
            "tilt_modulus_out": 1.0,
            "tilt_solve_mode": "fixed",
            "line_search_reduced_energy": False,
        }
    )
    minim = _build_minimizer(mesh, gp)

    # Baseline energy includes out-of-plane components if not projected.
    E_unprojected = float(minim.compute_energy())

    # Compute the expected projected energy by projecting explicitly.
    mesh.project_tilts_to_tangent()
    E_projected_expected = float(minim.compute_energy())

    # Restore the original (non-tangent) tilts and let the line-search callback
    # apply the projection internally.
    mesh.set_tilts_in_from_array(tin0)
    mesh.set_tilts_out_from_array(tout0)
    energy_fn = minim._line_search_energy_fn()
    E_projected = float(energy_fn())

    assert E_projected == pytest.approx(E_projected_expected, abs=1e-12)
    assert E_projected <= E_unprojected + 1e-12
    assert np.allclose(mesh.tilts_in_view()[:, 2], 0.0, atol=1e-12)
    assert np.allclose(mesh.tilts_out_view()[:, 2], 0.0, atol=1e-12)

from unittest.mock import patch

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from modules.energy import surface
from tests.sample_meshes import tetra_mesh_with_body as _tetra_mesh_with_body


def test_surface_energy_avoids_per_facet_area_calls_for_triangle_meshes():
    mesh = _tetra_mesh_with_body()
    gp = GlobalParameters({"surface_tension": 1.0})

    with patch(
        "geometry.entities.Facet.compute_area",
        autospec=True,
        side_effect=AssertionError("surface energy regressed to per-facet loop"),
    ):
        energy = surface.calculate_surface_energy(mesh, gp)

    assert float(energy) > 0.0


def test_surface_gradient_avoids_per_facet_gradient_calls_for_triangle_meshes():
    mesh = _tetra_mesh_with_body()
    gp = GlobalParameters({"surface_tension": 1.0})
    resolver = ParameterResolver(gp)

    with patch(
        "geometry.entities.Facet.compute_area_and_gradient",
        autospec=True,
        side_effect=AssertionError(
            "surface gradient regressed to per-facet compute_area_and_gradient"
        ),
    ):
        energy, grad = surface.compute_energy_and_gradient(mesh, gp, resolver)

    assert float(energy) > 0.0
    assert set(grad.keys()) == set(mesh.vertices.keys())

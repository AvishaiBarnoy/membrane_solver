import numpy as np

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def build_triangle_mesh():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.vertices[2] = Vertex(2, np.array([0.0, 1.0, 0.0]))
    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 2)
    mesh.edges[3] = Edge(3, 2, 0)
    mesh.facets[1] = Facet(1, [1, 2, 3])
    mesh.global_parameters = GlobalParameters()
    mesh.energy_modules = ["surface"]
    mesh.constraint_modules = []
    return mesh


def test_minimizer_soa_cache_updates_on_version_bump():
    mesh = build_triangle_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )

    positions1, index1, grad1 = minim._soa_views()
    assert index1
    assert grad1.shape == positions1.shape

    mesh.vertices[0].position = mesh.vertices[0].position + np.array([0.5, 0.0, 0.0])
    mesh.increment_version()

    positions2, index2, grad2 = minim._soa_views()
    assert np.isclose(positions2[index2[0], 0], 0.5)
    assert grad2.shape == positions2.shape

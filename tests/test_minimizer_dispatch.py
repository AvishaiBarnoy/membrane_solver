import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import MagicMock

import numpy as np

from geometry.entities import Edge, Facet, Mesh, Vertex
from parameters.global_parameters import GlobalParameters
from runtime.constraint_manager import ConstraintModuleManager
from runtime.minimizer import Minimizer


def test_minimizer_dispatches_to_array_path():
    """Verify that Minimizer calls compute_energy_and_gradient_array if available."""
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([0, 0, 0]))
    mesh.vertices[1] = Vertex(1, np.array([1, 0, 0]))
    mesh.vertices[2] = Vertex(2, np.array([0, 1, 0]))
    mesh.facets[0] = Facet(0, [1, 2, 3])
    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 2)
    mesh.edges[3] = Edge(3, 2, 0)
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()

    # Mock module with array support
    mock_mod = MagicMock()
    mock_mod.compute_energy_and_gradient_array.return_value = 10.0

    # Setup Minimizer manually
    gp = GlobalParameters()
    em = MagicMock()
    em.get_module.return_value = mock_mod
    cm = ConstraintModuleManager([])

    minim = Minimizer(mesh, gp, MagicMock(), em, cm, energy_modules=["mock"])

    # Trigger calculation
    minim.compute_energy_and_gradient()

    # Check if the array version was called
    assert mock_mod.compute_energy_and_gradient_array.called
    # Check if the legacy version was NOT called (or at least array was preferred)
    assert not mock_mod.compute_energy_and_gradient.called


def test_minimizer_falls_back_to_dict_path():
    """Verify that Minimizer still works with legacy modules."""
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([0, 0, 0]))
    mesh.build_connectivity_maps()

    # Mock module WITHOUT array support
    mock_mod = MagicMock()
    del mock_mod.compute_energy_and_gradient_array
    mock_mod.compute_energy_and_gradient.return_value = (5.0, {0: np.array([1, 1, 1])})

    gp = GlobalParameters()
    em = MagicMock()
    em.get_module.return_value = mock_mod
    cm = ConstraintModuleManager([])

    minim = Minimizer(mesh, gp, MagicMock(), em, cm, energy_modules=["mock"])

    E, grad = minim.compute_energy_and_gradient()

    assert E == 5.0
    assert 0 in grad
    assert mock_mod.compute_energy_and_gradient.called

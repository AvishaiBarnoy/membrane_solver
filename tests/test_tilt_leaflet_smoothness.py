import importlib
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.entities import Edge, Facet, Mesh, Vertex

LEAFLET_CASES = {
    "in": {
        "module": "modules.energy.tilt_smoothness_in",
        "field": "tilt_in",
        "touch": "touch_tilts_in",
    },
    "out": {
        "module": "modules.energy.tilt_smoothness_out",
        "field": "tilt_out",
        "touch": "touch_tilts_out",
    },
}


def _build_single_triangle_mesh() -> Mesh:
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


@pytest.mark.parametrize("leaflet", ["in", "out"])
def test_tilt_leaflet_smoothness_constant_field_has_zero_energy(
    leaflet: str,
) -> None:
    case = LEAFLET_CASES[leaflet]
    module = importlib.import_module(case["module"])
    mesh = _build_single_triangle_mesh()

    for v in mesh.vertices.values():
        setattr(v, case["field"], np.array([1.0, -0.5, 0.0], dtype=float))
    getattr(mesh, case["touch"])()

    gp = GlobalParameters({"bending_modulus": 1.0})
    resolver = ParameterResolver(gp)

    energy, shape_grad, tilt_grad = module.compute_energy_and_gradient(
        mesh, gp, resolver
    )

    assert float(energy) == pytest.approx(0.0, rel=0.0, abs=1e-12)
    assert all(np.allclose(g, 0.0, atol=1e-12) for g in shape_grad.values())
    assert all(np.allclose(g, 0.0, atol=1e-12) for g in tilt_grad.values())

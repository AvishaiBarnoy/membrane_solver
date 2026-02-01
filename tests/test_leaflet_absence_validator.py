import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.leaflet_validation import validate_leaflet_absence_topology


def _build_one_triangle_mesh(*, presets: list[str | None]) -> Mesh:
    if len(presets) != 3:
        raise ValueError("presets must have length 3")
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.vertices[2] = Vertex(2, np.array([0.0, 1.0, 0.0]))
    for i, preset in enumerate(presets):
        if preset is None:
            continue
        mesh.vertices[i].options = {"preset": preset}

    # Edges are 1-based IDs in this codebase.
    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 2)
    mesh.edges[3] = Edge(3, 2, 0)
    mesh.facets[0] = Facet(0, [1, 2, 3])
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def test_leaflet_absence_validator_passes_without_absence_flag() -> None:
    mesh = _build_one_triangle_mesh(presets=["disk", None, None])
    gp = GlobalParameters()
    validate_leaflet_absence_topology(mesh, gp)


def test_leaflet_absence_validator_passes_when_all_absent_or_all_present() -> None:
    gp = GlobalParameters({"leaflet_out_absent_presets": ["disk"]})

    mesh_all_absent = _build_one_triangle_mesh(presets=["disk", "disk", "disk"])
    validate_leaflet_absence_topology(mesh_all_absent, gp)

    mesh_all_present = _build_one_triangle_mesh(presets=[None, None, None])
    validate_leaflet_absence_topology(mesh_all_present, gp)


def test_leaflet_absence_validator_fails_on_straddling_triangle() -> None:
    mesh = _build_one_triangle_mesh(presets=["disk", None, None])
    gp = GlobalParameters({"leaflet_out_absent_presets": ["disk"]})
    with pytest.raises(ValueError, match="Leaflet absence topology invalid"):
        validate_leaflet_absence_topology(mesh, gp)

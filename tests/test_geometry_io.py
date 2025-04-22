import os
import numpy as np
import pytest
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geometry_io import load_data, parse_geometry
from parameters.global_parameters import GlobalParameters
from geometry.geometry_entities import Facet
from runtime.refinement import refine_polygonal_facets

SAMPLE_FILE = os.path.join("meshes", "sample_geometry.json")

def test_loaded_geometry_is_triangular():
    data = load_data(SAMPLE_FILE)
    vertices, edges, facets, bodies, global_params = parse_geometry(data)

    assert bodies, "At least one body should be defined"
    body = bodies[0]

    volume_before = bodies[0].calculate_volume()
    assert volume_before > 0, f"Volume should be positive, got {volume_before}"

    # Triangulate all non-triangles before testing
    vertices, edges, facets, bodies = refine_polygonal_facets(vertices, edges, facets, bodies, global_params)

    assert all(isinstance(f, Facet) for f in facets)
    assert all(len(f.edges) == 3 for f in facets), "All facets must be triangles"

    volume_after = bodies[0].calculate_volume()
    assert volume_after > 0, f"Refined volume should be positive, got {volume_after}"
    assert np.isclose(volume_before, volume_after, rtol=1e-6), (
        f"Volume changed after triangulation: before={volume_before}, after={volume_after}"
    )

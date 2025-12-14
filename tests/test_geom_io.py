import os
import numpy as np
import json
import pytest
import sys

# Adjust import paths for this testing environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry, save_geometry
from geometry.entities import Mesh
from runtime.refinement import refine_polygonal_facets
from sample_meshes import SAMPLE_GEOMETRY, write_sample_geometry


@pytest.fixture
def sample_geometry_file(tmp_path):
    return write_sample_geometry(tmp_path)


def test_geometry_loads_correctly(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)

    assert isinstance(mesh, Mesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.edges) > 0
    assert len(mesh.facets) > 0
    assert len(mesh.bodies) > 0

def test_mesh_validation_passes(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)
    assert mesh.validate_edge_indices()

def test_edge_orientation_signs_preserved(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)

    for facet_idx in mesh.facets.keys():
        for ei in mesh.facets[facet_idx].edge_indices:
            assert ei != 0, "Zero edge index should not exist (shifted system)"

def test_round_trip_consistency(sample_geometry_file, tmp_path):
    data = load_data(sample_geometry_file)
    mesh1 = parse_geometry(data)

    temp_file = tmp_path / "temp_geometry_output.json"
    save_geometry(mesh1, temp_file)
    data2 = load_data(temp_file)
    mesh2 = parse_geometry(data2)

    assert len(mesh1.vertices) == len(mesh2.vertices)
    assert len(mesh1.edges) == len(mesh2.edges)
    assert len(mesh1.facets) == len(mesh2.facets)

def test_facet_vertex_sequence_is_consistent(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)

    for facet in mesh.facets:
        verts = []
        for signed_index in mesh.facets[facet].edge_indices:
            edge = mesh.edges[abs(signed_index)]
            tail, head = (edge.tail_index, edge.head_index) if signed_index > 0 else (edge.head_index, edge.tail_index)
            if not verts:
                verts.append(tail)
            verts.append(head)
        # It should form a closed loop: first == last
        assert verts[0] == verts[-1], f"Facet {facet.index} is not closed: {verts}"

# TDD placeholder for future feature
def test_body_volume_is_positive(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)

    body = mesh.bodies[0]
    volume = body.compute_volume(mesh)
    assert round(volume, 3) == 1.0, f"Expected positive volume, got {volume}"

    mesh_volume = mesh.compute_total_volume()
    assert round(mesh_volume, 3) == 1.0, f"Expected positive volume, got {mesh_volume}"

def test_body_surface_area_positive(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)

    area = mesh.bodies[0].compute_surface_area(mesh)
    assert round(area, 3) == 6.0, f"Surface area should be positive, got {area}"

    mesh_tri = refine_polygonal_facets(mesh)
    mesh_area = mesh_tri.compute_total_surface_area()
    assert round(mesh_area, 3) == 6.0, f"Surface area should be positive, got {mesh_area}"

def test_default_energy_assignment(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)
    for facet in mesh.facets.values():
        assert len(facet.options["energy"]) != 0, "Energy should be assigned to each facet"
        assert isinstance(facet.options["energy"], list), f"Energy module list should be a list, but it is a {type(facet.options['energy'])}"


    #if len(mesh.bodies) > 0:
    #    for body in mesh.bodies.values():
    #        print(body.options["energy"])
    #        assert len(body.options["energy"]) != 0, "Energy should be assigned to each body"
    #        assert isinstance(body.options["energy"], list), f"Energy module list should be a list, but it is a {type(body.options['energy'])}"


def test_parse_geometry_allows_missing_faces_for_line_only_mesh():
    """parse_geometry should handle inputs that omit 'faces' for line-only meshes."""
    data = {
        "vertices": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        "edges": [
            [0, 1, {"energy": ["line_tension"], "line_tension": 2.0}],
        ],
        # No "faces" key: line-only geometry.
        "global_parameters": {"line_tension": 2.0},
    }
    mesh = parse_geometry(data)

    assert isinstance(mesh, Mesh)
    assert len(mesh.vertices) == 2
    assert len(mesh.edges) == 1
    assert len(mesh.facets) == 0

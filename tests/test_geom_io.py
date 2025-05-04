import os
import numpy as np
import json
import pytest
import sys

# Adjust import paths for this testing environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry, save_geometry
from geometry.entities import Mesh

SAMPLE_FILE = os.path.join("meshes", "sample_geometry.json")
TEMP_FILE = os.path.join("meshes", "temp_geometry_output.json")

def test_geometry_loads_correctly():
    data = load_data(SAMPLE_FILE)
    mesh = parse_geometry(data)

    assert isinstance(mesh, Mesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.edges) > 0
    assert len(mesh.facets) > 0
    assert len(mesh.bodies) > 0

def test_mesh_validation_passes():
    data = load_data(SAMPLE_FILE)
    mesh = parse_geometry(data)
    assert mesh.validate_edge_indices()

def test_edge_orientation_signs_preserved():
    data = load_data(SAMPLE_FILE)
    mesh = parse_geometry(data)

    for facet_idx in mesh.facets.keys():
        for ei in mesh.facets[facet_idx].edge_indices:
            assert ei != 0, "Zero edge index should not exist (shifted system)"

def test_round_trip_consistency():
    data = load_data(SAMPLE_FILE)
    mesh1 = parse_geometry(data)

    save_geometry(mesh1, TEMP_FILE)
    data2 = load_data(TEMP_FILE)
    mesh2 = parse_geometry(data2)

    assert len(mesh1.vertices) == len(mesh2.vertices)
    assert len(mesh1.edges) == len(mesh2.edges)
    assert len(mesh1.facets) == len(mesh2.facets)

def test_facet_vertex_sequence_is_consistent():
    data = load_data(SAMPLE_FILE)
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
def test_body_volume_is_positive():
    data = load_data(SAMPLE_FILE)
    mesh = parse_geometry(data)
    #sys.exit()

    body = mesh.bodies[0]
    volume = body.compute_volume(mesh)  # to be implemented
    # TODO: fix volume calculation currently gives wrong answer
    #print(f"volume: {volume}")
    #sys.exit()
    assert volume > 0, f"Expected positive volume, got {volume}"

def test_body_surface_area_positive():
    data = load_data(SAMPLE_FILE)
    mesh = parse_geometry(data)

    area = mesh.bodies[0].compute_surface_area(mesh)
    # TODO: fix area calculation currently gives wrong answer
    #print(f"area: {area}")
    #sys.exit()
    assert area > 0, f"Surface area should be positive, got {area}"

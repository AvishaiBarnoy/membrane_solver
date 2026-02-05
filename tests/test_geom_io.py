"""Consolidated tests for geometry IO (loading, parsing, saving, validation)."""

import math
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_meshes import write_sample_geometry

from geometry.entities import Mesh
from geometry.geom_io import load_data, parse_geometry, save_geometry
from runtime.refinement import refine_polygonal_facets


@pytest.fixture
def sample_geometry_file(tmp_path):
    return write_sample_geometry(tmp_path)


# --- Basic Loading and Roundtrip ---


def test_geometry_loads_correctly(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)

    assert isinstance(mesh, Mesh)
    assert len(mesh.vertices) > 0
    assert len(mesh.edges) > 0
    assert len(mesh.facets) > 0
    assert len(mesh.bodies) > 0


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


def test_load_data_reads_yaml(tmp_path):
    path = tmp_path / "mesh.yaml"
    yaml.safe_dump({"vertices": [[0, 0, 0]], "edges": [[0, 0]]}, path.open("w"))
    data = load_data(path)
    assert "vertices" in data


# --- Validation and Error Handling ---


def test_mesh_validation_passes(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)
    assert mesh.validate_edge_indices()


def test_load_data_rejects_unknown_extension(tmp_path):
    path = tmp_path / "mesh.txt"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_data(path)


def test_parse_geometry_rejects_nan_and_inf_vertices():
    data_nan = {"vertices": [[0.0, float("nan"), 0.0]], "edges": [[0, 0]]}
    with pytest.raises(ValueError, match="NaN"):
        parse_geometry(data_nan)

    data_inf = {"vertices": [[0.0, float("inf"), 0.0]], "edges": [[0, 0]]}
    with pytest.raises(ValueError, match="infinite"):
        parse_geometry(data_inf)


def test_parse_geometry_rejects_missing_edges_section():
    with pytest.raises(KeyError, match="missing required 'edges'"):
        parse_geometry({"vertices": [[0, 0, 0]]})


def test_parse_geometry_rejects_edges_referencing_unknown_vertices():
    data = {"vertices": [[0, 0, 0]], "edges": [[0, 1]]}
    with pytest.raises(ValueError, match="references missing head vertex"):
        parse_geometry(data)


# --- Feature-specific Parsing ---


def test_edge_orientation_signs_preserved(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)

    for facet_idx in mesh.facets.keys():
        for ei in mesh.facets[facet_idx].edge_indices:
            assert ei != 0, "Zero edge index should not exist (shifted system)"


def test_facet_vertex_sequence_is_consistent(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)

    for facet in mesh.facets:
        verts = []
        for signed_index in mesh.facets[facet].edge_indices:
            edge = mesh.edges[abs(signed_index)]
            tail, head = (
                (edge.tail_index, edge.head_index)
                if signed_index > 0
                else (edge.head_index, edge.tail_index)
            )
            if not verts:
                verts.append(tail)
            verts.append(head)
        # It should form a closed loop: first == last
        assert verts[0] == verts[-1], f"Facet {facet.index} is not closed: {verts}"


def test_body_volume_is_positive(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)
    body = mesh.bodies[0]
    volume = body.compute_volume(mesh)
    assert round(volume, 3) == 1.0, f"Expected positive volume, got {volume}"


def test_body_surface_area_positive(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)
    area = mesh.bodies[0].compute_surface_area(mesh)
    assert round(area, 3) == 6.0, f"Surface area should be positive, got {area}"
    mesh_tri = refine_polygonal_facets(mesh)
    mesh_area = mesh_tri.compute_total_surface_area()
    assert round(mesh_area, 3) == 6.0, (
        f"Surface area should be positive, got {mesh_area}"
    )


def test_default_energy_assignment(sample_geometry_file):
    data = load_data(sample_geometry_file)
    mesh = parse_geometry(data)
    for facet in mesh.facets.values():
        assert "energy" in facet.options
        assert isinstance(facet.options["energy"], list)


def test_parse_geometry_allows_missing_faces_for_line_only_mesh():
    data = {
        "vertices": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "edges": [[0, 1, {"energy": ["line_tension"], "line_tension": 2.0}]],
        "global_parameters": {"line_tension": 2.0},
    }
    mesh = parse_geometry(data)
    assert isinstance(mesh, Mesh)
    assert len(mesh.facets) == 0


def test_parse_geometry_preset_not_found():
    data = {
        "vertices": [[0, 0, 0], [1, 0, 0]],
        "edges": [[0, 1]],
        "faces": [[0, 0, 0, {"preset": "missing"}]],
        "definitions": {},
    }
    with pytest.raises(ValueError, match="Preset 'missing' not found"):
        parse_geometry(data)


def test_parse_geometry_parses_reversed_edge_token_r0():
    data = {
        "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [["r0", 1, 2]],
    }
    mesh = parse_geometry(data)
    assert list(mesh.facets.values())[0].edge_indices[0] == -1


def test_parse_geometry_evaluates_defines():
    data = {
        "global_parameters": {"angle": 60.0},
        "defines": {"WALLT": "-cos(angle*pi/180)"},
        "vertices": [[0, 0, 0], [1, 0, 0]],
        "edges": [[0, 1]],
    }
    mesh = parse_geometry(data)
    assert math.isclose(mesh.global_parameters.get("WALLT"), -0.5, rel_tol=1e-6)


def test_tilt_in_out_roundtrip(tmp_path):
    data = {
        "vertices": [
            [
                0.0,
                0.0,
                0.0,
                {
                    "tilt_in": [1.0, 0.0],
                    "tilt_out": [0.0, 1.0],
                    "tilt_fixed_in": True,
                    "tilt_fixed_out": True,
                },
            ],
            [1.0, 0.0, 0.0],
        ],
        "edges": [[0, 1]],
    }
    mesh = parse_geometry(data)
    assert mesh.vertices[0].tilt_fixed_in is True
    assert mesh.vertices[0].tilt_fixed_out is True
    assert np.allclose(mesh.vertices[0].tilt_in, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(mesh.vertices[0].tilt_out, np.array([0.0, 1.0, 0.0]))

    out_path = tmp_path / "tilt_in_out.json"
    save_geometry(mesh, str(out_path))
    roundtrip = parse_geometry(load_data(out_path))
    assert roundtrip.vertices[0].tilt_fixed_in is True
    assert roundtrip.vertices[0].tilt_fixed_out is True
    assert np.allclose(roundtrip.vertices[0].tilt_in, np.array([1.0, 0.0, 0.0]))
    assert np.allclose(roundtrip.vertices[0].tilt_out, np.array([0.0, 1.0, 0.0]))

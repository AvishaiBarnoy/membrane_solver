import math
import os
import sys

import numpy as np
import pytest
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry, save_geometry


def test_load_data_rejects_unknown_extension(tmp_path):
    path = tmp_path / "mesh.txt"
    path.write_text("{}")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_data(path)


def test_load_data_reads_yaml(tmp_path):
    path = tmp_path / "mesh.yaml"
    yaml.safe_dump({"vertices": [[0, 0, 0]], "edges": [[0, 0]]}, path.open("w"))
    data = load_data(path)
    assert "vertices" in data


def test_parse_geometry_rejects_nan_and_inf_vertices(tmp_path):
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


def test_parse_geometry_preset_not_found():
    data = {
        "vertices": [[0, 0, 0], [1, 0, 0]],
        "edges": [[0, 1]],
        "faces": [[0, 0, 0, {"preset": "missing"}]],
        "definitions": {},
    }
    with pytest.raises(ValueError, match="Preset 'missing' not found"):
        parse_geometry(data)


def test_parse_geometry_parses_reversed_edge_token_r0(tmp_path):
    data = {
        "vertices": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        "edges": [[0, 1], [1, 2], [2, 0]],
        # Faces use zero-based edge indices in the file format.
        # "r0" means "edge 0 reversed", which becomes internal -1.
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
        "instructions": [],
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

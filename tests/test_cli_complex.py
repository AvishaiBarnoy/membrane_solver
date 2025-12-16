import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geometry.entities import Body, Edge, Facet, Vertex
from main import process_complex_command
from parameters.global_parameters import GlobalParameters


class MockMesh:
    def __init__(self):
        self.vertices = {0: Vertex(0, np.array([0.,0.,0.])),
                         1: Vertex(1, np.array([1.,0.,0.]), options={'fixed': False})}
        self.edges = {0: Edge(0, 0, 1, options={'len': 1.0})}
        self.facets = {0: Facet(0, [0], options={'area': 0.5})}
        self.bodies = {0: Body(0, [0], options={'vol': 1.0})}
        self.global_parameters = GlobalParameters()
        self.global_parameters.set("surface_tension", 1.0)

    def compute_total_surface_area(self): return 1.0
    def compute_total_volume(self): return 1.0

def test_set_global_param():
    mesh = MockMesh()

    # Test setting a global parameter
    # set surface_tension 2.0
    tokens = ["set", "surface_tension", "2.0"]
    process_complex_command(tokens, mesh)

    assert mesh.global_parameters.get("surface_tension") == 2.0

def test_set_vertex_fixed():
    mesh = MockMesh()

    # Test setting vertex fixed property
    # set vertex 1 fixed true
    tokens = ["set", "vertex", "1", "fixed", "true"]
    process_complex_command(tokens, mesh)

    assert mesh.vertices[1].fixed is True

def test_set_edge_option():
    mesh = MockMesh()

    # set edge 0 line_tension 5.0
    tokens = ["set", "edge", "0", "line_tension", "5.0"]
    process_complex_command(tokens, mesh)

    assert mesh.edges[0].options["line_tension"] == 5.0

def test_print_commands(capsys):
    mesh = MockMesh()

    # print vertex 0
    process_complex_command(["print", "vertex", "0"], mesh)
    captured = capsys.readouterr()
    assert "Vertex 0" in captured.out

    # print edges
    process_complex_command(["print", "edges"], mesh)
    captured = capsys.readouterr()
    assert "List of edges" in captured.out

    # print facets
    process_complex_command(["print", "facets"], mesh)
    captured = capsys.readouterr()
    assert "List of facets" in captured.out

    # print bodies
    process_complex_command(["print", "bodies"], mesh)
    captured = capsys.readouterr()
    assert "List of bodies" in captured.out

def test_print_filter(capsys):
    mesh = MockMesh()
    # Mock compute_length for edge
    mesh.edges[0].compute_length = lambda m: 1.0

    # print edges len > 0.5
    process_complex_command(["print", "edges", "len", ">", "0.5"], mesh)
    captured = capsys.readouterr()
    assert "Found 1 edges matching filter" in captured.out

    # print edges len > 1.5 (should be 0)
    process_complex_command(["print", "edges", "len", ">", "1.5"], mesh)
    captured = capsys.readouterr()
    assert "Found 0 edges matching filter" in captured.out

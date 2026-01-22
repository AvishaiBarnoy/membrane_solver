import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from commands.context import CommandContext
from commands.meta import PrintEntityCommand, SetCommand
from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Body, Edge, Facet, Vertex


class MockMesh:
    def __init__(self):
        self.vertices = {
            0: Vertex(0, np.array([0.0, 0.0, 0.0])),
            1: Vertex(1, np.array([1.0, 0.0, 0.0]), options={"fixed": False}),
        }
        self.edges = {1: Edge(1, 0, 1, options={"len": 1.0})}
        self.facets = {0: Facet(0, [1], options={"area": 0.5})}
        self.bodies = {0: Body(0, [0], options={"vol": 1.0})}
        self.global_parameters = GlobalParameters()
        self.global_parameters.set("surface_tension", 1.0)

    def compute_total_surface_area(self):
        return 1.0

    def compute_total_volume(self):
        return 1.0


def _get_context(mesh):
    # Dummy minimizer/stepper as they aren't used here
    return CommandContext(mesh, None, None)


def test_set_global_param():
    mesh = MockMesh()
    ctx = _get_context(mesh)
    cmd = SetCommand()

    # set surface_tension 2.0
    cmd.execute(ctx, ["surface_tension", "2.0"])

    assert mesh.global_parameters.get("surface_tension") == 2.0


def test_set_vertex_fixed():
    mesh = MockMesh()
    ctx = _get_context(mesh)
    cmd = SetCommand()

    # set vertex 1 fixed true
    cmd.execute(ctx, ["vertex", "1", "fixed", "true"])

    assert mesh.vertices[1].fixed is True


def test_set_edge_option():
    mesh = MockMesh()
    ctx = _get_context(mesh)
    cmd = SetCommand()

    # set edge 1 line_tension 5.0
    cmd.execute(ctx, ["edge", "1", "line_tension", "5.0"])

    assert mesh.edges[1].options["line_tension"] == 5.0


def test_set_vertex_coordinate():
    mesh = MockMesh()
    ctx = _get_context(mesh)
    cmd = SetCommand()

    cmd.execute(ctx, ["vertex", "1", "z", "2.0"])

    assert float(mesh.vertices[1].position[2]) == 2.0


def test_set_vertices_all_where_option_filter():
    mesh = MockMesh()
    mesh.vertices[0].options["pin_to_circle_group"] = "outer"
    mesh.vertices[1].options["pin_to_circle_group"] = "inner"
    mesh.vertices[2] = Vertex(
        2,
        np.array([2.0, 0.0, 0.0]),
        options={"pin_to_circle_group": "inner"},
    )

    ctx = _get_context(mesh)
    cmd = SetCommand()

    cmd.execute(
        ctx,
        [
            "vertices",
            "all",
            "z",
            "3.0",
            "where",
            "pin_to_circle_group=inner",
        ],
    )

    assert float(mesh.vertices[0].position[2]) == 0.0
    assert float(mesh.vertices[1].position[2]) == 3.0
    assert float(mesh.vertices[2].position[2]) == 3.0


def test_set_edges_all_where_numeric_filter():
    mesh = MockMesh()
    ctx = _get_context(mesh)
    cmd = SetCommand()

    cmd.execute(ctx, ["edges", "all", "fixed", "true", "where", "len", ">", "0.5"])

    assert mesh.edges[1].fixed is True
    assert mesh.vertices[0].fixed is True
    assert mesh.vertices[1].fixed is True


def test_print_commands(capsys):
    mesh = MockMesh()
    ctx = _get_context(mesh)
    cmd = PrintEntityCommand()

    # print vertex 0
    cmd.execute(ctx, ["vertex", "0"])
    captured = capsys.readouterr()
    assert "Vertex 0" in captured.out

    # print edges
    cmd.execute(ctx, ["edges"])
    captured = capsys.readouterr()
    assert "List of edges" in captured.out

    # print facets
    cmd.execute(ctx, ["facets"])
    captured = capsys.readouterr()
    assert "List of facets" in captured.out

    # print bodies
    cmd.execute(ctx, ["bodies"])
    captured = capsys.readouterr()
    assert "List of bodies" in captured.out


def test_print_filter(capsys):
    mesh = MockMesh()
    # Mock compute_length for edge
    mesh.edges[1].compute_length = lambda m: 1.0
    ctx = _get_context(mesh)
    cmd = PrintEntityCommand()

    # print edges len > 0.5
    cmd.execute(ctx, ["edges", "len", ">", "0.5"])
    captured = capsys.readouterr()
    assert "Found 1 edges matching filter" in captured.out

    # print edges len > 1.5 (should be 0)
    cmd.execute(ctx, ["edges", "len", ">", "1.5"])
    captured = capsys.readouterr()
    assert "Found 0 edges matching filter" in captured.out

import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.meta import HelpCommand, PrintEntityCommand, QuitCommand, SetCommand
from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from parameters.global_parameters import GlobalParameters


def build_mesh():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.edges[1] = Edge(1, 0, 1)
    mesh.facets[0] = Facet(0, [1])
    mesh.global_parameters = GlobalParameters()
    return mesh


def test_quit_sets_flag_and_prints(capsys):
    ctx = SimpleNamespace(should_exit=False)
    QuitCommand().execute(ctx, [])
    assert ctx.should_exit is True
    assert "Exiting interactive mode" in capsys.readouterr().out


def test_help_prints_summary(capsys):
    HelpCommand().execute(SimpleNamespace(), [])
    out = capsys.readouterr().out
    assert "Interactive commands:" in out
    assert "print [entity]" in out


def test_set_command_usage_errors(capsys):
    ctx = SimpleNamespace(mesh=build_mesh())
    SetCommand().execute(ctx, [])
    assert "Usage: set" in capsys.readouterr().out

    SetCommand().execute(ctx, ["vertex", "0", "fixed"])
    assert "Usage: set [entity]" in capsys.readouterr().out

    SetCommand().execute(ctx, ["vertex", "not_an_int", "fixed", "true"])
    assert "ID must be an integer" in capsys.readouterr().out


def test_set_command_missing_entity(capsys):
    ctx = SimpleNamespace(mesh=build_mesh())
    SetCommand().execute(ctx, ["vertex", "999", "fixed", "true"])
    assert "Entity 999 not found." in capsys.readouterr().out


def test_print_entity_command_usage_and_unknown_type(capsys):
    ctx = SimpleNamespace(mesh=build_mesh())
    PrintEntityCommand().execute(ctx, [])
    assert "Usage: print" in capsys.readouterr().out

    PrintEntityCommand().execute(ctx, ["unknowns"])
    assert "Unknown entity type" in capsys.readouterr().out


def test_print_entity_command_filter_len(capsys):
    ctx = SimpleNamespace(mesh=build_mesh())
    PrintEntityCommand().execute(ctx, ["edges", "len", ">", "0.5"])
    out = capsys.readouterr().out
    assert "Found 1 edges matching filter." in out
    assert "List of edges" in out


def test_set_command_updates_body_target_volume(capsys):
    mesh = build_mesh()
    mesh.bodies[0] = Body(
        0, facet_indices=[], target_volume=1.0, options={"target_volume": 1.0}
    )
    ctx = SimpleNamespace(mesh=mesh)

    SetCommand().execute(ctx, ["body", "0", "target_volume", "1.2"])
    assert mesh.bodies[0].target_volume == 1.2
    assert mesh.bodies[0].options["target_volume"] == 1.2
    assert "target_volume=1.2" in capsys.readouterr().out

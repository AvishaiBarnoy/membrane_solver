import os
import sys
from types import SimpleNamespace

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from commands.mesh_ops import EquiangulateCommand, RefineCommand, VertexAverageCommand


def test_refine_command_runs_requested_passes_without_enforcement(monkeypatch):
    mesh = object()
    ctx = SimpleNamespace(mesh=mesh, minimizer=SimpleNamespace(mesh=mesh))
    calls = {"poly": 0, "tri": 0, "enforce": 0}

    def fake_refine_poly(m):
        calls["poly"] += 1
        return m

    def fake_refine_tri(m):
        calls["tri"] += 1
        return m

    def fake_enforce(m):
        calls["enforce"] += 1

    ctx.minimizer.enforce_constraints_after_mesh_ops = fake_enforce

    monkeypatch.setattr("commands.mesh_ops.refine_polygonal_facets", fake_refine_poly)
    monkeypatch.setattr("commands.mesh_ops.refine_triangle_mesh", fake_refine_tri)

    RefineCommand().execute(ctx, ["3"])
    assert calls == {"poly": 3, "tri": 3, "enforce": 0}


def test_vertex_average_command_runs_n_passes_and_enforces(monkeypatch):
    mesh = object()
    ctx = SimpleNamespace(mesh=mesh, minimizer=SimpleNamespace(mesh=mesh))
    calls = {"avg": 0, "enforce": 0}

    def fake_vertex_average(m):
        assert m is mesh
        calls["avg"] += 1

    def fake_enforce(m):
        calls["enforce"] += 1

    ctx.minimizer.enforce_constraints_after_mesh_ops = fake_enforce
    monkeypatch.setattr("commands.mesh_ops.vertex_average", fake_vertex_average)

    VertexAverageCommand().execute(ctx, ["2"])
    assert calls == {"avg": 2, "enforce": 1}


def test_equiangulate_command_updates_mesh_and_enforces(monkeypatch):
    mesh = object()
    new_mesh = object()
    ctx = SimpleNamespace(mesh=mesh, minimizer=SimpleNamespace(mesh=mesh))
    calls = {"equi": 0, "enforce": 0}

    def fake_equiangulate(m):
        assert m is mesh
        calls["equi"] += 1
        return new_mesh

    def fake_enforce(m):
        assert m is new_mesh
        calls["enforce"] += 1

    ctx.minimizer.enforce_constraints_after_mesh_ops = fake_enforce
    monkeypatch.setattr("commands.mesh_ops.equiangulate_mesh", fake_equiangulate)

    EquiangulateCommand().execute(ctx, [])
    assert ctx.mesh is new_mesh
    assert ctx.minimizer.mesh is new_mesh
    assert calls == {"equi": 1, "enforce": 1}

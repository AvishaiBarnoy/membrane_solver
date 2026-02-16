from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Edge, Facet, Mesh, Vertex
from modules.energy import surface


def _single_triangle_mesh() -> Mesh:
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([0.0, 1.0, 0.0])),
    }
    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 0),
    }
    mesh.facets = {0: Facet(0, [1, 2, 3], options={"surface_tension": 1.0})}
    mesh.build_facet_vertex_loops()
    mesh.build_position_cache()
    return mesh


def test_surface_strict_nocopy_rejects_non_fortran_positions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _single_triangle_mesh()
    positions = np.asarray(mesh.positions_view(), dtype=np.float64, order="C")
    grad = np.zeros_like(positions)
    gp = GlobalParameters({"surface_tension": 1.0})

    called = {"kernel": False}

    def _kernel(*args, **kwargs):
        called["kernel"] = True

    monkeypatch.setenv("MEMBRANE_FORTRAN_STRICT_NOCOPY", "1")
    monkeypatch.setattr(
        surface,
        "get_surface_energy_kernel",
        lambda: SimpleNamespace(func=_kernel, expects_transpose=False),
    )

    with pytest.raises(ValueError, match="F-contiguous"):
        surface.compute_energy_and_gradient_array(
            mesh,
            gp,
            None,
            positions=positions,
            index_map=mesh.vertex_index_to_row,
            grad_arr=grad,
        )
    assert called["kernel"] is False


def test_surface_strict_nocopy_rejects_wrong_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = _single_triangle_mesh()
    positions = np.asarray(mesh.positions_view(), dtype=np.float32, order="F")
    grad = np.zeros((positions.shape[0], 3), dtype=np.float64, order="F")
    gp = GlobalParameters({"surface_tension": 1.0})

    called = {"kernel": False}

    def _kernel(*args, **kwargs):
        called["kernel"] = True

    monkeypatch.setenv("MEMBRANE_FORTRAN_STRICT_NOCOPY", "1")
    monkeypatch.setattr(
        surface,
        "get_surface_energy_kernel",
        lambda: SimpleNamespace(func=_kernel, expects_transpose=False),
    )

    with pytest.raises(TypeError, match="float64"):
        surface.compute_energy_and_gradient_array(
            mesh,
            gp,
            None,
            positions=positions,
            index_map=mesh.vertex_index_to_row,
            grad_arr=grad,
        )
    assert called["kernel"] is False

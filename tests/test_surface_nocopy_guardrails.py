from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from core.parameters.global_parameters import GlobalParameters
from modules.energy import surface
from tests.sample_meshes import single_triangle_mesh as _base_single_triangle_mesh


def _single_triangle_mesh():
    mesh = _base_single_triangle_mesh()
    mesh.facets[0].options["surface_tension"] = 1.0
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

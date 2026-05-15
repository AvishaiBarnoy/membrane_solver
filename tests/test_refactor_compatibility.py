from __future__ import annotations

import importlib

import numpy as np
import pytest

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Mesh, Vertex
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _single_vertex_mesh() -> Mesh:
    mesh = Mesh()
    mesh.vertices = {0: Vertex(0, np.zeros(3, dtype=float))}
    mesh.edges = {}
    mesh.facets = {}
    mesh.energy_modules = []
    mesh.constraint_modules = []
    mesh.global_parameters = GlobalParameters()
    mesh.build_position_cache()
    return mesh


def test_tilt_front_modules_keep_legacy_helper_shims(monkeypatch) -> None:
    tilt_in = importlib.import_module("modules.energy.tilt_in")
    tilt_out = importlib.import_module("modules.energy.tilt_out")

    mesh = _single_vertex_mesh()
    resolver = object()

    def fake_outer_shell(mesh_arg, leaflet):
        assert mesh_arg is mesh
        return np.asarray([0 if leaflet == "in" else 1], dtype=int)

    def fake_trace_weights(mesh_arg, resolver_arg, leaflet):
        assert mesh_arg is mesh
        assert resolver_arg is resolver
        return np.asarray([2.0 if leaflet == "in" else 3.0])

    monkeypatch.setattr(
        tilt_in._tilt_utils, "_shared_rim_outer_shell_rows", fake_outer_shell
    )
    monkeypatch.setattr(
        tilt_in._tilt_utils,
        "_explicit_trace_layer_active_row_weights",
        fake_trace_weights,
    )

    assert tilt_in._shared_rim_outer_shell_rows(mesh).tolist() == [0]
    assert tilt_out._shared_rim_outer_shell_rows(mesh).tolist() == [1]
    assert tilt_in._explicit_trace_layer_active_row_weights(
        mesh, resolver
    ).tolist() == [2.0]
    assert tilt_out._explicit_trace_layer_active_row_weights(
        mesh, resolver
    ).tolist() == [3.0]
    assert hasattr(tilt_in, "build_local_interface_shell_data")
    assert hasattr(tilt_out, "build_local_interface_shell_data")


def test_tilt_smoothness_front_modules_keep_legacy_helper_shims(monkeypatch) -> None:
    smooth_in = importlib.import_module("modules.energy.tilt_smoothness_in")
    smooth_out = importlib.import_module("modules.energy.tilt_smoothness_out")

    resolver = object()

    def fake_rigidity(resolver_arg, leaflet):
        assert resolver_arg is resolver
        return 11.0 if leaflet == "in" else 13.0

    monkeypatch.setattr(smooth_in._utils, "_resolve_smoothness_rigidity", fake_rigidity)

    assert smooth_in._resolve_smoothness_rigidity(resolver) == pytest.approx(11.0)
    assert smooth_out._resolve_smoothness_rigidity(resolver) == pytest.approx(13.0)


def test_minimizer_evaluation_manager_bridge_syncs_mutable_state() -> None:
    mesh = _single_vertex_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )

    assert hasattr(minim, "_evaluation_manager")
    new_mesh = _single_vertex_mesh()
    modules = [object()]
    names = ["manual"]

    minim.mesh = new_mesh
    minim.energy_modules = modules
    minim.energy_module_names = names
    minim._sync_evaluation_manager()

    assert minim._evaluation_manager.mesh is new_mesh
    assert minim._evaluation_manager.global_params is minim.global_params
    assert minim._evaluation_manager.param_resolver is minim.param_resolver
    assert minim._evaluation_manager.energy_modules is modules
    assert minim._evaluation_manager.energy_module_names is names

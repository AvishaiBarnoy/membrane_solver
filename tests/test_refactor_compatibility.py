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


class _SingleTiltEnergyModule:
    USES_TILT = True

    def compute_energy_and_gradient_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        grad_arr,
        tilts=None,
    ) -> float:
        _ = mesh, global_params, param_resolver, positions, index_map, grad_arr
        return float(np.sum(np.asarray(tilts, dtype=float)))


class _SingleTiltGradientModule:
    USES_TILT = True

    def compute_energy_and_gradient_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        grad_arr,
        tilts=None,
        tilt_grad_arr=None,
    ) -> float:
        _ = mesh, global_params, param_resolver, positions, index_map, grad_arr
        tilts = np.asarray(tilts, dtype=float)
        if tilt_grad_arr is not None:
            tilt_grad_arr += tilts
        return float(np.sum(tilts))


class _SingleTiltEnergyArrayModule:
    USES_TILT = True

    def compute_energy_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        tilts=None,
    ) -> float:
        _ = mesh, global_params, param_resolver, positions, index_map
        return float(np.sum(np.asarray(tilts, dtype=float)))


class _NonTiltEnergyModule:
    def compute_energy_and_gradient_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        grad_arr,
    ) -> float:
        _ = mesh, global_params, param_resolver, positions, index_map, grad_arr
        return 100.0


class _LegacyEnergyModule:
    def compute_energy_and_gradient(
        self, mesh, global_params, param_resolver, *, compute_gradient=True
    ):
        _ = mesh, global_params, param_resolver, compute_gradient
        return 5.0, {}


class _NonTiltEnergyArrayRejectsTiltsModule:
    def compute_energy_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
    ) -> float:
        _ = mesh, global_params, param_resolver, positions, index_map
        return 40.0


class _ShapeEnergyGradientModule:
    def compute_energy_and_gradient_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        grad_arr,
    ) -> float:
        _ = mesh, global_params, param_resolver, positions, index_map
        grad_arr += 1.0
        return 2.0


class _LeafletEnergyArrayModule:
    USES_TILT_LEAFLETS = True

    def compute_energy_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        tilts_in=None,
        tilts_out=None,
    ) -> float:
        _ = mesh, global_params, param_resolver, positions, index_map
        return float(np.sum(tilts_in) + 2.0 * np.sum(tilts_out))


class _LeafletGradientEnergyModule:
    USES_TILT_LEAFLETS = True

    def compute_energy_and_gradient_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        grad_arr,
        tilts_in=None,
        tilts_out=None,
        tilt_in_grad_arr=None,
        tilt_out_grad_arr=None,
    ) -> float:
        _ = (
            mesh,
            global_params,
            param_resolver,
            positions,
            index_map,
            grad_arr,
            tilt_in_grad_arr,
            tilt_out_grad_arr,
        )
        return float(3.0 * np.sum(tilts_in) + 5.0 * np.sum(tilts_out))


class _LeafletGradientMutatingModule:
    USES_TILT_LEAFLETS = True

    def __init__(self) -> None:
        self.grad_args: list[np.ndarray | None] = []

    def compute_energy_and_gradient_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        grad_arr,
        tilts_in=None,
        tilts_out=None,
        tilt_in_grad_arr=None,
        tilt_out_grad_arr=None,
    ) -> float:
        _ = mesh, global_params, param_resolver, positions, index_map
        self.grad_args.append(grad_arr)
        if grad_arr is not None:
            grad_arr += 9.0
        if tilt_in_grad_arr is not None:
            tilt_in_grad_arr += tilts_in
        if tilt_out_grad_arr is not None:
            tilt_out_grad_arr += 2.0 * tilts_out
        return 13.0


class _LeafletGradientRejectsTrialTiltsModule:
    USES_TILT_LEAFLETS = True

    def compute_energy_and_gradient_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        grad_arr,
        **kwargs,
    ) -> float:
        _ = mesh, global_params, param_resolver, positions, index_map, grad_arr
        if "tilts_in" in kwargs or "tilts_out" in kwargs:
            raise TypeError("legacy gradient module reads tilts from mesh")
        return 17.0


class _LeafletRejectsTrialTiltsModule:
    USES_TILT_LEAFLETS = True

    def compute_energy_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        **kwargs,
    ) -> float:
        _ = mesh, global_params, param_resolver, positions, index_map
        if "tilts_in" in kwargs or "tilts_out" in kwargs:
            raise TypeError("legacy module reads tilts from mesh")
        return 7.0


class _LeafletMarkerModule:
    USES_TILT_LEAFLETS = True


class _NonLeafletExplodingModule:
    def compute_energy_array(self, *args, **kwargs) -> float:
        _ = args, kwargs
        raise AssertionError("non-leaflet module should be skipped")


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


def test_minimizer_delegates_single_tilt_energy_path_with_sync() -> None:
    mesh = _single_vertex_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [_NonTiltEnergyModule(), _SingleTiltEnergyModule()]
    minim.energy_module_names = ["non_tilt", "tilt"]

    tilts = np.asarray([[1.0, 2.0, 3.0]], dtype=float)
    energy = minim._compute_energy_array_with_tilts(
        positions=mesh.positions_view(),
        tilts=tilts,
    )

    assert energy == pytest.approx(6.0)
    assert minim._evaluation_manager.energy_modules is minim.energy_modules
    assert minim._evaluation_manager.energy_module_names is minim.energy_module_names


def test_minimizer_delegates_total_energy_path_with_sync() -> None:
    mesh = _single_vertex_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [
        _NonTiltEnergyArrayRejectsTiltsModule(),
        _ShapeEnergyGradientModule(),
    ]
    minim.energy_module_names = ["array", "gradient"]

    energy = minim._compute_energy_array_total(positions=mesh.positions_view())

    assert energy == pytest.approx(42.0)
    assert minim._evaluation_manager.energy_modules is minim.energy_modules
    assert minim._evaluation_manager.energy_module_names is minim.energy_module_names


def test_minimizer_delegates_shape_gradient_assembly_with_sync() -> None:
    mesh = _single_vertex_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [_ShapeEnergyGradientModule()]
    minim.energy_module_names = ["shape"]

    energy, grad_arr = minim.compute_energy_and_gradient_array()

    assert energy == pytest.approx(2.0)
    np.testing.assert_allclose(grad_arr, np.ones_like(mesh.positions_view()))
    assert minim._evaluation_manager.energy_modules is minim.energy_modules
    assert minim._evaluation_manager.energy_module_names is minim.energy_module_names


def test_minimizer_delegates_energy_breakdown_with_sync() -> None:
    mesh = _single_vertex_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [_ShapeEnergyGradientModule(), _LegacyEnergyModule()]
    minim.energy_module_names = ["shape", "legacy"]
    mesh._curvature_cache = {"stale": object()}
    mesh._curvature_version = 99

    breakdown = minim.compute_energy_breakdown()

    assert breakdown == pytest.approx({"shape": 2.0, "legacy": 5.0})
    assert mesh._curvature_cache == {}
    assert mesh._curvature_version == -1
    assert minim._evaluation_manager.energy_modules is minim.energy_modules
    assert minim._evaluation_manager.energy_module_names is minim.energy_module_names


def test_minimizer_delegates_total_energy_with_projected_tilts_path() -> None:
    mesh = _single_vertex_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [
        _NonTiltEnergyArrayRejectsTiltsModule(),
        _SingleTiltEnergyArrayModule(),
        _SingleTiltGradientModule(),
    ]
    minim.energy_module_names = ["non_tilt", "tilt_array", "tilt_gradient"]

    tilts = np.asarray([[1.0, 2.0, 3.0]], dtype=float)
    energy = minim._compute_total_energy_array_with_tilts(
        positions=mesh.positions_view(),
        tilts=tilts,
    )

    assert energy == pytest.approx(52.0)
    assert minim._evaluation_manager.energy_modules is minim.energy_modules


def test_minimizer_delegates_leaflet_energy_path_with_legacy_fallback() -> None:
    mesh = _single_vertex_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [
        _LeafletRejectsTrialTiltsModule(),
        _LeafletEnergyArrayModule(),
        _LeafletGradientEnergyModule(),
    ]
    minim.energy_module_names = ["legacy_leaflet", "leaflet_array", "leaflet_gradient"]

    tilts_in = np.asarray([[1.0, 2.0, 3.0]], dtype=float)
    tilts_out = np.asarray([[4.0, 5.0, 6.0]], dtype=float)
    energy = minim._compute_energy_array_with_leaflet_tilts(
        positions=mesh.positions_view(),
        tilts_in=tilts_in,
        tilts_out=tilts_out,
    )

    assert energy == pytest.approx(7.0 + 36.0 + 93.0)
    assert minim._evaluation_manager.energy_modules is minim.energy_modules
    assert minim._evaluation_manager.energy_module_names is minim.energy_module_names


def test_minimizer_delegates_leaflet_tilt_dependent_fast_paths() -> None:
    mesh = _single_vertex_mesh()
    mesh.global_parameters.set("tilt_modulus_in", 2.0)
    mesh.global_parameters.set("tilt_modulus_out", 3.0)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [
        _NonLeafletExplodingModule(),
        _LeafletMarkerModule(),
        _LeafletMarkerModule(),
    ]
    minim.energy_module_names = ["shape", "tilt_in", "tilt_out"]

    tilts_in = np.asarray([[1.0, 2.0, 2.0]], dtype=float)
    tilts_out = np.asarray([[2.0, 0.0, 1.0]], dtype=float)
    energy = minim._compute_tilt_dependent_energy_with_leaflet_tilts(
        positions=mesh.positions_view(),
        tilts_in=tilts_in,
        tilts_out=tilts_out,
        tilt_vertex_areas_in=np.asarray([0.5], dtype=float),
        tilt_vertex_areas_out=np.asarray([2.0], dtype=float),
    )

    assert energy == pytest.approx(19.5)
    assert minim._evaluation_manager.energy_modules is minim.energy_modules


def test_minimizer_delegates_leaflet_tilt_dependent_legacy_fallback() -> None:
    mesh = _single_vertex_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [
        _NonLeafletExplodingModule(),
        _LeafletRejectsTrialTiltsModule(),
    ]
    minim.energy_module_names = ["shape", "legacy_leaflet"]

    energy = minim._compute_tilt_dependent_energy_with_leaflet_tilts(
        positions=mesh.positions_view(),
        tilts_in=np.asarray([[1.0, 2.0, 3.0]], dtype=float),
        tilts_out=np.asarray([[4.0, 5.0, 6.0]], dtype=float),
    )

    assert energy == pytest.approx(7.0)
    assert minim._evaluation_manager.energy_modules is minim.energy_modules


def test_minimizer_delegates_leaflet_gradient_path_with_shape_energy() -> None:
    mesh = _single_vertex_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [
        _ShapeEnergyGradientModule(),
        _LeafletGradientMutatingModule(),
    ]
    minim.energy_module_names = ["shape", "leaflet"]

    tilts_in = np.asarray([[1.0, 2.0, 3.0]], dtype=float)
    tilts_out = np.asarray([[4.0, 5.0, 6.0]], dtype=float)
    grad_dummy = np.zeros_like(tilts_in)
    tilt_in_grad = np.zeros_like(tilts_in)
    tilt_out_grad = np.zeros_like(tilts_out)
    energy = minim._compute_energy_and_leaflet_tilt_gradients_array(
        positions=mesh.positions_view(),
        tilts_in=tilts_in,
        tilts_out=tilts_out,
        tilt_in_grad_arr=tilt_in_grad,
        tilt_out_grad_arr=tilt_out_grad,
        grad_dummy=grad_dummy,
    )

    assert energy == pytest.approx(15.0)
    np.testing.assert_allclose(grad_dummy, np.full_like(grad_dummy, 10.0))
    np.testing.assert_allclose(tilt_in_grad, tilts_in)
    np.testing.assert_allclose(tilt_out_grad, 2.0 * tilts_out)
    assert minim._evaluation_manager.energy_modules is minim.energy_modules


def test_minimizer_delegates_leaflet_gradient_tilt_only_keeps_shape_gradient() -> None:
    mesh = _single_vertex_mesh()
    leaflet = _LeafletGradientMutatingModule()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [_ShapeEnergyGradientModule(), leaflet]
    minim.energy_module_names = ["shape", "leaflet"]

    grad_dummy = np.zeros((1, 3), dtype=float)
    energy = minim._compute_energy_and_leaflet_tilt_gradients_array(
        positions=mesh.positions_view(),
        tilts_in=np.asarray([[1.0, 2.0, 3.0]], dtype=float),
        tilts_out=np.asarray([[4.0, 5.0, 6.0]], dtype=float),
        tilt_in_grad_arr=np.zeros((1, 3), dtype=float),
        tilt_out_grad_arr=np.zeros((1, 3), dtype=float),
        grad_dummy=grad_dummy,
        tilt_only=True,
    )

    assert energy == pytest.approx(15.0)
    np.testing.assert_allclose(grad_dummy, np.ones_like(grad_dummy))
    assert leaflet.grad_args == [None]


def test_minimizer_delegates_leaflet_gradient_fast_paths() -> None:
    mesh = _single_vertex_mesh()
    mesh.global_parameters.set("tilt_modulus_in", 2.0)
    mesh.global_parameters.set("tilt_modulus_out", 3.0)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [_LeafletMarkerModule(), _LeafletMarkerModule()]
    minim.energy_module_names = ["tilt_in", "tilt_out"]

    tilts_in = np.asarray([[1.0, 2.0, 2.0]], dtype=float)
    tilts_out = np.asarray([[2.0, 0.0, 1.0]], dtype=float)
    tilt_in_grad = np.zeros_like(tilts_in)
    tilt_out_grad = np.zeros_like(tilts_out)
    energy = minim._compute_energy_and_leaflet_tilt_gradients_array(
        positions=mesh.positions_view(),
        tilts_in=tilts_in,
        tilts_out=tilts_out,
        tilt_in_grad_arr=tilt_in_grad,
        tilt_out_grad_arr=tilt_out_grad,
        tilt_vertex_areas_in=np.asarray([0.5], dtype=float),
        tilt_vertex_areas_out=np.asarray([2.0], dtype=float),
    )

    assert energy == pytest.approx(19.5)
    np.testing.assert_allclose(tilt_in_grad, tilts_in)
    np.testing.assert_allclose(tilt_out_grad, 6.0 * tilts_out)


def test_minimizer_delegates_leaflet_gradient_legacy_fallback() -> None:
    mesh = _single_vertex_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [_LeafletGradientRejectsTrialTiltsModule()]
    minim.energy_module_names = ["legacy_leaflet"]

    tilt_in_grad = np.ones((1, 3), dtype=float)
    tilt_out_grad = np.ones((1, 3), dtype=float)
    energy = minim._compute_energy_and_leaflet_tilt_gradients_array(
        positions=mesh.positions_view(),
        tilts_in=np.asarray([[1.0, 2.0, 3.0]], dtype=float),
        tilts_out=np.asarray([[4.0, 5.0, 6.0]], dtype=float),
        tilt_in_grad_arr=tilt_in_grad,
        tilt_out_grad_arr=tilt_out_grad,
    )

    assert energy == pytest.approx(17.0)
    np.testing.assert_allclose(tilt_in_grad, np.zeros_like(tilt_in_grad))
    np.testing.assert_allclose(tilt_out_grad, np.zeros_like(tilt_out_grad))


def test_minimizer_delegates_leaflet_gradient_scaled_deltas() -> None:
    mesh = _single_vertex_mesh()
    mesh.global_parameters.set(
        "curved_theta_objective_ablation_mode", "inner_outer_rescaled"
    )
    mesh.global_parameters.set("benchmark_geometry_lane", "free_z")
    mesh.global_parameters.set("benchmark_parameterization", "kh_physical")
    mesh.global_parameters.set("curved_theta_objective_ablation_inner_scale", 2.0)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [_LeafletGradientMutatingModule()]
    minim.energy_module_names = ["tilt_smoothness_in"]

    tilts_in = np.asarray([[1.0, 2.0, 3.0]], dtype=float)
    tilts_out = np.asarray([[4.0, 5.0, 6.0]], dtype=float)
    tilt_in_grad = np.zeros_like(tilts_in)
    tilt_out_grad = np.zeros_like(tilts_out)
    energy = minim._compute_energy_and_leaflet_tilt_gradients_array(
        positions=mesh.positions_view(),
        tilts_in=tilts_in,
        tilts_out=tilts_out,
        tilt_in_grad_arr=tilt_in_grad,
        tilt_out_grad_arr=tilt_out_grad,
    )

    assert energy == pytest.approx(26.0)
    np.testing.assert_allclose(tilt_in_grad, 2.0 * tilts_in)
    np.testing.assert_allclose(tilt_out_grad, 4.0 * tilts_out)


def test_minimizer_delegates_single_tilt_gradient_path_with_sync() -> None:
    mesh = _single_vertex_mesh()
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager([]),
        ConstraintModuleManager([]),
        quiet=True,
    )
    minim.energy_modules = [_NonTiltEnergyModule(), _SingleTiltGradientModule()]
    minim.energy_module_names = ["non_tilt", "tilt"]

    tilts = np.asarray([[1.0, 2.0, 3.0]], dtype=float)
    tilt_grad = np.zeros_like(tilts)
    energy = minim._compute_energy_and_tilt_gradient_array(
        positions=mesh.positions_view(),
        tilts=tilts,
        tilt_grad_arr=tilt_grad,
    )

    assert energy == pytest.approx(6.0)
    np.testing.assert_allclose(tilt_grad, tilts)
    assert minim._evaluation_manager.energy_modules is minim.energy_modules

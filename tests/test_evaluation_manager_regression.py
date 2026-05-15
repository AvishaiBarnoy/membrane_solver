from __future__ import annotations

import numpy as np
import pytest

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.entities import Mesh, Vertex
from runtime.energy_context import EnergyContext
from runtime.evaluation_manager import EvaluationManager


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


def _manager(
    mesh: Mesh,
    modules: list[object],
    names: list[str],
    *,
    ctx: EnergyContext | None = None,
    scale_fn=None,
) -> EvaluationManager:
    if ctx is None:
        ctx = EnergyContext()
    if scale_fn is None:

        def scale_fn(_name):
            return 1.0

    return EvaluationManager(
        mesh=mesh,
        global_params=mesh.global_parameters,
        param_resolver=ParameterResolver(mesh.global_parameters),
        energy_modules=modules,
        energy_module_names=names,
        energy_context_fn=lambda: ctx,
        experimental_energy_scale_fn=scale_fn,
    )


class _EnergyArrayWithResolverAliasAndCtx:
    def __init__(self) -> None:
        self.seen_ctx: EnergyContext | None = None
        self.seen_resolver: ParameterResolver | None = None

    def compute_energy_array(
        self,
        mesh,
        global_params,
        *,
        positions,
        index_map,
        resolver,
        ctx,
    ) -> float:
        _ = mesh, global_params, positions, index_map
        self.seen_ctx = ctx
        self.seen_resolver = resolver
        return 3.0


class _GradientModuleAssertsFreshScratch:
    def __init__(self, energy: float, fill_value: float) -> None:
        self.energy = float(energy)
        self.fill_value = float(fill_value)

    def compute_energy_and_gradient_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        grad_arr,
    ) -> np.ndarray:
        _ = mesh, global_params, param_resolver, positions, index_map
        np.testing.assert_allclose(grad_arr, np.zeros_like(grad_arr))
        grad_arr += self.fill_value
        return np.asarray([self.energy], dtype=float)


class _SingleTiltDictGradientModule:
    USES_TILT = True

    def compute_energy_and_gradient(self, mesh, global_params, param_resolver):
        _ = mesh, global_params, param_resolver
        return (
            4.0,
            {},
            {
                0: np.asarray([1.0, 2.0, 3.0], dtype=float),
                99: np.asarray([9.0, 9.0, 9.0], dtype=float),
            },
        )


class _ScaledTiltArrayGradientModule:
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
        tilts,
        tilt_grad_arr,
    ) -> float:
        _ = mesh, global_params, param_resolver, positions, index_map, grad_arr
        tilt_grad_arr += tilts
        return 5.0


def test_evaluation_manager_adapts_resolver_alias_and_ctx() -> None:
    mesh = _single_vertex_mesh()
    ctx = EnergyContext()
    module = _EnergyArrayWithResolverAliasAndCtx()
    manager = _manager(mesh, [module], ["array"], ctx=ctx)

    energy = manager.compute_energy_array_total(positions=mesh.positions_view())

    assert energy == pytest.approx(3.0)
    assert module.seen_ctx is ctx
    assert module.seen_resolver is manager.param_resolver


def test_evaluation_manager_resets_energy_only_scratch_between_modules() -> None:
    mesh = _single_vertex_mesh()
    first = _GradientModuleAssertsFreshScratch(energy=2.0, fill_value=7.0)
    second = _GradientModuleAssertsFreshScratch(energy=3.0, fill_value=11.0)
    manager = _manager(mesh, [first, second], ["first", "second"])

    energy = manager.compute_energy_array_total(positions=mesh.positions_view())

    assert energy == pytest.approx(5.0)


def test_evaluation_manager_single_tilt_dict_gradient_fallback_maps_known_rows() -> (
    None
):
    mesh = _single_vertex_mesh()
    manager = _manager(mesh, [_SingleTiltDictGradientModule()], ["tilt"])
    tilt_grad = np.zeros((1, 3), dtype=float)

    energy = manager.compute_energy_and_tilt_gradient_array(
        positions=mesh.positions_view(),
        tilts=np.zeros((1, 3), dtype=float),
        tilt_grad_arr=tilt_grad,
    )

    assert energy == pytest.approx(4.0)
    np.testing.assert_allclose(tilt_grad, [[1.0, 2.0, 3.0]])


def test_evaluation_manager_single_tilt_scaled_array_gradient_scales_delta() -> None:
    mesh = _single_vertex_mesh()
    manager = _manager(
        mesh,
        [_ScaledTiltArrayGradientModule()],
        ["scaled"],
        scale_fn=lambda name: 2.0 if name == "scaled" else 1.0,
    )
    tilts = np.asarray([[1.0, 2.0, 3.0]], dtype=float)
    tilt_grad = np.full_like(tilts, 100.0)

    energy = manager.compute_energy_and_tilt_gradient_array(
        positions=mesh.positions_view(),
        tilts=tilts,
        tilt_grad_arr=tilt_grad,
    )

    assert energy == pytest.approx(10.0)
    np.testing.assert_allclose(tilt_grad, 2.0 * tilts)

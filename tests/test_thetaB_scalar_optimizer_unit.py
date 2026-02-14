import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Mesh, Vertex
from runtime.minimizer import Minimizer


class _DummyEnergyManager:
    def __init__(self, module):
        self._module = module
        self.modules = {"dummy": module}

    def get_module(self, name: str):
        assert name == "dummy"
        return self._module


class _DummyConstraintManager:
    def get_constraint(self, name: str):
        raise AssertionError(f"unexpected constraint: {name}")


class _DummyStepper:
    pass


class _ThetaBQuadraticEnergy:
    """Quadratic energy in thetaB only (used to unit-test scalar optimization)."""

    USES_TILT_LEAFLETS = True

    def __init__(self, target: float):
        self._target = float(target)

    def compute_energy_and_gradient_array(
        self,
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        grad_arr,
    ):
        thetaB = float(global_params.get("tilt_thetaB_value") or 0.0)
        return (thetaB - self._target) ** 2


def _minimizer_with_dummy_energy(
    *, target: float, global_params: GlobalParameters
) -> Minimizer:
    mesh = Mesh()
    mesh.global_parameters = global_params
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.vertices[2] = Vertex(2, np.array([0.0, 1.0, 0.0]))

    energy = _ThetaBQuadraticEnergy(target=target)
    energy_manager = _DummyEnergyManager(energy)
    constraint_manager = _DummyConstraintManager()
    stepper = _DummyStepper()
    return Minimizer(
        mesh,
        global_params,
        stepper,
        energy_manager,  # type: ignore[arg-type]
        constraint_manager,  # type: ignore[arg-type]
        energy_modules=["dummy"],
        constraint_modules=[],
        quiet=True,
    )


@pytest.mark.unit
def test_thetaB_scalar_optimizer_moves_thetaB_toward_lower_energy():
    gp = GlobalParameters(
        {
            "tilt_thetaB_value": 0.0,
            "tilt_thetaB_optimize": True,
            "tilt_thetaB_optimize_every": 1,
            "tilt_thetaB_optimize_delta": 0.1,
            "tilt_thetaB_optimize_inner_steps": 1,
        }
    )
    minimizer = _minimizer_with_dummy_energy(target=0.25, global_params=gp)

    e0 = minimizer.compute_energy()
    minimizer._optimize_thetaB_scalar(tilt_mode="fixed", iteration=0)
    e1 = minimizer.compute_energy()

    assert e1 < e0
    assert float(gp.get("tilt_thetaB_value")) != 0.0


@pytest.mark.unit
def test_thetaB_scalar_optimizer_restores_thetaB_when_best_is_current_point():
    gp = GlobalParameters(
        {
            "tilt_thetaB_value": 0.0,
            "tilt_thetaB_optimize": True,
            "tilt_thetaB_optimize_every": 1,
            "tilt_thetaB_optimize_delta": 0.1,
            "tilt_thetaB_optimize_inner_steps": 1,
        }
    )
    minimizer = _minimizer_with_dummy_energy(target=0.0, global_params=gp)

    minimizer._optimize_thetaB_scalar(tilt_mode="fixed", iteration=0)
    assert float(gp.get("tilt_thetaB_value")) == 0.0


@pytest.mark.unit
def test_thetaB_scalar_optimizer_restores_tilt_inner_steps_semantics():
    gp = GlobalParameters(
        {
            "tilt_thetaB_value": 0.0,
            "tilt_thetaB_optimize": True,
            "tilt_thetaB_optimize_every": 1,
            "tilt_thetaB_optimize_delta": 0.1,
            "tilt_thetaB_optimize_inner_steps": 1,
        }
    )
    minimizer = _minimizer_with_dummy_energy(target=0.25, global_params=gp)

    assert "tilt_inner_steps" not in gp
    minimizer._optimize_thetaB_scalar(tilt_mode="fixed", iteration=0)
    assert "tilt_inner_steps" not in gp

    gp.set("tilt_inner_steps", 123)
    minimizer._optimize_thetaB_scalar(tilt_mode="fixed", iteration=1)
    assert int(gp.get("tilt_inner_steps")) == 123


@pytest.mark.unit
def test_set_leaflet_tilts_from_arrays_fast_updates_mesh_views_and_vertices():
    gp = GlobalParameters({"tilt_thetaB_optimize": False})
    minimizer = _minimizer_with_dummy_energy(target=0.0, global_params=gp)
    mesh = minimizer.mesh

    tin = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]], dtype=float)
    tout = np.asarray(
        [[-0.1, -0.2, -0.3], [-0.4, -0.5, -0.6], [-0.7, -0.8, -0.9]], dtype=float
    )

    minimizer._set_leaflet_tilts_from_arrays_fast(tin, tout)

    assert np.allclose(mesh.tilts_in_view(), tin)
    assert np.allclose(mesh.tilts_out_view(), tout)
    # Vertex accessors must reflect cache-backed values.
    assert np.allclose(mesh.vertices[0].tilt_in, tin[0])
    assert np.allclose(mesh.vertices[2].tilt_out, tout[2])

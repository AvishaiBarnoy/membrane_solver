"""Test that _optimize_thetaB_scalar rolls back when candidates worsen energy."""

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


class _QuadraticEnergyWithMinimum:
    """Energy that is quadratic in thetaB with a tunable minimum.

    When ``spike_on_perturb`` is True, any thetaB != target returns a very
    high energy, simulating an unstable regime after mesh refinement.
    """

    USES_TILT_LEAFLETS = True

    def __init__(self, target: float, *, spike_on_perturb: bool = False):
        self._target = float(target)
        self._spike = spike_on_perturb

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
        if self._spike and abs(thetaB - self._target) > 1e-12:
            return 1e6  # Very high energy for any perturbation
        return (thetaB - self._target) ** 2


def _build_minimizer(
    *, target: float, global_params: GlobalParameters, spike_on_perturb: bool = False
) -> Minimizer:
    mesh = Mesh()
    mesh.global_parameters = global_params
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.vertices[2] = Vertex(2, np.array([0.0, 1.0, 0.0]))

    energy = _QuadraticEnergyWithMinimum(
        target=target, spike_on_perturb=spike_on_perturb
    )
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
def test_thetaB_optimizer_rollback_when_candidates_worsen_energy():
    """When both ±delta candidates increase energy, the optimizer should
    roll back to the original thetaB and tilts."""
    gp = GlobalParameters(
        {
            "tilt_thetaB_value": 0.5,
            "tilt_thetaB_optimize": True,
            "tilt_thetaB_optimize_every": 1,
            "tilt_thetaB_optimize_delta": 0.1,
            "tilt_thetaB_optimize_inner_steps": 1,
        }
    )
    # Energy is quadratic around 0.5, AND spikes for any perturbation.
    # This simulates the post-refinement regime where the optimizer
    # should not change thetaB.
    minimizer = _build_minimizer(target=0.5, global_params=gp, spike_on_perturb=True)

    e0 = minimizer.compute_energy()
    assert e0 == pytest.approx(0.0, abs=1e-12)

    minimizer._optimize_thetaB_scalar(tilt_mode="fixed", iteration=0)

    # thetaB should remain unchanged
    assert float(gp.get("tilt_thetaB_value")) == pytest.approx(0.5)
    # Energy should still be the baseline
    e1 = minimizer.compute_energy()
    assert e1 == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_thetaB_optimizer_accepts_improving_candidates():
    """When a candidate improves energy, it should be accepted (no rollback)."""
    gp = GlobalParameters(
        {
            "tilt_thetaB_value": 0.0,
            "tilt_thetaB_optimize": True,
            "tilt_thetaB_optimize_every": 1,
            "tilt_thetaB_optimize_delta": 0.1,
            "tilt_thetaB_optimize_inner_steps": 1,
        }
    )
    # Minimum at 0.25 — one of ±0.1 candidates should improve.
    minimizer = _build_minimizer(target=0.25, global_params=gp, spike_on_perturb=False)

    e0 = minimizer.compute_energy()
    minimizer._optimize_thetaB_scalar(tilt_mode="fixed", iteration=0)
    e1 = minimizer.compute_energy()

    assert e1 < e0
    assert float(gp.get("tilt_thetaB_value")) != 0.0


@pytest.mark.unit
def test_thetaB_optimizer_rollback_preserves_tilts():
    """When rolling back, the original tilt arrays should be restored."""
    gp = GlobalParameters(
        {
            "tilt_thetaB_value": 0.5,
            "tilt_thetaB_optimize": True,
            "tilt_thetaB_optimize_every": 1,
            "tilt_thetaB_optimize_delta": 0.1,
            "tilt_thetaB_optimize_inner_steps": 1,
        }
    )
    minimizer = _build_minimizer(target=0.5, global_params=gp, spike_on_perturb=True)
    mesh = minimizer.mesh

    # Set known tilt values
    tin_original = np.array([[0.01, 0.02, 0.0], [0.03, 0.04, 0.0], [0.05, 0.06, 0.0]])
    tout_original = np.array(
        [[-0.01, -0.02, 0.0], [-0.03, -0.04, 0.0], [-0.05, -0.06, 0.0]]
    )
    minimizer._set_leaflet_tilts_from_arrays_fast(tin_original, tout_original)

    minimizer._optimize_thetaB_scalar(tilt_mode="fixed", iteration=0)

    # Tilts should be restored to the original values
    np.testing.assert_allclose(mesh.tilts_in_view(), tin_original, atol=1e-12)
    np.testing.assert_allclose(mesh.tilts_out_view(), tout_original, atol=1e-12)

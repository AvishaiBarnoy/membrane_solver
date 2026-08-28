import numpy as np
import pytest

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

    def __init__(self, target: float, *, spike_on_perturb: bool = False):
        self._target = float(target)
        self._spike_on_perturb = bool(spike_on_perturb)

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
        if self._spike_on_perturb and abs(thetaB - self._target) > 1e-12:
            return 1e6
        return (thetaB - self._target) ** 2


def _minimizer_with_dummy_energy(
    *,
    target: float,
    global_params: GlobalParameters,
    spike_on_perturb: bool = False,
) -> Minimizer:
    mesh = Mesh()
    mesh.global_parameters = global_params
    mesh.vertices[0] = Vertex(0, np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(1, np.array([1.0, 0.0, 0.0]))
    mesh.vertices[2] = Vertex(2, np.array([0.0, 1.0, 0.0]))

    energy = _ThetaBQuadraticEnergy(
        target=target,
        spike_on_perturb=spike_on_perturb,
    )
    energy_manager = _DummyEnergyManager(energy)
    constraint_manager = _DummyConstraintManager()
    stepper = _DummyStepper()
    minimizer = Minimizer(
        mesh,
        global_params,
        stepper,
        energy_manager,  # type: ignore[arg-type]
        constraint_manager,  # type: ignore[arg-type]
        energy_modules=["dummy"],
        constraint_modules=[],
        quiet=True,
    )
    minimizer.compute_energy_breakdown = lambda: {
        "bending_tilt_in": 1.0,
        "bending_tilt_out": 2.0,
        "tilt_in": 3.0,
        "tilt_out": 4.0,
        "tilt_thetaB_contact_in": 5.0,
    }
    return minimizer


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
    trace = getattr(minimizer.mesh, "_thetaB_scan_trace")
    assert trace[-1]["status"] == "evaluated"
    assert len(trace[-1]["candidate_energies"]) == 3
    for cand in trace[-1]["candidate_energies"]:
        assert "bending_tilt_in" in cand
        assert cand["bending_tilt_in"] == 1.0
        assert "bending_tilt_out" in cand
        assert cand["bending_tilt_out"] == 2.0
        assert "tilt_in" in cand
        assert cand["tilt_in"] == 3.0
        assert "tilt_out" in cand
        assert cand["tilt_out"] == 4.0
        assert "tilt_thetaB_contact_in" in cand
        assert cand["tilt_thetaB_contact_in"] == 5.0


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


@pytest.mark.unit
def test_thetaB_optimizer_rollback_when_candidates_worsen_energy():
    gp = GlobalParameters(
        {
            "tilt_thetaB_value": 0.5,
            "tilt_thetaB_optimize": True,
            "tilt_thetaB_optimize_every": 1,
            "tilt_thetaB_optimize_delta": 0.1,
            "tilt_thetaB_optimize_inner_steps": 1,
        }
    )
    minimizer = _minimizer_with_dummy_energy(
        target=0.5,
        global_params=gp,
        spike_on_perturb=True,
    )

    assert minimizer.compute_energy() == pytest.approx(0.0, abs=1e-12)
    minimizer._optimize_thetaB_scalar(tilt_mode="fixed", iteration=0)
    assert float(gp.get("tilt_thetaB_value")) == pytest.approx(0.5)
    assert minimizer.compute_energy() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_thetaB_optimizer_accepts_improving_candidates():
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

    energy_before = minimizer.compute_energy()
    minimizer._optimize_thetaB_scalar(tilt_mode="fixed", iteration=0)
    assert minimizer.compute_energy() < energy_before
    assert float(gp.get("tilt_thetaB_value")) != 0.0


@pytest.mark.unit
def test_thetaB_optimizer_rollback_preserves_tilts():
    gp = GlobalParameters(
        {
            "tilt_thetaB_value": 0.5,
            "tilt_thetaB_optimize": True,
            "tilt_thetaB_optimize_every": 1,
            "tilt_thetaB_optimize_delta": 0.1,
            "tilt_thetaB_optimize_inner_steps": 1,
        }
    )
    minimizer = _minimizer_with_dummy_energy(
        target=0.5,
        global_params=gp,
        spike_on_perturb=True,
    )
    tin_original = np.array([[0.01, 0.02, 0.0], [0.03, 0.04, 0.0], [0.05, 0.06, 0.0]])
    tout_original = np.array(
        [[-0.01, -0.02, 0.0], [-0.03, -0.04, 0.0], [-0.05, -0.06, 0.0]]
    )
    minimizer._set_leaflet_tilts_from_arrays_fast(tin_original, tout_original)

    minimizer._optimize_thetaB_scalar(tilt_mode="fixed", iteration=0)

    np.testing.assert_allclose(minimizer.mesh.tilts_in_view(), tin_original, atol=1e-12)
    np.testing.assert_allclose(
        minimizer.mesh.tilts_out_view(), tout_original, atol=1e-12
    )

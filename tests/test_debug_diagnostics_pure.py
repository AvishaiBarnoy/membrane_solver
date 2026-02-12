import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


class _MutatingModule:
    USES_TILT_LEAFLETS = False

    @staticmethod
    def compute_energy_and_gradient_array(
        mesh,
        global_params,
        param_resolver,
        *,
        positions,
        index_map,
        grad_arr,
        **kwargs,
    ) -> float:
        _ = param_resolver, positions, index_map, kwargs
        tilts = mesh.tilts_view()
        tilts[0, 0] += 1.0
        global_params.set(
            "debug_purity_probe", global_params.get("debug_purity_probe", 0) + 1
        )
        grad_arr[:, 0] += 0.0
        return 0.0


def test_debug_diagnostics_restore_state(caplog) -> None:
    caplog.set_level(logging.DEBUG, logger="membrane_solver")
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_free_disk_coarse_refinable.yaml")
    )
    mesh.energy_modules = ["mutator"]
    mesh.global_parameters.set("debug_purity_probe", 0)

    energy_manager = EnergyModuleManager([])
    energy_manager.modules["mutator"] = _MutatingModule

    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        energy_manager,
        ConstraintModuleManager([]),
        energy_modules=["mutator"],
        constraint_modules=[],
        quiet=True,
    )

    tilts_before = mesh.tilts_view().copy()
    params_before = dict(mesh.global_parameters.to_dict())

    minim._log_debug_energy_context(0)

    tilts_after = mesh.tilts_view()
    params_after = dict(mesh.global_parameters.to_dict())

    assert np.array_equal(tilts_before, tilts_after)
    assert params_before == params_after


def test_debug_energy_context_does_not_call_breakdown(caplog, monkeypatch) -> None:
    caplog.set_level(logging.DEBUG, logger="membrane_solver")
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_free_disk_coarse_refinable.yaml")
    )
    energy_manager = EnergyModuleManager(mesh.energy_modules)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        energy_manager,
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )

    def _boom(self):
        raise AssertionError("breakdown should not be called from debug context")

    monkeypatch.setattr(Minimizer, "_diagnostic_energy_breakdown", _boom)
    minim._log_debug_energy_context(0)


def test_debug_minimize_loop_does_not_probe_diagnostic_energy(monkeypatch) -> None:
    logging.getLogger("membrane_solver").setLevel(logging.DEBUG)
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_free_disk_coarse_refinable.yaml")
    )
    energy_manager = EnergyModuleManager(mesh.energy_modules)
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        energy_manager,
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )

    def _boom(self):
        raise AssertionError("_diagnostic_energy should not be called in minimize loop")

    monkeypatch.setattr(Minimizer, "_diagnostic_energy", _boom)
    minim.minimize(n_steps=1)

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _relax_rect_tilt_source(
    *, tilt_rigidity: float, inner_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Relax the single-source rectangle benchmark and return (x, |t|) arrays."""
    mesh = parse_geometry(
        load_data("meshes/tilt_benchmarks/tilt_source_rect_single.yaml")
    )
    mesh.global_parameters.update(
        {
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.05,
            "tilt_inner_steps": int(inner_steps),
            "tilt_tol": 0.0,
            "tilt_smoothness_rigidity": 1.0,
            "tilt_rigidity": float(tilt_rigidity),
        }
    )

    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    minim._relax_tilts(positions=mesh.positions_view(), mode="nested")

    positions = mesh.positions_view()
    x = positions[:, 0].copy()
    mags = np.linalg.norm(mesh.tilts_view(), axis=1)
    return x, mags


def test_single_source_tilt_decays_across_rectangle() -> None:
    """E2E: relax tilts on a fixed rectangle and assert decay away from a source edge.

    The benchmark mesh `tilt_source_rect_single.yaml` enables `tilt_smoothness`
    and `tilt` energies. For the continuum proxy energy

        E = ∫ (k_s/2)|∇t|^2 + (k_t/2)|t|^2 dA

    the characteristic decay length is λ ≈ sqrt(k_s / k_t). With k_s=k_t=1,
    we expect |t| to decay over O(1) distance in mesh units.
    """

    mesh = parse_geometry(
        load_data("meshes/tilt_benchmarks/tilt_source_rect_single.yaml")
    )
    mesh.global_parameters.update(
        {
            "tilt_solve_mode": "nested",
            "tilt_step_size": 0.05,
            "tilt_inner_steps": 400,
            "tilt_tol": 0.0,
        }
    )

    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )
    minim._relax_tilts(positions=mesh.positions_view(), mode="nested")

    positions = mesh.positions_view()
    mags = np.linalg.norm(mesh.tilts_view(), axis=1)
    x = positions[:, 0]

    def mean_mag_at(x0: float) -> float:
        idx = np.where(np.isclose(x, x0))[0]
        assert idx.size > 0
        return float(mags[idx].mean())

    m0 = mean_mag_at(0.0)
    m2 = mean_mag_at(2.0)
    m3 = mean_mag_at(3.0)
    m4 = mean_mag_at(4.0)

    assert m0 == pytest.approx(1.0, abs=1e-12)
    assert m0 > m2 > m3 > m4
    assert m4 < 0.08


def test_higher_tilt_modulus_shortens_decay_length() -> None:
    """E2E regression: increasing k_t makes tilt decay more quickly."""

    def mean_mag_at(x: np.ndarray, mags: np.ndarray, x0: float) -> float:
        idx = np.where(np.isclose(x, x0))[0]
        assert idx.size > 0
        return float(mags[idx].mean())

    x_lo, mags_lo = _relax_rect_tilt_source(tilt_rigidity=1.0, inner_steps=800)
    x_hi, mags_hi = _relax_rect_tilt_source(tilt_rigidity=4.0, inner_steps=800)
    assert np.allclose(x_lo, x_hi)

    m2_lo = mean_mag_at(x_lo, mags_lo, 2.0)
    m4_lo = mean_mag_at(x_lo, mags_lo, 4.0)
    m2_hi = mean_mag_at(x_hi, mags_hi, 2.0)
    m4_hi = mean_mag_at(x_hi, mags_hi, 4.0)

    assert m2_hi < 0.15 * m2_lo
    assert m4_hi < 0.15 * m4_lo

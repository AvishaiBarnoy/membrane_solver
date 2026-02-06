import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


def _run_free_disk(*, base_term_boundary_group: str | None) -> tuple[float, float]:
    """Return (bending_tilt_in, tilt_in) after a short deterministic relax."""
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    gp = mesh.global_parameters
    # This regression targets the base-term boundary-ring knob specifically.
    # Ensure the newer theory-mode J=0-on-disk masking is disabled here so the
    # boundary-group effect remains observable.
    try:
        gp.unset("bending_tilt_assume_J0_presets_in")
    except Exception:
        gp.set("bending_tilt_assume_J0_presets_in", None)

    # Keep the run deterministic and bounded.
    gp.set("tilt_thetaB_optimize", False)
    gp.set("tilt_thetaB_value", 0.03)

    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 10)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)

    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 1.0e-3)

    if base_term_boundary_group is None:
        try:
            gp.unset("bending_tilt_base_term_boundary_group_in")
        except Exception:
            gp.set("bending_tilt_base_term_boundary_group_in", "")
    else:
        gp.set("bending_tilt_base_term_boundary_group_in", base_term_boundary_group)

    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )
    minim.minimize(n_steps=3)
    bd = minim.compute_energy_breakdown()
    return float(bd.get("bending_tilt_in") or 0.0), float(bd.get("tilt_in") or 0.0)


@pytest.mark.regression
def test_bending_tilt_in_base_term_boundary_group_drops_spurious_disk_ring_curvature() -> (
    None
):
    """Disk interface ring should not carry bulk curvature penalty in 'hard disk' models."""
    bend_off, tilt_off = _run_free_disk(base_term_boundary_group=None)
    bend_on, tilt_on = _run_free_disk(base_term_boundary_group="disk")

    # Sanity: driven inner tilt should remain non-trivial.
    assert tilt_off > 0.1 and tilt_on > 0.1

    # Enabling the base-term boundary group must dramatically reduce the
    # curvature-driven bending_tilt_in contribution.
    assert bend_on < 0.2 * bend_off

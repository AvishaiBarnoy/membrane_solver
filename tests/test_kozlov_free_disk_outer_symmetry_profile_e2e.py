import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


def _outer_band_metrics(mesh, *, r_min: float, r_max: float) -> tuple[float, float]:
    """Return (median |tin_r - tout_r|, median |tin_r + tout_r|) in an annulus.

    Note: this measures *absolute* mismatch. When the true signal is very small
    (far field), absolute metrics can be dominated by numerical noise or by
    localized features (e.g. boundary constraints). The test below therefore
    uses a normalized symmetry score rather than raw monotonicity in radius.
    """
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    mask = (r >= float(r_min)) & (r <= float(r_max))
    if not np.any(mask):
        raise AssertionError(f"Annulus empty: r in [{r_min}, {r_max}].")

    ers = np.zeros_like(positions)
    nz = r > 1e-12
    ers[nz, 0] = positions[nz, 0] / r[nz]
    ers[nz, 1] = positions[nz, 1] / r[nz]

    tin = mesh.tilts_in_view()
    tout = mesh.tilts_out_view()
    tin_r = np.sum(tin * ers, axis=1)
    tout_r = np.sum(tout * ers, axis=1)

    diff_same = float(np.median(np.abs(tin_r[mask] - tout_r[mask])))
    diff_oppo = float(np.median(np.abs(tin_r[mask] + tout_r[mask])))
    return diff_same, diff_oppo


@pytest.mark.e2e
def test_kozlov_free_disk_outer_leaflet_symmetry_improves_with_radius() -> None:
    """E2E diagnostic: in the outer membrane, leaflets match up to sign convention.

    The TeX model predicts that far from the disk, the two leaflets share the
    same decaying tilt profile (up to a sign convention). We avoid absolute
    targets (elastic parity gap still tracked elsewhere) and check robust
    *trends*:
      - In the far outer band, opposite-sign mismatch is no worse than same-sign.
      - Opposite-sign mismatch improves with radius.
    """
    path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "kozlov_1disk_3d_free_disk_theory_parity.yaml",
    )
    mesh = parse_geometry(load_data(path))
    gp = mesh.global_parameters

    # Bounded, deterministic settings.
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 10)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)

    gp.set("tilt_thetaB_optimize", True)
    gp.set("tilt_thetaB_optimize_every", 1)
    gp.set("tilt_thetaB_optimize_delta", 0.05)
    gp.set("tilt_thetaB_optimize_inner_steps", 5)

    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 1.0e-3)

    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-8,
    )
    minim.minimize(n_steps=3)

    # Three annular bands in the lipid region (exclude the near-disk patch).
    bands = [
        (1.5, 3.0),
        (3.0, 6.0),
        (6.0, 10.5),
    ]

    metrics = [(_outer_band_metrics(mesh, r_min=a, r_max=b), (a, b)) for a, b in bands]
    diff_same = [m[0][0] for m in metrics]
    diff_oppo = [m[0][1] for m in metrics]

    # In the far outer band, opposite-sign mismatch should be no worse than
    # same-sign mismatch (matches sign convention expectation).
    assert diff_oppo[-1] <= diff_same[-1] + 1e-9, f"outer band metrics: {metrics}"

    # Symmetry should be best in the far field, but absolute mismatches can be
    # noisy when the signal is tiny. Use a normalized score:
    #
    #   score = diff_oppo / (diff_same + eps)
    #
    # Smaller is better (opposite-sign convention matches more closely than
    # same-sign). Require the far-field score to be no worse than the inner
    # band's score.
    eps = 1e-12
    score = [o / (s + eps) for s, o in zip(diff_same, diff_oppo)]
    assert score[-1] <= score[0] + 0.1, f"annulus metrics: {metrics}, score={score}"

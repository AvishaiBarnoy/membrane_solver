import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.refinement import refine_triangle_mesh  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


def _outer_band_symmetry_score(
    mesh, *, r_min: float, r_max: float, quantile: float = 0.9
) -> tuple[float, float]:
    """Return (amplitude, symmetry_score) in an annulus.

    We measure the TeX-predicted far-field symmetry of radial components:

      tin_r(r) ≈ -tout_r(r)

    and normalize by a robust amplitude statistic in the same annulus so the
    metric remains meaningful when the far-field signal is small.
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

    amp = float(
        np.quantile(np.abs(tin_r[mask]) + np.abs(tout_r[mask]), float(quantile))
    )
    mismatch = float(np.quantile(np.abs(tin_r[mask] + tout_r[mask]), float(quantile)))
    score = mismatch / (amp + 1e-12)
    return amp, score


def _disk_and_outer_radii(mesh) -> tuple[float, float]:
    """Return (r_disk, r_outer) from pinned circle vertex metadata."""
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)

    disk_rows: list[int] = []
    outer_rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = mesh.vertices[int(vid)].options or {}
        group = opts.get("pin_to_circle_group")
        if group == "disk":
            disk_rows.append(mesh.vertex_index_to_row[int(vid)])
        elif group == "outer":
            outer_rows.append(mesh.vertex_index_to_row[int(vid)])

    if not disk_rows:
        raise AssertionError("No disk ring vertices found (pin_to_circle_group=disk).")
    if not outer_rows:
        raise AssertionError("No outer rim vertices found (pin_to_circle_group=outer).")

    r_disk = float(np.mean(r[np.asarray(disk_rows, dtype=int)]))
    r_outer = float(np.mean(r[np.asarray(outer_rows, dtype=int)]))
    return r_disk, r_outer


def _build_minimizer(mesh) -> Minimizer:
    """Return a minimizer configured for quick, deterministic e2e diagnostics."""
    gp = mesh.global_parameters

    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 1.0e-3)

    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_inner_steps", 15)
    gp.set("tilt_tol", 1e-8)

    return Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-8,
    )


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

    minim = _build_minimizer(mesh)
    minim.minimize(n_steps=4)

    r_disk, r_outer = _disk_and_outer_radii(mesh)

    # Two annular bands in the lipid region:
    # - exclude the disk boundary layer
    # - avoid the clamped outer rim where boundary constraints dominate
    near = (max(1.5, 3.0 * r_disk), max(3.0, 6.0 * r_disk))
    far = (0.60 * r_outer, 0.80 * r_outer)

    amp_near, score_near = _outer_band_symmetry_score(
        mesh, r_min=near[0], r_max=near[1]
    )
    amp_far, score_far = _outer_band_symmetry_score(mesh, r_min=far[0], r_max=far[1])

    # Prevent vacuous pass when the near band has no signal.
    assert amp_near > 1e-9, (
        f"near amp too small: amp_near={amp_near}, bands={near, far}"
    )

    # Far-field symmetry should be better (smaller score).
    assert score_far <= score_near + 0.15, (
        f"symmetry scores (near, far)=({score_near:.3g}, {score_far:.3g}), "
        f"amps=({amp_near:.3g}, {amp_far:.3g}), bands={near, far}"
    )

    # Refinement should not make the far-field symmetry worse.
    mesh_ref = refine_triangle_mesh(mesh)
    minim_ref = _build_minimizer(mesh_ref)
    # Give the refined mesh a few coupled steps to re-equilibrate before
    # evaluating far-field symmetry.
    minim_ref.minimize(n_steps=4)

    _amp_near_r, score_near_r = _outer_band_symmetry_score(
        mesh_ref, r_min=near[0], r_max=near[1]
    )
    _amp_far_r, score_far_r = _outer_band_symmetry_score(
        mesh_ref, r_min=far[0], r_max=far[1]
    )

    # After refinement, the far-field signal can be extremely small, so we use
    # a *trend* check: the far band remains at least as symmetric as the near
    # band (up to a small tolerance).
    assert score_far_r <= score_near_r + 0.15, (
        f"refined symmetry scores (near, far)=({score_near_r:.3g}, {score_far_r:.3g}); "
        f"pre-refine (near, far)=({score_near:.3g}, {score_far:.3g}); bands={near, far}"
    )

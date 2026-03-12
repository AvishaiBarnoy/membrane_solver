import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.refinement import refine_triangle_mesh  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402
from tools.diagnostics.free_disk_profile_protocol import (  # noqa: E402
    configure_free_disk_curved_bilayer_stage2,
    load_free_disk_curved_bilayer_mesh,
    measure_free_disk_curved_bilayer_near_rim,
    run_free_disk_curved_bilayer_theta_sweep,
)

pytestmark = pytest.mark.regression


def _build_minimizer(mesh) -> Minimizer:
    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def _free_disk_rows(mesh) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("preset") == "disk":
            rows.append(mesh.vertex_index_to_row[int(vid)])
    return np.asarray(rows, dtype=int)


def test_curved_bilayer_loader_keeps_shared_rim_and_resolves_first_outer_ring() -> None:
    mesh = load_free_disk_curved_bilayer_mesh()
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)

    rim_rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("rim_slope_match_group") == "rim":
            rim_rows.append(mesh.vertex_index_to_row[int(vid)])

    rim_rows_arr = np.asarray(rim_rows, dtype=int)
    assert rim_rows_arr.size == 24
    assert np.allclose(r[rim_rows_arr], 7.0 / 15.0, atol=1e-12, rtol=0.0)

    free_radii = sorted(
        {
            round(float(rr), 6)
            for rr in r
            if rr > (7.0 / 15.0 + 1.0e-6) and rr < 12.0 - 1.0e-6
        }
    )
    assert free_radii
    assert free_radii[0] == pytest.approx(0.582, abs=1.0e-6)


def test_curved_bilayer_loader_allows_outer_leaflet_tilt_on_disk_patch() -> None:
    mesh = load_free_disk_curved_bilayer_mesh()
    assert list(mesh.global_parameters.get("leaflet_out_absent_presets") or []) == []

    disk_rows = _free_disk_rows(mesh)
    assert disk_rows.size > 0
    free_disk_rows = [
        int(row)
        for row in disk_rows
        if not bool(
            (
                getattr(mesh.vertices[int(mesh.vertex_ids[int(row)])], "options", None)
                or {}
            ).get("tilt_fixed_out", False)
        )
    ]
    assert free_disk_rows

    minim = _build_minimizer(mesh)
    baseline = minim.compute_energy_breakdown()

    row = int(free_disk_rows[0])
    tout = mesh.tilts_out_view().copy(order="F")
    tout[row] = np.array([0.25, 0.0, 0.0], dtype=float)
    mesh.set_tilts_out_from_array(tout)
    perturbed = minim.compute_energy_breakdown()

    assert perturbed["tilt_out"] != baseline["tilt_out"]
    assert perturbed["bending_tilt_out"] != baseline["bending_tilt_out"]


def test_curved_bilayer_loader_uses_sliding_outer_height_gauge() -> None:
    mesh = load_free_disk_curved_bilayer_mesh()

    outer_rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("pin_to_circle_group") != "outer":
            continue
        outer_rows.append(mesh.vertex_index_to_row[int(vid)])
        assert opts.get("pin_to_plane_mode") == "slide"
        assert opts.get("pin_to_plane_group") == "outer_height_gauge"
        assert opts.get("pin_to_circle_mode") == "slide"
    assert len(outer_rows) == 24


def test_curved_bilayer_stage2_uses_outer_support_ring_as_inner_base_term_boundary() -> (
    None
):
    mesh = load_free_disk_curved_bilayer_mesh()
    shell_radius = configure_free_disk_curved_bilayer_stage2(mesh, theta_b=0.02)

    gp = mesh.global_parameters
    assert gp.get("bending_tilt_base_term_boundary_group_in") == "outer"
    assert gp.get("bending_tilt_base_term_boundary_group_out") == "rim"

    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    outer_rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("rim_slope_match_group") == "outer":
            outer_rows.append(mesh.vertex_index_to_row[int(vid)])

    outer_rows_arr = np.asarray(outer_rows, dtype=int)
    assert outer_rows_arr.size > 0
    assert np.allclose(r[outer_rows_arr], shell_radius, atol=1.0e-3, rtol=0.0)


def test_curved_bilayer_stage2_refined_shell_selection_skips_physical_rim() -> None:
    mesh = refine_triangle_mesh(load_free_disk_curved_bilayer_mesh())
    shell_radius = configure_free_disk_curved_bilayer_stage2(mesh, theta_b=0.02)

    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    rim_rows: list[int] = []
    outer_rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        row = mesh.vertex_index_to_row[int(vid)]
        if opts.get("rim_slope_match_group") == "rim":
            rim_rows.append(row)
        if opts.get("rim_slope_match_group") == "outer":
            outer_rows.append(row)

    rim_rows_arr = np.asarray(rim_rows, dtype=int)
    outer_rows_arr = np.asarray(outer_rows, dtype=int)
    assert rim_rows_arr.size > 0
    assert outer_rows_arr.size > 0
    assert shell_radius > float(np.max(r[rim_rows_arr])) + 1.0e-3
    assert np.allclose(r[outer_rows_arr], shell_radius, atol=1.0e-3, rtol=0.0)


def test_curved_bilayer_imposed_theta_sweep_reveals_near_rim_under_response() -> None:
    rows = run_free_disk_curved_bilayer_theta_sweep(
        [0.02, 0.05, 0.10, 0.15, 0.18],
        shape_steps=60,
        z_bump=3.0e-4,
    )

    assert len(rows) == 5

    # Shared-rim closure is already correct: the first free ring splits the
    # imposed drive between inner and outer channels.
    for row in rows:
        assert row["closure_error"] == pytest.approx(0.0, abs=1.0e-3)
        assert row["theta_out_phi_gap"] == pytest.approx(0.0, abs=5.0e-4)
        assert row["theta_disk"] == pytest.approx(row["theta_b"], rel=0.02, abs=1.0e-3)

    # The remaining mismatch is a curvature/outer-tilt under-response that
    # grows with imposed thetaB on the current mesh/branch.
    deficits = np.asarray([row["phi_deficit"] for row in rows], dtype=float)
    assert np.all(deficits > 0.0)
    assert deficits[-1] > deficits[0]

    last = rows[-1]
    assert last["phi_abs"] < (0.40 * last["theta_b"])
    assert last["theta_outer_out"] < (0.40 * last["theta_b"])


def test_curved_bilayer_refined_auto_seed_tracks_half_theta() -> None:
    for theta_b in (0.02, 0.04, 0.10):
        mesh = refine_triangle_mesh(load_free_disk_curved_bilayer_mesh())
        configure_free_disk_curved_bilayer_stage2(mesh, theta_b=theta_b, z_bump=None)

        minim = _build_minimizer(mesh)
        gp = mesh.global_parameters
        gp.set("tilt_solve_mode", "coupled")
        gp.set("tilt_solver", "gd")
        gp.set("tilt_step_size", 0.15)
        gp.set("tilt_inner_steps", 10)
        gp.set("tilt_tol", 1e-8)
        gp.set("tilt_kkt_projection_during_relaxation", False)
        gp.set("step_size_mode", "fixed")
        gp.set("step_size", 0.01)
        minim.minimize(n_steps=60)

        row = measure_free_disk_curved_bilayer_near_rim(mesh, theta_b=theta_b)
        target = 0.5 * theta_b
        assert row["closure_error"] == pytest.approx(0.0, abs=1.0e-3)
        assert row["phi_abs"] == pytest.approx(target, rel=0.05, abs=1.0e-4)
        assert row["theta_outer_out"] == pytest.approx(target, rel=0.08, abs=1.0e-4)

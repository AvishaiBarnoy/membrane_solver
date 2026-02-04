import os
import sys

import numpy as np
import pytest
from scipy import special

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from runtime.constraint_manager import ConstraintModuleManager  # noqa: E402
from runtime.energy_manager import EnergyModuleManager  # noqa: E402
from runtime.minimizer import Minimizer  # noqa: E402
from runtime.steppers.gradient_descent import GradientDescent  # noqa: E402


def _disk_group_rows(mesh, group: str) -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if (
            opts.get("rim_slope_match_group") == group
            or opts.get("tilt_thetaB_group") == group
        ):
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    return np.asarray(rows, dtype=int)


def _order_by_angle(pts: np.ndarray, *, center: np.ndarray) -> np.ndarray:
    rel = pts - center[None, :]
    ang = np.arctan2(rel[:, 1], rel[:, 0])
    return np.argsort(ang)


def _arc_length_weights(pts: np.ndarray) -> np.ndarray:
    n = len(pts)
    if n == 0:
        return np.zeros(0, dtype=float)
    diffs_next = pts[(np.arange(n) + 1) % n] - pts
    diffs_prev = pts - pts[(np.arange(n) - 1) % n]
    l_next = np.linalg.norm(diffs_next, axis=1)
    l_prev = np.linalg.norm(diffs_prev, axis=1)
    return 0.5 * (l_next + l_prev)


def _outer_radial_tilt_symmetry_metric(
    mesh, *, r_min: float, r_max: float
) -> tuple[float, float]:
    """Return (median |tin_r - tout_r|, median |tin_r + tout_r|) in an outer band."""
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    mask = (r >= float(r_min)) & (r <= float(r_max))
    if not np.any(mask):
        raise AssertionError(f"Outer band empty: r in [{r_min}, {r_max}].")

    ers = np.zeros_like(positions)
    nz = r > 1e-12
    ers[nz, 0] = positions[nz, 0] / r[nz]
    ers[nz, 1] = positions[nz, 1] / r[nz]

    tin = mesh.tilts_in_view()
    tout = mesh.tilts_out_view()
    tin_r = np.sum(tin * ers, axis=1)
    tout_r = np.sum(tout * ers, axis=1)
    return (
        float(np.median(np.abs(tin_r[mask] - tout_r[mask]))),
        float(np.median(np.abs(tin_r[mask] + tout_r[mask]))),
    )


def _tensionless_thetaB_prediction(
    *, kappa: float, kappa_t: float, drive: float, R: float
):
    """Return (thetaB*, Fin_el*, Fout_el*, Fcontact*, Ftot*) for tensionless theory."""
    lam = float(np.sqrt(kappa_t / kappa))
    x = lam * float(R)
    ratio_i = float(special.iv(0, x) / special.iv(1, x))
    ratio_k = float(special.kv(0, x) / special.kv(1, x))
    den = float(ratio_i + 0.5 * ratio_k)

    theta_star = float(drive / (np.sqrt(kappa * kappa_t) * den))
    # Split elastic coefficient into the two additive parts (inner/outer regions in TeX).
    fin = float(np.pi * kappa * R * lam * ratio_i * theta_star**2)
    fout = float(np.pi * kappa * R * lam * 0.5 * ratio_k * theta_star**2)
    fcont = float(-2.0 * np.pi * R * drive * theta_star)
    return theta_star, fin, fout, fcont, float(fin + fout + fcont)


@pytest.mark.e2e
def test_kozlov_free_disk_thetaB_matches_tensionless_theory() -> None:
    """E2E: thetaB optimization produces theory-scale thetaB on the free-disk mesh.

    This test focuses on robust, low-cost observables:
      - thetaB is nontrivial and close to the tensionless prediction (with
        elastic coefficients from both leaflets, since contact work is applied
        only to the inner leaflet)
      - contact energy term matches -2π R_eff * drive * thetaB (using the discrete R_eff)

    The mesh is loaded from tests/data to keep this test independent of the
    hand-authored YAMLs under meshes/.
    """
    path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "kozlov_1disk_3d_free_disk_theory_parity.yaml",
    )
    mesh = parse_geometry(load_data(path))
    gp = mesh.global_parameters

    # Fast tilt relaxation and frequent thetaB scans keep this test quick.
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 5)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)

    gp.set("tilt_thetaB_optimize", True)
    gp.set("tilt_thetaB_optimize_every", 1)
    # Use a larger delta and a tiny inner budget so we can move thetaB away from
    # zero in a single scan while keeping runtime bounded.
    gp.set("tilt_thetaB_optimize_delta", 0.05)
    gp.set("tilt_thetaB_optimize_inner_steps", 2)

    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 0.01)

    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )

    # --- Tensionless theory (docs/tex/1_disk_3d.tex) ---
    kappa_in = float(gp.get("bending_modulus_in") or gp.get("bending_modulus") or 0.0)
    kappa_out = float(gp.get("bending_modulus_out") or gp.get("bending_modulus") or 0.0)
    kappa = float(kappa_in + kappa_out)

    kappa_t_in = float(gp.get("tilt_modulus_in") or 0.0)
    kappa_t_out = float(gp.get("tilt_modulus_out") or 0.0)
    kappa_t = float(kappa_t_in + kappa_t_out)
    drive = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)
    assert kappa > 0.0 and kappa_t > 0.0 and drive != 0.0

    # Avoid expensive shape-gradient evaluations; for this theory-parity test we
    # only need the reduced-energy thetaB scan + tilt relaxation on fixed
    # geometry.
    tilt_mode = str(gp.get("tilt_solve_mode") or "coupled")
    for i in range(1):
        minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode=tilt_mode)
        minim._optimize_thetaB_scalar(tilt_mode=tilt_mode, iteration=i)

    thetaB = float(gp.get("tilt_thetaB_value") or 0.0)
    assert thetaB > 1.0e-2

    R = 7.0 / 15.0
    thetaB_star, fin_star, fout_star, fcont_star, ftot_star = (
        _tensionless_thetaB_prediction(kappa=kappa, kappa_t=kappa_t, drive=drive, R=R)
    )

    assert thetaB == pytest.approx(
        thetaB_star,
        rel=0.40,
        abs=0.03,
    )

    # --- Contact term check using the same discrete R_eff convention ---
    rows = _disk_group_rows(mesh, "disk")
    assert rows.size > 0
    positions = mesh.positions_view()
    center = np.asarray(
        gp.get("tilt_thetaB_center") or [0.0, 0.0, 0.0], dtype=float
    ).reshape(3)

    pts = positions[rows]
    order = _order_by_angle(pts, center=center)
    pts = pts[order]
    weights = _arc_length_weights(pts)
    wsum = float(np.sum(weights))
    assert wsum > 1e-12

    r_len = np.linalg.norm((pts - center[None, :])[:, :2], axis=1)
    R_eff = float(np.sum(weights * r_len) / wsum)

    breakdown = minim.compute_energy_breakdown()
    contact_energy = float(breakdown.get("tilt_thetaB_contact_in") or 0.0)
    contact_pred = float(-2.0 * np.pi * R_eff * drive * thetaB)

    assert contact_energy == pytest.approx(contact_pred, rel=0.05, abs=0.02)

    # --- Outer-region symmetry: tilts should match up to sign convention. ---
    diff_same, diff_oppo = _outer_radial_tilt_symmetry_metric(
        mesh, r_min=1.5, r_max=10.5
    )
    # Opposite-sign mismatch should be no worse than same-sign mismatch, up to
    # tiny numerical noise.
    assert diff_oppo <= diff_same + 1e-9


@pytest.mark.e2e
@pytest.mark.xfail(
    reason=(
        "Elastic energy magnitude does not yet match the continuum tensionless "
        "prediction; thetaB/contact parity is validated separately. This xfail "
        "tracks the remaining physics/discretization gap."
    ),
    strict=False,
)
def test_kozlov_free_disk_elastic_energy_matches_tensionless_theory_xfail() -> None:
    """Diagnostic parity check for elastic energy vs theory (expected to fail currently)."""
    from geometry.curvature import compute_curvature_fields
    from geometry.tilt_operators import p1_vertex_divergence

    path = os.path.join(
        os.path.dirname(__file__),
        "fixtures",
        "kozlov_1disk_3d_free_disk_theory_parity.yaml",
    )
    mesh = parse_geometry(load_data(path))
    gp = mesh.global_parameters

    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 5)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)

    gp.set("tilt_thetaB_optimize", True)
    gp.set("tilt_thetaB_optimize_every", 1)
    gp.set("tilt_thetaB_optimize_delta", 0.05)
    gp.set("tilt_thetaB_optimize_inner_steps", 2)

    minim = Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )

    kappa_in = float(gp.get("bending_modulus_in") or gp.get("bending_modulus") or 0.0)
    kappa_out = float(gp.get("bending_modulus_out") or gp.get("bending_modulus") or 0.0)
    kappa = float(kappa_in + kappa_out)
    kappa_t_in = float(gp.get("tilt_modulus_in") or 0.0)
    kappa_t_out = float(gp.get("tilt_modulus_out") or 0.0)
    kappa_t = float(kappa_t_in + kappa_t_out)
    drive = float(gp.get("tilt_thetaB_contact_strength_in") or 0.0)

    tilt_mode = str(gp.get("tilt_solve_mode") or "coupled")
    minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode=tilt_mode)
    minim._optimize_thetaB_scalar(tilt_mode=tilt_mode, iteration=0)

    thetaB = float(gp.get("tilt_thetaB_value") or 0.0)
    breakdown = minim.compute_energy_breakdown()
    contact_energy = float(breakdown.get("tilt_thetaB_contact_in") or 0.0)
    fel_meas = float(sum(float(v) for v in breakdown.values()) - contact_energy)
    ftot_meas = float(fel_meas + contact_energy)

    R = 7.0 / 15.0
    theta_star, fin_star, fout_star, fcont_star, ftot_star = (
        _tensionless_thetaB_prediction(kappa=kappa, kappa_t=kappa_t, drive=drive, R=R)
    )
    fel_star = float(fin_star + fout_star)

    # Extra diagnostics to make the remaining gap actionable.
    positions = mesh.positions_view()
    r_all = np.linalg.norm(positions[:, :2], axis=1)
    z_all = positions[:, 2]
    z_max = float(np.max(np.abs(z_all)))

    tri_rows, _ = mesh.triangle_row_cache()
    div_in = None
    div_out = None
    div_tri_in = None
    div_tri_out = None
    if tri_rows is not None and len(tri_rows) > 0:
        div_in, _ = p1_vertex_divergence(
            n_vertices=len(mesh.vertex_ids),
            positions=positions,
            tilts=mesh.tilts_in_view(),
            tri_rows=tri_rows,
        )
        div_out, _ = p1_vertex_divergence(
            n_vertices=len(mesh.vertex_ids),
            positions=positions,
            tilts=mesh.tilts_out_view(),
            tri_rows=tri_rows,
        )
        # Triangle divergence stats help detect whether large vertex divergence
        # is a per-triangle effect or an accumulation/normalization artifact.
        from geometry.tilt_operators import p1_triangle_divergence  # noqa: PLC0415

        div_tri_in, *_ = p1_triangle_divergence(
            positions=positions, tilts=mesh.tilts_in_view(), tri_rows=tri_rows
        )
        div_tri_out, *_ = p1_triangle_divergence(
            positions=positions, tilts=mesh.tilts_out_view(), tri_rows=tri_rows
        )

    fields = compute_curvature_fields(mesh, positions, mesh.vertex_index_to_row)
    H = np.asarray(fields.mean_curvature, dtype=float)
    absH = np.abs(H)
    # Exclude the disk boundary itself to avoid counting interface singularities.
    mask_mem = (r_all >= 0.55) & (r_all <= 11.5)
    absH_mem = absH[mask_mem] if np.any(mask_mem) else absH

    # Region masks: near-disk ring, mid membrane, far field. These are used for
    # stats only (not for energy partitioning yet).
    mask_near = (r_all >= 0.55) & (r_all < 1.5)
    mask_mid = (r_all >= 1.5) & (r_all < 6.0)
    mask_far = (r_all >= 6.0) & (r_all <= 11.5)

    def qstats(arr: np.ndarray) -> str:
        if arr.size == 0:
            return "empty"
        q = np.quantile(arr, [0.5, 0.9, 0.99, 1.0])
        return f"med={q[0]:.3e} p90={q[1]:.3e} p99={q[2]:.3e} max={q[3]:.3e}"

    def _masked(arr: np.ndarray | None, mask: np.ndarray) -> np.ndarray:
        if arr is None:
            return np.zeros(0, dtype=float)
        return np.asarray(arr, dtype=float)[mask]

    report = "\n".join(
        [
            "Elastic-energy parity gap diagnostics (expected xfail):",
            f"  thetaB_meas={thetaB:.6g} thetaB_star={theta_star:.6g}",
            f"  fel_meas={fel_meas:.6g} fel_star={fel_star:.6g} (ratio {fel_meas / max(fel_star, 1e-12):.3e})",
            f"  ftot_meas={ftot_meas:.6g} ftot_star={ftot_star:.6g}",
            "  breakdown:",
            "    "
            + ", ".join(f"{k}={float(v):.6g}" for k, v in sorted(breakdown.items())),
            f"  |z|_max={z_max:.3e}",
            f"  |H| (membrane band) stats: {qstats(absH_mem)}",
            "  |H| by region:",
            f"    near [0.55,1.5): {qstats(absH[mask_near])}",
            f"    mid  [1.5,6.0): {qstats(absH[mask_mid])}",
            f"    far  [6.0,11.5]: {qstats(absH[mask_far])}",
            f"  |div t_in| stats: {qstats(np.abs(div_in)) if div_in is not None else 'n/a'}",
            f"  |div t_out| stats: {qstats(np.abs(div_out)) if div_out is not None else 'n/a'}",
            f"  |div_tri t_in| stats: {qstats(np.abs(np.asarray(div_tri_in, dtype=float))) if div_tri_in is not None else 'n/a'}",
            f"  |div_tri t_out| stats: {qstats(np.abs(np.asarray(div_tri_out, dtype=float))) if div_tri_out is not None else 'n/a'}",
            "  |div t_in| by region:",
            f"    near [0.55,1.5): {qstats(np.abs(_masked(div_in, mask_near)))}",
            f"    mid  [1.5,6.0): {qstats(np.abs(_masked(div_in, mask_mid)))}",
            f"    far  [6.0,11.5]: {qstats(np.abs(_masked(div_in, mask_far)))}",
            "  |div t_out| by region:",
            f"    near [0.55,1.5): {qstats(np.abs(_masked(div_out, mask_near)))}",
            f"    mid  [1.5,6.0): {qstats(np.abs(_masked(div_out, mask_mid)))}",
            f"    far  [6.0,11.5]: {qstats(np.abs(_masked(div_out, mask_far)))}",
        ]
    )

    theta_ok = thetaB == pytest.approx(theta_star, rel=0.40, abs=0.03)
    fel_ok = fel_meas == pytest.approx(fel_star, rel=2.0, abs=0.2)
    ftot_ok = ftot_meas == pytest.approx(ftot_star, rel=2.0, abs=0.2)
    if not (theta_ok and fel_ok and ftot_ok):
        pytest.xfail(report)

    # If we ever close the gap, keep these assertions so the test becomes a
    # normal pass without edits.
    assert theta_ok
    assert fel_ok
    assert ftot_ok

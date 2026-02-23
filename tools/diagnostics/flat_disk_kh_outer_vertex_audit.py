#!/usr/bin/env python3
"""Strict-KH outer-field vertex-distribution audit at fixed theta_B."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FIXTURE = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)
DEFAULT_OUT = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "diagnostics"
    / "flat_disk_kh_outer_vertex_audit.yaml"
)


def _resolve_controls(optimize_preset: str) -> dict[str, float | int]:
    p = str(optimize_preset).lower()
    if p == "kh_strict_outertail_balanced":
        return {
            "refine_level": 2,
            "rim_local_refine_steps": 1,
            "rim_local_refine_band_lambda": 3.0,
            "outer_local_refine_steps": 1,
            "outer_local_refine_rmin_lambda": 1.0,
            "outer_local_refine_rmax_lambda": 10.0,
        }
    if p == "kh_strict_outerfield_tight":
        return {
            "refine_level": 2,
            "rim_local_refine_steps": 1,
            "rim_local_refine_band_lambda": 3.0,
            "outer_local_refine_steps": 1,
            "outer_local_refine_rmin_lambda": 1.0,
            "outer_local_refine_rmax_lambda": 8.0,
        }
    return {
        "refine_level": 1,
        "rim_local_refine_steps": 1,
        "rim_local_refine_band_lambda": 4.0,
        "outer_local_refine_steps": 0,
        "outer_local_refine_rmin_lambda": 0.0,
        "outer_local_refine_rmax_lambda": 0.0,
    }


def _vertex_bands(
    *,
    positions: np.ndarray,
    tri_rows: np.ndarray,
    tri_area: np.ndarray,
    tilts: np.ndarray,
    radius: float,
    lambda_value: float,
) -> list[dict[str, float | int | str]]:
    from tools.diagnostics.flat_disk_kh_term_audit import _radial_frames

    r, r_hat, phi_hat = _radial_frames(positions)

    dual = np.zeros(positions.shape[0], dtype=float)
    share = np.asarray(tri_area, dtype=float) / 3.0
    np.add.at(dual, tri_rows[:, 0], share)
    np.add.at(dual, tri_rows[:, 1], share)
    np.add.at(dual, tri_rows[:, 2], share)

    t_rad = np.einsum("ij,ij->i", tilts, r_hat)
    t_phi = np.einsum("ij,ij->i", tilts, phi_hat)

    rim_end = float(radius + lambda_value)
    near_end = float(radius + (4.0 * lambda_value))
    masks = {
        "outer_near": (r > rim_end) & (r <= near_end),
        "outer_far": r > near_end,
    }

    rows: list[dict[str, float | int | str]] = []
    for name, vm in masks.items():
        vm = np.asarray(vm, dtype=bool)
        if int(np.count_nonzero(vm)) == 0:
            raise ValueError(f"Empty vertex band: {name}")
        trad = float(np.median(np.abs(t_rad[vm])))
        tphi = float(np.median(np.abs(t_phi[vm])))
        rows.append(
            {
                "band": str(name),
                "vertex_count": int(np.count_nonzero(vm)),
                "dual_area_total": float(np.sum(dual[vm])),
                "vertex_density_per_dual_area": float(
                    np.count_nonzero(vm) / max(float(np.sum(dual[vm])), 1e-18)
                ),
                "t_phi_over_t_rad_median": float(tphi / max(trad, 1e-18)),
            }
        )
    return rows


def run_flat_disk_kh_outer_vertex_audit(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    optimize_preset: str = "kh_strict_outerfield_tight",
    theta: float = 0.138,
) -> dict[str, Any]:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from runtime.refinement import refine_triangle_mesh
    from tools.diagnostics.flat_disk_kh_term_audit import (
        _mesh_internal_band_split,
        _theory_term_band_split,
    )
    from tools.diagnostics.flat_disk_one_leaflet_theory import (
        compute_flat_disk_kh_physical_theory,
        physical_to_dimensionless_theory_params,
    )
    from tools.reproduce_flat_disk_one_leaflet import (
        _build_minimizer,
        _configure_benchmark_mesh,
        _load_mesh_from_fixture,
        _refine_mesh_locally_in_outer_annulus,
        _refine_mesh_locally_near_rim,
        _run_theta_relaxation,
    )

    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    controls = _resolve_controls(optimize_preset)
    params = physical_to_dimensionless_theory_params(
        kappa_physical=10.0,
        kappa_t_physical=10.0,
        radius_physical=7.0,
        drive_physical=(2.0 / 0.7),
        length_scale=15.0,
    )
    theory = compute_flat_disk_kh_physical_theory(params)
    radius = float(theory.radius)
    lam = float(theory.lambda_value)

    mesh = _load_mesh_from_fixture(fixture_path)
    for _ in range(int(controls["refine_level"])):
        mesh = refine_triangle_mesh(mesh)
    if int(controls["rim_local_refine_steps"]) > 0:
        mesh = _refine_mesh_locally_near_rim(
            mesh,
            local_steps=int(controls["rim_local_refine_steps"]),
            rim_radius=radius,
            band_half_width=float(controls["rim_local_refine_band_lambda"]) * lam,
        )
    if int(controls["outer_local_refine_steps"]) > 0:
        mesh = _refine_mesh_locally_in_outer_annulus(
            mesh,
            local_steps=int(controls["outer_local_refine_steps"]),
            r_min=radius + float(controls["outer_local_refine_rmin_lambda"]) * lam,
            r_max=radius + float(controls["outer_local_refine_rmax_lambda"]) * lam,
        )

    _configure_benchmark_mesh(
        mesh,
        theory_params=params,
        parameterization="kh_physical",
        outer_mode="disabled",
        smoothness_model="splay_twist",
        splay_modulus_scale_in=1.0,
        tilt_mass_mode_in="consistent",
    )
    total_energy = float(
        _run_theta_relaxation(
            _build_minimizer(mesh), theta_value=float(theta), reset_outer=True
        )
    )

    positions = mesh.positions_view()
    tri_area, _, _, _, tri_rows = mesh.p1_triangle_shape_gradient_cache(
        positions=positions
    )
    if tri_rows is None or tri_rows.size == 0:
        raise ValueError("Mesh has no triangles after refinement")
    bands = _vertex_bands(
        positions=positions,
        tri_rows=np.asarray(tri_rows, dtype=int),
        tri_area=np.asarray(tri_area, dtype=float),
        tilts=mesh.tilts_in_view(),
        radius=radius,
        lambda_value=lam,
    )

    mesh_bands = _mesh_internal_band_split(
        mesh,
        smoothness_model="splay_twist",
        radius=radius,
        lambda_value=lam,
        rim_half_width_lambda=1.0,
        outer_near_width_lambda=4.0,
    )
    theory_bands = _theory_term_band_split(
        theta=float(theta),
        kappa=float(theory.kappa),
        kappa_t=float(theory.kappa_t),
        radius=radius,
        lambda_value=lam,
        rim_half_width_lambda=1.0,
        outer_near_width_lambda=4.0,
        outer_r_max=float(np.max(np.linalg.norm(positions[:, :2], axis=1))),
    )
    near = float(
        mesh_bands["mesh_internal_outer_near"]
        / max(float(theory_bands["theory_internal_outer_near"]), 1e-18)
    )
    far = float(
        mesh_bands["mesh_internal_outer_far"]
        / max(float(theory_bands["theory_internal_outer_far"]), 1e-18)
    )

    return {
        "meta": {
            "mode": "flat_disk_kh_outer_vertex_audit",
            "fixture": str(fixture_path.relative_to(ROOT)),
            "optimize_preset": str(optimize_preset),
            "theta": float(theta),
            "controls_effective": controls,
        },
        "parity": {
            "mesh_total_energy": total_energy,
            "outer_near_ratio_mesh_over_theory": near,
            "outer_far_ratio_mesh_over_theory": far,
            "outer_tail_balance_score": float(
                np.hypot(np.log(max(near, 1e-18)), np.log(max(far, 1e-18)))
            ),
        },
        "bands": bands,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--optimize-preset", default="kh_strict_outerfield_tight")
    ap.add_argument("--theta", type=float, default=0.138)
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    report = run_flat_disk_kh_outer_vertex_audit(
        fixture=args.fixture,
        optimize_preset=args.optimize_preset,
        theta=args.theta,
    )
    out = Path(args.output)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

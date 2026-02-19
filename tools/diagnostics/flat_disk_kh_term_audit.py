#!/usr/bin/env python3
"""Per-theta KH physical lane audit for flat one-leaflet disk benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_FIXTURE = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)


def _ensure_repo_root_on_sys_path() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


DEFAULT_OUT = (
    ROOT / "benchmarks" / "outputs" / "diagnostics" / "flat_disk_kh_term_audit.yaml"
)


def _theory_split_coeffs(theory: Any) -> tuple[float, float, float]:
    """Return (c_in, c_out, b_contact) for theory energy split at fixed theta."""
    c_in = float(
        np.pi
        * float(theory.kappa_t)
        * float(theory.radius)
        * float(theory.lambda_inverse)
        * float(theory.ratio_i1_i0)
    )
    c_out = float(
        np.pi
        * float(theory.kappa_t)
        * float(theory.radius)
        * float(theory.lambda_inverse)
        * float(theory.ratio_k1_k0)
    )
    b_contact = float(theory.coeff_B)
    return c_in, c_out, b_contact


def run_flat_disk_kh_term_audit(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    refine_level: int = 1,
    outer_mode: str = "disabled",
    smoothness_model: str = "splay_twist",
    kappa_physical: float = 10.0,
    kappa_t_physical: float = 10.0,
    radius_nm: float = 7.0,
    length_scale_nm: float = 15.0,
    drive_physical: float = (2.0 / 0.7),
    theta_values: Sequence[float] = (0.0, 6.366e-4, 0.004),
) -> dict[str, Any]:
    """Evaluate per-theta mesh/theory split terms in KH physical lane."""
    _ensure_repo_root_on_sys_path()
    from runtime.refinement import refine_triangle_mesh
    from tools.diagnostics.flat_disk_one_leaflet_theory import (
        compute_flat_disk_theory,
        physical_to_dimensionless_theory_params,
    )
    from tools.reproduce_flat_disk_one_leaflet import (
        _build_minimizer,
        _configure_benchmark_mesh,
        _load_mesh_from_fixture,
        _run_theta_relaxation,
    )

    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    params = physical_to_dimensionless_theory_params(
        kappa_physical=float(kappa_physical),
        kappa_t_physical=float(kappa_t_physical),
        radius_physical=float(radius_nm),
        drive_physical=float(drive_physical),
        length_scale=float(length_scale_nm),
    )
    theory = compute_flat_disk_theory(params)
    c_in, c_out, b_contact = _theory_split_coeffs(theory)

    mesh = _load_mesh_from_fixture(fixture_path)
    for _ in range(int(refine_level)):
        mesh = refine_triangle_mesh(mesh)

    _configure_benchmark_mesh(
        mesh,
        theory_params=params,
        parameterization="kh_physical",
        outer_mode=outer_mode,
        smoothness_model=smoothness_model,
        splay_modulus_scale_in=1.0,
    )
    minim = _build_minimizer(mesh)
    minim.enforce_constraints_after_mesh_ops(mesh)
    mesh.project_tilts_to_tangent()

    rows: list[dict[str, float]] = []
    for theta in np.asarray(theta_values, dtype=float).tolist():
        theta_f = float(theta)
        mesh_total = float(
            _run_theta_relaxation(minim, theta_value=theta_f, reset_outer=True)
        )
        breakdown = minim.compute_energy_breakdown()
        mesh_contact = float(breakdown.get("tilt_thetaB_contact_in", 0.0))
        mesh_internal = float(mesh_total - mesh_contact)

        th_in = float(c_in * theta_f * theta_f)
        th_out = float(c_out * theta_f * theta_f)
        th_contact = float(-b_contact * theta_f)
        th_total = float(th_in + th_out + th_contact)
        th_internal = float(th_in + th_out)

        rows.append(
            {
                "theta": theta_f,
                "mesh_total": mesh_total,
                "mesh_contact": mesh_contact,
                "mesh_internal": mesh_internal,
                "theory_total": th_total,
                "theory_contact": th_contact,
                "theory_internal": th_internal,
                "total_error": float(mesh_total - th_total),
                "contact_error": float(mesh_contact - th_contact),
                "internal_error": float(mesh_internal - th_internal),
                "contact_ratio_mesh_over_theory": (
                    float(mesh_contact / th_contact)
                    if abs(th_contact) > 1e-18
                    else float("nan")
                ),
            }
        )

    return {
        "meta": {
            "fixture": str(fixture_path.relative_to(ROOT)),
            "refine_level": int(refine_level),
            "outer_mode": str(outer_mode),
            "smoothness_model": str(smoothness_model),
            "parameterization": "kh_physical",
            "kappa_physical": float(kappa_physical),
            "kappa_t_physical": float(kappa_t_physical),
            "radius_nm": float(radius_nm),
            "length_scale_nm": float(length_scale_nm),
            "drive_physical": float(drive_physical),
        },
        "theory": {
            "kappa": float(theory.kappa),
            "kappa_t": float(theory.kappa_t),
            "radius": float(theory.radius),
            "lambda_inverse": float(theory.lambda_inverse),
            "coeff_B": float(theory.coeff_B),
            "coeff_c_inner": float(c_in),
            "coeff_c_outer": float(c_out),
        },
        "rows": rows,
    }


def main() -> int:
    _ensure_repo_root_on_sys_path()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--refine-level", type=int, default=1)
    ap.add_argument("--outer-mode", choices=("disabled", "free"), default="disabled")
    ap.add_argument(
        "--smoothness-model",
        choices=("dirichlet", "splay_twist"),
        default="splay_twist",
    )
    ap.add_argument("--kappa-physical", type=float, default=10.0)
    ap.add_argument("--kappa-t-physical", type=float, default=10.0)
    ap.add_argument("--radius-nm", type=float, default=7.0)
    ap.add_argument("--length-scale-nm", type=float, default=15.0)
    ap.add_argument("--drive-physical", type=float, default=(2.0 / 0.7))
    ap.add_argument(
        "--theta-values", type=float, nargs="+", default=[0.0, 6.366e-4, 0.004]
    )
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    report = run_flat_disk_kh_term_audit(
        fixture=args.fixture,
        refine_level=args.refine_level,
        outer_mode=args.outer_mode,
        smoothness_model=args.smoothness_model,
        kappa_physical=args.kappa_physical,
        kappa_t_physical=args.kappa_t_physical,
        radius_nm=args.radius_nm,
        length_scale_nm=args.length_scale_nm,
        drive_physical=args.drive_physical,
        theta_values=args.theta_values,
    )

    out = Path(args.output)
    if not out.is_absolute():
        out = (ROOT / out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(report, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

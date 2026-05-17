#!/usr/bin/env python3
"""Audit thetaB reduced-energy normalization for parity fixtures."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy import special

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tools.diagnostics.parity_acceptance_triage import (  # noqa: E402
    DEFAULT_FIXTURE,
    FIXED_THETA_SWEEP_VALUES,
    GHOST_FIXTURE,
    _as_float,
    _run_fixed_theta_sweep,
)
from tools.reproduce_theory_parity import (  # noqa: E402
    DEFAULT_TEX_BENDING_MODULUS,
    DEFAULT_TEX_TILT_MODULUS,
    DEFAULT_THEORY_RADIUS,
)


def _ratio(measured: float, expected: float) -> float:
    if abs(expected) < 1.0e-16:
        return 0.0
    return float(measured / expected)


def _fit_quadratic(theta: np.ndarray, values: np.ndarray) -> dict[str, float]:
    coeff = np.polyfit(theta, values, deg=2)
    a = float(coeff[0])
    b = float(coeff[1])
    c = float(coeff[2])
    theta_min = 0.0 if abs(a) < 1.0e-16 else float(-b / (2.0 * a))
    return {"quadratic": a, "linear": b, "constant": c, "theta_min": theta_min}


def _fit_linear(theta: np.ndarray, values: np.ndarray) -> dict[str, float]:
    coeff = np.polyfit(theta, values, deg=1)
    return {"slope": float(coeff[0]), "intercept": float(coeff[1])}


def _theory_terms_for_fixture(fixture: Path) -> dict[str, float]:
    doc = yaml.safe_load(fixture.read_text(encoding="utf-8")) or {}
    gp = doc.get("global_parameters") or {}
    radius = _as_float(gp.get("theory_radius"), DEFAULT_THEORY_RADIUS)
    drive = _as_float(gp.get("tilt_thetaB_contact_strength_in"), 0.0)
    kappa = float(DEFAULT_TEX_BENDING_MODULUS)
    kappa_t = float(DEFAULT_TEX_TILT_MODULUS)
    if radius <= 0.0 or drive == 0.0:
        return {
            "radius": radius,
            "drive": drive,
            "elastic_A": 0.0,
            "contact_B": 0.0,
            "theta_min": 0.0,
        }

    lam = float(np.sqrt(kappa_t / kappa))
    x = float(lam * radius)
    ratio_i = float(special.iv(0, x) / special.iv(1, x))
    ratio_k = float(special.kv(0, x) / special.kv(1, x))
    den = float(ratio_i + 0.5 * ratio_k)
    elastic_a = float(np.pi * kappa * radius * lam * den)
    contact_b = float(2.0 * np.pi * radius * drive)
    theta_min = float(contact_b / (2.0 * elastic_a))
    return {
        "radius": radius,
        "drive": drive,
        "elastic_A": elastic_a,
        "contact_B": contact_b,
        "theta_min": theta_min,
    }


def summarize_fixed_theta_sweep(
    sweep: dict[str, Any], *, fixture: Path | None = None
) -> dict[str, Any]:
    rows = [sweep[f"{theta:.2f}"] for theta in FIXED_THETA_SWEEP_VALUES]
    theta = np.asarray(
        [_as_float(row.get("thetaB_value")) for row in rows], dtype=float
    )
    elastic = np.asarray(
        [
            _as_float(row.get("reduced_terms", {}).get("elastic_measured"))
            for row in rows
        ],
        dtype=float,
    )
    contact = np.asarray(
        [
            _as_float(row.get("reduced_terms", {}).get("contact_measured"))
            for row in rows
        ],
        dtype=float,
    )
    total = np.asarray(
        [_as_float(row.get("reduced_terms", {}).get("total_measured")) for row in rows],
        dtype=float,
    )

    elastic_fit = _fit_quadratic(theta, elastic)
    contact_fit = _fit_linear(theta, contact)
    total_fit = _fit_quadratic(theta, total)
    module_fits = _module_fits(rows=rows, theta=theta)
    measured = {
        "elastic_A": float(elastic_fit["quadratic"]),
        "contact_B": float(-contact_fit["slope"]),
        "theta_min": float(total_fit["theta_min"]),
        "total_quadratic": float(total_fit["quadratic"]),
        "total_linear": float(total_fit["linear"]),
    }
    theory = _theory_terms_for_fixture(fixture) if fixture is not None else {}

    ratios: dict[str, float] = {}
    if theory:
        ratios = {
            "elastic_A": _ratio(measured["elastic_A"], theory["elastic_A"]),
            "contact_B": _ratio(measured["contact_B"], theory["contact_B"]),
            "theta_min": _ratio(measured["theta_min"], theory["theta_min"]),
        }

    return {
        "theta_values": [float(x) for x in theta],
        "fits": {
            "elastic": elastic_fit,
            "contact": contact_fit,
            "total": total_fit,
        },
        "module_fits": module_fits,
        "measured": measured,
        "theory": theory,
        "ratios": ratios,
        "classification": classify_balance(ratios=ratios, measured=measured),
    }


def _module_fits(*, rows: list[dict[str, Any]], theta: np.ndarray) -> dict[str, Any]:
    keys: set[str] = set()
    for row in rows:
        breakdown = row.get("energy_breakdown", {})
        if isinstance(breakdown, dict):
            keys.update(str(key) for key in breakdown)
    out: dict[str, Any] = {}
    for key in sorted(keys):
        values = np.asarray(
            [_as_float(row.get("energy_breakdown", {}).get(key)) for row in rows],
            dtype=float,
        )
        if key == "tilt_thetaB_contact_in":
            fit = _fit_linear(theta, values)
            out[key] = {
                "slope": fit["slope"],
                "intercept": fit["intercept"],
                "contact_B": float(-fit["slope"]),
            }
        else:
            fit = _fit_quadratic(theta, values)
            out[key] = {
                "quadratic": fit["quadratic"],
                "linear": fit["linear"],
                "constant": fit["constant"],
            }
    return out


def classify_balance(*, ratios: dict[str, float], measured: dict[str, float]) -> str:
    if not ratios:
        return "insufficient_theory_reference"
    contact_ratio = abs(float(ratios.get("contact_B", 0.0)))
    elastic_ratio = abs(float(ratios.get("elastic_A", 0.0)))
    theta_ratio = abs(float(ratios.get("theta_min", 0.0)))
    if contact_ratio > 1.25 and elastic_ratio <= 1.25:
        return "contact_scale_high"
    if elastic_ratio < 0.75 and contact_ratio >= 0.75:
        return "elastic_response_low"
    if theta_ratio > 1.25:
        return "theta_min_high_from_combined_balance"
    if theta_ratio < 0.75:
        return "theta_min_low_from_combined_balance"
    if measured.get("total_quadratic", 0.0) <= 0.0:
        return "nonconvex_or_unbounded_total_fit"
    return "balanced"


def run_audit() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="thetaB_norm_audit_") as tmp:
        tmpdir = Path(tmp)
        ghost = _run_fixed_theta_sweep(
            base_fixture=GHOST_FIXTURE, label="ghost_norm", tmpdir=tmpdir
        )
        default = _run_fixed_theta_sweep(
            base_fixture=DEFAULT_FIXTURE, label="default_norm", tmpdir=tmpdir
        )
    return {
        "meta": {
            "theta_values": [float(theta) for theta in FIXED_THETA_SWEEP_VALUES],
            "fit_model": "E(theta)=A*theta^2-B*theta+C",
        },
        "cases": {
            "ghost": summarize_fixed_theta_sweep(ghost, fixture=GHOST_FIXTURE),
            "default": summarize_fixed_theta_sweep(default, fixture=DEFAULT_FIXTURE),
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    audit = run_audit()
    text = yaml.safe_dump(audit, sort_keys=False)
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

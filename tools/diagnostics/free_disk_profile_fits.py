#!/usr/bin/env python3
"""Fit tilt and curvature profiles to analytic forms for free-disk runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.optimize import curve_fit
from scipy.special import i1, k0, k1

from geometry.curvature import compute_curvature_fields
from geometry.geom_io import parse_geometry


def _load_output(path: Path) -> dict:
    text = path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        import yaml

        return yaml.safe_load(text)


def _ensure_definitions(data: dict) -> dict:
    if "definitions" in data and isinstance(data["definitions"], dict):
        return data
    presets: set[str] = set()
    for v in data.get("vertices", []):
        if isinstance(v, list) and v and isinstance(v[-1], dict):
            preset = v[-1].get("preset")
            if preset:
                presets.add(str(preset))
    data["definitions"] = {name: {} for name in sorted(presets)}
    return data


def _center_normal(params: dict) -> Tuple[np.ndarray, np.ndarray]:
    center = np.asarray(params.get("tilt_thetaB_center", [0.0, 0.0, 0.0]), dtype=float)
    normal = np.asarray(params.get("tilt_thetaB_normal", [0.0, 0.0, 1.0]), dtype=float)
    nrm = float(np.linalg.norm(normal))
    if nrm < 1e-12:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = normal / nrm
    return center, normal


def _disk_radius(data: dict, center: np.ndarray, normal: np.ndarray) -> float:
    rs = []
    for v in data.get("vertices", []):
        if isinstance(v, list) and v and isinstance(v[-1], dict):
            opt = v[-1]
            if opt.get("pin_to_circle_group") != "disk":
                continue
            p = np.asarray(v[:3], dtype=float)
            rvec = p - center
            rvec = rvec - np.dot(rvec, normal) * normal
            rs.append(float(np.linalg.norm(rvec)))
    if rs:
        return float(np.median(rs))
    return float(data.get("global_parameters", {}).get("pin_to_circle_radius", 0.0))


def _radial_components(
    data: dict, *, center: np.ndarray, normal: np.ndarray
) -> Dict[str, np.ndarray]:
    r_all: list[float] = []
    tin_rad: list[float] = []
    tout_rad: list[float] = []
    for v in data.get("vertices", []):
        if not (isinstance(v, list) and v):
            continue
        opts = v[-1] if isinstance(v[-1], dict) else {}
        p = np.asarray(v[:3], dtype=float)
        rvec = p - center
        rvec = rvec - np.dot(rvec, normal) * normal
        r = float(np.linalg.norm(rvec))
        if r < 1e-12:
            continue
        rhat = rvec / r
        r_all.append(r)
        tin = opts.get("tilt_in")
        if tin is not None:
            tin_rad.append(float(np.dot(np.asarray(tin, dtype=float), rhat)))
        else:
            tin_rad.append(np.nan)
        tout = opts.get("tilt_out")
        if tout is not None:
            tout_rad.append(float(np.dot(np.asarray(tout, dtype=float), rhat)))
        else:
            tout_rad.append(np.nan)
    return {
        "r": np.asarray(r_all, dtype=float),
        "tin": np.asarray(tin_rad, dtype=float),
        "tout": np.asarray(tout_rad, dtype=float),
    }


def _bin_profile(
    r: np.ndarray, y: np.ndarray, *, rmin: float, rmax: float, bins: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.isfinite(y) & (r >= rmin) & (r <= rmax)
    if not np.any(mask):
        return np.array([]), np.array([]), np.array([])
    r = r[mask]
    y = y[mask]
    edges = np.linspace(rmin, rmax, bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    vals = np.full(bins, np.nan, dtype=float)
    counts = np.zeros(bins, dtype=int)
    for i in range(bins):
        sel = (r >= edges[i]) & (r < edges[i + 1])
        counts[i] = int(np.sum(sel))
        if counts[i]:
            vals[i] = float(np.median(y[sel]))
    ok = np.isfinite(vals)
    return centers[ok], vals[ok], counts[ok]


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    if y.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def _fit_k1(r: np.ndarray, y: np.ndarray, R: float) -> Tuple[float, float, float]:
    def model(x, theta_r, lam):
        return theta_r * k1(lam * x) / k1(lam * R)

    p0 = (float(y[0]), 10.0)
    bounds = ([-np.inf, 1e-6], [np.inf, 1e3])
    popt, _ = curve_fit(model, r, y, p0=p0, bounds=bounds, maxfev=10000)
    yhat = model(r, *popt)
    return float(popt[0]), float(popt[1]), _rmse(y, yhat)


def _fit_i1(r: np.ndarray, y: np.ndarray, R: float) -> Tuple[float, float, float]:
    def model(x, theta_r, lam):
        return theta_r * i1(lam * x) / i1(lam * R)

    p0 = (float(y[-1]), 10.0)
    bounds = ([-np.inf, 1e-6], [np.inf, 1e3])
    popt, _ = curve_fit(model, r, y, p0=p0, bounds=bounds, maxfev=10000)
    yhat = model(r, *popt)
    return float(popt[0]), float(popt[1]), _rmse(y, yhat)


def _fit_k0(r: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    def model(x, amp, psi):
        return amp * k0(psi * x)

    p0 = (float(y[0]), 1.0)
    bounds = ([-np.inf, 1e-6], [np.inf, 1e3])
    popt, _ = curve_fit(model, r, y, p0=p0, bounds=bounds, maxfev=10000)
    yhat = model(r, *popt)
    return float(popt[0]), float(popt[1]), _rmse(y, yhat)


def _fit_k0_offset(r: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    def model(x, z0, amp, psi):
        return z0 + amp * k0(psi * x)

    z0 = float(np.median(y))
    p0 = (z0, float(y[0] - z0), 1.0)
    bounds = ([-np.inf, -np.inf, 1e-6], [np.inf, np.inf, 1e3])
    popt, _ = curve_fit(model, r, y, p0=p0, bounds=bounds, maxfev=10000)
    yhat = model(r, *popt)
    return float(popt[0]), float(popt[1]), float(popt[2]), _rmse(y, yhat)


def _fit_log(r: np.ndarray, y: np.ndarray, R: float) -> Tuple[float, float, float]:
    def model(x, z0, a):
        return z0 + a * np.log(x / R)

    z0 = float(y[0])
    denom = float(np.log(max(r[-1] / R, 1.000001)))
    p0 = (z0, float((y[-1] - y[0]) / denom))
    popt, _ = curve_fit(model, r, y, p0=p0, maxfev=10000)
    yhat = model(r, *popt)
    return float(popt[0]), float(popt[1]), _rmse(y, yhat)


def _curvature_profile(
    mesh, *, center: np.ndarray, normal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    mesh.build_position_cache()
    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row
    fields = compute_curvature_fields(mesh, positions, idx_map)
    r = []
    h = []
    for row, vid in enumerate(mesh.vertex_ids):
        pos = positions[row]
        rvec = pos - center
        rvec = rvec - np.dot(rvec, normal) * normal
        rr = float(np.linalg.norm(rvec))
        if rr < 1e-12:
            continue
        r.append(rr)
        h.append(float(fields.mean_curvature[row]))
    return np.asarray(r, dtype=float), np.asarray(h, dtype=float)


def _height_profile(
    data: dict, *, center: np.ndarray, normal: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    r = []
    z = []
    for v in data.get("vertices", []):
        if not (isinstance(v, list) and v):
            continue
        p = np.asarray(v[:3], dtype=float)
        rvec = p - center
        rvec = rvec - np.dot(rvec, normal) * normal
        rr = float(np.linalg.norm(rvec))
        if rr < 1e-12:
            continue
        r.append(rr)
        z.append(float(p[2]))
    return np.asarray(r, dtype=float), np.asarray(z, dtype=float)


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="/tmp/out.yaml", help="Output mesh file")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--no-curvature", action="store_true")
    ap.add_argument(
        "--flip-tilt-out",
        dest="flip_tilt_out",
        action="store_true",
        default=True,
        help="Flip tilt_out sign to align with theory conventions (default).",
    )
    ap.add_argument(
        "--no-flip-tilt-out",
        dest="flip_tilt_out",
        action="store_false",
        help="Do not flip tilt_out sign; report raw values.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    data = _load_output(Path(args.input))
    params = data.get("global_parameters", {})
    center, normal = _center_normal(params)
    R = _disk_radius(data, center, normal)

    radial = _radial_components(data, center=center, normal=normal)
    r = radial["r"]
    tin = radial["tin"]
    tout = radial["tout"]
    if args.flip_tilt_out:
        tout = -tout

    # Inner fit (r <= R): tilt_in ~ theta_R * I1(lambda r) / I1(lambda R)
    rin, tin_med, _ = _bin_profile(r, tin, rmin=0.0, rmax=R, bins=args.bins)
    inner_fit = None
    if rin.size >= 3 and np.any(np.isfinite(tin_med)):
        theta_r, lam, rmse = _fit_i1(rin, tin_med, R)
        inner_fit = {"theta_R": theta_r, "lambda": lam, "rmse": rmse}

    # Outer fit (r >= R): tilt_out ~ theta_R * K1(lambda r) / K1(lambda R)
    rout, tout_med, _ = _bin_profile(
        r, tout, rmin=R, rmax=float(np.max(r)), bins=args.bins
    )
    outer_fit = None
    if rout.size >= 3 and np.any(np.isfinite(tout_med)):
        theta_r, lam, rmse = _fit_k1(rout, tout_med, R)
        outer_fit = {"theta_R": theta_r, "lambda": lam, "rmse": rmse}

    curvature_fit = None
    if not args.no_curvature:
        data = _ensure_definitions(data)
        mesh = parse_geometry(data)
        r_h, h = _curvature_profile(mesh, center=center, normal=normal)
        r_hb, h_med, _ = _bin_profile(
            r_h, h, rmin=R, rmax=float(np.max(r_h)), bins=args.bins
        )
        if r_hb.size >= 3 and np.any(np.isfinite(h_med)):
            amp, psi, rmse = _fit_k0(r_hb, h_med)
            curvature_fit = {"amp": amp, "psi": psi, "rmse": rmse}

    # Height fit (tensionless: log, finite tension: K0)
    tension = float(params.get("surface_tension") or 0.0)
    r_z, z = _height_profile(data, center=center, normal=normal)
    z_fit = None
    if r_z.size >= 3:
        rzb, z_med, _ = _bin_profile(
            r_z, z, rmin=R, rmax=float(np.max(r_z)), bins=args.bins
        )
        if rzb.size >= 3 and np.any(np.isfinite(z_med)):
            if tension <= 0.0:
                z0, a, rmse = _fit_log(rzb, z_med, R)
                z_fit = {"model": "log", "z0": z0, "a": a, "rmse": rmse}
            else:
                z0, amp, psi, rmse = _fit_k0_offset(rzb, z_med)
                z_fit = {"model": "k0", "z0": z0, "amp": amp, "psi": psi, "rmse": rmse}

    inner_mask = (r_z > 0) & (r_z <= R)
    z_flat = None
    if np.any(inner_mask):
        z_inner = z[inner_mask]
        z_flat = {
            "z_mean": float(np.mean(z_inner)),
            "z_std": float(np.std(z_inner)),
            "count": int(z_inner.size),
        }

    out = {
        "R": R,
        "inner_fit_I1": inner_fit,
        "outer_fit_K1": outer_fit,
        "curvature_fit_K0": curvature_fit,
        "height_fit": z_fit,
        "disk_flatness": z_flat,
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

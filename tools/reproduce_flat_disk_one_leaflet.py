#!/usr/bin/env python3
"""Reproduce the flat one-leaflet disk benchmark from docs/tex/1_disk_flat.tex."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent

if TYPE_CHECKING:
    from runtime.minimizer import Minimizer
    from tools.diagnostics.flat_disk_one_leaflet_theory import FlatDiskTheoryParams


def _ensure_repo_root_on_sys_path() -> None:
    import sys

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))


DEFAULT_FIXTURE = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)
DEFAULT_OUT = (
    ROOT
    / "benchmarks"
    / "outputs"
    / "diagnostics"
    / "flat_disk_one_leaflet_report.yaml"
)


@dataclass(frozen=True)
class BenchmarkScanConfig:
    """Scan configuration for theta_B reduced-energy sampling."""

    theta_min: float
    theta_max: float
    theta_count: int

    def validate(self) -> None:
        """Validate scan domain."""
        if int(self.theta_count) < 3:
            raise ValueError("theta_count must be >= 3.")
        if float(self.theta_max) <= float(self.theta_min):
            raise ValueError("theta_max must be > theta_min.")


@dataclass(frozen=True)
class BenchmarkOptimizeConfig:
    """Configuration for scalar theta_B optimization mode."""

    theta_initial: float
    optimize_steps: int
    optimize_every: int
    optimize_delta: float
    optimize_inner_steps: int

    def validate(self) -> None:
        """Validate optimizer controls."""
        if int(self.optimize_steps) < 1:
            raise ValueError("theta_optimize_steps must be >= 1.")
        if int(self.optimize_every) < 1:
            raise ValueError("theta_optimize_every must be >= 1.")
        if float(self.optimize_delta) <= 0.0:
            raise ValueError("theta_optimize_delta must be > 0.")
        if int(self.optimize_inner_steps) < 1:
            raise ValueError("theta_optimize_inner_steps must be >= 1.")


def _resolve_optimize_preset(
    *,
    optimize_preset: str,
    refine_level: int,
    optimize_cfg: BenchmarkOptimizeConfig,
) -> tuple[BenchmarkOptimizeConfig, str]:
    """Resolve benchmark optimize controls from a named preset."""
    preset = str(optimize_preset).lower()
    if preset == "none":
        return optimize_cfg, "none"
    if preset == "fast_r3":
        if int(refine_level) >= 3:
            return (
                BenchmarkOptimizeConfig(
                    theta_initial=float(optimize_cfg.theta_initial),
                    optimize_steps=10,
                    optimize_every=1,
                    optimize_delta=float(optimize_cfg.optimize_delta),
                    optimize_inner_steps=10,
                ),
                "fast_r3",
            )
        return optimize_cfg, "fast_r3_inactive"
    raise ValueError("optimize_preset must be 'none' or 'fast_r3'.")


def _load_mesh_from_fixture(path: Path):
    _ensure_repo_root_on_sys_path()
    from geometry.geom_io import load_data, parse_geometry

    data = load_data(str(path))
    return parse_geometry(data)


def _collect_disk_boundary_rows(mesh, *, group: str = "disk") -> np.ndarray:
    rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if (
            opts.get("rim_slope_match_group") == group
            or opts.get("tilt_thetaB_group") == group
            or opts.get("tilt_thetaB_group_in") == group
        ):
            row = mesh.vertex_index_to_row.get(int(vid))
            if row is not None:
                rows.append(int(row))
    out = np.asarray(rows, dtype=int)
    if out.size == 0:
        raise AssertionError(f"Missing or empty disk boundary group: {group!r}")
    return out


def _radial_unit_vectors(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]
    return r, r_hat


def _factor_difference(measured: float, target: float) -> float:
    t = abs(float(target))
    m = abs(float(measured))
    if t < 1e-18:
        return 1.0 if m < 1e-18 else float("inf")
    ratio = m / t
    if ratio <= 0.0:
        return float("inf")
    return float(max(ratio, 1.0 / ratio))


def _configure_benchmark_mesh(
    mesh,
    *,
    theory_params: FlatDiskTheoryParams,
    outer_mode: str,
    smoothness_model: str,
) -> None:
    _ensure_repo_root_on_sys_path()
    from tools.diagnostics.flat_disk_one_leaflet_theory import (
        solver_mapping_from_theory,
    )

    gp = mesh.global_parameters
    mapping = solver_mapping_from_theory(theory_params)

    gp.set("surface_tension", 0.0)
    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 0.0)
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_step_size", 0.08)
    gp.set("tilt_inner_steps", 250)
    gp.set("tilt_tol", 1e-12)
    gp.set("tilt_kkt_projection_during_relaxation", False)
    gp.set("tilt_thetaB_optimize", False)
    gp.set("tilt_thetaB_group_in", "disk")
    gp.set("rim_slope_match_disk_group", "disk")
    gp.set("tilt_thetaB_contact_penalty_mode", "off")
    gp.set("tilt_thetaB_contact_strength_in", float(theory_params.drive))
    gp.set("tilt_thetaB_value", 0.0)
    gp.set("tilt_thetaB_center", [0.0, 0.0, 0.0])
    gp.set("tilt_thetaB_normal", [0.0, 0.0, 1.0])
    gp.set("pin_to_plane_normal", [0.0, 0.0, 1.0])
    gp.set("pin_to_plane_point", [0.0, 0.0, 0.0])
    gp.set("leaflet_out_absent_presets", [])

    gp.set("bending_modulus_in", float(mapping["bending_modulus_in"]))
    gp.set("tilt_modulus_in", float(mapping["tilt_modulus_in"]))
    gp.set("tilt_twist_modulus_in", 0.0)

    if smoothness_model == "dirichlet":
        smoothness_in_module = "tilt_smoothness_in"
        smoothness_out_module = "tilt_smoothness_out"
    elif smoothness_model == "splay_twist":
        smoothness_in_module = "tilt_splay_twist_in"
        smoothness_out_module = "tilt_smoothness_out"
    else:
        raise ValueError("smoothness_model must be 'dirichlet' or 'splay_twist'.")

    if outer_mode == "disabled":
        mesh.energy_modules = [
            "tilt_in",
            smoothness_in_module,
            "tilt_thetaB_contact_in",
        ]
        for vid in mesh.vertex_ids:
            v = mesh.vertices[int(vid)]
            v.tilt_out = np.zeros(3, dtype=float)
            v.tilt_fixed_out = True
    elif outer_mode == "free":
        mesh.energy_modules = [
            "tilt_in",
            smoothness_in_module,
            "tilt_out",
            smoothness_out_module,
            "tilt_thetaB_contact_in",
        ]
        gp.set("bending_modulus_out", float(mapping["bending_modulus_in"]))
        gp.set("tilt_modulus_out", float(mapping["tilt_modulus_in"]))
        gp.set("tilt_twist_modulus_out", 0.0)
        for vid in mesh.vertex_ids:
            v = mesh.vertices[int(vid)]
            v.tilt_out = np.zeros(3, dtype=float)
    else:
        raise ValueError("outer_mode must be 'disabled' or 'free'.")

    mesh.constraint_modules = [
        "pin_to_plane",
        "pin_to_circle",
        "tilt_thetaB_boundary_in",
    ]


def _build_minimizer(mesh) -> Minimizer:
    _ensure_repo_root_on_sys_path()
    from runtime.constraint_manager import ConstraintModuleManager
    from runtime.energy_manager import EnergyModuleManager
    from runtime.minimizer import Minimizer
    from runtime.steppers.gradient_descent import GradientDescent

    return Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
    )


def _run_theta_relaxation(
    minim: Minimizer,
    *,
    theta_value: float,
    reset_outer: bool,
) -> float:
    mesh = minim.mesh
    gp = mesh.global_parameters
    gp.set("tilt_thetaB_value", float(theta_value))

    tin = np.zeros_like(mesh.tilts_in_view())
    mesh.set_tilts_in_from_array(tin)
    if reset_outer:
        tout = np.zeros_like(mesh.tilts_out_view())
        mesh.set_tilts_out_from_array(tout)

    minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode="coupled")
    energy = float(minim.compute_energy())
    if not np.isfinite(energy):
        raise ValueError(f"Non-finite energy during theta scan at theta={theta_value}.")
    return energy


def _run_theta_optimize(
    minim: Minimizer,
    *,
    optimize_cfg: BenchmarkOptimizeConfig,
    reset_outer: bool,
) -> float:
    mesh = minim.mesh
    gp = mesh.global_parameters
    gp.set("tilt_thetaB_optimize", True)
    gp.set("tilt_thetaB_value", float(optimize_cfg.theta_initial))
    gp.set("tilt_thetaB_optimize_every", int(optimize_cfg.optimize_every))
    gp.set("tilt_thetaB_optimize_delta", float(optimize_cfg.optimize_delta))
    gp.set("tilt_thetaB_optimize_inner_steps", int(optimize_cfg.optimize_inner_steps))

    tin = np.zeros_like(mesh.tilts_in_view())
    mesh.set_tilts_in_from_array(tin)
    if reset_outer:
        tout = np.zeros_like(mesh.tilts_out_view())
        mesh.set_tilts_out_from_array(tout)

    positions = mesh.positions_view()
    tilt_mode = str(gp.get("tilt_solve_mode", "coupled") or "coupled")
    for i in range(int(optimize_cfg.optimize_steps)):
        minim._relax_leaflet_tilts(positions=positions, mode=tilt_mode)
        minim._optimize_thetaB_scalar(tilt_mode=tilt_mode, iteration=i)

    theta_opt = float(gp.get("tilt_thetaB_value") or 0.0)
    if not np.isfinite(theta_opt):
        raise ValueError("Non-finite optimized theta_B value.")
    return theta_opt


def _profile_metrics(mesh, *, radius: float) -> dict[str, float]:
    positions = mesh.positions_view()
    r, r_hat = _radial_unit_vectors(positions)
    t_in = mesh.tilts_in_view()
    t_in_rad = np.einsum("ij,ij->i", t_in, r_hat)

    inner_mask = r < (0.60 * float(radius))
    rim_mask = (r >= (0.90 * float(radius))) & (r <= (1.10 * float(radius)))
    outer_mask = (r >= (2.00 * float(radius))) & (r <= (4.00 * float(radius)))
    if not np.any(rim_mask):
        raise AssertionError(
            "Rim profile band is empty; cannot evaluate benchmark profile."
        )
    if not np.any(outer_mask):
        raise AssertionError(
            "Outer profile band is empty; cannot evaluate benchmark profile."
        )

    def _median_abs(mask: np.ndarray) -> float:
        if not np.any(mask):
            return float("nan")
        return float(np.median(np.abs(t_in_rad[mask])))

    return {
        "inner_abs_median": _median_abs(inner_mask),
        "rim_abs_median": _median_abs(rim_mask),
        "outer_abs_median": _median_abs(outer_mask),
        "tilt_in_radial_max_abs": float(np.max(np.abs(t_in_rad))),
    }


def _rim_continuity_metrics(
    mesh,
    *,
    radius: float,
) -> dict[str, float]:
    """Compute rim continuity diagnostics by matching nearest angles across r=R."""
    positions = mesh.positions_view()
    r, r_hat = _radial_unit_vectors(positions)
    phi = np.mod(np.arctan2(positions[:, 1], positions[:, 0]), 2.0 * np.pi)
    t_in_rad = np.einsum("ij,ij->i", mesh.tilts_in_view(), r_hat)

    inner_candidates = r < (float(radius) - 1e-12)
    outer_candidates = r > (float(radius) + 1e-12)
    if not np.any(inner_candidates) or not np.any(outer_candidates):
        return {
            "matched_bins": 0,
            "jump_abs_median": float("nan"),
            "jump_abs_max": float("nan"),
            "jump_signed_mean": float("nan"),
        }

    r_in_shell = float(np.max(r[inner_candidates]))
    r_out_shell = float(np.min(r[outer_candidates]))
    tol_in = max(1e-9, 1e-5 * max(1.0, abs(r_in_shell)))
    tol_out = max(1e-9, 1e-5 * max(1.0, abs(r_out_shell)))
    inner_mask = np.abs(r - r_in_shell) <= tol_in
    outer_mask = np.abs(r - r_out_shell) <= tol_out
    inner_rows = np.flatnonzero(inner_mask)
    outer_rows = np.flatnonzero(outer_mask)
    if inner_rows.size == 0 or outer_rows.size == 0:
        return {
            "matched_bins": 0,
            "jump_abs_median": float("nan"),
            "jump_abs_max": float("nan"),
            "jump_signed_mean": float("nan"),
        }

    phi_in = phi[inner_rows]
    phi_out = phi[outer_rows]
    dphi = np.abs(phi_out[:, None] - phi_in[None, :])
    dphi = np.minimum(dphi, 2.0 * np.pi - dphi)
    nearest_in = np.argmin(dphi, axis=1)
    jumps = t_in_rad[outer_rows] - t_in_rad[inner_rows[nearest_in]]
    arr = np.asarray(jumps, dtype=float)
    return {
        "matched_bins": int(arr.size),
        "jump_abs_median": float(np.median(np.abs(arr))),
        "jump_abs_max": float(np.max(np.abs(arr))),
        "jump_signed_mean": float(np.mean(arr)),
    }


def _contact_diagnostics(
    *,
    breakdown: dict[str, float],
    theory,
    radius: float,
) -> dict[str, float]:
    """Return contact energy diagnostics in both absolute and per-length units."""
    perimeter = 2.0 * np.pi * float(radius)
    if perimeter <= 0.0:
        raise ValueError("radius must be > 0 for contact diagnostics.")
    mesh_contact = float(breakdown.get("tilt_thetaB_contact_in", 0.0))
    theory_contact = float(theory.contact)
    return {
        "mesh_contact_energy": mesh_contact,
        "theory_contact_energy": theory_contact,
        "mesh_contact_per_length": float(mesh_contact / perimeter),
        "theory_contact_per_length": float(theory_contact / perimeter),
        "contact_factor": float(_factor_difference(mesh_contact, theory_contact)),
    }


def run_flat_disk_one_leaflet_benchmark(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    refine_level: int = 2,
    outer_mode: str = "disabled",
    smoothness_model: str = "dirichlet",
    theta_min: float = 0.0,
    theta_max: float = 0.0014,
    theta_count: int = 8,
    theta_mode: str = "scan",
    theta_initial: float = 0.0,
    theta_optimize_steps: int = 20,
    theta_optimize_every: int = 1,
    theta_optimize_delta: float = 2.0e-4,
    theta_optimize_inner_steps: int = 20,
    optimize_preset: str = "none",
    theory_params: FlatDiskTheoryParams | None = None,
) -> dict[str, Any]:
    """Run the flat one-leaflet benchmark and return a report dict."""
    _ensure_repo_root_on_sys_path()
    from runtime.refinement import refine_triangle_mesh
    from tools.diagnostics.flat_disk_one_leaflet_theory import (
        compute_flat_disk_theory,
        quadratic_min_from_scan,
        tex_reference_params,
    )

    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = (ROOT / fixture_path).resolve()
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture not found: {fixture_path}")

    if int(refine_level) < 0:
        raise ValueError("refine_level must be >= 0.")

    params = theory_params if theory_params is not None else tex_reference_params()
    theory = compute_flat_disk_theory(params)
    theta_mode_str = str(theta_mode).lower()
    if theta_mode_str not in {"scan", "optimize"}:
        raise ValueError("theta_mode must be 'scan' or 'optimize'.")

    scan_cfg = None
    optimize_cfg = None
    effective_optimize_preset = "none"
    if theta_mode_str == "scan":
        scan_cfg = BenchmarkScanConfig(
            theta_min=float(theta_min),
            theta_max=float(theta_max),
            theta_count=int(theta_count),
        )
        scan_cfg.validate()
    else:
        optimize_cfg = BenchmarkOptimizeConfig(
            theta_initial=float(theta_initial),
            optimize_steps=int(theta_optimize_steps),
            optimize_every=int(theta_optimize_every),
            optimize_delta=float(theta_optimize_delta),
            optimize_inner_steps=int(theta_optimize_inner_steps),
        )
        optimize_cfg.validate()
        optimize_cfg, effective_optimize_preset = _resolve_optimize_preset(
            optimize_preset=str(optimize_preset),
            refine_level=int(refine_level),
            optimize_cfg=optimize_cfg,
        )
        optimize_cfg.validate()

    mesh = _load_mesh_from_fixture(fixture_path)
    for _ in range(int(refine_level)):
        mesh = refine_triangle_mesh(mesh)

    _configure_benchmark_mesh(
        mesh,
        theory_params=params,
        outer_mode=outer_mode,
        smoothness_model=smoothness_model,
    )
    _collect_disk_boundary_rows(mesh, group="disk")

    minim = _build_minimizer(mesh)
    minim.enforce_constraints_after_mesh_ops(mesh)
    mesh.project_tilts_to_tangent()

    scan_report: dict[str, Any] | None = None
    optimize_report: dict[str, Any] | None = None
    theta_star: float
    if theta_mode_str == "scan":
        assert scan_cfg is not None
        theta_values = np.linspace(
            float(scan_cfg.theta_min),
            float(scan_cfg.theta_max),
            int(scan_cfg.theta_count),
        )
        energies = np.zeros_like(theta_values)
        for i, theta_value in enumerate(theta_values):
            energies[i] = _run_theta_relaxation(
                minim,
                theta_value=float(theta_value),
                reset_outer=True,
            )

        min_idx = int(np.argmin(energies))
        if min_idx == 0 or min_idx == int(theta_values.size - 1):
            raise ValueError(
                "Empty interior scan bracket: minimum lies on theta scan boundary; "
                "expand [theta_min, theta_max]."
            )

        local_theta = theta_values[min_idx - 1 : min_idx + 2]
        local_energy = energies[min_idx - 1 : min_idx + 2]
        qfit = quadratic_min_from_scan(local_theta, local_energy)
        theta_star = float(qfit.theta_star)
        scan_report = {
            "theta_min": float(scan_cfg.theta_min),
            "theta_max": float(scan_cfg.theta_max),
            "theta_count": int(scan_cfg.theta_count),
            "theta_values": [float(x) for x in theta_values.tolist()],
            "energy_values": [float(x) for x in energies.tolist()],
            "grid_min_theta": float(theta_values[min_idx]),
            "grid_min_energy": float(energies[min_idx]),
            "local_quadratic_fit": qfit.to_dict(),
        }
    else:
        assert optimize_cfg is not None
        t0 = perf_counter()
        theta_star = _run_theta_optimize(
            minim,
            optimize_cfg=optimize_cfg,
            reset_outer=True,
        )
        optimize_seconds = float(perf_counter() - t0)
        optimize_report = {
            "theta_initial": float(optimize_cfg.theta_initial),
            "optimize_steps": int(optimize_cfg.optimize_steps),
            "optimize_every": int(optimize_cfg.optimize_every),
            "optimize_delta": float(optimize_cfg.optimize_delta),
            "optimize_inner_steps": int(optimize_cfg.optimize_inner_steps),
            "optimize_seconds": optimize_seconds,
            "optimize_preset_effective": str(effective_optimize_preset),
            "theta_star": float(theta_star),
        }

    total_energy = _run_theta_relaxation(
        minim,
        theta_value=float(theta_star),
        reset_outer=True,
    )
    breakdown = minim.compute_energy_breakdown()

    profile = _profile_metrics(mesh, radius=float(theory.radius))
    rim_continuity = _rim_continuity_metrics(mesh, radius=float(theory.radius))
    positions = mesh.positions_view()
    z_span = float(np.ptp(positions[:, 2]))
    t_out = mesh.tilts_out_view()
    outer_free_rows = np.array(
        [
            row
            for row, vid in enumerate(mesh.vertex_ids)
            if not bool(getattr(mesh.vertices[int(vid)], "tilt_fixed_out", False))
        ],
        dtype=int,
    )
    if outer_free_rows.size:
        outer_mag = np.linalg.norm(t_out[outer_free_rows], axis=1)
        outer_max = float(np.max(outer_mag))
        outer_mean = float(np.mean(outer_mag))
    else:
        outer_max = 0.0
        outer_mean = 0.0

    outer_decay_probe_before = 0.0
    outer_decay_probe_after = 0.0
    if outer_mode == "free" and outer_free_rows.size:
        rng = np.random.default_rng(12345)
        t_probe = mesh.tilts_out_view().copy(order="F")
        t_probe[outer_free_rows] += 1e-3 * rng.standard_normal(
            (outer_free_rows.size, 3)
        )
        mesh.set_tilts_out_from_array(t_probe)
        outer_decay_probe_before = float(
            np.max(np.linalg.norm(mesh.tilts_out_view()[outer_free_rows], axis=1))
        )
        gp = mesh.global_parameters
        gp.set("tilt_thetaB_value", float(theta_star))
        orig_step = gp.get("tilt_step_size")
        orig_inner = gp.get("tilt_inner_steps")
        try:
            # Use a conservative probe relaxation so the test checks physical
            # decay instead of line-search stiffness sensitivity.
            gp.set("tilt_step_size", 1e-3)
            gp.set("tilt_inner_steps", 600)
            minim._relax_leaflet_tilts(positions=mesh.positions_view(), mode="coupled")
        finally:
            if orig_step is None:
                gp.unset("tilt_step_size")
            else:
                gp.set("tilt_step_size", orig_step)
            if orig_inner is None:
                gp.unset("tilt_inner_steps")
            else:
                gp.set("tilt_inner_steps", orig_inner)
        outer_decay_probe_after = float(
            np.max(np.linalg.norm(mesh.tilts_out_view()[outer_free_rows], axis=1))
        )

    theta_factor = _factor_difference(float(theta_star), float(theory.theta_star))
    energy_factor = _factor_difference(
        float(abs(total_energy)), float(abs(theory.total))
    )

    report = {
        "meta": {
            "fixture": str(fixture_path.relative_to(ROOT)),
            "refine_level": int(refine_level),
            "outer_mode": str(outer_mode),
            "smoothness_model": str(smoothness_model),
            "theta_mode": str(theta_mode_str),
            "optimize_preset": str(optimize_preset).lower(),
            "optimize_preset_effective": str(effective_optimize_preset),
            "theory_source": "docs/tex/1_disk_flat.tex",
        },
        "theory": theory.to_dict(),
        "scan": scan_report,
        "optimize": optimize_report,
        "mesh": {
            "theta_star": float(theta_star),
            "total_energy": float(total_energy),
            "energy_breakdown": {str(k): float(v) for k, v in breakdown.items()},
            "planarity_z_span": z_span,
            "profile": profile,
            "rim_continuity": rim_continuity,
            "outer_tilt_max_free_rows": outer_max,
            "outer_tilt_mean_free_rows": outer_mean,
            "outer_decay_probe_max_before": outer_decay_probe_before,
            "outer_decay_probe_max_after": outer_decay_probe_after,
        },
        "diagnostics": {
            "contact": _contact_diagnostics(
                breakdown={str(k): float(v) for k, v in breakdown.items()},
                theory=theory,
                radius=float(theory.radius),
            )
        },
        "parity": {
            "theta_factor": float(theta_factor),
            "energy_factor": float(energy_factor),
            "meets_factor_2": bool(theta_factor <= 2.0 and energy_factor <= 2.0),
        },
    }
    return report


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(DEFAULT_FIXTURE))
    ap.add_argument("--refine-level", type=int, default=2)
    ap.add_argument("--outer-mode", choices=("disabled", "free"), default="disabled")
    ap.add_argument(
        "--smoothness-model",
        choices=("dirichlet", "splay_twist"),
        default="dirichlet",
    )
    ap.add_argument("--theta-mode", choices=("scan", "optimize"), default="scan")
    ap.add_argument("--theta-min", type=float, default=0.0)
    ap.add_argument("--theta-max", type=float, default=0.0014)
    ap.add_argument("--theta-count", type=int, default=8)
    ap.add_argument("--theta-initial", type=float, default=0.0)
    ap.add_argument("--theta-optimize-steps", type=int, default=20)
    ap.add_argument("--theta-optimize-every", type=int, default=1)
    ap.add_argument("--theta-optimize-delta", type=float, default=2.0e-4)
    ap.add_argument("--theta-optimize-inner-steps", type=int, default=20)
    ap.add_argument(
        "--optimize-preset",
        choices=("none", "fast_r3"),
        default="none",
    )
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args(list(argv) if argv is not None else None)

    report = run_flat_disk_one_leaflet_benchmark(
        fixture=args.fixture,
        refine_level=args.refine_level,
        outer_mode=args.outer_mode,
        smoothness_model=args.smoothness_model,
        theta_mode=args.theta_mode,
        theta_min=args.theta_min,
        theta_max=args.theta_max,
        theta_count=args.theta_count,
        theta_initial=args.theta_initial,
        theta_optimize_steps=args.theta_optimize_steps,
        theta_optimize_every=args.theta_optimize_every,
        theta_optimize_delta=args.theta_optimize_delta,
        theta_optimize_inner_steps=args.theta_optimize_inner_steps,
        optimize_preset=args.optimize_preset,
    )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    _write_yaml(out_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

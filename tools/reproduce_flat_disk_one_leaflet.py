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


@dataclass(frozen=True)
class BenchmarkPolishConfig:
    """Configuration for local theta_B reduced-energy polish."""

    polish_delta: float
    polish_points: int

    def validate(self) -> None:
        """Validate polish controls."""
        if float(self.polish_delta) <= 0.0:
            raise ValueError("theta_polish_delta must be > 0.")
        if int(self.polish_points) < 3:
            raise ValueError("theta_polish_points must be >= 3.")
        if int(self.polish_points) % 2 == 0:
            raise ValueError("theta_polish_points must be odd.")


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
    if preset == "full_accuracy_r3":
        if int(refine_level) >= 3:
            return (
                BenchmarkOptimizeConfig(
                    theta_initial=float(optimize_cfg.theta_initial),
                    optimize_steps=40,
                    optimize_every=1,
                    optimize_delta=1.0e-4,
                    optimize_inner_steps=20,
                ),
                "full_accuracy_r3",
            )
        return optimize_cfg, "full_accuracy_r3_inactive"
    if preset == "kh_wide":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=120,
                optimize_every=1,
                optimize_delta=2.0e-3,
                optimize_inner_steps=20,
            ),
            "kh_wide",
        )
    raise ValueError(
        "optimize_preset must be 'none', 'fast_r3', 'full_accuracy_r3', or 'kh_wide'."
    )


def _load_mesh_from_fixture(path: Path):
    _ensure_repo_root_on_sys_path()
    from geometry.geom_io import load_data, parse_geometry

    data = load_data(str(path))
    return parse_geometry(data)


def _refine_mesh_locally_near_rim(
    mesh,
    *,
    local_steps: int,
    rim_radius: float,
    band_half_width: float,
):
    """Refine only facets in a radial band around the disk rim.

    This uses facet-level ``no_refine`` flags before each call to
    ``refine_triangle_mesh``. It is benchmark-harness-only and does not modify
    runtime refinement behavior.
    """
    _ensure_repo_root_on_sys_path()
    from runtime.refinement import refine_triangle_mesh

    steps = int(local_steps)
    if steps <= 0:
        return mesh
    if float(band_half_width) <= 0.0:
        raise ValueError("rim_local_refine_band_half_width must be > 0.")
    if float(rim_radius) <= 0.0:
        raise ValueError("rim_radius must be > 0 for local rim refinement.")

    out = mesh
    for _ in range(steps):
        out.build_connectivity_maps()
        out.build_facet_vertex_loops()
        positions = out.positions_view()
        selected = 0
        for facet in out.facets.values():
            loop = out.facet_vertex_loops.get(int(facet.index))
            if loop is None or len(loop) == 0:
                continue
            rows = np.asarray(
                [out.vertex_index_to_row[int(vid)] for vid in loop], dtype=int
            )
            centroid = np.mean(positions[rows], axis=0)
            r_cent = float(np.linalg.norm(centroid[:2]))
            if abs(r_cent - float(rim_radius)) <= float(band_half_width):
                facet.options.pop("no_refine", None)
                selected += 1
            else:
                facet.options["no_refine"] = True
        if selected == 0:
            raise AssertionError(
                "Rim local refinement selected no facets. Increase "
                "rim_local_refine_band_lambda."
            )
        out = refine_triangle_mesh(out)

    for facet in out.facets.values():
        facet.options.pop("no_refine", None)
    return out


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
    parameterization: str,
    outer_mode: str,
    smoothness_model: str,
    splay_modulus_scale_in: float,
    tilt_mass_mode_in: str,
) -> None:
    _ensure_repo_root_on_sys_path()
    from tools.diagnostics.flat_disk_one_leaflet_theory import (
        solver_mapping_from_theory,
    )

    gp = mesh.global_parameters
    mapping = solver_mapping_from_theory(
        theory_params, parameterization=str(parameterization)
    )

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
    gp.set("tilt_mass_mode_in", str(tilt_mass_mode_in))
    gp.set("tilt_twist_modulus_in", 0.0)

    if smoothness_model == "dirichlet":
        smoothness_in_module = "tilt_smoothness_in"
        smoothness_out_module = "tilt_smoothness_out"
    elif smoothness_model == "splay_twist":
        smoothness_in_module = "tilt_splay_twist_in"
        smoothness_out_module = "tilt_smoothness_out"
        gp.set(
            "tilt_splay_modulus_in",
            float(mapping["bending_modulus_in"]) * float(splay_modulus_scale_in),
        )
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


def _run_theta_local_polish(
    minim: Minimizer,
    *,
    theta_center: float,
    polish_cfg: BenchmarkPolishConfig,
    reset_outer: bool,
) -> tuple[float, dict[str, Any]]:
    """Refine theta_B by local reduced-energy sampling near a center value."""
    _ensure_repo_root_on_sys_path()
    from tools.diagnostics.flat_disk_one_leaflet_theory import quadratic_min_from_scan

    n = int(polish_cfg.polish_points)
    d = float(polish_cfg.polish_delta)
    offsets = np.linspace(-d, d, n)
    theta_values = float(theta_center) + offsets
    energies = np.zeros_like(theta_values)
    for i, theta_value in enumerate(theta_values):
        energies[i] = _run_theta_relaxation(
            minim,
            theta_value=float(theta_value),
            reset_outer=reset_outer,
        )
    min_idx = int(np.argmin(energies))
    theta_star = float(theta_values[min_idx])
    method = "grid_min"
    qfit_report = None
    if 0 < min_idx < int(theta_values.size - 1):
        local_theta = theta_values[min_idx - 1 : min_idx + 2]
        local_energy = energies[min_idx - 1 : min_idx + 2]
        try:
            qfit = quadratic_min_from_scan(local_theta, local_energy)
            theta_star = float(qfit.theta_star)
            method = "quadratic_local"
            qfit_report = qfit.to_dict()
        except ValueError:
            method = "grid_min"
    report = {
        "polish_delta": float(d),
        "polish_points": int(n),
        "theta_values": [float(x) for x in theta_values.tolist()],
        "energy_values": [float(x) for x in energies.tolist()],
        "grid_min_theta": float(theta_values[min_idx]),
        "grid_min_energy": float(energies[min_idx]),
        "method": method,
        "local_quadratic_fit": qfit_report,
    }
    return float(theta_star), report


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
    theta_polish_delta: float = 1.0e-4,
    theta_polish_points: int = 3,
    optimize_preset: str = "none",
    parameterization: str = "legacy",
    length_scale_nm: float = 15.0,
    radius_nm: float = 7.0,
    kappa_physical: float = 10.0,
    kappa_t_physical: float = 10.0,
    drive_physical: float = (2.0 / 0.7),
    splay_modulus_scale_in: float = 1.0,
    tilt_mass_mode_in: str = "auto",
    rim_local_refine_steps: int = 0,
    rim_local_refine_band_lambda: float = 0.0,
    theory_params: FlatDiskTheoryParams | None = None,
) -> dict[str, Any]:
    """Run the flat one-leaflet benchmark and return a report dict."""
    _ensure_repo_root_on_sys_path()
    from runtime.refinement import refine_triangle_mesh
    from tools.diagnostics.flat_disk_one_leaflet_theory import (
        compute_flat_disk_kh_physical_theory,
        compute_flat_disk_theory,
        physical_to_dimensionless_theory_params,
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
    if int(rim_local_refine_steps) < 0:
        raise ValueError("rim_local_refine_steps must be >= 0.")
    if float(rim_local_refine_band_lambda) < 0.0:
        raise ValueError("rim_local_refine_band_lambda must be >= 0.")
    if float(splay_modulus_scale_in) <= 0.0:
        raise ValueError("splay_modulus_scale_in must be > 0.")
    mode = str(parameterization).lower()
    if mode not in {"legacy", "kh_physical"}:
        raise ValueError("parameterization must be 'legacy' or 'kh_physical'.")
    mass_mode_raw = str(tilt_mass_mode_in).strip().lower()
    if mass_mode_raw == "auto":
        mass_mode = "consistent" if mode == "kh_physical" else "lumped"
    elif mass_mode_raw in {"lumped", "consistent"}:
        mass_mode = mass_mode_raw
    else:
        raise ValueError("tilt_mass_mode_in must be 'auto', 'lumped', or 'consistent'.")

    using_physical_scaling = theory_params is None and mode == "kh_physical"
    if theory_params is not None:
        params = theory_params
    elif using_physical_scaling:
        params = physical_to_dimensionless_theory_params(
            kappa_physical=float(kappa_physical),
            kappa_t_physical=float(kappa_t_physical),
            radius_physical=float(radius_nm),
            drive_physical=float(drive_physical),
            length_scale=float(length_scale_nm),
        )
    else:
        params = tex_reference_params()
    if mode == "legacy":
        theory = compute_flat_disk_theory(params)
        theory_model = "legacy_scalar_reduced"
        theory_source = "docs/tex/1_disk_flat.tex"
    else:
        theory = compute_flat_disk_kh_physical_theory(params)
        theory_model = "kh_physical_strict_kh"
        theory_source = "kh_physical_closed_form"
    theta_mode_str = str(theta_mode).lower()
    if theta_mode_str not in {"scan", "optimize", "optimize_full"}:
        raise ValueError("theta_mode must be 'scan', 'optimize', or 'optimize_full'.")

    scan_cfg = None
    optimize_cfg = None
    polish_cfg = None
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
        if theta_mode_str == "optimize_full":
            polish_delta = float(theta_polish_delta)
            polish_points = int(theta_polish_points)
            if effective_optimize_preset == "full_accuracy_r3":
                polish_delta = min(polish_delta, 5.0e-5)
                polish_points = max(polish_points, 5)
            polish_cfg = BenchmarkPolishConfig(
                polish_delta=polish_delta,
                polish_points=polish_points,
            )
            polish_cfg.validate()

    mesh = _load_mesh_from_fixture(fixture_path)
    for _ in range(int(refine_level)):
        mesh = refine_triangle_mesh(mesh)
    if int(rim_local_refine_steps) > 0:
        band_half_width = float(rim_local_refine_band_lambda) * float(
            theory.lambda_value
        )
        mesh = _refine_mesh_locally_near_rim(
            mesh,
            local_steps=int(rim_local_refine_steps),
            rim_radius=float(theory.radius),
            band_half_width=band_half_width,
        )

    _configure_benchmark_mesh(
        mesh,
        theory_params=params,
        parameterization=mode,
        outer_mode=outer_mode,
        smoothness_model=smoothness_model,
        splay_modulus_scale_in=float(splay_modulus_scale_in),
        tilt_mass_mode_in=mass_mode,
    )
    _collect_disk_boundary_rows(mesh, group="disk")

    minim = _build_minimizer(mesh)
    minim.enforce_constraints_after_mesh_ops(mesh)
    mesh.project_tilts_to_tangent()

    scan_report: dict[str, Any] | None = None
    optimize_report: dict[str, Any] | None = None
    theta_factor_raw: float | None = None
    energy_factor_raw: float | None = None
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
        theta_opt_raw = _run_theta_optimize(
            minim,
            optimize_cfg=optimize_cfg,
            reset_outer=True,
        )
        theta_star = float(theta_opt_raw)
        polish_report = None
        if theta_mode_str == "optimize_full":
            assert polish_cfg is not None
            theta_star, polish_report = _run_theta_local_polish(
                minim,
                theta_center=float(theta_opt_raw),
                polish_cfg=polish_cfg,
                reset_outer=True,
            )
            polish_theta = np.asarray(polish_report["theta_values"], dtype=float)
            polish_energy = np.asarray(polish_report["energy_values"], dtype=float)
            center_idx = int(np.argmin(np.abs(polish_theta - float(theta_opt_raw))))
            raw_energy = float(polish_energy[center_idx])
            theta_factor_raw = _factor_difference(
                float(theta_opt_raw), float(theory.theta_star)
            )
            energy_factor_raw = _factor_difference(
                float(abs(raw_energy)), float(abs(theory.total))
            )
        optimize_seconds = float(perf_counter() - t0)
        optimize_theta_span = float(
            int(optimize_cfg.optimize_steps) * float(optimize_cfg.optimize_delta)
        )
        hit_step_limit = bool(
            abs(float(theta_opt_raw) - float(optimize_cfg.theta_initial))
            >= (optimize_theta_span - 1e-12)
        )
        optimize_report = {
            "theta_initial": float(optimize_cfg.theta_initial),
            "optimize_steps": int(optimize_cfg.optimize_steps),
            "optimize_every": int(optimize_cfg.optimize_every),
            "optimize_delta": float(optimize_cfg.optimize_delta),
            "optimize_inner_steps": int(optimize_cfg.optimize_inner_steps),
            "optimize_theta_span": optimize_theta_span,
            "hit_step_limit": hit_step_limit,
            "optimize_seconds": optimize_seconds,
            "optimize_preset_effective": str(effective_optimize_preset),
            "theta_star_raw": float(theta_opt_raw),
            "theta_factor_raw": (
                None if theta_factor_raw is None else float(theta_factor_raw)
            ),
            "energy_factor_raw": (
                None if energy_factor_raw is None else float(energy_factor_raw)
            ),
            "polish": polish_report,
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
    if theta_mode_str == "optimize_full":
        assert theta_factor_raw is not None
        assert energy_factor_raw is not None
        raw_score = float(
            np.hypot(
                np.log(max(theta_factor_raw, 1e-18)),
                np.log(max(energy_factor_raw, 1e-18)),
            )
        )
        full_score = float(
            np.hypot(
                np.log(max(theta_factor, 1e-18)), np.log(max(energy_factor, 1e-18))
            )
        )
        recommended_mode_for_theory = (
            "optimize" if raw_score <= full_score else "optimize_full"
        )
    else:
        recommended_mode_for_theory = "scan" if theta_mode_str == "scan" else "optimize"

    report = {
        "meta": {
            "fixture": str(fixture_path.relative_to(ROOT)),
            "refine_level": int(refine_level),
            "outer_mode": str(outer_mode),
            "smoothness_model": str(smoothness_model),
            "parameterization": str(mode),
            "using_physical_scaling": bool(using_physical_scaling),
            "kappa_physical": float(kappa_physical),
            "kappa_t_physical": float(kappa_t_physical),
            "length_scale_nm": float(length_scale_nm),
            "radius_nm": float(radius_nm),
            "drive_physical": float(drive_physical),
            "radius_dimless": float(params.radius),
            "theta_mode": str(theta_mode_str),
            "optimize_preset": str(optimize_preset).lower(),
            "optimize_preset_effective": str(effective_optimize_preset),
            "splay_modulus_scale_in": float(splay_modulus_scale_in),
            "tilt_mass_mode_in": str(mass_mode),
            "rim_local_refine_steps": int(rim_local_refine_steps),
            "rim_local_refine_band_lambda": float(rim_local_refine_band_lambda),
            "theory_model": theory_model,
            "theory_source": theory_source,
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
            "lane": str(mode),
            "theta_factor": float(theta_factor),
            "energy_factor": float(energy_factor),
            "meets_factor_2": bool(theta_factor <= 2.0 and energy_factor <= 2.0),
            "recommended_mode_for_theory": str(recommended_mode_for_theory),
        },
    }
    return report


def run_flat_disk_lane_comparison(
    *,
    fixture: Path | str = DEFAULT_FIXTURE,
    refine_level: int = 2,
    outer_mode: str = "disabled",
    legacy_smoothness_model: str = "dirichlet",
    legacy_theta_mode: str = "scan",
    legacy_theta_min: float = 0.0,
    legacy_theta_max: float = 0.0014,
    legacy_theta_count: int = 8,
    legacy_theta_initial: float = 0.0,
    legacy_theta_optimize_steps: int = 20,
    legacy_theta_optimize_every: int = 1,
    legacy_theta_optimize_delta: float = 2.0e-4,
    legacy_theta_optimize_inner_steps: int = 20,
    kh_smoothness_model: str = "splay_twist",
    kh_theta_mode: str = "optimize",
    kh_theta_min: float = 0.0,
    kh_theta_max: float = 0.0014,
    kh_theta_count: int = 8,
    kh_theta_initial: float = 0.0,
    kh_theta_optimize_steps: int = 20,
    kh_theta_optimize_every: int = 1,
    kh_theta_optimize_delta: float = 2.0e-4,
    kh_theta_optimize_inner_steps: int = 20,
    kh_length_scale_nm: float = 15.0,
    kh_radius_nm: float = 7.0,
    kh_kappa_physical: float = 10.0,
    kh_kappa_t_physical: float = 10.0,
    kh_drive_physical: float = (2.0 / 0.7),
    splay_modulus_scale_in: float = 1.0,
    tilt_mass_mode_in: str = "auto",
) -> dict[str, Any]:
    """Run both legacy and kh_physical benchmark lanes and summarize differences."""
    legacy = run_flat_disk_one_leaflet_benchmark(
        fixture=fixture,
        refine_level=refine_level,
        outer_mode=outer_mode,
        smoothness_model=legacy_smoothness_model,
        theta_mode=legacy_theta_mode,
        theta_min=legacy_theta_min,
        theta_max=legacy_theta_max,
        theta_count=legacy_theta_count,
        theta_initial=legacy_theta_initial,
        theta_optimize_steps=legacy_theta_optimize_steps,
        theta_optimize_every=legacy_theta_optimize_every,
        theta_optimize_delta=legacy_theta_optimize_delta,
        theta_optimize_inner_steps=legacy_theta_optimize_inner_steps,
        parameterization="legacy",
        splay_modulus_scale_in=splay_modulus_scale_in,
        tilt_mass_mode_in=tilt_mass_mode_in,
    )

    kh = run_flat_disk_one_leaflet_benchmark(
        fixture=fixture,
        refine_level=refine_level,
        outer_mode=outer_mode,
        smoothness_model=kh_smoothness_model,
        theta_mode=kh_theta_mode,
        theta_min=kh_theta_min,
        theta_max=kh_theta_max,
        theta_count=kh_theta_count,
        theta_initial=kh_theta_initial,
        theta_optimize_steps=kh_theta_optimize_steps,
        theta_optimize_every=kh_theta_optimize_every,
        theta_optimize_delta=kh_theta_optimize_delta,
        theta_optimize_inner_steps=kh_theta_optimize_inner_steps,
        parameterization="kh_physical",
        length_scale_nm=kh_length_scale_nm,
        radius_nm=kh_radius_nm,
        kappa_physical=kh_kappa_physical,
        kappa_t_physical=kh_kappa_t_physical,
        drive_physical=kh_drive_physical,
        splay_modulus_scale_in=splay_modulus_scale_in,
        tilt_mass_mode_in=tilt_mass_mode_in,
    )

    legacy_theta = float(legacy["mesh"]["theta_star"])
    kh_theta = float(kh["mesh"]["theta_star"])
    legacy_energy = float(legacy["mesh"]["total_energy"])
    kh_energy = float(kh["mesh"]["total_energy"])
    legacy_contact = float(legacy["diagnostics"]["contact"]["mesh_contact_energy"])
    kh_contact = float(kh["diagnostics"]["contact"]["mesh_contact_energy"])

    theta_ratio = (
        float(kh_theta / legacy_theta) if abs(legacy_theta) > 1e-18 else float("inf")
    )
    energy_ratio = (
        float(abs(kh_energy) / abs(legacy_energy))
        if abs(legacy_energy) > 1e-18
        else float("inf")
    )
    contact_ratio = (
        float(abs(kh_contact) / abs(legacy_contact))
        if abs(legacy_contact) > 1e-18
        else float("inf")
    )

    return {
        "meta": {
            "mode": "compare_lanes",
            "fixture": legacy["meta"]["fixture"],
            "refine_level": int(refine_level),
            "outer_mode": str(outer_mode),
        },
        "legacy": legacy,
        "kh_physical": kh,
        "comparison": {
            "legacy_theta_star": legacy_theta,
            "kh_theta_star": kh_theta,
            "kh_over_legacy_theta_star_ratio": theta_ratio,
            "legacy_total_energy": legacy_energy,
            "kh_total_energy": kh_energy,
            "kh_over_legacy_abs_total_energy_ratio": energy_ratio,
            "legacy_contact_energy": legacy_contact,
            "kh_contact_energy": kh_contact,
            "kh_over_legacy_abs_contact_energy_ratio": contact_ratio,
            "legacy_theta_factor": float(legacy["parity"]["theta_factor"]),
            "kh_theta_factor": float(kh["parity"]["theta_factor"]),
            "legacy_energy_factor": float(legacy["parity"]["energy_factor"]),
            "kh_energy_factor": float(kh["parity"]["energy_factor"]),
        },
    }


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
    ap.add_argument(
        "--theta-mode", choices=("scan", "optimize", "optimize_full"), default="scan"
    )
    ap.add_argument("--theta-min", type=float, default=0.0)
    ap.add_argument("--theta-max", type=float, default=0.0014)
    ap.add_argument("--theta-count", type=int, default=8)
    ap.add_argument("--theta-initial", type=float, default=0.0)
    ap.add_argument("--theta-optimize-steps", type=int, default=20)
    ap.add_argument("--theta-optimize-every", type=int, default=1)
    ap.add_argument("--theta-optimize-delta", type=float, default=2.0e-4)
    ap.add_argument("--theta-optimize-inner-steps", type=int, default=20)
    ap.add_argument("--theta-polish-delta", type=float, default=1.0e-4)
    ap.add_argument("--theta-polish-points", type=int, default=3)
    ap.add_argument(
        "--optimize-preset",
        choices=("none", "fast_r3", "full_accuracy_r3", "kh_wide"),
        default="none",
    )
    ap.add_argument(
        "--parameterization",
        choices=("legacy", "kh_physical"),
        default="legacy",
    )
    ap.add_argument("--length-scale-nm", type=float, default=15.0)
    ap.add_argument("--radius-nm", type=float, default=7.0)
    ap.add_argument("--kappa-physical", type=float, default=10.0)
    ap.add_argument("--kappa-t-physical", type=float, default=10.0)
    ap.add_argument("--drive-physical", type=float, default=(2.0 / 0.7))
    ap.add_argument("--splay-modulus-scale-in", type=float, default=1.0)
    ap.add_argument(
        "--tilt-mass-mode-in",
        choices=("auto", "lumped", "consistent"),
        default="auto",
    )
    ap.add_argument("--rim-local-refine-steps", type=int, default=0)
    ap.add_argument("--rim-local-refine-band-lambda", type=float, default=0.0)
    ap.add_argument("--compare-lanes", action="store_true")
    ap.add_argument(
        "--compare-kh-smoothness-model",
        choices=("dirichlet", "splay_twist"),
        default="splay_twist",
    )
    ap.add_argument(
        "--compare-kh-theta-mode",
        choices=("scan", "optimize", "optimize_full"),
        default="optimize",
    )
    ap.add_argument("--compare-kh-theta-min", type=float, default=0.0)
    ap.add_argument("--compare-kh-theta-max", type=float, default=0.0014)
    ap.add_argument("--compare-kh-theta-count", type=int, default=8)
    ap.add_argument("--compare-kh-theta-initial", type=float, default=0.0)
    ap.add_argument("--compare-kh-theta-optimize-steps", type=int, default=20)
    ap.add_argument("--compare-kh-theta-optimize-every", type=int, default=1)
    ap.add_argument("--compare-kh-theta-optimize-delta", type=float, default=2.0e-4)
    ap.add_argument("--compare-kh-theta-optimize-inner-steps", type=int, default=20)
    ap.add_argument("--output", default=str(DEFAULT_OUT))
    args = ap.parse_args(list(argv) if argv is not None else None)

    if bool(args.compare_lanes):
        report = run_flat_disk_lane_comparison(
            fixture=args.fixture,
            refine_level=args.refine_level,
            outer_mode=args.outer_mode,
            legacy_smoothness_model=args.smoothness_model,
            legacy_theta_mode=args.theta_mode,
            legacy_theta_min=args.theta_min,
            legacy_theta_max=args.theta_max,
            legacy_theta_count=args.theta_count,
            legacy_theta_initial=args.theta_initial,
            legacy_theta_optimize_steps=args.theta_optimize_steps,
            legacy_theta_optimize_every=args.theta_optimize_every,
            legacy_theta_optimize_delta=args.theta_optimize_delta,
            legacy_theta_optimize_inner_steps=args.theta_optimize_inner_steps,
            kh_smoothness_model=args.compare_kh_smoothness_model,
            kh_theta_mode=args.compare_kh_theta_mode,
            kh_theta_min=args.compare_kh_theta_min,
            kh_theta_max=args.compare_kh_theta_max,
            kh_theta_count=args.compare_kh_theta_count,
            kh_theta_initial=args.compare_kh_theta_initial,
            kh_theta_optimize_steps=args.compare_kh_theta_optimize_steps,
            kh_theta_optimize_every=args.compare_kh_theta_optimize_every,
            kh_theta_optimize_delta=args.compare_kh_theta_optimize_delta,
            kh_theta_optimize_inner_steps=args.compare_kh_theta_optimize_inner_steps,
            kh_length_scale_nm=args.length_scale_nm,
            kh_radius_nm=args.radius_nm,
            kh_kappa_physical=args.kappa_physical,
            kh_kappa_t_physical=args.kappa_t_physical,
            kh_drive_physical=args.drive_physical,
            splay_modulus_scale_in=args.splay_modulus_scale_in,
            tilt_mass_mode_in=args.tilt_mass_mode_in,
            rim_local_refine_steps=args.rim_local_refine_steps,
            rim_local_refine_band_lambda=args.rim_local_refine_band_lambda,
        )
    else:
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
            theta_polish_delta=args.theta_polish_delta,
            theta_polish_points=args.theta_polish_points,
            optimize_preset=args.optimize_preset,
            parameterization=args.parameterization,
            length_scale_nm=args.length_scale_nm,
            radius_nm=args.radius_nm,
            kappa_physical=args.kappa_physical,
            kappa_t_physical=args.kappa_t_physical,
            drive_physical=args.drive_physical,
            splay_modulus_scale_in=args.splay_modulus_scale_in,
            tilt_mass_mode_in=args.tilt_mass_mode_in,
            rim_local_refine_steps=args.rim_local_refine_steps,
            rim_local_refine_band_lambda=args.rim_local_refine_band_lambda,
        )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    _write_yaml(out_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

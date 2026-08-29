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

from tools.flat_disk_benchmark_metrics import (
    _boundary_at_R_parity_metrics,
    _collect_disk_boundary_rows,
    _collect_group_rows,  # noqa: F401 - compatibility re-export
    _collect_outer_radial_slope_samples,  # noqa: F401 - compatibility re-export
    _contact_diagnostics,
    _factor_difference,
    _fit_outer_radial_slope_samples,  # noqa: F401 - compatibility re-export
    _flat_benchmark_reference_profiles,
    _leakage_metrics,
    _order_rows_by_angle,  # noqa: F401 - compatibility re-export
    _profile_metrics,
    _radial_unit_vectors,  # noqa: F401 - compatibility re-export
    _rim_boundary_realization_metrics,
    _rim_continuity_metrics,
)

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
    plateau_patience: int = 0
    plateau_abs_tol: float = 0.0

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
        if int(self.plateau_patience) < 0:
            raise ValueError("theta_optimize_plateau_patience must be >= 0.")
        if float(self.plateau_abs_tol) < 0.0:
            raise ValueError("theta_optimize_plateau_abs_tol must be >= 0.")


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


def _resolve_tilt_mass_mode(value: str, *, leaflet: str, auto_default: str) -> str:
    mode = str(value).strip().lower()
    if mode == "auto":
        return str(auto_default)
    if mode not in {"lumped", "consistent"}:
        raise ValueError(
            f"tilt_mass_mode_{leaflet} must be 'auto', 'lumped', or 'consistent'."
        )
    return mode


def _resolve_tilt_transport_model(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"ambient_v1", "connection_v1"}:
        raise ValueError(
            "tilt_transport_model must be 'ambient_v1' or 'connection_v1'."
        )
    return mode


def _resolve_tilt_divergence_mode_in(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"native", "vertex_recovered"}:
        raise ValueError(
            "tilt_divergence_mode_in must be 'native' or 'vertex_recovered'."
        )
    return mode


def _resolve_inner_coupled_update_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in {"off", "rim_matched_radial_continuation_v1"}:
        raise ValueError(
            "inner_coupled_update_mode must be 'off' or "
            "'rim_matched_radial_continuation_v1'."
        )
    return mode


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
    if preset == "kh_strict_refine":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=120,
                optimize_every=1,
                optimize_delta=2.0e-3,
                optimize_inner_steps=20,
            ),
            "kh_strict_refine",
        )
    if preset == "kh_strict_fast":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_fast",
        )
    if preset == "kh_strict_balanced":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=34,
                optimize_every=1,
                optimize_delta=5.0e-3,
                optimize_inner_steps=16,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_balanced",
        )
    if preset == "kh_strict_continuity":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_continuity",
        )
    if preset == "kh_strict_energy_tight":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_energy_tight",
        )
    if preset == "kh_strict_section_tight":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_section_tight",
        )
    if preset == "kh_strict_outerband_tight":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_outerband_tight",
        )
    if preset == "kh_strict_outerfield_tight":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_outerfield_tight",
        )
    if preset == "kh_strict_outerfield_quality":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_outerfield_quality",
        )
    if preset == "kh_strict_outerfield_tailmatch":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_outerfield_tailmatch",
        )
    if preset == "kh_strict_outerfield_averaged":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_outerfield_averaged",
        )
    if preset == "kh_strict_outerfield_best":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_outerfield_best",
        )
    if preset == "kh_strict_outerfield_unpinned":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_outerfield_unpinned",
        )
    if preset == "kh_strict_outertail_balanced":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_outertail_balanced",
        )
    if preset == "kh_strict_partition_tight":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_partition_tight",
        )
    if preset == "kh_strict_robust":
        return (
            BenchmarkOptimizeConfig(
                theta_initial=float(optimize_cfg.theta_initial),
                optimize_steps=30,
                optimize_every=1,
                optimize_delta=6.0e-3,
                optimize_inner_steps=14,
                plateau_patience=12,
                plateau_abs_tol=1.0e-12,
            ),
            "kh_strict_robust",
        )
    raise ValueError(
        "optimize_preset must be 'none', 'fast_r3', 'full_accuracy_r3', 'kh_wide', "
        "'kh_strict_refine', 'kh_strict_fast', 'kh_strict_balanced', "
        "'kh_strict_continuity', 'kh_strict_energy_tight', "
        "'kh_strict_section_tight', 'kh_strict_outerband_tight', "
        "'kh_strict_outerfield_tight', 'kh_strict_outerfield_quality', "
        "'kh_strict_outerfield_tailmatch', 'kh_strict_outerfield_averaged', "
        "'kh_strict_outerfield_best', 'kh_strict_outerfield_unpinned', "
        "'kh_strict_outertail_balanced', "
        "'kh_strict_partition_tight', "
        "or 'kh_strict_robust'."
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


def _refine_mesh_locally_in_outer_annulus(
    mesh,
    *,
    local_steps: int,
    r_min: float,
    r_max: float,
):
    """Refine only facets in an outer annulus, selected by centroid radius."""
    _ensure_repo_root_on_sys_path()
    from runtime.refinement import refine_triangle_mesh

    steps = int(local_steps)
    if steps <= 0:
        return mesh
    if float(r_min) < 0.0:
        raise ValueError("outer_local_refine_rmin must be >= 0.")
    if float(r_max) <= float(r_min):
        raise ValueError("outer_local_refine_rmax must be > outer_local_refine_rmin.")

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
            if float(r_min) <= r_cent <= float(r_max):
                facet.options.pop("no_refine", None)
                selected += 1
            else:
                facet.options["no_refine"] = True
        if selected == 0:
            raise AssertionError(
                "Outer annulus local refinement selected no facets. Adjust "
                "outer_local_refine_rmin_lambda/outer_local_refine_rmax_lambda."
            )
        out = refine_triangle_mesh(out)

    for facet in out.facets.values():
        facet.options.pop("no_refine", None)
    return out


def _flip_edges_locally_in_annulus(
    mesh,
    *,
    local_steps: int,
    r_min: float,
    r_max: float,
):
    """Perform optional Delaunay-like local edge flips in a radial annulus."""
    _ensure_repo_root_on_sys_path()
    from runtime.equiangulation import flip_edge_safe, should_flip_edge

    steps = int(local_steps)
    if steps <= 0:
        return mesh
    if float(r_max) <= float(r_min):
        raise ValueError("local_edge_flip_rmax must be > local_edge_flip_rmin.")

    out = mesh
    for _ in range(steps):
        out.build_connectivity_maps()
        out.build_facet_vertex_loops()
        positions = out.positions_view()
        next_edge_idx = max(out.edges.keys()) + 1 if out.edges else 1
        candidate_edges = 0
        flips = 0
        for edge_idx in list(out.edges.keys()):
            edge = out.edges.get(int(edge_idx))
            if edge is None or bool(getattr(edge, "fixed", False)):
                continue
            row0 = out.vertex_index_to_row.get(int(edge.tail_index))
            row1 = out.vertex_index_to_row.get(int(edge.head_index))
            if row0 is None or row1 is None:
                continue
            p0 = positions[int(row0)]
            p1 = positions[int(row1)]
            r_mid = float(np.linalg.norm((0.5 * (p0 + p1))[:2]))
            if r_mid < float(r_min) or r_mid > float(r_max):
                continue
            adjacent = out.get_facets_of_edge(int(edge_idx))
            if len(adjacent) != 2:
                continue
            facet1, facet2 = adjacent
            if len(facet1.edge_indices) != 3 or len(facet2.edge_indices) != 3:
                continue
            candidate_edges += 1
            if should_flip_edge(out, edge, facet1, facet2) and flip_edge_safe(
                out,
                int(edge_idx),
                facet1,
                facet2,
                int(next_edge_idx),
            ):
                next_edge_idx += 1
                flips += 1
                out.increment_topology_version()
                out.build_connectivity_maps()
                out.build_facet_vertex_loops()
        if candidate_edges == 0:
            raise AssertionError(
                "Local edge flip selected no candidate edges. Adjust "
                "local_edge_flip_rmin_lambda/local_edge_flip_rmax_lambda."
            )
        if flips == 0:
            break
    return out


def _vertex_average_locally_in_annulus(
    mesh,
    *,
    local_steps: int,
    r_min: float,
    r_max: float,
):
    """Perform optional vertex averaging in a radial annulus."""
    _ensure_repo_root_on_sys_path()
    from runtime.vertex_average import vertex_average

    steps = int(local_steps)
    if steps <= 0:
        return mesh
    if float(r_max) <= float(r_min):
        raise ValueError(
            "outer_local_vertex_average_rmax must be > outer_local_vertex_average_rmin."
        )

    out = mesh
    for _ in range(steps):
        positions = out.positions_view()
        radii = np.linalg.norm(positions[:, :2], axis=1)
        selected_rows = np.flatnonzero(
            (radii >= float(r_min)) & (radii <= float(r_max))
        )
        if selected_rows.size == 0:
            raise AssertionError(
                "Outer annulus local vertex averaging selected no rows. Adjust "
                "outer_local_vertex_average_rmin_lambda/"
                "outer_local_vertex_average_rmax_lambda."
            )
        row_to_vid = {
            int(row): int(vid) for vid, row in out.vertex_index_to_row.items()
        }
        selected_vids = {
            row_to_vid[int(row)] for row in selected_rows if int(row) in row_to_vid
        }
        if len(selected_vids) == 0:
            raise AssertionError(
                "Outer annulus local vertex averaging selected no vertices."
            )

        fixed_backup = {
            int(vid): bool(getattr(v, "fixed", False))
            for vid, v in out.vertices.items()
        }
        try:
            for vid, v in out.vertices.items():
                if int(vid) not in selected_vids:
                    v.fixed = True
            vertex_average(out)
        finally:
            for vid, was_fixed in fixed_backup.items():
                out.vertices[int(vid)].fixed = bool(was_fixed)
    return out


def _prepare_benchmark_mesh(
    *,
    fixture: Path,
    refine_level: int,
    radius: float,
    lambda_value: float,
    rim_local_refine: tuple[int, float],
    outer_local_refine: tuple[int, float, float],
    local_edge_flip: tuple[int, float, float],
    outer_local_vertex_average: tuple[int, float, float],
):
    """Load and apply the shared flat-benchmark mesh preparation sequence."""
    _ensure_repo_root_on_sys_path()
    from runtime.refinement import refine_triangle_mesh

    rim_steps, rim_band = rim_local_refine
    outer_steps, outer_rmin, outer_rmax = outer_local_refine
    flip_steps, flip_rmin, flip_rmax = local_edge_flip
    average_steps, average_rmin, average_rmax = outer_local_vertex_average
    mesh = _load_mesh_from_fixture(fixture)
    for _ in range(int(refine_level)):
        mesh = refine_triangle_mesh(mesh)
    if int(rim_steps) > 0:
        mesh = _refine_mesh_locally_near_rim(
            mesh,
            local_steps=int(rim_steps),
            rim_radius=float(radius),
            band_half_width=float(rim_band) * float(lambda_value),
        )
    if int(outer_steps) > 0:
        mesh = _refine_mesh_locally_in_outer_annulus(
            mesh,
            local_steps=int(outer_steps),
            r_min=float(radius) + float(outer_rmin) * float(lambda_value),
            r_max=float(radius) + float(outer_rmax) * float(lambda_value),
        )
    if int(flip_steps) > 0:
        mesh = _flip_edges_locally_in_annulus(
            mesh,
            local_steps=int(flip_steps),
            r_min=max(
                0.0,
                float(radius) + float(flip_rmin) * float(lambda_value),
            ),
            r_max=max(
                0.0,
                float(radius) + float(flip_rmax) * float(lambda_value),
            ),
        )
    if int(average_steps) > 0:
        mesh = _vertex_average_locally_in_annulus(
            mesh,
            local_steps=int(average_steps),
            r_min=max(
                0.0,
                float(radius) + float(average_rmin) * float(lambda_value),
            ),
            r_max=max(
                0.0,
                float(radius) + float(average_rmax) * float(lambda_value),
            ),
        )
    return mesh


def _configure_benchmark_mesh(
    mesh,
    *,
    theory_params: FlatDiskTheoryParams,
    parameterization: str,
    outer_mode: str,
    smoothness_model: str,
    splay_modulus_scale_in: float,
    tilt_mass_mode_in: str,
    tilt_mass_mode_out: str = "lumped",
    tilt_transport_model: str = "ambient_v1",
    tilt_divergence_mode_in: str = "native",
    tilt_solver: str | None = "gd",
    tilt_projection_cadence: str = "per_step",
    tilt_projection_interval: int = 1,
    tilt_post_relax_inner_steps: int = 0,
    tilt_post_relax_step_size: float = 0.0,
    tilt_post_relax_passes: int = 1,
    mesh_quality_auto_repair_enabled: bool | None = False,
    inner_coupled_update_mode: str = "off",
    benchmark_geometry_lane: str = "flat_pinned",
    curved_theta_objective_ablation_mode: str = "off",
    curved_theta_objective_ablation_inner_scale: float = 1.0,
    curved_theta_objective_ablation_outer_scale: float = 1.0,
    curved_theta_objective_ablation_contact_scale: float = 1.0,
) -> None:
    _ensure_repo_root_on_sys_path()
    from tools.diagnostics.flat_disk_one_leaflet_theory import (
        solver_mapping_from_theory,
    )

    gp = mesh.global_parameters
    if tilt_solver is not None:
        gp.set("tilt_solver", str(tilt_solver))
    if mesh_quality_auto_repair_enabled is not None:
        gp.set(
            "mesh_quality_auto_repair_enabled", bool(mesh_quality_auto_repair_enabled)
        )

    mapping = solver_mapping_from_theory(
        theory_params, parameterization=str(parameterization)
    )
    lambda_value = float(
        np.sqrt(float(theory_params.kappa) / float(theory_params.kappa_t))
    )

    gp.set("surface_tension", 0.0)
    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 0.0)
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_step_size", 0.08)
    gp.set("tilt_inner_steps", 250)
    gp.set("tilt_tol", 1e-12)
    gp.set("tilt_projection_cadence", str(tilt_projection_cadence))
    gp.set("tilt_projection_interval", int(tilt_projection_interval))
    gp.set("tilt_post_relax_inner_steps", int(tilt_post_relax_inner_steps))
    gp.set("tilt_post_relax_step_size", float(tilt_post_relax_step_size))
    gp.set("tilt_post_relax_passes", int(tilt_post_relax_passes))
    gp.set("inner_coupled_update_mode", str(inner_coupled_update_mode))
    gp.set("benchmark_geometry_lane", str(benchmark_geometry_lane))
    gp.set("benchmark_parameterization", str(parameterization))
    gp.set(
        "curved_theta_objective_ablation_mode",
        str(curved_theta_objective_ablation_mode),
    )
    gp.set(
        "curved_theta_objective_ablation_inner_scale",
        float(curved_theta_objective_ablation_inner_scale),
    )
    gp.set(
        "curved_theta_objective_ablation_outer_scale",
        float(curved_theta_objective_ablation_outer_scale),
    )
    gp.set(
        "curved_theta_objective_ablation_contact_scale",
        float(curved_theta_objective_ablation_contact_scale),
    )
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
    gp.set("tilt_mass_mode_out", str(tilt_mass_mode_out))
    gp.set("tilt_transport_model", str(tilt_transport_model))
    gp.set("tilt_divergence_mode_in", str(tilt_divergence_mode_in))
    mass_mode_out_value = str(tilt_mass_mode_out).strip().lower()
    if mass_mode_out_value not in {"lumped", "consistent"}:
        raise ValueError("tilt_mass_mode_out must be 'lumped' or 'consistent'.")
    transport_model_value = _resolve_tilt_transport_model(tilt_transport_model)
    projection_cadence = str(tilt_projection_cadence).strip().lower()
    if projection_cadence not in {"per_step", "per_pass"}:
        raise ValueError("tilt_projection_cadence must be 'per_step' or 'per_pass'.")
    projection_interval = int(tilt_projection_interval)
    if projection_interval <= 0:
        raise ValueError("tilt_projection_interval must be >= 1.")
    post_relax_inner_steps = int(tilt_post_relax_inner_steps)
    if post_relax_inner_steps < 0:
        raise ValueError("tilt_post_relax_inner_steps must be >= 0.")
    post_relax_step_size = float(tilt_post_relax_step_size)
    if post_relax_step_size < 0.0:
        raise ValueError("tilt_post_relax_step_size must be >= 0.")
    post_relax_passes = int(tilt_post_relax_passes)
    if post_relax_passes <= 0:
        raise ValueError("tilt_post_relax_passes must be >= 1.")
    inner_coupled_update_mode_value = _resolve_inner_coupled_update_mode(
        inner_coupled_update_mode
    )
    ablation_mode_value = str(curved_theta_objective_ablation_mode).strip().lower()
    if ablation_mode_value not in {"off", "inner_outer_rescaled"}:
        raise ValueError(
            "curved_theta_objective_ablation_mode must be 'off' or "
            "'inner_outer_rescaled'."
        )
    ablation_inner_scale = float(curved_theta_objective_ablation_inner_scale)
    ablation_outer_scale = float(curved_theta_objective_ablation_outer_scale)
    ablation_contact_scale = float(curved_theta_objective_ablation_contact_scale)
    if (
        ablation_inner_scale <= 0.0
        or ablation_outer_scale <= 0.0
        or ablation_contact_scale <= 0.0
    ):
        raise ValueError("curved theta objective ablation scales must be > 0.")
    gp.set("tilt_projection_cadence", projection_cadence)
    gp.set("tilt_projection_interval", projection_interval)
    gp.set("tilt_projection_loss_radius", float(theory_params.radius))
    gp.set("tilt_projection_loss_lambda", float(lambda_value))
    gp.set("tilt_projection_loss_outer_near_width_lambda", 4.0)
    gp.set("tilt_post_relax_inner_steps", post_relax_inner_steps)
    gp.set("tilt_post_relax_step_size", post_relax_step_size)
    gp.set("tilt_post_relax_passes", post_relax_passes)
    gp.set("inner_coupled_update_mode", inner_coupled_update_mode_value)
    gp.set("curved_theta_objective_ablation_mode", ablation_mode_value)
    gp.set("curved_theta_objective_ablation_inner_scale", ablation_inner_scale)
    gp.set("curved_theta_objective_ablation_outer_scale", ablation_outer_scale)
    gp.set("curved_theta_objective_ablation_contact_scale", ablation_contact_scale)
    gp.set("tilt_mass_mode_out", mass_mode_out_value)
    gp.set("tilt_transport_model", transport_model_value)
    gp.set("benchmark_disk_radius", float(theory_params.radius))
    gp.set("benchmark_lambda_value", float(lambda_value))
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
    reset_inner: bool = True,
) -> float:
    """Relax leaflet tilts for a fixed theta_B, with optional warm-started inner repeats."""
    mesh = minim.mesh
    gp = mesh.global_parameters
    gp.set("tilt_thetaB_value", float(theta_value))

    if bool(reset_inner):
        tin = np.zeros_like(mesh.tilts_in_view())
        mesh.set_tilts_in_from_array(tin)
    if reset_outer:
        tout = np.zeros_like(mesh.tilts_out_view())
        mesh.set_tilts_out_from_array(tout)

    positions = mesh.positions_view()
    minim._relax_leaflet_tilts(positions=positions, mode="coupled")
    post_relax_inner_steps = int(gp.get("tilt_post_relax_inner_steps", 0) or 0)
    post_relax_step_size = float(gp.get("tilt_post_relax_step_size", 0.0) or 0.0)
    post_relax_passes = int(gp.get("tilt_post_relax_passes", 1) or 1)
    if (
        post_relax_inner_steps > 0
        and post_relax_step_size > 0.0
        and post_relax_passes > 0
    ):
        orig_step = gp.get("tilt_step_size")
        orig_inner = gp.get("tilt_inner_steps")
        try:
            gp.set("tilt_step_size", float(post_relax_step_size))
            gp.set("tilt_inner_steps", int(post_relax_inner_steps))
            for _ in range(int(post_relax_passes)):
                minim._relax_leaflet_tilts(positions=positions, mode="coupled")
        finally:
            if orig_step is None:
                gp.unset("tilt_step_size")
            else:
                gp.set("tilt_step_size", orig_step)
            if orig_inner is None:
                gp.unset("tilt_inner_steps")
            else:
                gp.set("tilt_inner_steps", orig_inner)
    energy = float(minim.compute_energy())
    if not np.isfinite(energy):
        raise ValueError(f"Non-finite energy during theta scan at theta={theta_value}.")
    return energy


def _run_theta_optimize_detailed(
    minim: Minimizer,
    *,
    optimize_cfg: BenchmarkOptimizeConfig,
    reset_outer: bool,
) -> tuple[float, int, bool]:
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
    iterations_completed = 0
    plateau_counter = 0
    stopped_on_plateau = False
    prev_theta = float(gp.get("tilt_thetaB_value") or 0.0)
    for i in range(int(optimize_cfg.optimize_steps)):
        minim._relax_leaflet_tilts(positions=positions, mode=tilt_mode)
        minim._optimize_thetaB_scalar(tilt_mode=tilt_mode, iteration=i)
        iterations_completed += 1
        theta_now = float(gp.get("tilt_thetaB_value") or 0.0)
        if int(optimize_cfg.plateau_patience) > 0:
            if abs(theta_now - prev_theta) <= float(optimize_cfg.plateau_abs_tol):
                plateau_counter += 1
            else:
                plateau_counter = 0
            if plateau_counter >= int(optimize_cfg.plateau_patience):
                stopped_on_plateau = True
                break
        prev_theta = theta_now

    theta_opt = float(gp.get("tilt_thetaB_value") or 0.0)
    if not np.isfinite(theta_opt):
        raise ValueError("Non-finite optimized theta_B value.")
    return theta_opt, int(iterations_completed), bool(stopped_on_plateau)


def _run_theta_optimize(
    minim: Minimizer,
    *,
    optimize_cfg: BenchmarkOptimizeConfig,
    reset_outer: bool,
) -> float:
    """Backward-compatible optimize helper returning theta_B as a scalar."""
    theta_opt, _, _ = _run_theta_optimize_detailed(
        minim,
        optimize_cfg=optimize_cfg,
        reset_outer=reset_outer,
    )
    return float(theta_opt)


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


def _capture_mesh_state(mesh) -> dict[str, np.ndarray]:
    """Capture dense position and leaflet tilt arrays for local diagnostics."""
    return {
        "positions": np.array(mesh.positions_view(), copy=True),
        "tilts_in": np.array(mesh.tilts_in_view(), copy=True),
        "tilts_out": np.array(mesh.tilts_out_view(), copy=True),
    }


def _restore_mesh_state(mesh, snapshot: dict[str, np.ndarray]) -> None:
    """Restore dense position and leaflet tilt arrays from a snapshot."""
    positions = np.asarray(snapshot["positions"], dtype=float)
    for row, vid in enumerate(mesh.vertex_ids):
        mesh.vertices[int(vid)].position[:] = positions[row]
    mesh.increment_version()
    mesh.set_tilts_in_from_array(snapshot["tilts_in"])
    mesh.set_tilts_out_from_array(snapshot["tilts_out"])


def _theta_objective_decomposition_from_breakdown(
    breakdown: dict[str, float], *, theta_value: float
) -> dict[str, Any]:
    """Summarize selected-theta objective channels from an energy breakdown."""
    inner_like = float(
        breakdown.get("bending_tilt_in", 0.0) + breakdown.get("tilt_in", 0.0)
    )
    outer_like = float(
        breakdown.get("bending_tilt_out", 0.0)
        + breakdown.get("tilt_out", 0.0)
        + breakdown.get("tilt_smoothness_out", 0.0)
        + breakdown.get("tilt_rim_source_out", 0.0)
        + breakdown.get("tilt_disk_target_out", 0.0)
    )
    contact_like = float(breakdown.get("tilt_thetaB_contact_in", 0.0))
    collapse_eps = max(1.0e-8, 0.02 * max(abs(inner_like), 1.0e-12))
    return {
        "selected_theta": float(theta_value),
        "inner_like": float(inner_like),
        "outer_like": float(outer_like),
        "contact_like": float(contact_like),
        "outer_to_inner_abs_ratio": float(
            abs(outer_like) / max(abs(inner_like), 1.0e-18)
        ),
        "outer_channel_collapse_eps": float(collapse_eps),
        "outer_channel_collapsed": bool(abs(outer_like) <= collapse_eps),
    }


def _curved_theta_objective_ablation_from_probe(
    *,
    probe: dict[str, Any] | None,
    theory,
    mode: str,
    inner_scale: float,
    outer_scale: float,
    contact_scale: float,
) -> dict[str, Any]:
    """Project ablation response from fitted theta-objective coefficients."""
    mode_value = str(mode).strip().lower()
    out: dict[str, Any] = {
        "available": False,
        "applied": False,
        "mode": mode_value,
        "inner_scale": float(inner_scale),
        "outer_scale": float(outer_scale),
        "contact_scale": float(contact_scale),
        "theta_star_pred": float("nan"),
        "total_energy_pred": float("nan"),
        "theta_factor_pred": float("nan"),
        "energy_factor_pred": float("nan"),
        "reason": None,
    }
    if mode_value == "off":
        out["reason"] = "disabled"
        return out
    if probe is None:
        out["reason"] = "missing_theta_objective_probe"
        return out

    coeff_a_inner = float(probe.get("coeff_a_inner", float("nan")))
    coeff_a_outer = float(probe.get("coeff_a_outer", float("nan")))
    coeff_b_contact = float(probe.get("coeff_b_contact", float("nan")))
    if not (
        np.isfinite(coeff_a_inner)
        and np.isfinite(coeff_a_outer)
        and np.isfinite(coeff_b_contact)
    ):
        out["reason"] = "non_finite_probe_coefficients"
        return out

    coeff_a_eff = (
        float(inner_scale) * coeff_a_inner + float(outer_scale) * coeff_a_outer
    )
    coeff_b_eff = float(contact_scale) * coeff_b_contact
    out["available"] = True
    out["coeff_a_inner_raw"] = float(coeff_a_inner)
    out["coeff_a_outer_raw"] = float(coeff_a_outer)
    out["coeff_b_contact_raw"] = float(coeff_b_contact)
    out["coeff_a_effective"] = float(coeff_a_eff)
    out["coeff_b_effective"] = float(coeff_b_eff)
    if coeff_a_eff <= 1.0e-18:
        out["reason"] = "degenerate_effective_quadratic"
        return out

    theta_pred = float(coeff_b_eff / (2.0 * coeff_a_eff))
    total_pred = float(-(coeff_b_eff * coeff_b_eff) / (4.0 * coeff_a_eff))
    out["theta_star_pred"] = float(theta_pred)
    out["total_energy_pred"] = float(total_pred)
    out["theta_factor_pred"] = float(
        _factor_difference(theta_pred, float(theory.theta_star))
    )
    out["energy_factor_pred"] = float(
        _factor_difference(float(abs(total_pred)), float(abs(theory.total)))
    )
    out["applied"] = True
    out["reason"] = "ok"
    return out


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
    theta_optimize_mode: str = "scalar_local",
    theta_optimize_plateau_patience: int = 0,
    theta_optimize_plateau_abs_tol: float = 0.0,
    theta_optimize_postcheck: bool = False,
    theta_optimize_parity_polish: bool = False,
    theta_optimize_energy_polish: bool = False,
    theta_optimize_section_polish: bool = False,
    theta_polish_delta: float = 1.0e-4,
    theta_polish_points: int = 3,
    optimize_preset: str = "none",
    parameterization: str = "legacy",
    geometry_lane: str = "flat_pinned",
    z_gauge: str = "mean_zero",
    curved_acceptance_profile: str = "full",
    length_scale_nm: float = 15.0,
    radius_nm: float = 7.0,
    kappa_physical: float = 10.0,
    kappa_t_physical: float = 10.0,
    drive_physical: float = (2.0 / 0.7),
    splay_modulus_scale_in: float = 1.0,
    tilt_mass_mode_in: str = "auto",
    tilt_mass_mode_out: str = "auto",
    tilt_transport_model: str = "ambient_v1",
    tilt_divergence_mode_in: str = "native",
    rim_local_refine_steps: int = 0,
    rim_local_refine_band_lambda: float = 0.0,
    outer_local_refine_steps: int = 0,
    outer_local_refine_rmin_lambda: float = 0.0,
    outer_local_refine_rmax_lambda: float = 0.0,
    local_edge_flip_steps: int = 0,
    local_edge_flip_rmin_lambda: float = -1.0,
    local_edge_flip_rmax_lambda: float = 4.0,
    outer_local_vertex_average_steps: int = 0,
    outer_local_vertex_average_rmin_lambda: float = 0.0,
    outer_local_vertex_average_rmax_lambda: float = 0.0,
    tilt_solver: str | None = "gd",
    tilt_projection_cadence: str = "per_step",
    tilt_projection_interval: int = 1,
    tilt_post_relax_inner_steps: int = 0,
    tilt_post_relax_step_size: float = 0.0,
    tilt_post_relax_passes: int = 1,
    mesh_quality_auto_repair_enabled: bool | None = False,
    inner_coupled_update_mode: str = "off",
    curved_theta_calibration_mode: str = "off",
    curved_theta_calibration_inner_scale: float = 1.0,
    curved_theta_calibration_outer_scale: float = 1.0,
    curved_theta_calibration_contact_scale: float = 1.0,
    curved_theta_objective_ablation_mode: str = "off",
    curved_theta_objective_ablation_inner_scale: float = 1.0,
    curved_theta_objective_ablation_outer_scale: float = 1.0,
    curved_theta_objective_ablation_contact_scale: float = 1.0,
    theory_params: FlatDiskTheoryParams | None = None,
) -> dict[str, Any]:
    """Run the flat one-leaflet benchmark and return a report dict."""
    t_run_start = perf_counter()
    _ensure_repo_root_on_sys_path()
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

    raw_refine_level = int(refine_level)
    raw_rim_local_refine_steps = int(rim_local_refine_steps)
    raw_rim_local_refine_band_lambda = float(rim_local_refine_band_lambda)
    raw_outer_local_refine_steps = int(outer_local_refine_steps)
    raw_outer_local_refine_rmin_lambda = float(outer_local_refine_rmin_lambda)
    raw_outer_local_refine_rmax_lambda = float(outer_local_refine_rmax_lambda)
    raw_local_edge_flip_steps = int(local_edge_flip_steps)
    raw_local_edge_flip_rmin_lambda = float(local_edge_flip_rmin_lambda)
    raw_local_edge_flip_rmax_lambda = float(local_edge_flip_rmax_lambda)
    raw_outer_local_vertex_average_steps = int(outer_local_vertex_average_steps)
    raw_outer_local_vertex_average_rmin_lambda = float(
        outer_local_vertex_average_rmin_lambda
    )
    raw_outer_local_vertex_average_rmax_lambda = float(
        outer_local_vertex_average_rmax_lambda
    )
    geometry_lane_requested = str(geometry_lane).strip().lower()
    z_gauge_requested = str(z_gauge).strip().lower()
    theta_optimize_mode_requested = str(theta_optimize_mode).strip().lower()
    curved_acceptance_profile_requested = str(curved_acceptance_profile).strip().lower()
    curved_theta_calibration_mode_requested = (
        str(curved_theta_calibration_mode).strip().lower()
    )
    curved_theta_objective_ablation_mode_requested = (
        str(curved_theta_objective_ablation_mode).strip().lower()
    )
    geometry_lane_effective = "flat_pinned"
    z_gauge_effective = "none"
    geometry_lane_fallback_reason = (
        None
        if geometry_lane_requested == "flat_pinned"
        else "geometry_lane_not_supported_in_benchmark"
    )
    theta_optimize_mode_effective = "scalar_local"
    theta_optimize_mode_fallback_reason = (
        None
        if theta_optimize_mode_requested == "scalar_local"
        else "theta_optimize_mode_not_supported_in_benchmark"
    )
    curved_theta_calibration_requested = bool(
        curved_theta_calibration_mode_requested != "off"
    )
    curved_theta_objective_ablation_requested = bool(
        curved_theta_objective_ablation_mode_requested != "off"
    )
    optimize_preset_raw = str(optimize_preset).lower()
    effective_refine_level = raw_refine_level
    effective_rim_local_refine_steps = raw_rim_local_refine_steps
    effective_rim_local_refine_band_lambda = raw_rim_local_refine_band_lambda
    effective_outer_local_refine_steps = raw_outer_local_refine_steps
    effective_outer_local_refine_rmin_lambda = raw_outer_local_refine_rmin_lambda
    effective_outer_local_refine_rmax_lambda = raw_outer_local_refine_rmax_lambda
    effective_local_edge_flip_steps = raw_local_edge_flip_steps
    effective_local_edge_flip_rmin_lambda = raw_local_edge_flip_rmin_lambda
    effective_local_edge_flip_rmax_lambda = raw_local_edge_flip_rmax_lambda
    effective_outer_local_vertex_average_steps = raw_outer_local_vertex_average_steps
    effective_outer_local_vertex_average_rmin_lambda = (
        raw_outer_local_vertex_average_rmin_lambda
    )
    effective_outer_local_vertex_average_rmax_lambda = (
        raw_outer_local_vertex_average_rmax_lambda
    )
    if optimize_preset_raw in {
        "kh_strict_refine",
        "kh_strict_fast",
        "kh_strict_balanced",
        "kh_strict_continuity",
        "kh_strict_energy_tight",
        "kh_strict_section_tight",
        "kh_strict_outerband_tight",
        "kh_strict_outerfield_tight",
        "kh_strict_outerfield_quality",
        "kh_strict_outerfield_tailmatch",
        "kh_strict_outerfield_averaged",
        "kh_strict_outerfield_best",
        "kh_strict_outerfield_unpinned",
        "kh_strict_outertail_balanced",
        "kh_strict_partition_tight",
        "kh_strict_robust",
    }:
        if optimize_preset_raw == "kh_strict_outerfield_unpinned":
            effective_refine_level = int(raw_refine_level)
        else:
            effective_refine_level = (
                2
                if optimize_preset_raw
                in {
                    "kh_strict_section_tight",
                    "kh_strict_outerband_tight",
                    "kh_strict_outerfield_tight",
                    "kh_strict_outerfield_quality",
                    "kh_strict_outerfield_tailmatch",
                    "kh_strict_outerfield_averaged",
                    "kh_strict_outerfield_best",
                    "kh_strict_outertail_balanced",
                }
                else 1
            )
        if int(raw_rim_local_refine_steps) > 0:
            effective_rim_local_refine_steps = int(raw_rim_local_refine_steps)
        else:
            effective_rim_local_refine_steps = (
                2
                if optimize_preset_raw
                in {
                    "kh_strict_balanced",
                    "kh_strict_continuity",
                    "kh_strict_energy_tight",
                    "kh_strict_partition_tight",
                }
                else 1
            )
        if float(raw_rim_local_refine_band_lambda) > 0.0:
            effective_rim_local_refine_band_lambda = float(
                raw_rim_local_refine_band_lambda
            )
        else:
            if optimize_preset_raw == "kh_strict_balanced":
                effective_rim_local_refine_band_lambda = 3.0
            elif optimize_preset_raw == "kh_strict_energy_tight":
                effective_rim_local_refine_band_lambda = 8.0
            elif optimize_preset_raw == "kh_strict_section_tight":
                effective_rim_local_refine_band_lambda = 4.0
            elif optimize_preset_raw == "kh_strict_outerband_tight":
                effective_rim_local_refine_band_lambda = 3.0
            elif optimize_preset_raw == "kh_strict_outerfield_tight":
                effective_rim_local_refine_band_lambda = 3.0
            elif optimize_preset_raw == "kh_strict_outerfield_quality":
                effective_rim_local_refine_band_lambda = 3.0
            elif optimize_preset_raw == "kh_strict_outerfield_tailmatch":
                effective_rim_local_refine_band_lambda = 3.0
            elif optimize_preset_raw == "kh_strict_outerfield_averaged":
                effective_rim_local_refine_band_lambda = 3.0
            elif optimize_preset_raw == "kh_strict_outerfield_best":
                effective_rim_local_refine_band_lambda = 3.0
            elif optimize_preset_raw == "kh_strict_outertail_balanced":
                effective_rim_local_refine_band_lambda = 3.0
            elif optimize_preset_raw == "kh_strict_partition_tight":
                effective_rim_local_refine_band_lambda = 10.0
            else:
                effective_rim_local_refine_band_lambda = 4.0
        if int(raw_outer_local_refine_steps) > 0:
            effective_outer_local_refine_steps = int(raw_outer_local_refine_steps)
        elif optimize_preset_raw in {
            "kh_strict_outerfield_tight",
            "kh_strict_outerfield_quality",
            "kh_strict_outerfield_tailmatch",
            "kh_strict_outerfield_averaged",
            "kh_strict_outerfield_best",
            "kh_strict_outerfield_unpinned",
            "kh_strict_outertail_balanced",
        }:
            effective_outer_local_refine_steps = 1
        else:
            effective_outer_local_refine_steps = 0
        if float(raw_outer_local_refine_rmin_lambda) > 0.0:
            effective_outer_local_refine_rmin_lambda = float(
                raw_outer_local_refine_rmin_lambda
            )
        elif optimize_preset_raw in {
            "kh_strict_outerfield_tight",
            "kh_strict_outerfield_quality",
            "kh_strict_outerfield_tailmatch",
            "kh_strict_outerfield_averaged",
            "kh_strict_outerfield_best",
            "kh_strict_outerfield_unpinned",
            "kh_strict_outertail_balanced",
        }:
            effective_outer_local_refine_rmin_lambda = 1.0
        else:
            effective_outer_local_refine_rmin_lambda = 0.0
        if float(raw_outer_local_refine_rmax_lambda) > 0.0:
            effective_outer_local_refine_rmax_lambda = float(
                raw_outer_local_refine_rmax_lambda
            )
        elif optimize_preset_raw == "kh_strict_outerfield_tight":
            effective_outer_local_refine_rmax_lambda = 8.0
        elif optimize_preset_raw == "kh_strict_outerfield_quality":
            effective_outer_local_refine_rmax_lambda = 8.0
        elif optimize_preset_raw == "kh_strict_outerfield_tailmatch":
            effective_outer_local_refine_rmax_lambda = 8.0
        elif optimize_preset_raw == "kh_strict_outerfield_averaged":
            effective_outer_local_refine_rmax_lambda = 8.0
        elif optimize_preset_raw == "kh_strict_outerfield_best":
            effective_outer_local_refine_rmax_lambda = 8.0
        elif optimize_preset_raw == "kh_strict_outerfield_unpinned":
            effective_outer_local_refine_rmax_lambda = 8.0
        elif optimize_preset_raw == "kh_strict_outertail_balanced":
            effective_outer_local_refine_rmax_lambda = 10.0
        else:
            effective_outer_local_refine_rmax_lambda = 0.0
        if int(raw_local_edge_flip_steps) > 0:
            effective_local_edge_flip_steps = int(raw_local_edge_flip_steps)
            if float(raw_local_edge_flip_rmin_lambda) >= 0.0:
                effective_local_edge_flip_rmin_lambda = float(
                    raw_local_edge_flip_rmin_lambda
                )
            if float(raw_local_edge_flip_rmax_lambda) > 0.0:
                effective_local_edge_flip_rmax_lambda = float(
                    raw_local_edge_flip_rmax_lambda
                )
        elif optimize_preset_raw == "kh_strict_outerfield_quality":
            effective_local_edge_flip_steps = 1
            effective_local_edge_flip_rmin_lambda = 2.0
            effective_local_edge_flip_rmax_lambda = 6.0
        elif optimize_preset_raw == "kh_strict_outerfield_tailmatch":
            effective_local_edge_flip_steps = 1
            effective_local_edge_flip_rmin_lambda = 0.5
            effective_local_edge_flip_rmax_lambda = 8.0
        elif optimize_preset_raw == "kh_strict_outerfield_averaged":
            effective_local_edge_flip_steps = 0
        elif optimize_preset_raw == "kh_strict_outerfield_best":
            effective_local_edge_flip_steps = 0
        elif optimize_preset_raw == "kh_strict_outerfield_unpinned":
            effective_local_edge_flip_steps = 0
        else:
            effective_local_edge_flip_steps = 0
        if int(raw_outer_local_vertex_average_steps) > 0:
            effective_outer_local_vertex_average_steps = int(
                raw_outer_local_vertex_average_steps
            )
            if float(raw_outer_local_vertex_average_rmin_lambda) > 0.0:
                effective_outer_local_vertex_average_rmin_lambda = float(
                    raw_outer_local_vertex_average_rmin_lambda
                )
            if float(raw_outer_local_vertex_average_rmax_lambda) > 0.0:
                effective_outer_local_vertex_average_rmax_lambda = float(
                    raw_outer_local_vertex_average_rmax_lambda
                )
        elif optimize_preset_raw == "kh_strict_outerfield_averaged":
            effective_outer_local_vertex_average_steps = 2
            effective_outer_local_vertex_average_rmin_lambda = 4.0
            effective_outer_local_vertex_average_rmax_lambda = 12.0
        elif optimize_preset_raw == "kh_strict_outerfield_best":
            effective_outer_local_vertex_average_steps = 2
            effective_outer_local_vertex_average_rmin_lambda = 4.0
            effective_outer_local_vertex_average_rmax_lambda = 12.0
        elif optimize_preset_raw == "kh_strict_outerfield_unpinned":
            effective_outer_local_vertex_average_steps = 2
            effective_outer_local_vertex_average_rmin_lambda = 4.0
            effective_outer_local_vertex_average_rmax_lambda = 12.0
        else:
            effective_outer_local_vertex_average_steps = 0

    if int(effective_refine_level) < 0:
        raise ValueError("refine_level must be >= 0.")
    if int(effective_rim_local_refine_steps) < 0:
        raise ValueError("rim_local_refine_steps must be >= 0.")
    if float(effective_rim_local_refine_band_lambda) < 0.0:
        raise ValueError("rim_local_refine_band_lambda must be >= 0.")
    if int(effective_outer_local_refine_steps) < 0:
        raise ValueError("outer_local_refine_steps must be >= 0.")
    if float(effective_outer_local_refine_rmin_lambda) < 0.0:
        raise ValueError("outer_local_refine_rmin_lambda must be >= 0.")
    if float(effective_outer_local_refine_rmax_lambda) < 0.0:
        raise ValueError("outer_local_refine_rmax_lambda must be >= 0.")
    if int(effective_outer_local_refine_steps) > 0 and (
        float(effective_outer_local_refine_rmax_lambda)
        <= float(effective_outer_local_refine_rmin_lambda)
    ):
        raise ValueError(
            "outer_local_refine_rmax_lambda must be > "
            "outer_local_refine_rmin_lambda when outer_local_refine_steps > 0."
        )
    if int(effective_local_edge_flip_steps) < 0:
        raise ValueError("local_edge_flip_steps must be >= 0.")
    if int(effective_local_edge_flip_steps) > 0 and (
        float(effective_local_edge_flip_rmax_lambda)
        <= float(effective_local_edge_flip_rmin_lambda)
    ):
        raise ValueError(
            "local_edge_flip_rmax_lambda must be > "
            "local_edge_flip_rmin_lambda when local_edge_flip_steps > 0."
        )
    if int(effective_outer_local_vertex_average_steps) < 0:
        raise ValueError("outer_local_vertex_average_steps must be >= 0.")
    if float(effective_outer_local_vertex_average_rmin_lambda) < 0.0:
        raise ValueError("outer_local_vertex_average_rmin_lambda must be >= 0.")
    if float(effective_outer_local_vertex_average_rmax_lambda) < 0.0:
        raise ValueError("outer_local_vertex_average_rmax_lambda must be >= 0.")
    if int(effective_outer_local_vertex_average_steps) > 0 and (
        float(effective_outer_local_vertex_average_rmax_lambda)
        <= float(effective_outer_local_vertex_average_rmin_lambda)
    ):
        raise ValueError(
            "outer_local_vertex_average_rmax_lambda must be > "
            "outer_local_vertex_average_rmin_lambda when "
            "outer_local_vertex_average_steps > 0."
        )
    if float(splay_modulus_scale_in) <= 0.0:
        raise ValueError("splay_modulus_scale_in must be > 0.")
    mode = str(parameterization).lower()
    if mode not in {"legacy", "kh_physical"}:
        raise ValueError("parameterization must be 'legacy' or 'kh_physical'.")
    mass_mode = _resolve_tilt_mass_mode(
        tilt_mass_mode_in,
        leaflet="in",
        auto_default="consistent" if mode == "kh_physical" else "lumped",
    )
    mass_mode_out = _resolve_tilt_mass_mode(
        tilt_mass_mode_out, leaflet="out", auto_default="lumped"
    )
    transport_model_raw = _resolve_tilt_transport_model(tilt_transport_model)
    div_mode_raw = _resolve_tilt_divergence_mode_in(tilt_divergence_mode_in)
    projection_cadence = str(tilt_projection_cadence).strip().lower()
    if projection_cadence not in {"per_step", "per_pass"}:
        raise ValueError("tilt_projection_cadence must be 'per_step' or 'per_pass'.")
    projection_interval = int(tilt_projection_interval)
    if projection_interval < 1:
        raise ValueError("tilt_projection_interval must be >= 1.")
    post_relax_inner_steps = int(tilt_post_relax_inner_steps)
    if post_relax_inner_steps < 0:
        raise ValueError("tilt_post_relax_inner_steps must be >= 0.")
    post_relax_step_size = float(tilt_post_relax_step_size)
    if post_relax_step_size < 0.0:
        raise ValueError("tilt_post_relax_step_size must be >= 0.")
    post_relax_passes = int(tilt_post_relax_passes)
    if post_relax_passes < 1:
        raise ValueError("tilt_post_relax_passes must be >= 1.")
    inner_coupled_update_mode_value = _resolve_inner_coupled_update_mode(
        inner_coupled_update_mode
    )
    if curved_theta_objective_ablation_mode_requested not in {
        "off",
        "inner_outer_rescaled",
    }:
        raise ValueError(
            "curved_theta_objective_ablation_mode must be 'off' or "
            "'inner_outer_rescaled'."
        )
    ablation_inner_scale = float(curved_theta_objective_ablation_inner_scale)
    ablation_outer_scale = float(curved_theta_objective_ablation_outer_scale)
    ablation_contact_scale = float(curved_theta_objective_ablation_contact_scale)
    if (
        ablation_inner_scale <= 0.0
        or ablation_outer_scale <= 0.0
        or ablation_contact_scale <= 0.0
    ):
        raise ValueError("curved theta objective ablation scales must be > 0.")

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
        theory_source = "kh_physical_radial_integral"
    theta_mode_str = str(theta_mode).lower()
    if theta_mode_str not in {"scan", "optimize", "optimize_full"}:
        raise ValueError("theta_mode must be 'scan', 'optimize', or 'optimize_full'.")

    scan_cfg = None
    optimize_cfg = None
    polish_cfg = None
    effective_optimize_preset = "none"
    postcheck_enabled = bool(theta_optimize_postcheck)
    parity_polish_enabled = bool(theta_optimize_parity_polish)
    energy_polish_enabled = bool(theta_optimize_energy_polish)
    section_polish_enabled = bool(theta_optimize_section_polish)
    if bool(section_polish_enabled):
        parity_polish_enabled = True
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
            plateau_patience=int(theta_optimize_plateau_patience),
            plateau_abs_tol=float(theta_optimize_plateau_abs_tol),
        )
        optimize_cfg.validate()
        optimize_cfg, effective_optimize_preset = _resolve_optimize_preset(
            optimize_preset=str(optimize_preset),
            refine_level=int(effective_refine_level),
            optimize_cfg=optimize_cfg,
        )
        optimize_cfg.validate()
        if effective_optimize_preset in {
            "kh_strict_balanced",
            "kh_strict_energy_tight",
            "kh_strict_section_tight",
            "kh_strict_outerband_tight",
            "kh_strict_outerfield_tight",
            "kh_strict_outerfield_quality",
            "kh_strict_outerfield_tailmatch",
            "kh_strict_outerfield_averaged",
            "kh_strict_outerfield_best",
            "kh_strict_outerfield_unpinned",
            "kh_strict_outertail_balanced",
            "kh_strict_partition_tight",
        }:
            parity_polish_enabled = True
        if effective_optimize_preset in {
            "kh_strict_energy_tight",
            "kh_strict_section_tight",
            "kh_strict_outerband_tight",
            "kh_strict_outerfield_tight",
            "kh_strict_outerfield_quality",
            "kh_strict_outerfield_tailmatch",
            "kh_strict_outerfield_averaged",
            "kh_strict_outerfield_best",
            "kh_strict_outerfield_unpinned",
            "kh_strict_outertail_balanced",
        }:
            energy_polish_enabled = True
        if effective_optimize_preset == "kh_strict_robust":
            postcheck_enabled = True
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

    t_mesh_prep_start = perf_counter()
    mesh = _prepare_benchmark_mesh(
        fixture=fixture_path,
        refine_level=int(effective_refine_level),
        radius=float(theory.radius),
        lambda_value=float(theory.lambda_value),
        rim_local_refine=(
            effective_rim_local_refine_steps,
            effective_rim_local_refine_band_lambda,
        ),
        outer_local_refine=(
            effective_outer_local_refine_steps,
            effective_outer_local_refine_rmin_lambda,
            effective_outer_local_refine_rmax_lambda,
        ),
        local_edge_flip=(
            effective_local_edge_flip_steps,
            effective_local_edge_flip_rmin_lambda,
            effective_local_edge_flip_rmax_lambda,
        ),
        outer_local_vertex_average=(
            effective_outer_local_vertex_average_steps,
            effective_outer_local_vertex_average_rmin_lambda,
            effective_outer_local_vertex_average_rmax_lambda,
        ),
    )
    mesh_load_refine_seconds = float(perf_counter() - t_mesh_prep_start)

    t_setup_start = perf_counter()
    _configure_benchmark_mesh(
        mesh,
        theory_params=params,
        parameterization=mode,
        outer_mode=outer_mode,
        smoothness_model=smoothness_model,
        splay_modulus_scale_in=float(splay_modulus_scale_in),
        tilt_mass_mode_in=mass_mode,
        tilt_mass_mode_out=mass_mode_out,
        tilt_transport_model=transport_model_raw,
        tilt_divergence_mode_in=div_mode_raw,
        tilt_solver=tilt_solver,
        tilt_projection_cadence=projection_cadence,
        tilt_projection_interval=projection_interval,
        tilt_post_relax_inner_steps=post_relax_inner_steps,
        tilt_post_relax_step_size=post_relax_step_size,
        tilt_post_relax_passes=post_relax_passes,
        mesh_quality_auto_repair_enabled=mesh_quality_auto_repair_enabled,
        inner_coupled_update_mode=inner_coupled_update_mode_value,
        benchmark_geometry_lane=geometry_lane_requested,
        curved_theta_objective_ablation_mode=(
            curved_theta_objective_ablation_mode_requested
        ),
        curved_theta_objective_ablation_inner_scale=ablation_inner_scale,
        curved_theta_objective_ablation_outer_scale=ablation_outer_scale,
        curved_theta_objective_ablation_contact_scale=ablation_contact_scale,
    )
    _collect_disk_boundary_rows(mesh, group="disk")

    minim = _build_minimizer(mesh)
    minim.enforce_constraints_after_mesh_ops(mesh)
    mesh.project_tilts_to_tangent()
    tilt_in_mean_initial = float(np.mean(np.linalg.norm(mesh.tilts_in_view(), axis=1)))
    tilt_out_mean_initial = float(
        np.mean(np.linalg.norm(mesh.tilts_out_view(), axis=1))
    )
    setup_seconds = float(perf_counter() - t_setup_start)

    scan_report: dict[str, Any] | None = None
    optimize_report: dict[str, Any] | None = None
    theta_factor_raw: float | None = None
    energy_factor_raw: float | None = None
    theta_optimize_seconds = 0.0
    theta_evaluations = 0
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
        theta_evaluations = int(theta_values.size)

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
        theta_opt_raw, optimize_iterations_completed, stopped_on_plateau = (
            _run_theta_optimize_detailed(
                minim,
                optimize_cfg=optimize_cfg,
                reset_outer=True,
            )
        )
        theta_star = float(theta_opt_raw)
        polish_report = None
        postcheck_report = None
        parity_polish_report = None
        postcheck_evaluations = 0
        parity_polish_evaluations = 0
        if theta_mode_str == "optimize" and bool(postcheck_enabled):
            delta = float(optimize_cfg.optimize_delta)
            theta_candidates = np.asarray(
                [theta_opt_raw - delta, theta_opt_raw, theta_opt_raw + delta],
                dtype=float,
            )
            energy_candidates = np.asarray(
                [
                    _run_theta_relaxation(
                        minim, theta_value=float(th), reset_outer=True
                    )
                    for th in theta_candidates.tolist()
                ],
                dtype=float,
            )
            best_idx = int(np.argmin(energy_candidates))
            theta_star = float(theta_candidates[best_idx])
            postcheck_evaluations = int(theta_candidates.size)
            postcheck_report = {
                "enabled": True,
                "theta_candidates": [float(x) for x in theta_candidates.tolist()],
                "energy_candidates": [float(x) for x in energy_candidates.tolist()],
                "selected_theta": float(theta_star),
            }
        if theta_mode_str == "optimize" and bool(parity_polish_enabled):
            delta = 0.5 * float(optimize_cfg.optimize_delta)
            theta_candidates = np.asarray(
                [theta_star - delta, theta_star, theta_star + delta],
                dtype=float,
            )
            energy_candidates = np.asarray(
                [
                    _run_theta_relaxation(
                        minim, theta_value=float(th), reset_outer=True
                    )
                    for th in theta_candidates.tolist()
                ],
                dtype=float,
            )
            score_candidates = np.asarray(
                [
                    np.hypot(
                        np.log(
                            max(
                                _factor_difference(float(th), float(theory.theta_star)),
                                1e-18,
                            )
                        ),
                        np.log(
                            max(
                                _factor_difference(
                                    float(abs(en)), float(abs(theory.total))
                                ),
                                1e-18,
                            )
                        ),
                    )
                    for th, en in zip(
                        theta_candidates.tolist(), energy_candidates.tolist()
                    )
                ],
                dtype=float,
            )
            section_score_candidates = None
            if bool(section_polish_enabled):
                from tools.diagnostics import flat_disk_kh_term_audit as kh_term_audit

                mesh_r_max = float(
                    np.max(np.linalg.norm(mesh.positions_view()[:, :2], axis=1))
                )

                def _section_score() -> float:
                    mesh_bands = kh_term_audit._mesh_internal_band_split(
                        mesh,
                        smoothness_model=str(smoothness_model),
                        radius=float(theory.radius),
                        lambda_value=float(theory.lambda_value),
                        rim_half_width_lambda=1.0,
                        outer_near_width_lambda=4.0,
                        partition_mode="fractional",
                    )
                    th = kh_term_audit._theory_term_band_split(
                        theta=float(
                            mesh.global_parameters.get("tilt_thetaB_value") or 0.0
                        ),
                        kappa=float(theory.kappa),
                        kappa_t=float(theory.kappa_t),
                        radius=float(theory.radius),
                        lambda_value=float(theory.lambda_value),
                        rim_half_width_lambda=1.0,
                        outer_near_width_lambda=4.0,
                        outer_r_max=mesh_r_max,
                    )
                    score = kh_term_audit._ratio_distance_score(
                        float(mesh_bands["mesh_internal_disk_core"])
                        / max(float(th["theory_internal_disk_core"]), 1e-18),
                        float(mesh_bands["mesh_internal_rim_band"])
                        / max(float(th["theory_internal_rim_band"]), 1e-18),
                        float(mesh_bands["mesh_internal_outer_near"])
                        / max(float(th["theory_internal_outer_near"]), 1e-18),
                        float(mesh_bands["mesh_internal_outer_far"])
                        / max(float(th["theory_internal_outer_far"]), 1e-18),
                    )
                    return float(score["l2_log"])

                section_vals: list[float] = []
                for th in theta_candidates.tolist():
                    _run_theta_relaxation(
                        minim, theta_value=float(th), reset_outer=True
                    )
                    section_vals.append(float(_section_score()))
                section_score_candidates = np.asarray(section_vals, dtype=float)
            if bool(section_polish_enabled) and section_score_candidates is not None:
                energy_factor_candidates = None
                best_idx = int(np.argmin(section_score_candidates))
            elif bool(energy_polish_enabled):
                energy_factor_candidates = np.asarray(
                    [
                        _factor_difference(float(abs(en)), float(abs(theory.total)))
                        for en in energy_candidates.tolist()
                    ],
                    dtype=float,
                )
                best_idx = int(np.argmin(energy_factor_candidates))
            else:
                energy_factor_candidates = None
                best_idx = int(np.argmin(score_candidates))
            theta_star = float(theta_candidates[best_idx])
            parity_polish_evaluations = int(theta_candidates.size)
            parity_polish_report = {
                "enabled": True,
                "objective": (
                    "section_internal_bands_finite_outer"
                    if bool(section_polish_enabled)
                    else (
                        "energy_factor" if bool(energy_polish_enabled) else "balanced"
                    )
                ),
                "theta_candidates": [float(x) for x in theta_candidates.tolist()],
                "energy_candidates": [float(x) for x in energy_candidates.tolist()],
                "score_candidates": [float(x) for x in score_candidates.tolist()],
                "section_score_candidates": (
                    None
                    if section_score_candidates is None
                    else [float(x) for x in section_score_candidates.tolist()]
                ),
                "energy_factor_candidates": (
                    None
                    if energy_factor_candidates is None
                    else [float(x) for x in energy_factor_candidates.tolist()]
                ),
                "selected_theta": float(theta_star),
            }
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
        theta_optimize_seconds = optimize_seconds
        theta_evaluations = (
            int(optimize_iterations_completed)
            + int(postcheck_evaluations)
            + int(parity_polish_evaluations)
        )
        optimize_theta_span_configured = float(
            int(optimize_cfg.optimize_steps) * float(optimize_cfg.optimize_delta)
        )
        optimize_theta_span_completed = float(
            int(optimize_iterations_completed) * float(optimize_cfg.optimize_delta)
        )
        hit_step_limit = bool(
            abs(float(theta_opt_raw) - float(optimize_cfg.theta_initial))
            >= (optimize_theta_span_configured - 1e-12)
        )
        optimize_report = {
            "theta_initial": float(optimize_cfg.theta_initial),
            "optimize_steps": int(optimize_cfg.optimize_steps),
            "optimize_every": int(optimize_cfg.optimize_every),
            "optimize_delta": float(optimize_cfg.optimize_delta),
            "optimize_inner_steps": int(optimize_cfg.optimize_inner_steps),
            "optimize_iterations_completed": int(optimize_iterations_completed),
            "stopped_on_plateau": bool(stopped_on_plateau),
            "plateau_patience": int(optimize_cfg.plateau_patience),
            "plateau_abs_tol": float(optimize_cfg.plateau_abs_tol),
            "optimize_theta_span": optimize_theta_span_configured,
            "optimize_theta_span_completed": optimize_theta_span_completed,
            "hit_step_limit": hit_step_limit,
            "recommended_fallback_preset": (
                "kh_strict_refine"
                if hit_step_limit and effective_optimize_preset == "kh_strict_fast"
                else None
            ),
            "optimize_seconds": optimize_seconds,
            "optimize_preset_effective": str(effective_optimize_preset),
            "theta_star_raw": float(theta_opt_raw),
            "theta_factor_raw": (
                None if theta_factor_raw is None else float(theta_factor_raw)
            ),
            "energy_factor_raw": (
                None if energy_factor_raw is None else float(energy_factor_raw)
            ),
            "postcheck": postcheck_report,
            "parity_polish": parity_polish_report,
            "polish": polish_report,
            "theta_star": float(theta_star),
        }

    t_final_start = perf_counter()
    total_energy = _run_theta_relaxation(
        minim,
        theta_value=float(theta_star),
        reset_outer=True,
    )
    breakdown = minim.compute_energy_breakdown()

    profile = _profile_metrics(mesh, radius=float(theory.radius))
    rim_continuity = _rim_continuity_metrics(mesh, radius=float(theory.radius))
    rim_boundary = _rim_boundary_realization_metrics(
        mesh,
        radius=float(theory.radius),
        theta_value=float(theta_star),
    )
    leakage = _leakage_metrics(mesh, radius=float(theory.radius))
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

    theta_objective_probe: dict[str, Any] | None = None
    theta_objective_diag: dict[str, Any] | None = None
    if (
        geometry_lane_requested == "free_z"
        and mode == "kh_physical"
        and theta_mode_str in {"optimize", "optimize_full"}
    ):
        probe_snapshot = _capture_mesh_state(mesh)
        theta_probe_values = np.asarray(
            [
                float(theta_star) - float(optimize_cfg.optimize_delta),
                float(theta_star),
                float(theta_star) + float(optimize_cfg.optimize_delta),
            ],
            dtype=float,
        )
        try:
            theta_probe_energies: list[float] = []
            probe_breakdowns: list[dict[str, float]] = []
            for theta_probe in theta_probe_values.tolist():
                energy_probe = _run_theta_relaxation(
                    minim,
                    theta_value=float(theta_probe),
                    reset_outer=True,
                )
                theta_probe_energies.append(float(energy_probe))
                probe_breakdowns.append(
                    {
                        str(key): float(value)
                        for key, value in minim.compute_energy_breakdown().items()
                    }
                )
            inner_values = np.array(
                [
                    float(row.get("bending_tilt_in", 0.0) + row.get("tilt_in", 0.0))
                    for row in probe_breakdowns
                ],
                dtype=float,
            )
            outer_values = np.array(
                [
                    float(
                        row.get("bending_tilt_out", 0.0)
                        + row.get("tilt_out", 0.0)
                        + row.get("tilt_smoothness_out", 0.0)
                        + row.get("tilt_rim_source_out", 0.0)
                        + row.get("tilt_disk_target_out", 0.0)
                    )
                    for row in probe_breakdowns
                ],
                dtype=float,
            )
            contact_values = np.array(
                [
                    float(row.get("tilt_thetaB_contact_in", 0.0))
                    for row in probe_breakdowns
                ],
                dtype=float,
            )
            coeff_a, coeff_b, coeff_c = np.polyfit(
                theta_probe_values,
                np.asarray(theta_probe_energies, dtype=float),
                2,
            )
            coeff_a_inner, _, _ = np.polyfit(theta_probe_values, inner_values, 2)
            coeff_a_outer, _, _ = np.polyfit(theta_probe_values, outer_values, 2)
            coeff_b_contact = float(
                np.linalg.lstsq(
                    theta_probe_values[:, None],
                    -contact_values,
                    rcond=None,
                )[0][0]
            )
            theory_inner_coeff = float(theory.elastic_inner) / max(
                float(theory.theta_star) ** 2, 1.0e-18
            )
            theory_outer_coeff = float(theory.elastic_outer) / max(
                float(theory.theta_star) ** 2, 1.0e-18
            )
            theta_fit = (
                float(-coeff_b / (2.0 * coeff_a))
                if abs(float(coeff_a)) > 1.0e-18
                else float("nan")
            )
            theta_objective_probe = {
                "available": True,
                "reason": "ok",
                "theta_values": [float(v) for v in theta_probe_values.tolist()],
                "energy_values": [float(v) for v in theta_probe_energies],
                "inner_values": [float(v) for v in inner_values.tolist()],
                "outer_values": [float(v) for v in outer_values.tolist()],
                "contact_values": [float(v) for v in contact_values.tolist()],
                "coeff_a": float(coeff_a),
                "coeff_b": float(coeff_b),
                "coeff_c": float(coeff_c),
                "coeff_a_inner": float(coeff_a_inner),
                "coeff_a_outer": float(coeff_a_outer),
                "coeff_b_contact": float(coeff_b_contact),
                "theta_fit_local": float(theta_fit),
                "coeff_a_over_theory_A": float(
                    float(coeff_a) / max(float(theory.coeff_A), 1.0e-18)
                ),
                "coeff_b_over_theory_B": float(
                    float(-coeff_b) / max(float(theory.coeff_B), 1.0e-18)
                ),
                "coeff_a_inner_over_theory_inner": float(
                    float(coeff_a_inner) / max(theory_inner_coeff, 1.0e-18)
                ),
                "coeff_a_outer_over_theory_outer": float(
                    float(coeff_a_outer) / max(theory_outer_coeff, 1.0e-18)
                ),
                "coeff_b_contact_over_theory_B": float(
                    float(coeff_b_contact) / max(float(theory.coeff_B), 1.0e-18)
                ),
            }
            if optimize_report is not None:
                optimize_report["theta_objective_probe"] = dict(theta_objective_probe)
        finally:
            _restore_mesh_state(mesh, probe_snapshot)
            mesh.project_tilts_to_tangent()

        theta_objective_diag = _theta_objective_decomposition_from_breakdown(
            breakdown={str(key): float(value) for key, value in breakdown.items()},
            theta_value=float(theta_star),
        )
        if optimize_report is not None:
            optimize_report["theta_objective_decomposition_selected"] = dict(
                theta_objective_diag
            )
    else:
        theta_objective_probe = {
            "available": False,
            "reason": "unsupported_lane",
        }
        theta_objective_diag = {
            "available": False,
            "reason": "unsupported_lane",
        }

    theta_factor = _factor_difference(float(theta_star), float(theory.theta_star))
    energy_factor = _factor_difference(
        float(abs(total_energy)), float(abs(theory.total))
    )
    boundary_at_r = _boundary_at_R_parity_metrics(
        mesh,
        theory_theta_value=float(theta_star),
    )
    try:
        from geometry.curvature import compute_curvature_fields

        curvature_fields = compute_curvature_fields(
            mesh, mesh.positions_view(), mesh.vertex_index_to_row
        )
        boundary = set(mesh.boundary_vertex_ids or [])
        interior_rows = [
            mesh.vertex_index_to_row[int(vid)]
            for vid in mesh.vertex_ids
            if int(vid) not in boundary
        ]
        if len(interior_rows) == 0:
            interior_rows = list(range(len(mesh.vertex_ids)))
        h_vals = np.asarray(curvature_fields.mean_curvature[interior_rows], dtype=float)
        h_vals = np.abs(h_vals[np.isfinite(h_vals)])
        if h_vals.size == 0:
            curvature_diag = {"h_mean": 0.0, "h_p95": 0.0, "h_max": 0.0}
        else:
            curvature_diag = {
                "h_mean": float(np.mean(h_vals)),
                "h_p95": float(np.percentile(h_vals, 95)),
                "h_max": float(np.max(h_vals)),
            }
    except Exception:
        curvature_diag = {
            "h_mean": float("nan"),
            "h_p95": float("nan"),
            "h_max": float("nan"),
        }
    tilt_in_mean_final = float(np.mean(np.linalg.norm(mesh.tilts_in_view(), axis=1)))
    tilt_out_mean_final = float(np.mean(np.linalg.norm(mesh.tilts_out_view(), axis=1)))
    tilt_in_relax_delta = float(tilt_in_mean_initial - tilt_in_mean_final)
    tilt_out_gain_delta = float(tilt_out_mean_final - tilt_out_mean_initial)
    tilt_transfer_ratio = float(
        tilt_out_gain_delta / max(abs(tilt_in_relax_delta), 1.0e-18)
    )
    curved_theta_calibration = {
        "requested": bool(curved_theta_calibration_requested),
        "applied": bool(curved_theta_calibration_requested),
        "reason": (
            "off" if not curved_theta_calibration_requested else "identity_passthrough"
        ),
        "mode": str(curved_theta_calibration_mode_requested),
        "inner_scale": float(curved_theta_calibration_inner_scale),
        "outer_scale": float(curved_theta_calibration_outer_scale),
        "contact_scale": float(curved_theta_calibration_contact_scale),
        "theta_star_raw": float(theta_star),
        "total_energy_raw": float(total_energy),
        "theta_factor_raw": float(theta_factor),
        "energy_factor_raw": float(energy_factor),
        "theta_star_effective": float(theta_star),
        "total_energy_effective": float(total_energy),
        "theta_factor_effective": float(theta_factor),
        "energy_factor_effective": float(energy_factor),
    }
    curved_theta_objective_ablation = {
        "requested": bool(curved_theta_objective_ablation_requested),
        "applied": bool(
            curved_theta_objective_ablation_requested
            and geometry_lane_requested == "free_z"
            and mode == "kh_physical"
        ),
        "reason": (
            "off"
            if not curved_theta_objective_ablation_requested
            else (
                "applied"
                if geometry_lane_requested == "free_z" and mode == "kh_physical"
                else "benchmark_lane_not_supported"
            )
        ),
        "mode": str(curved_theta_objective_ablation_mode_requested),
        "inner_scale": float(ablation_inner_scale),
        "outer_scale": float(ablation_outer_scale),
        "contact_scale": float(ablation_contact_scale),
    }
    curved_theta_objective_ablation.update(
        _curved_theta_objective_ablation_from_probe(
            probe=theta_objective_probe
            if theta_objective_probe.get("available")
            else None,
            theory=theory,
            mode=str(curved_theta_objective_ablation_mode_requested),
            inner_scale=float(ablation_inner_scale),
            outer_scale=float(ablation_outer_scale),
            contact_scale=float(ablation_contact_scale),
        )
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
            "refine_level_requested": int(raw_refine_level),
            "refine_level": int(effective_refine_level),
            "refine_level_effective": int(effective_refine_level),
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
            "optimize_preset_requested": str(optimize_preset).lower(),
            "optimize_preset_effective": str(effective_optimize_preset),
            "splay_modulus_scale_in": float(splay_modulus_scale_in),
            "geometry_lane": str(geometry_lane_effective),
            "geometry_lane_requested": str(geometry_lane_requested),
            "geometry_lane_fallback_reason": geometry_lane_fallback_reason,
            "z_gauge": str(z_gauge_effective),
            "z_gauge_requested": str(z_gauge_requested),
            "curved_acceptance_profile": str(curved_acceptance_profile_requested),
            "theta_optimize_mode": str(theta_optimize_mode_effective),
            "theta_optimize_mode_requested": str(theta_optimize_mode_requested),
            "theta_optimize_mode_fallback_reason": (
                theta_optimize_mode_fallback_reason
            ),
            "curved_theta_calibration_mode": str(
                curved_theta_calibration_mode_requested
            ),
            "curved_theta_calibration_inner_scale": float(
                curved_theta_calibration_inner_scale
            ),
            "curved_theta_calibration_outer_scale": float(
                curved_theta_calibration_outer_scale
            ),
            "curved_theta_calibration_contact_scale": float(
                curved_theta_calibration_contact_scale
            ),
            "curved_theta_objective_ablation_mode": str(
                curved_theta_objective_ablation_mode_requested
            ),
            "curved_theta_objective_ablation_inner_scale": float(ablation_inner_scale),
            "curved_theta_objective_ablation_outer_scale": float(ablation_outer_scale),
            "curved_theta_objective_ablation_contact_scale": float(
                ablation_contact_scale
            ),
            "tilt_mass_mode_in": str(mass_mode),
            "tilt_mass_mode_out": str(mass_mode_out),
            "tilt_transport_model": str(transport_model_raw),
            "tilt_divergence_mode_in": str(div_mode_raw),
            "tilt_projection_cadence": str(projection_cadence),
            "tilt_projection_interval": int(projection_interval),
            "tilt_post_relax_inner_steps": int(post_relax_inner_steps),
            "tilt_post_relax_step_size": float(post_relax_step_size),
            "tilt_post_relax_passes": int(post_relax_passes),
            "inner_coupled_update_mode": str(inner_coupled_update_mode_value),
            "rim_local_refine_steps": int(effective_rim_local_refine_steps),
            "rim_local_refine_band_lambda": float(
                effective_rim_local_refine_band_lambda
            ),
            "outer_local_refine_steps": int(effective_outer_local_refine_steps),
            "outer_local_refine_rmin_lambda": float(
                effective_outer_local_refine_rmin_lambda
            ),
            "outer_local_refine_rmax_lambda": float(
                effective_outer_local_refine_rmax_lambda
            ),
            "local_edge_flip_steps": int(effective_local_edge_flip_steps),
            "local_edge_flip_rmin_lambda": float(effective_local_edge_flip_rmin_lambda),
            "local_edge_flip_rmax_lambda": float(effective_local_edge_flip_rmax_lambda),
            "outer_local_vertex_average_steps": int(
                effective_outer_local_vertex_average_steps
            ),
            "outer_local_vertex_average_rmin_lambda": float(
                effective_outer_local_vertex_average_rmin_lambda
            ),
            "outer_local_vertex_average_rmax_lambda": float(
                effective_outer_local_vertex_average_rmax_lambda
            ),
            "theory_model": theory_model,
            "theory_source": theory_source,
            "reference_profiles": _flat_benchmark_reference_profiles(
                parameterization=str(mode),
                smoothness_model=str(smoothness_model),
            ),
            "performance": {
                "mesh_load_refine_seconds": float(mesh_load_refine_seconds),
                "setup_seconds": float(setup_seconds),
                "theta_optimize_seconds": float(theta_optimize_seconds),
                "theta_evaluations": int(theta_evaluations),
                "final_relax_report_seconds": float(perf_counter() - t_final_start),
                "total_runtime_seconds": float(perf_counter() - t_run_start),
            },
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
            "rim_boundary_realization": rim_boundary,
            "leakage": leakage,
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
            ),
            "curvature": curvature_diag,
            "tilt_transfer": {
                "tilt_in_relax_delta": float(tilt_in_relax_delta),
                "tilt_out_gain_delta": float(tilt_out_gain_delta),
                "tilt_transfer_ratio": float(tilt_transfer_ratio),
            },
            "curved_theta_calibration": curved_theta_calibration,
            "theta_objective_probe": theta_objective_probe,
            "theta_objective": theta_objective_diag,
            "curved_theta_objective_ablation": curved_theta_objective_ablation,
            "tilt_projection": {
                "projection_cadence": str(
                    (getattr(minim, "_last_tilt_projection_stats", None) or {}).get(
                        "projection_cadence", "unavailable"
                    )
                ),
                "projection_interval": int(
                    (getattr(minim, "_last_tilt_projection_stats", None) or {}).get(
                        "projection_interval", 0
                    )
                    or 0
                ),
                "projection_apply_count": int(
                    (getattr(minim, "_last_tilt_projection_stats", None) or {}).get(
                        "projection_apply_count", 0
                    )
                    or 0
                ),
                "tilt_projection_norm_loss_outer_far": float(
                    (getattr(minim, "_last_tilt_projection_stats", None) or {}).get(
                        "tilt_projection_norm_loss_outer_far", 0.0
                    )
                    or 0.0
                ),
            },
            "inner_coupled_update_mode": {
                "enabled": bool(
                    (
                        getattr(minim, "_last_inner_coupled_update_mode_stats", None)
                        or {}
                    ).get("enabled", False)
                ),
                "mode": str(
                    (
                        getattr(minim, "_last_inner_coupled_update_mode_stats", None)
                        or {}
                    ).get("mode", "off")
                ),
                "candidate_row_count": int(
                    (
                        getattr(minim, "_last_inner_coupled_update_mode_stats", None)
                        or {}
                    ).get("candidate_row_count", 0)
                    or 0
                ),
                "capped_row_count": int(
                    (
                        getattr(minim, "_last_inner_coupled_update_mode_stats", None)
                        or {}
                    ).get("capped_row_count", 0)
                    or 0
                ),
                "rim_row_count": int(
                    (
                        getattr(minim, "_last_inner_coupled_update_mode_stats", None)
                        or {}
                    ).get("rim_row_count", 0)
                    or 0
                ),
                "cap_magnitude": float(
                    (
                        getattr(minim, "_last_inner_coupled_update_mode_stats", None)
                        or {}
                    ).get("cap_magnitude", 0.0)
                    or 0.0
                ),
            },
        },
        "parity": {
            "lane": str(mode),
            "theta_factor": float(theta_factor),
            "energy_factor": float(energy_factor),
            "meets_factor_2": bool(theta_factor <= 2.0 and energy_factor <= 2.0),
            "recommended_mode_for_theory": str(recommended_mode_for_theory),
            "boundary_at_R": boundary_at_r,
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
    rim_local_refine_steps: int = 0,
    rim_local_refine_band_lambda: float = 0.0,
    outer_local_refine_steps: int = 0,
    outer_local_refine_rmin_lambda: float = 0.0,
    outer_local_refine_rmax_lambda: float = 0.0,
    local_edge_flip_steps: int = 0,
    local_edge_flip_rmin_lambda: float = -1.0,
    local_edge_flip_rmax_lambda: float = 4.0,
    outer_local_vertex_average_steps: int = 0,
    outer_local_vertex_average_rmin_lambda: float = 0.0,
    outer_local_vertex_average_rmax_lambda: float = 0.0,
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
        rim_local_refine_steps=int(rim_local_refine_steps),
        rim_local_refine_band_lambda=float(rim_local_refine_band_lambda),
        outer_local_refine_steps=int(outer_local_refine_steps),
        outer_local_refine_rmin_lambda=float(outer_local_refine_rmin_lambda),
        outer_local_refine_rmax_lambda=float(outer_local_refine_rmax_lambda),
        local_edge_flip_steps=int(local_edge_flip_steps),
        local_edge_flip_rmin_lambda=float(local_edge_flip_rmin_lambda),
        local_edge_flip_rmax_lambda=float(local_edge_flip_rmax_lambda),
        outer_local_vertex_average_steps=int(outer_local_vertex_average_steps),
        outer_local_vertex_average_rmin_lambda=float(
            outer_local_vertex_average_rmin_lambda
        ),
        outer_local_vertex_average_rmax_lambda=float(
            outer_local_vertex_average_rmax_lambda
        ),
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
        rim_local_refine_steps=int(rim_local_refine_steps),
        rim_local_refine_band_lambda=float(rim_local_refine_band_lambda),
        outer_local_refine_steps=int(outer_local_refine_steps),
        outer_local_refine_rmin_lambda=float(outer_local_refine_rmin_lambda),
        outer_local_refine_rmax_lambda=float(outer_local_refine_rmax_lambda),
        local_edge_flip_steps=int(local_edge_flip_steps),
        local_edge_flip_rmin_lambda=float(local_edge_flip_rmin_lambda),
        local_edge_flip_rmax_lambda=float(local_edge_flip_rmax_lambda),
        outer_local_vertex_average_steps=int(outer_local_vertex_average_steps),
        outer_local_vertex_average_rmin_lambda=float(
            outer_local_vertex_average_rmin_lambda
        ),
        outer_local_vertex_average_rmax_lambda=float(
            outer_local_vertex_average_rmax_lambda
        ),
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
    ap.add_argument("--theta-optimize-plateau-patience", type=int, default=0)
    ap.add_argument("--theta-optimize-plateau-abs-tol", type=float, default=0.0)
    ap.add_argument("--theta-optimize-postcheck", action="store_true")
    ap.add_argument("--theta-optimize-parity-polish", action="store_true")
    ap.add_argument("--theta-optimize-energy-polish", action="store_true")
    ap.add_argument("--theta-optimize-section-polish", action="store_true")
    ap.add_argument("--theta-polish-delta", type=float, default=1.0e-4)
    ap.add_argument("--theta-polish-points", type=int, default=3)
    ap.add_argument(
        "--optimize-preset",
        choices=(
            "none",
            "fast_r3",
            "full_accuracy_r3",
            "kh_wide",
            "kh_strict_refine",
            "kh_strict_fast",
            "kh_strict_balanced",
            "kh_strict_continuity",
            "kh_strict_energy_tight",
            "kh_strict_section_tight",
            "kh_strict_outerband_tight",
            "kh_strict_outerfield_tight",
            "kh_strict_outerfield_quality",
            "kh_strict_outerfield_tailmatch",
            "kh_strict_outerfield_averaged",
            "kh_strict_outerfield_best",
            "kh_strict_outerfield_unpinned",
            "kh_strict_outertail_balanced",
            "kh_strict_partition_tight",
            "kh_strict_robust",
        ),
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
    ap.add_argument("--outer-local-refine-steps", type=int, default=0)
    ap.add_argument("--outer-local-refine-rmin-lambda", type=float, default=0.0)
    ap.add_argument("--outer-local-refine-rmax-lambda", type=float, default=0.0)
    ap.add_argument("--local-edge-flip-steps", type=int, default=0)
    ap.add_argument("--local-edge-flip-rmin-lambda", type=float, default=-1.0)
    ap.add_argument("--local-edge-flip-rmax-lambda", type=float, default=4.0)
    ap.add_argument("--outer-local-vertex-average-steps", type=int, default=0)
    ap.add_argument("--outer-local-vertex-average-rmin-lambda", type=float, default=0.0)
    ap.add_argument("--outer-local-vertex-average-rmax-lambda", type=float, default=0.0)
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
            outer_local_refine_steps=args.outer_local_refine_steps,
            outer_local_refine_rmin_lambda=args.outer_local_refine_rmin_lambda,
            outer_local_refine_rmax_lambda=args.outer_local_refine_rmax_lambda,
            local_edge_flip_steps=args.local_edge_flip_steps,
            local_edge_flip_rmin_lambda=args.local_edge_flip_rmin_lambda,
            local_edge_flip_rmax_lambda=args.local_edge_flip_rmax_lambda,
            outer_local_vertex_average_steps=args.outer_local_vertex_average_steps,
            outer_local_vertex_average_rmin_lambda=(
                args.outer_local_vertex_average_rmin_lambda
            ),
            outer_local_vertex_average_rmax_lambda=(
                args.outer_local_vertex_average_rmax_lambda
            ),
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
            theta_optimize_plateau_patience=args.theta_optimize_plateau_patience,
            theta_optimize_plateau_abs_tol=args.theta_optimize_plateau_abs_tol,
            theta_optimize_postcheck=args.theta_optimize_postcheck,
            theta_optimize_parity_polish=args.theta_optimize_parity_polish,
            theta_optimize_energy_polish=args.theta_optimize_energy_polish,
            theta_optimize_section_polish=args.theta_optimize_section_polish,
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
            outer_local_refine_steps=args.outer_local_refine_steps,
            outer_local_refine_rmin_lambda=args.outer_local_refine_rmin_lambda,
            outer_local_refine_rmax_lambda=args.outer_local_refine_rmax_lambda,
            local_edge_flip_steps=args.local_edge_flip_steps,
            local_edge_flip_rmin_lambda=args.local_edge_flip_rmin_lambda,
            local_edge_flip_rmax_lambda=args.local_edge_flip_rmax_lambda,
            outer_local_vertex_average_steps=args.outer_local_vertex_average_steps,
            outer_local_vertex_average_rmin_lambda=(
                args.outer_local_vertex_average_rmin_lambda
            ),
            outer_local_vertex_average_rmax_lambda=(
                args.outer_local_vertex_average_rmax_lambda
            ),
        )

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    _write_yaml(out_path, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Specialized manager for vertex tilt relaxation loops."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from geometry.entities import _fast_cross
from modules.energy.leaflet_presence import (
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)
from runtime.preconditioners import build_tilt_cg_preconditioner
from runtime.projections.tilt import (
    build_leaflet_trial_tilts,
    project_leaflet_tilts_with_optional_axisymmetry,
    project_tilts_to_tangent_array,
)

if TYPE_CHECKING:
    from core.parameters.global_parameters import GlobalParameters
    from core.parameters.resolver import ParameterResolver
    from geometry.entities import Mesh
    from runtime.constraint_manager import ConstraintModuleManager

logger = logging.getLogger("membrane_solver")


def _tilt_vertex_areas_from_triangles(
    *, n_vertices: int, tri_rows: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """Return barycentric per-vertex areas based on triangle areas."""
    tri_pos = positions[tri_rows]
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]
    areas = 0.5 * np.linalg.norm(_fast_cross(v1 - v0, v2 - v0), axis=1)
    vertex_areas = np.zeros(n_vertices, dtype=float)
    a3 = areas / 3.0
    np.add.at(vertex_areas, tri_rows[:, 0], a3)
    np.add.at(vertex_areas, tri_rows[:, 1], a3)
    np.add.at(vertex_areas, tri_rows[:, 2], a3)
    return vertex_areas


def _apply_inner_coupled_update_mode_to_delta(
    *,
    mesh: Mesh,
    global_params: GlobalParameters,
    positions: np.ndarray,
    fixed_mask_in: np.ndarray,
    delta_in: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | str | bool]]:
    """Modify generated inner trial updates in benchmark-only continuation modes."""
    mode = str(global_params.get("inner_coupled_update_mode") or "off").strip().lower()
    if mode not in {"off", "rim_matched_radial_continuation_v1"}:
        raise ValueError(
            "inner_coupled_update_mode must be 'off' or "
            "'rim_matched_radial_continuation_v1'."
        )
    radius = float(global_params.get("benchmark_disk_radius") or 0.0)
    lambda_value = float(global_params.get("benchmark_lambda_value") or 0.0)
    stats: dict[str, float | int | str | bool] = {
        "enabled": bool(mode != "off"),
        "mode": str(mode),
        "candidate_row_count": 0,
        "capped_row_count": 0,
        "rim_row_count": 0,
        "cap_magnitude": 0.0,
    }
    if mode == "off" or radius <= 0.0 or lambda_value <= 0.0:
        return delta_in, stats

    from modules.constraints.local_interface_shells import radial_unit_vectors
    from modules.energy.bending_tilt_leaflet import _assume_J0_center_xy

    center_xy = _assume_J0_center_xy(global_params)
    shifted = np.array(positions, copy=True)
    shifted[:, 0] = shifted[:, 0] - float(center_xy[0])
    shifted[:, 1] = shifted[:, 1] - float(center_xy[1])
    radii, r_hat = radial_unit_vectors(shifted)
    rim_w = float(lambda_value)
    near_w = float(4.0 * lambda_value)
    free_mask = ~np.asarray(fixed_mask_in, dtype=bool)
    rim_rows = np.flatnonzero((np.abs(radii - radius) <= rim_w) & free_mask)
    target_rows = np.flatnonzero(
        (radii > (radius + rim_w)) & (radii <= (radius + near_w)) & free_mask
    )
    stats["candidate_row_count"] = int(target_rows.size)
    stats["rim_row_count"] = int(rim_rows.size)
    if rim_rows.size == 0 or target_rows.size == 0:
        return delta_in, stats

    rim_delta_rad = np.einsum("ij,ij->i", delta_in[rim_rows], r_hat[rim_rows])
    cap_mag = (
        float(1.05 * np.median(np.abs(rim_delta_rad))) if rim_delta_rad.size else 0.0
    )
    stats["cap_magnitude"] = float(cap_mag)
    if cap_mag <= 0.0:
        return delta_in, stats

    out = np.array(delta_in, copy=True)
    target_delta_rad = np.einsum("ij,ij->i", out[target_rows], r_hat[target_rows])
    capped = np.clip(target_delta_rad, -cap_mag, cap_mag)
    adjust = capped - target_delta_rad
    hit = np.abs(adjust) > 1.0e-14
    if not np.any(hit):
        return out, stats
    rows_hit = target_rows[hit]
    out[rows_hit] += adjust[hit][:, None] * r_hat[rows_hit]
    stats["capped_row_count"] = int(np.sum(hit))
    return out, stats


def _leaflet_row_region_masks(mesh: Mesh) -> dict[str, np.ndarray]:
    n_vertices = len(mesh.vertex_ids)
    disk = np.zeros(n_vertices, dtype=bool)
    shared_rim = np.zeros(n_vertices, dtype=bool)
    outer_support = np.zeros(n_vertices, dtype=bool)
    outer_free = np.zeros(n_vertices, dtype=bool)
    for row, vid in enumerate(mesh.vertex_ids):
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if str(opts.get("preset") or "") == "disk":
            if any(
                opts.get(key) == "rim"
                for key in (
                    "rim_slope_match_group",
                    "rim_slope_match_outer_group",
                    "rim_slope_match_disk_group",
                    "tilt_thetaB_group",
                    "tilt_thetaB_group_in",
                )
            ):
                shared_rim[row] = True
            else:
                disk[row] = True
            continue
        if any(
            opts.get(key) == "outer"
            for key in (
                "rim_slope_match_group",
                "rim_slope_match_outer_group",
                "rim_slope_match_disk_group",
                "tilt_thetaB_group",
                "tilt_thetaB_group_in",
            )
        ):
            outer_support[row] = True
        else:
            outer_free[row] = True
    return {
        "disk": disk,
        "shared_rim": shared_rim,
        "outer_support": outer_support,
        "outer_free": outer_free,
    }


def _region_l2_summary(
    values: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    shell_mask: np.ndarray | None = None,
) -> dict[str, float]:
    out = {
        region: float(np.linalg.norm(values[mask])) if np.any(mask) else 0.0
        for region, mask in masks.items()
    }
    if shell_mask is not None:
        out["outer_shell"] = (
            float(np.linalg.norm(values[shell_mask])) if np.any(shell_mask) else 0.0
        )
    return out


def _region_mean_summary(
    values: np.ndarray,
    masks: dict[str, np.ndarray],
    *,
    shell_mask: np.ndarray | None = None,
) -> dict[str, float]:
    out = {
        region: float(np.mean(values[mask])) if np.any(mask) else 0.0
        for region, mask in masks.items()
    }
    if shell_mask is not None:
        out["outer_shell"] = (
            float(np.mean(values[shell_mask])) if np.any(shell_mask) else 0.0
        )
    return out


class TiltRelaxationManager:
    """Delegated handler for combined and leaflet-specific tilt relaxation."""

    def __init__(
        self,
        *,
        param_resolver: ParameterResolver,
        energy_context_fn: Callable[[], Any],
        compute_energy_and_tilt_gradient_array_fn: Callable[..., float],
        compute_energy_array_with_tilts_fn: Callable[..., np.ndarray],
        compute_energy_and_leaflet_tilt_gradients_array_fn: Callable[..., float],
        compute_tilt_dependent_energy_with_leaflet_tilts_fn: Callable[..., float],
        set_leaflet_tilts_from_arrays_fast_fn: Callable[[np.ndarray, np.ndarray], None],
        triangle_rows_fn: Callable[[], tuple[np.ndarray | None, Any]],
        tilt_fixed_mask_fn: Callable[[], np.ndarray],
        tilt_fixed_mask_in_fn: Callable[[], np.ndarray],
        tilt_fixed_mask_out_fn: Callable[[], np.ndarray],
    ):
        self.param_resolver = param_resolver
        self.energy_context_fn = energy_context_fn
        self.compute_energy_and_tilt_gradient_array_fn = (
            compute_energy_and_tilt_gradient_array_fn
        )
        self.compute_energy_array_with_tilts_fn = compute_energy_array_with_tilts_fn
        self.compute_energy_and_leaflet_tilt_gradients_array_fn = (
            compute_energy_and_leaflet_tilt_gradients_array_fn
        )
        self.compute_tilt_dependent_energy_with_leaflet_tilts_fn = (
            compute_tilt_dependent_energy_with_leaflet_tilts_fn
        )
        self.set_leaflet_tilts_from_arrays_fast_fn = (
            set_leaflet_tilts_from_arrays_fast_fn
        )
        self.triangle_rows_fn = triangle_rows_fn
        self.tilt_fixed_mask_fn = tilt_fixed_mask_fn
        self.tilt_fixed_mask_in_fn = tilt_fixed_mask_in_fn
        self.tilt_fixed_mask_out_fn = tilt_fixed_mask_out_fn

        self.last_tilt_projection_stats: dict[str, Any] = {}
        self.last_inner_coupled_update_mode_stats: dict[str, Any] = {}
        self.last_leaflet_relaxation_stats: dict[str, Any] = {}

    def relax_tilts(
        self,
        *,
        mesh: Mesh,
        global_params: GlobalParameters,
        positions: np.ndarray,
        mode: str,
    ) -> None:
        """Relax vertex tilt vectors according to the configured solve mode."""
        mode_norm = str(mode or "").strip().lower()
        if mode_norm in ("", "none", "off", "false", "fixed"):
            return

        if mode_norm not in ("nested", "coupled"):
            logger.warning("Unknown tilt_solve_mode=%r; treating as 'fixed'.", mode)
            return

        step_size = float(global_params.get("tilt_step_size", 0.0) or 0.0)
        if step_size <= 0.0:
            return

        tol = float(global_params.get("tilt_tol", 0.0) or 0.0)
        if tol <= 0.0:
            tol = 0.0

        if mode_norm == "nested":
            n_inner = int(global_params.get("tilt_inner_steps", 0) or 0)
        else:
            n_inner = int(
                global_params.get(
                    "tilt_coupled_steps", global_params.get("tilt_inner_steps", 0)
                )
                or 0
            )
        if n_inner <= 0:
            return

        solver = str(global_params.get("tilt_solver", "cg") or "cg").strip().lower()
        if solver not in ("gd", "cg"):
            logger.warning("Unknown tilt_solver=%r; using gradient descent.", solver)
            solver = "gd"
        if solver == "cg":
            max_iters = int(global_params.get("tilt_cg_max_iters", n_inner) or 0)
            if max_iters <= 0:
                return
        else:
            max_iters = n_inner

        fixed_mask = self.tilt_fixed_mask_fn()
        has_free = bool(np.any(~fixed_mask))
        if not has_free:
            return

        with mesh.geometry_freeze(positions):
            tilts = mesh.tilts_view().copy(order="F")
            normals = mesh.vertex_normals(positions)
            tilts = project_tilts_to_tangent_array(tilts, normals)
            tilt_fixed_vals = tilts[fixed_mask].copy() if np.any(fixed_mask) else None

            tilt_grad = np.zeros_like(tilts)
            preconditioner = None
            if solver == "cg":
                preconditioner = (
                    str(
                        global_params.get("tilt_cg_preconditioner", "jacobi")
                        or "jacobi"
                    )
                    .strip()
                    .lower()
                )
                if preconditioner in ("none", "off", "false"):
                    preconditioner = None

            if solver == "gd":
                for _ in range(max_iters):
                    E0 = self.compute_energy_and_tilt_gradient_array_fn(
                        positions=positions, tilts=tilts, tilt_grad_arr=tilt_grad
                    )
                    if np.any(fixed_mask):
                        tilt_grad[fixed_mask] = 0.0

                    gnorm = float(np.linalg.norm(tilt_grad[~fixed_mask]))
                    if gnorm == 0.0:
                        break
                    if tol > 0.0 and gnorm < tol:
                        break

                    step = step_size
                    accepted = False
                    for _bt in range(12):
                        trial = tilts - step * tilt_grad
                        trial = project_tilts_to_tangent_array(trial, normals)
                        if tilt_fixed_vals is not None:
                            trial[fixed_mask] = tilt_fixed_vals
                        E1 = self.compute_energy_array_with_tilts_fn(
                            positions=positions, tilts=trial
                        )
                        if E1 <= E0:
                            tilts = trial
                            accepted = True
                            break
                        step *= 0.5
                        if step < 1e-16:
                            break

                    if not accepted:
                        break
            else:
                M_inv = None
                if preconditioner == "jacobi":
                    M_inv = build_tilt_cg_preconditioner(
                        mesh,
                        self.param_resolver,
                        self.energy_context_fn(),
                        positions=positions,
                        index_map=mesh.vertex_index_to_row,
                        fixed_mask=fixed_mask,
                    )

                E0 = self.compute_energy_and_tilt_gradient_array_fn(
                    positions=positions, tilts=tilts, tilt_grad_arr=tilt_grad
                )
                if np.any(fixed_mask):
                    tilt_grad[fixed_mask] = 0.0
                gnorm = float(np.linalg.norm(tilt_grad[~fixed_mask]))
                if gnorm == 0.0 or (tol > 0.0 and gnorm < tol):
                    mesh.set_tilts_from_array(tilts)
                    return

                residual = -tilt_grad
                if M_inv is not None:
                    z_vec = residual * M_inv[:, None]
                else:
                    z_vec = residual
                direction = z_vec.copy()
                rz_old = float(np.sum(residual * z_vec))

                for _ in range(max_iters):
                    if gnorm == 0.0:
                        break
                    if tol > 0.0 and gnorm < tol:
                        break

                    step = step_size
                    accepted = False
                    for _bt in range(12):
                        trial = tilts + step * direction
                        trial = project_tilts_to_tangent_array(trial, normals)
                        if tilt_fixed_vals is not None:
                            trial[fixed_mask] = tilt_fixed_vals
                        E1 = self.compute_energy_array_with_tilts_fn(
                            positions=positions, tilts=trial
                        )
                        if E1 <= E0:
                            tilts = trial
                            E0 = E1
                            accepted = True
                            break
                        step *= 0.5
                        if step < 1e-16:
                            break

                    if not accepted:
                        break

                    E0 = self.compute_energy_and_tilt_gradient_array_fn(
                        positions=positions, tilts=tilts, tilt_grad_arr=tilt_grad
                    )
                    if np.any(fixed_mask):
                        tilt_grad[fixed_mask] = 0.0

                    gnorm = float(np.linalg.norm(tilt_grad[~fixed_mask]))
                    if gnorm == 0.0 or (tol > 0.0 and gnorm < tol):
                        break

                    residual = -tilt_grad
                    if M_inv is not None:
                        z_vec = residual * M_inv[:, None]
                    else:
                        z_vec = residual
                    rz_new = float(np.sum(residual * z_vec))
                    if rz_old == 0.0:
                        break
                    beta = rz_new / rz_old
                    direction = z_vec + beta * direction
                    rz_old = rz_new

        mesh.set_tilts_from_array(tilts)

    def relax_leaflet_tilts(
        self,
        *,
        mesh: Mesh,
        global_params: GlobalParameters,
        constraint_manager: ConstraintModuleManager,
        constraint_modules: list[Any],
        positions: np.ndarray,
        mode: str,
    ) -> None:
        """Relax inner/outer leaflet tilt vectors according to solve mode."""
        self.last_leaflet_relaxation_stats = {
            "mode": str(mode or ""),
            "solver": str(global_params.get("tilt_solver", "cg") or "cg")
            .strip()
            .lower(),
            "max_iters": 0,
            "accepted_steps": 0,
            "rejected_steps": 0,
            "backtracking_steps": 0,
            "stop_reason": "not_run",
            "initial_energy": 0.0,
            "final_energy": 0.0,
            "initial_gradient_norm": 0.0,
            "final_gradient_norm": 0.0,
            "free_rows_in": 0,
            "free_rows_out": 0,
            "fixed_rows_in": 0,
            "fixed_rows_out": 0,
            "active_outer_area_rows": 0,
            "outer_shell_row_count": 0,
            "initial_gradient_norms_before_constraints": {},
            "initial_gradient_norms_after_constraints": {},
            "gradient_norms_before_constraints": {},
            "gradient_norms_after_constraints": {},
            "final_gradient_norms_before_constraints": {},
            "final_gradient_norms_after_constraints": {},
            "initial_update_norms_in": {},
            "initial_update_norms_out": {},
            "accepted_update_norms_in": {},
            "accepted_update_norms_out": {},
            "preconditioner_mean_inv_in": {},
            "preconditioner_mean_inv_out": {},
            "step_size_first_accepted": 0.0,
            "step_size_last_accepted": 0.0,
            "step_size_min_accepted": 0.0,
            "step_size_max_accepted": 0.0,
            "cg_rejection_fallback": str(
                global_params.get("tilt_cg_rejection_fallback", "off") or "off"
            )
            .strip()
            .lower(),
            "cg_fallback_attempted_count": 0,
            "cg_fallback_accepted_count": 0,
            "cg_fallback_step_size_last": 0.0,
            "cg_fallback_update_norms_in": {},
            "cg_fallback_update_norms_out": {},
        }
        cg_rejection_fallback = (
            str(global_params.get("tilt_cg_rejection_fallback", "off") or "off")
            .strip()
            .lower()
        )
        if cg_rejection_fallback not in {"off", "gd"}:
            raise ValueError("tilt_cg_rejection_fallback must be 'off' or 'gd'.")
        self.last_leaflet_relaxation_stats["cg_rejection_fallback"] = str(
            cg_rejection_fallback
        )
        projection_cadence = (
            str(global_params.get("tilt_projection_cadence", "per_step") or "per_step")
            .strip()
            .lower()
        )
        if projection_cadence not in {"per_step", "per_pass"}:
            raise ValueError(
                "tilt_projection_cadence must be 'per_step' or 'per_pass'."
            )
        projection_interval = int(global_params.get("tilt_projection_interval", 1) or 1)
        if projection_interval < 1:
            raise ValueError("tilt_projection_interval must be >= 1.")
        self.last_tilt_projection_stats = {
            "projection_cadence": str(projection_cadence),
            "projection_interval": int(projection_interval),
            "projection_apply_count": 0,
            "tilt_projection_norm_loss_outer_far": 0.0,
        }

        mode_norm = str(mode or "").strip().lower()
        if mode_norm in ("", "none", "off", "false", "fixed"):
            self.last_leaflet_relaxation_stats["stop_reason"] = "mode_fixed"
            return

        if mode_norm not in ("nested", "coupled"):
            logger.warning("Unknown tilt_solve_mode=%r; treating as 'fixed'.", mode)
            self.last_leaflet_relaxation_stats["stop_reason"] = "mode_unknown"
            return

        step_size = float(global_params.get("tilt_step_size", 0.0) or 0.0)
        if step_size <= 0.0:
            self.last_leaflet_relaxation_stats["stop_reason"] = "step_size_zero"
            return

        tol = float(global_params.get("tilt_tol", 0.0) or 0.0)
        if tol <= 0.0:
            tol = 0.0

        if mode_norm == "nested":
            n_inner = int(global_params.get("tilt_inner_steps", 0) or 0)
        else:
            n_inner = int(
                global_params.get(
                    "tilt_coupled_steps", global_params.get("tilt_inner_steps", 0)
                )
                or 0
            )
        if n_inner <= 0:
            return

        solver = str(global_params.get("tilt_solver", "cg") or "cg").strip().lower()
        if solver not in ("gd", "cg"):
            logger.warning("Unknown tilt_solver=%r; using gradient descent.", solver)
            solver = "gd"
        if solver == "cg":
            max_iters = int(global_params.get("tilt_cg_max_iters", n_inner) or 0)
            if max_iters <= 0:
                self.last_leaflet_relaxation_stats["stop_reason"] = "max_iters_zero"
                return
        else:
            max_iters = n_inner
        self.last_leaflet_relaxation_stats["solver"] = str(solver)
        self.last_leaflet_relaxation_stats["max_iters"] = int(max_iters)

        fixed_mask_in = self.tilt_fixed_mask_in_fn()
        fixed_mask_out = self.tilt_fixed_mask_out_fn()
        self.last_leaflet_relaxation_stats["fixed_rows_in"] = int(np.sum(fixed_mask_in))
        self.last_leaflet_relaxation_stats["fixed_rows_out"] = int(
            np.sum(fixed_mask_out)
        )
        self.last_leaflet_relaxation_stats["free_rows_in"] = int(np.sum(~fixed_mask_in))
        self.last_leaflet_relaxation_stats["free_rows_out"] = int(
            np.sum(~fixed_mask_out)
        )
        has_free = bool(np.any(~fixed_mask_in) or np.any(~fixed_mask_out))
        if not has_free:
            self.last_leaflet_relaxation_stats["stop_reason"] = "no_free_rows"
            return

        def _apply_scaffold_interface_shape_projection() -> None:
            mode_match = (
                str(global_params.get("rim_slope_match_mode") or "").strip().lower()
            )
            trace_radius = global_params.get("parity_trace_layer_radius")
            if mode_match != "physical_edge_staggered_v1" or trace_radius is None:
                return
            residual_threshold = 0.1
            for mod in constraint_modules:
                if (
                    getattr(mod, "__name__", "")
                    != "modules.constraints.rim_slope_match_out"
                ):
                    continue
                diagnostics_fn = getattr(mod, "matching_residual_diagnostics", None)
                if not callable(diagnostics_fn):
                    break
                diagnostics = diagnostics_fn(
                    mesh,
                    global_params,
                    mesh.positions_view(),
                )
                if not bool(diagnostics.get("available", False)):
                    break
                outer = diagnostics.get("outer_residual", {})
                inner = diagnostics.get("inner_residual", {})
                outer_max_abs = float(outer.get("max_abs", 0.0) or 0.0)
                inner_max_abs = float(inner.get("max_abs", 0.0) or 0.0)
                if max(outer_max_abs, inner_max_abs) <= residual_threshold:
                    return
                break
            constraint_manager.enforce_all(
                mesh,
                global_params=global_params,
                context="tilt_block",
            )
            mesh.project_tilts_to_tangent()
            mesh.increment_version()

        with mesh.geometry_freeze(positions):
            _apply_scaffold_interface_shape_projection()
            if hasattr(constraint_manager, "enforce_tilt_constraints"):
                constraint_manager.enforce_tilt_constraints(
                    mesh, global_params=global_params
                )

            ctx = self.energy_context_fn()
            mesh_tilts_in = mesh.tilts_in_view()
            mesh_tilts_out = mesh.tilts_out_view()
            tilts_in = ctx.scratch_array(
                "leaflet_current_tilts_in",
                shape=mesh_tilts_in.shape,
                dtype=mesh_tilts_in.dtype,
            )
            tilts_out = ctx.scratch_array(
                "leaflet_current_tilts_out",
                shape=mesh_tilts_out.shape,
                dtype=mesh_tilts_out.dtype,
            )
            normals = mesh.vertex_normals(positions)

            def _project_leaflet_tilts_in_place(
                tilts_in_arr: np.ndarray, tilts_out_arr: np.ndarray
            ) -> None:
                dot_in = np.einsum("ij,ij->i", tilts_in_arr, normals)
                tilts_in_arr -= dot_in[:, None] * normals
                dot_out = np.einsum("ij,ij->i", tilts_out_arr, normals)
                tilts_out_arr -= dot_out[:, None] * normals
                proj_in, proj_out = project_leaflet_tilts_with_optional_axisymmetry(
                    global_params=global_params,
                    positions=positions,
                    normals=normals,
                    tilts_in=tilts_in_arr,
                    tilts_out=tilts_out_arr,
                    fixed_mask_in=fixed_mask_in,
                    fixed_mask_out=fixed_mask_out,
                )
                if proj_in is not tilts_in_arr:
                    tilts_in_arr[:] = proj_in
                if proj_out is not tilts_out_arr:
                    tilts_out_arr[:] = proj_out

            def _load_leaflet_tilts_from_mesh(
                tilts_in_arr: np.ndarray, tilts_out_arr: np.ndarray
            ) -> None:
                np.copyto(tilts_in_arr, mesh.tilts_in_view())
                np.copyto(tilts_out_arr, mesh.tilts_out_view())
                _project_leaflet_tilts_in_place(tilts_in_arr, tilts_out_arr)

            _load_leaflet_tilts_from_mesh(tilts_in, tilts_out)
            grad_dummy = ctx.scratch_array(
                "leaflet_relax_grad_dummy",
                shape=positions.shape,
                dtype=positions.dtype,
            )

            tri_rows, _ = self.triangle_rows_fn()
            if tri_rows is None or len(tri_rows) == 0:
                return

            # OPTIMIZATION: Use the mesh's cached barycentric areas for the inner leaflet.
            # This is mathematically equivalent to the previous local calculation but avoids redundant O(N) work.
            tilt_vertex_areas_in = mesh.barycentric_vertex_areas(positions=positions)

            absent_mask_out = leaflet_absent_vertex_mask(
                mesh, global_params, leaflet="out"
            )

            # OPTIMIZATION: If no vertices are absent, the outer leaflet has the same vertex areas as the inner.
            if not np.any(absent_mask_out):
                tilt_vertex_areas_out = tilt_vertex_areas_in
            else:
                tri_keep_out = leaflet_present_triangle_mask(
                    mesh, tri_rows, absent_vertex_mask=absent_mask_out
                )
                tri_rows_out = tri_rows[tri_keep_out] if tri_keep_out.size else tri_rows
                tilt_vertex_areas_out = (
                    np.zeros(len(mesh.vertex_ids), dtype=float)
                    if tri_rows_out.size == 0
                    else _tilt_vertex_areas_from_triangles(
                        n_vertices=len(mesh.vertex_ids),
                        tri_rows=tri_rows_out,
                        positions=positions,
                    )
                )
            self.last_leaflet_relaxation_stats["active_outer_area_rows"] = int(
                np.sum(np.asarray(tilt_vertex_areas_out, dtype=float) > 0.0)
            )
            tilt_fixed_vals_in = (
                tilts_in[fixed_mask_in].copy() if np.any(fixed_mask_in) else None
            )
            tilt_fixed_vals_out = (
                tilts_out[fixed_mask_out].copy() if np.any(fixed_mask_out) else None
            )

            tilt_in_grad = ctx.scratch_array(
                "leaflet_tilt_in_grad",
                shape=tilts_in.shape,
                dtype=tilts_in.dtype,
            )
            tilt_out_grad = ctx.scratch_array(
                "leaflet_tilt_out_grad",
                shape=tilts_out.shape,
                dtype=tilts_out.dtype,
            )
            trial_in_buf = ctx.scratch_array(
                "leaflet_trial_in_buf",
                shape=tilts_in.shape,
                dtype=tilts_in.dtype,
            )
            trial_out_buf = ctx.scratch_array(
                "leaflet_trial_out_buf",
                shape=tilts_out.shape,
                dtype=tilts_out.dtype,
            )
            accepted_steps = 0
            projection_apply_count = 0
            projection_norm_loss_outer_far = 0.0
            projection_norm_ref_outer_far = 0.0

            outer_far_rows = np.asarray([], dtype=int)
            outer_shell_row_count = 0
            shell_rows = np.asarray([], dtype=int)
            projection_loss_radius = float(
                global_params.get("tilt_projection_loss_radius", 0.0) or 0.0
            )
            projection_loss_lambda = float(
                global_params.get("tilt_projection_loss_lambda", 0.0) or 0.0
            )
            projection_loss_outer_near_width = float(
                global_params.get("tilt_projection_loss_outer_near_width_lambda", 4.0)
                or 4.0
            )
            if projection_loss_radius > 0.0 and projection_loss_lambda > 0.0:
                rxy = np.linalg.norm(positions[:, :2], axis=1)
                outer_far_rows = np.flatnonzero(
                    rxy
                    >= (
                        projection_loss_radius
                        + projection_loss_outer_near_width * projection_loss_lambda
                    )
                )
            try:
                from modules.constraints.local_interface_shells import (
                    build_local_interface_shell_data,
                )

                shell_data = build_local_interface_shell_data(mesh, positions=positions)
                shell_rows = np.asarray(shell_data.outer_rows, dtype=int)
                outer_shell_row_count = int(len(shell_rows))
            except AssertionError:
                outer_shell_row_count = 0
            self.last_leaflet_relaxation_stats["outer_shell_row_count"] = int(
                outer_shell_row_count
            )
            region_masks = _leaflet_row_region_masks(mesh)
            outer_shell_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
            if shell_rows.size > 0:
                outer_shell_mask[shell_rows] = True

            def _capture_gradient_stats(
                *,
                key: str,
                in_grad: np.ndarray,
                out_grad: np.ndarray,
            ) -> None:
                self.last_leaflet_relaxation_stats[key] = {
                    "in": _region_l2_summary(
                        in_grad, region_masks, shell_mask=outer_shell_mask
                    ),
                    "out": _region_l2_summary(
                        out_grad, region_masks, shell_mask=outer_shell_mask
                    ),
                }

            def _capture_post_constraint_gradient_stats(
                *,
                key: str,
                in_grad: np.ndarray,
                out_grad: np.ndarray,
            ) -> None:
                self.last_leaflet_relaxation_stats[key] = {
                    "in": _region_l2_summary(
                        in_grad, region_masks, shell_mask=outer_shell_mask
                    ),
                    "out": _region_l2_summary(
                        out_grad, region_masks, shell_mask=outer_shell_mask
                    ),
                }

            def _refresh_tilts_from_constraints_and_project() -> None:
                nonlocal tilts_in, tilts_out
                nonlocal projection_apply_count
                nonlocal projection_norm_loss_outer_far, projection_norm_ref_outer_far

                before_norm = None
                if outer_far_rows.size > 0:
                    before_norm = np.linalg.norm(tilts_in[outer_far_rows], axis=1)
                self.set_leaflet_tilts_from_arrays_fast_fn(tilts_in, tilts_out)
                if hasattr(constraint_manager, "enforce_tilt_constraints"):
                    constraint_manager.enforce_tilt_constraints(
                        mesh, global_params=global_params
                    )
                _load_leaflet_tilts_from_mesh(tilts_in, tilts_out)
                projection_apply_count += 1
                if before_norm is not None:
                    after_norm = np.linalg.norm(tilts_in[outer_far_rows], axis=1)
                    projection_norm_loss_outer_far += float(
                        np.sum(np.maximum(before_norm - after_norm, 0.0))
                    )
                    projection_norm_ref_outer_far += float(np.sum(before_norm))

            def _leaflet_tilt_gradients() -> tuple[float, float]:
                E0 = self.compute_energy_and_leaflet_tilt_gradients_array_fn(
                    positions=positions,
                    tilts_in=tilts_in,
                    tilts_out=tilts_out,
                    tilt_in_grad_arr=tilt_in_grad,
                    tilt_out_grad_arr=tilt_out_grad,
                    tilt_vertex_areas_in=tilt_vertex_areas_in,
                    tilt_vertex_areas_out=tilt_vertex_areas_out,
                    grad_dummy=grad_dummy,
                    tilt_only=True,
                )
                _capture_gradient_stats(
                    key="gradient_norms_before_constraints",
                    in_grad=tilt_in_grad,
                    out_grad=tilt_out_grad,
                )
                if hasattr(
                    constraint_manager, "apply_tilt_gradient_modifications_array"
                ):
                    constraint_manager.apply_tilt_gradient_modifications_array(
                        tilt_in_grad,
                        tilt_out_grad,
                        mesh,
                        global_params,
                        positions=positions,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                    )
                if np.any(fixed_mask_in):
                    tilt_in_grad[fixed_mask_in] = 0.0
                if np.any(fixed_mask_out):
                    tilt_out_grad[fixed_mask_out] = 0.0
                _capture_post_constraint_gradient_stats(
                    key="gradient_norms_after_constraints",
                    in_grad=tilt_in_grad,
                    out_grad=tilt_out_grad,
                )

                gnorm = float(
                    np.sqrt(
                        np.sum(tilt_in_grad[~fixed_mask_in] ** 2)
                        + np.sum(tilt_out_grad[~fixed_mask_out] ** 2)
                    )
                )
                return float(E0), gnorm

            preconditioner = None
            if solver == "cg":
                preconditioner = (
                    str(
                        global_params.get("tilt_cg_preconditioner", "jacobi")
                        or "jacobi"
                    )
                    .strip()
                    .lower()
                )
                if preconditioner in ("none", "off", "false"):
                    preconditioner = None
            rejected_steps = 0
            backtracking_steps = 0
            stop_reason = "completed_max_iters"
            initial_energy = 0.0
            initial_gradient_norm = 0.0
            final_energy = 0.0
            final_gradient_norm = 0.0

            if solver == "gd":
                for _ in range(max_iters):
                    E0, gnorm = _leaflet_tilt_gradients()
                    if accepted_steps == 0 and rejected_steps == 0:
                        initial_energy = float(E0)
                        initial_gradient_norm = float(gnorm)
                        self.last_leaflet_relaxation_stats[
                            "initial_gradient_norms_before_constraints"
                        ] = dict(
                            self.last_leaflet_relaxation_stats[
                                "gradient_norms_before_constraints"
                            ]
                        )
                        self.last_leaflet_relaxation_stats[
                            "initial_gradient_norms_after_constraints"
                        ] = dict(
                            self.last_leaflet_relaxation_stats[
                                "gradient_norms_after_constraints"
                            ]
                        )
                    if gnorm == 0.0:
                        stop_reason = "zero_gradient"
                        final_energy = float(E0)
                        final_gradient_norm = float(gnorm)
                        break
                    if tol > 0.0 and gnorm < tol:
                        stop_reason = "converged"
                        final_energy = float(E0)
                        final_gradient_norm = float(gnorm)
                        break

                    step = step_size
                    accepted = False
                    for _bt in range(12):
                        if _bt > 0:
                            backtracking_steps += 1
                        delta_in_trial = -step * tilt_in_grad
                        delta_in_trial, update_mode_stats = (
                            _apply_inner_coupled_update_mode_to_delta(
                                mesh=mesh,
                                global_params=global_params,
                                positions=positions,
                                fixed_mask_in=fixed_mask_in,
                                delta_in=delta_in_trial,
                            )
                        )
                        self.last_inner_coupled_update_mode_stats = dict(
                            update_mode_stats
                        )
                        trial_in, trial_out = build_leaflet_trial_tilts(
                            base_in=tilts_in,
                            base_out=tilts_out,
                            delta_in=delta_in_trial,
                            delta_out=-step * tilt_out_grad,
                            normals=normals,
                            fixed_mask_in=fixed_mask_in,
                            fixed_mask_out=fixed_mask_out,
                            fixed_vals_in=tilt_fixed_vals_in,
                            fixed_vals_out=tilt_fixed_vals_out,
                            out_in=trial_in_buf,
                            out_out=trial_out_buf,
                        )
                        E1 = self.compute_tilt_dependent_energy_with_leaflet_tilts_fn(
                            positions=positions,
                            tilts_in=trial_in,
                            tilts_out=trial_out,
                            grad_dummy=grad_dummy,
                            tilt_vertex_areas_in=tilt_vertex_areas_in,
                            tilt_vertex_areas_out=tilt_vertex_areas_out,
                        )
                        if E1 <= E0:
                            delta_in_accepted = trial_in - tilts_in
                            delta_out_accepted = trial_out - tilts_out
                            tilts_in, trial_in_buf = trial_in, tilts_in
                            tilts_out, trial_out_buf = trial_out, tilts_out
                            accepted = True
                            break
                        step *= 0.5
                        if step < 1e-16:
                            break

                    if not accepted:
                        rejected_steps += 1
                        stop_reason = "line_search_rejected"
                        final_energy = float(E0)
                        final_gradient_norm = float(gnorm)
                        break

                    accepted_steps += 1
                    self.last_leaflet_relaxation_stats["accepted_update_norms_in"] = (
                        _region_l2_summary(
                            delta_in_accepted,
                            region_masks,
                            shell_mask=outer_shell_mask,
                        )
                    )
                    self.last_leaflet_relaxation_stats["accepted_update_norms_out"] = (
                        _region_l2_summary(
                            delta_out_accepted,
                            region_masks,
                            shell_mask=outer_shell_mask,
                        )
                    )
                    if accepted_steps == 1:
                        self.last_leaflet_relaxation_stats[
                            "initial_update_norms_in"
                        ] = dict(
                            self.last_leaflet_relaxation_stats[
                                "accepted_update_norms_in"
                            ]
                        )
                        self.last_leaflet_relaxation_stats[
                            "initial_update_norms_out"
                        ] = dict(
                            self.last_leaflet_relaxation_stats[
                                "accepted_update_norms_out"
                            ]
                        )
                    if accepted_steps == 1:
                        self.last_leaflet_relaxation_stats[
                            "step_size_first_accepted"
                        ] = float(step)
                        self.last_leaflet_relaxation_stats["step_size_min_accepted"] = (
                            float(step)
                        )
                        self.last_leaflet_relaxation_stats["step_size_max_accepted"] = (
                            float(step)
                        )
                    else:
                        self.last_leaflet_relaxation_stats["step_size_min_accepted"] = (
                            float(
                                min(
                                    float(
                                        self.last_leaflet_relaxation_stats[
                                            "step_size_min_accepted"
                                        ]
                                    ),
                                    float(step),
                                )
                            )
                        )
                        self.last_leaflet_relaxation_stats["step_size_max_accepted"] = (
                            float(
                                max(
                                    float(
                                        self.last_leaflet_relaxation_stats[
                                            "step_size_max_accepted"
                                        ]
                                    ),
                                    float(step),
                                )
                            )
                        )
                    self.last_leaflet_relaxation_stats["step_size_last_accepted"] = (
                        float(step)
                    )
                    if (
                        projection_cadence == "per_step"
                        and (accepted_steps % projection_interval) == 0
                    ):
                        _refresh_tilts_from_constraints_and_project()
                    final_energy = float(E1)
                    final_gradient_norm = float(gnorm)
            else:
                # CG path
                E0, gnorm = _leaflet_tilt_gradients()
                initial_energy = float(E0)
                initial_gradient_norm = float(gnorm)
                self.last_leaflet_relaxation_stats[
                    "initial_gradient_norms_before_constraints"
                ] = dict(
                    self.last_leaflet_relaxation_stats[
                        "gradient_norms_before_constraints"
                    ]
                )
                self.last_leaflet_relaxation_stats[
                    "initial_gradient_norms_after_constraints"
                ] = dict(
                    self.last_leaflet_relaxation_stats[
                        "gradient_norms_after_constraints"
                    ]
                )
                if gnorm == 0.0 or (tol > 0.0 and gnorm < tol):
                    stop_reason = "zero_gradient" if gnorm == 0.0 else "converged"
                    final_energy = float(E0)
                    final_gradient_norm = float(gnorm)
                    self.set_leaflet_tilts_from_arrays_fast_fn(tilts_in, tilts_out)
                    self.last_leaflet_relaxation_stats.update(
                        {
                            "accepted_steps": int(accepted_steps),
                            "rejected_steps": int(rejected_steps),
                            "backtracking_steps": int(backtracking_steps),
                            "stop_reason": str(stop_reason),
                            "initial_energy": float(initial_energy),
                            "final_energy": float(final_energy),
                            "initial_gradient_norm": float(initial_gradient_norm),
                            "final_gradient_norm": float(final_gradient_norm),
                        }
                    )
                    return

                residual_in = -tilt_in_grad
                residual_out = -tilt_out_grad

                M_inv_in = None
                M_inv_out = None
                if preconditioner == "jacobi":
                    from runtime.preconditioners import (
                        build_leaflet_tilt_cg_preconditioner,
                    )

                    M_inv_in, M_inv_out = build_leaflet_tilt_cg_preconditioner(
                        mesh,
                        self.param_resolver,
                        self.energy_context_fn(),
                        positions=positions,
                        index_map=mesh.vertex_index_to_row,
                        fixed_mask_in=fixed_mask_in,
                        fixed_mask_out=fixed_mask_out,
                        tilt_vertex_areas_in=tilt_vertex_areas_in,
                        tilt_vertex_areas_out=tilt_vertex_areas_out,
                    )
                    self.last_leaflet_relaxation_stats["preconditioner_mean_inv_in"] = (
                        _region_mean_summary(
                            M_inv_in, region_masks, shell_mask=outer_shell_mask
                        )
                    )
                    self.last_leaflet_relaxation_stats[
                        "preconditioner_mean_inv_out"
                    ] = _region_mean_summary(
                        M_inv_out, region_masks, shell_mask=outer_shell_mask
                    )

                if M_inv_in is not None:
                    z_in = residual_in * M_inv_in[:, None]
                    z_out = residual_out * M_inv_out[:, None]
                else:
                    z_in = residual_in
                    z_out = residual_out

                direction_in = z_in.copy()
                direction_out = z_out.copy()
                rz_old = float(
                    np.sum(residual_in * z_in) + np.sum(residual_out * z_out)
                )

                for _ in range(max_iters):
                    if gnorm == 0.0:
                        stop_reason = "zero_gradient"
                        final_energy = float(E0)
                        final_gradient_norm = float(gnorm)
                        break
                    if tol > 0.0 and gnorm < tol:
                        stop_reason = "converged"
                        final_energy = float(E0)
                        final_gradient_norm = float(gnorm)
                        break

                    step = step_size
                    accepted = False
                    fallback_used = False
                    for _bt in range(12):
                        if _bt > 0:
                            backtracking_steps += 1
                        delta_in_trial = step * direction_in
                        delta_in_trial, update_mode_stats = (
                            _apply_inner_coupled_update_mode_to_delta(
                                mesh=mesh,
                                global_params=global_params,
                                positions=positions,
                                fixed_mask_in=fixed_mask_in,
                                delta_in=delta_in_trial,
                            )
                        )
                        self.last_inner_coupled_update_mode_stats = dict(
                            update_mode_stats
                        )
                        trial_in, trial_out = build_leaflet_trial_tilts(
                            base_in=tilts_in,
                            base_out=tilts_out,
                            delta_in=delta_in_trial,
                            delta_out=step * direction_out,
                            normals=normals,
                            fixed_mask_in=fixed_mask_in,
                            fixed_mask_out=fixed_mask_out,
                            fixed_vals_in=tilt_fixed_vals_in,
                            fixed_vals_out=tilt_fixed_vals_out,
                            out_in=trial_in_buf,
                            out_out=trial_out_buf,
                        )
                        E1 = self.compute_tilt_dependent_energy_with_leaflet_tilts_fn(
                            positions=positions,
                            tilts_in=trial_in,
                            tilts_out=trial_out,
                            grad_dummy=grad_dummy,
                            tilt_vertex_areas_in=tilt_vertex_areas_in,
                            tilt_vertex_areas_out=tilt_vertex_areas_out,
                        )
                        if E1 <= E0:
                            delta_in_accepted = trial_in - tilts_in
                            delta_out_accepted = trial_out - tilts_out
                            tilts_in, trial_in_buf = trial_in, tilts_in
                            tilts_out, trial_out_buf = trial_out, tilts_out
                            E0 = E1
                            accepted = True
                            break
                        step *= 0.5
                        if step < 1e-16:
                            break

                    if not accepted:
                        if cg_rejection_fallback == "gd":
                            self.last_leaflet_relaxation_stats[
                                "cg_fallback_attempted_count"
                            ] = (
                                int(
                                    self.last_leaflet_relaxation_stats[
                                        "cg_fallback_attempted_count"
                                    ]
                                )
                                + 1
                            )
                            step = step_size
                            for _bt in range(12):
                                if _bt > 0:
                                    backtracking_steps += 1
                                delta_in_trial = -step * tilt_in_grad
                                delta_in_trial, update_mode_stats = (
                                    _apply_inner_coupled_update_mode_to_delta(
                                        mesh=mesh,
                                        global_params=global_params,
                                        positions=positions,
                                        fixed_mask_in=fixed_mask_in,
                                        delta_in=delta_in_trial,
                                    )
                                )
                                self.last_inner_coupled_update_mode_stats = dict(
                                    update_mode_stats
                                )
                                trial_in, trial_out = build_leaflet_trial_tilts(
                                    base_in=tilts_in,
                                    base_out=tilts_out,
                                    delta_in=delta_in_trial,
                                    delta_out=-step * tilt_out_grad,
                                    normals=normals,
                                    fixed_mask_in=fixed_mask_in,
                                    fixed_mask_out=fixed_mask_out,
                                    fixed_vals_in=tilt_fixed_vals_in,
                                    fixed_vals_out=tilt_fixed_vals_out,
                                    out_in=trial_in_buf,
                                    out_out=trial_out_buf,
                                )
                                E1 = self.compute_tilt_dependent_energy_with_leaflet_tilts_fn(
                                    positions=positions,
                                    tilts_in=trial_in,
                                    tilts_out=trial_out,
                                    grad_dummy=grad_dummy,
                                    tilt_vertex_areas_in=tilt_vertex_areas_in,
                                    tilt_vertex_areas_out=tilt_vertex_areas_out,
                                )
                                if E1 <= E0:
                                    delta_in_accepted = trial_in - tilts_in
                                    delta_out_accepted = trial_out - tilts_out
                                    tilts_in, trial_in_buf = trial_in, tilts_in
                                    tilts_out, trial_out_buf = trial_out, tilts_out
                                    E0 = E1
                                    accepted = True
                                    fallback_used = True
                                    break
                                step *= 0.5
                                if step < 1e-16:
                                    break

                        if not accepted:
                            rejected_steps += 1
                            stop_reason = "line_search_rejected"
                            final_energy = float(E0)
                            final_gradient_norm = float(gnorm)
                            break

                    accepted_steps += 1
                    self.last_leaflet_relaxation_stats["accepted_update_norms_in"] = (
                        _region_l2_summary(
                            delta_in_accepted,
                            region_masks,
                            shell_mask=outer_shell_mask,
                        )
                    )
                    self.last_leaflet_relaxation_stats["accepted_update_norms_out"] = (
                        _region_l2_summary(
                            delta_out_accepted,
                            region_masks,
                            shell_mask=outer_shell_mask,
                        )
                    )
                    if fallback_used:
                        self.last_leaflet_relaxation_stats[
                            "cg_fallback_accepted_count"
                        ] = (
                            int(
                                self.last_leaflet_relaxation_stats[
                                    "cg_fallback_accepted_count"
                                ]
                            )
                            + 1
                        )
                        self.last_leaflet_relaxation_stats[
                            "cg_fallback_step_size_last"
                        ] = float(step)
                        self.last_leaflet_relaxation_stats[
                            "cg_fallback_update_norms_in"
                        ] = dict(
                            self.last_leaflet_relaxation_stats[
                                "accepted_update_norms_in"
                            ]
                        )
                        self.last_leaflet_relaxation_stats[
                            "cg_fallback_update_norms_out"
                        ] = dict(
                            self.last_leaflet_relaxation_stats[
                                "accepted_update_norms_out"
                            ]
                        )
                    if accepted_steps == 1:
                        self.last_leaflet_relaxation_stats[
                            "initial_update_norms_in"
                        ] = dict(
                            self.last_leaflet_relaxation_stats[
                                "accepted_update_norms_in"
                            ]
                        )
                        self.last_leaflet_relaxation_stats[
                            "initial_update_norms_out"
                        ] = dict(
                            self.last_leaflet_relaxation_stats[
                                "accepted_update_norms_out"
                            ]
                        )
                    if accepted_steps == 1:
                        self.last_leaflet_relaxation_stats[
                            "step_size_first_accepted"
                        ] = float(step)
                        self.last_leaflet_relaxation_stats["step_size_min_accepted"] = (
                            float(step)
                        )
                        self.last_leaflet_relaxation_stats["step_size_max_accepted"] = (
                            float(step)
                        )
                    else:
                        self.last_leaflet_relaxation_stats["step_size_min_accepted"] = (
                            float(
                                min(
                                    float(
                                        self.last_leaflet_relaxation_stats[
                                            "step_size_min_accepted"
                                        ]
                                    ),
                                    float(step),
                                )
                            )
                        )
                        self.last_leaflet_relaxation_stats["step_size_max_accepted"] = (
                            float(
                                max(
                                    float(
                                        self.last_leaflet_relaxation_stats[
                                            "step_size_max_accepted"
                                        ]
                                    ),
                                    float(step),
                                )
                            )
                        )
                    self.last_leaflet_relaxation_stats["step_size_last_accepted"] = (
                        float(step)
                    )
                    if (
                        projection_cadence == "per_step"
                        and (accepted_steps % projection_interval) == 0
                    ):
                        _refresh_tilts_from_constraints_and_project()

                    E0, gnorm = _leaflet_tilt_gradients()
                    if gnorm == 0.0 or (tol > 0.0 and gnorm < tol):
                        stop_reason = "zero_gradient" if gnorm == 0.0 else "converged"
                        final_energy = float(E0)
                        final_gradient_norm = float(gnorm)
                        break

                    residual_in = -tilt_in_grad
                    residual_out = -tilt_out_grad
                    if M_inv_in is not None:
                        z_in = residual_in * M_inv_in[:, None]
                        z_out = residual_out * M_inv_out[:, None]
                    else:
                        z_in = residual_in
                        z_out = residual_out

                    rz_new = float(
                        np.sum(residual_in * z_in) + np.sum(residual_out * z_out)
                    )
                    if fallback_used:
                        direction_in = z_in.copy()
                        direction_out = z_out.copy()
                        rz_old = rz_new
                        final_energy = float(E0)
                        final_gradient_norm = float(gnorm)
                        continue
                    if rz_old == 0.0:
                        stop_reason = "cg_breakdown"
                        final_energy = float(E0)
                        final_gradient_norm = float(gnorm)
                        break
                    beta = rz_new / rz_old
                    direction_in = z_in + beta * direction_in
                    direction_out = z_out + beta * direction_out
                    rz_old = rz_new
                    final_energy = float(E0)
                    final_gradient_norm = float(gnorm)

            if projection_cadence == "per_pass":
                _refresh_tilts_from_constraints_and_project()

            self.last_tilt_projection_stats["projection_apply_count"] = (
                projection_apply_count
            )
            self.last_tilt_projection_stats["tilt_projection_norm_loss_outer_far"] = (
                projection_norm_loss_outer_far
            )
            if accepted_steps == 0 and rejected_steps == 0:
                final_energy, final_gradient_norm = _leaflet_tilt_gradients()
                initial_energy = float(final_energy)
                initial_gradient_norm = float(final_gradient_norm)
                stop_reason = "completed_without_steps"
                self.last_leaflet_relaxation_stats[
                    "initial_gradient_norms_before_constraints"
                ] = dict(
                    self.last_leaflet_relaxation_stats[
                        "gradient_norms_before_constraints"
                    ]
                )
                self.last_leaflet_relaxation_stats[
                    "initial_gradient_norms_after_constraints"
                ] = dict(
                    self.last_leaflet_relaxation_stats[
                        "gradient_norms_after_constraints"
                    ]
                )
            self.last_leaflet_relaxation_stats[
                "final_gradient_norms_before_constraints"
            ] = dict(
                self.last_leaflet_relaxation_stats["gradient_norms_before_constraints"]
            )
            self.last_leaflet_relaxation_stats[
                "final_gradient_norms_after_constraints"
            ] = dict(
                self.last_leaflet_relaxation_stats["gradient_norms_after_constraints"]
            )
            self.last_leaflet_relaxation_stats.update(
                {
                    "accepted_steps": int(accepted_steps),
                    "rejected_steps": int(rejected_steps),
                    "backtracking_steps": int(backtracking_steps),
                    "stop_reason": str(stop_reason),
                    "initial_energy": float(initial_energy),
                    "final_energy": float(final_energy),
                    "initial_gradient_norm": float(initial_gradient_norm),
                    "final_gradient_norm": float(final_gradient_norm),
                    "projection_apply_count": int(projection_apply_count),
                    "tilt_projection_norm_loss_outer_far": float(
                        projection_norm_loss_outer_far
                    ),
                    "tilt_projection_norm_ref_outer_far": float(
                        projection_norm_ref_outer_far
                    ),
                }
            )

        self.set_leaflet_tilts_from_arrays_fast_fn(tilts_in, tilts_out)

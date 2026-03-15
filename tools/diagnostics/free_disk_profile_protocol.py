#!/usr/bin/env python3
"""Reusable two-stage protocol for free-disk coupled profile diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from geometry.curvature import compute_curvature_data
from geometry.geom_io import load_data, parse_geometry
from geometry.tilt_operators import _resolve_transport_model, p1_triangle_divergence
from modules.energy import bending_tilt_leaflet as _bending_tilt_leaflet
from modules.energy.leaflet_presence import (
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.refinement import refine_triangle_mesh
from runtime.steppers.gradient_descent import GradientDescent

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FREE_DISK_FIXTURE = (
    ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
)
DEFAULT_FREE_DISK_CURVED_BILAYER_SOURCE = (
    ROOT
    / "meshes"
    / "caveolin"
    / "kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12_free_disk.yaml"
)


def _apply_global_parameter_overrides(
    mesh, overrides: dict[str, object] | None
) -> None:
    """Apply optional global-parameter overrides to ``mesh`` in place."""
    if not overrides:
        return
    gp = mesh.global_parameters
    for key, value in overrides.items():
        gp.set(str(key), value)


def _configure_theta_scan(mesh) -> Minimizer:
    gp = mesh.global_parameters
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 10)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)
    gp.set("tilt_thetaB_optimize", True)
    gp.set("tilt_thetaB_optimize_every", 1)
    gp.set("tilt_thetaB_optimize_delta", 0.02)
    gp.set("tilt_thetaB_optimize_inner_steps", 2)
    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 1.0e-3)
    return Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )


def _configure_shape_relax(mesh, *, theta_b: float) -> Minimizer:
    gp = mesh.global_parameters
    gp.set("tilt_solve_mode", "coupled")
    gp.set("tilt_solver", "gd")
    gp.set("tilt_step_size", 0.15)
    gp.set("tilt_inner_steps", 10)
    gp.set("tilt_tol", 1e-8)
    gp.set("tilt_kkt_projection_during_relaxation", False)
    gp.set("tilt_thetaB_optimize", False)
    gp.set("tilt_thetaB_value", float(theta_b))
    gp.set("step_size_mode", "fixed")
    gp.set("step_size", 0.01)
    return Minimizer(
        mesh,
        gp,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )


def _energy_total(breakdown: dict[str, float]) -> float:
    """Return total energy from an energy-breakdown mapping."""
    return float(sum(float(v) for v in breakdown.values()))


def _triangle_region_masks(
    mesh,
    tri_rows_eff: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return standard free-disk triangle region masks."""
    rim_rows: set[int] = set()
    outer_rows: set[int] = set()
    disk_rows: set[int] = set()
    for vid in mesh.vertex_ids:
        row = mesh.vertex_index_to_row[int(vid)]
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if str(opts.get("preset") or "") == "disk":
            disk_rows.add(row)
        if str(opts.get("rim_slope_match_group") or "") == "rim":
            rim_rows.add(row)
        if str(opts.get("rim_slope_match_group") or "") == "outer":
            outer_rows.add(row)

    has_disk = np.array(
        [any(int(row) in disk_rows for row in tri) for tri in tri_rows_eff], dtype=bool
    )
    has_rim = np.array(
        [any(int(row) in rim_rows for row in tri) for tri in tri_rows_eff], dtype=bool
    )
    has_outer = np.array(
        [any(int(row) in outer_rows for row in tri) for tri in tri_rows_eff], dtype=bool
    )

    return {
        "disk_core": has_disk & (~has_rim) & (~has_outer),
        "disk_rim": has_disk & has_rim & (~has_outer),
        "rim_outer": has_rim & has_outer & (~has_disk),
        "outer_support_band": has_outer & (~has_rim) & (~has_disk),
        "outer_far": (~has_disk) & (~has_rim) & (~has_outer),
        "outer_membrane": (~has_disk) & (~has_rim),
    }


def _tilt_leaflet_region_split(mesh, *, leaflet: str) -> dict[str, float]:
    """Return a regional split of a leaflet tilt-magnitude energy."""
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return {"disk_core": 0.0, "near_rim": 0.0, "outer_membrane": 0.0}

    absent_mask = leaflet_absent_vertex_mask(
        mesh, mesh.global_parameters, leaflet=leaflet
    )
    tri_keep = leaflet_present_triangle_mask(
        mesh, tri_rows, absent_vertex_mask=absent_mask
    )
    if tri_keep.size and not np.any(tri_keep):
        return {"disk_core": 0.0, "near_rim": 0.0, "outer_membrane": 0.0}

    tri_rows_eff = tri_rows[tri_keep] if tri_keep.size else tri_rows
    tri_pos = positions[tri_rows_eff]
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]
    n = np.cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm >= 1.0e-12
    if not np.any(mask):
        return {"disk_core": 0.0, "near_rim": 0.0, "outer_membrane": 0.0}

    tri_rows_eff = tri_rows_eff[mask]
    areas = 0.5 * n_norm[mask]
    if str(leaflet) == "out":
        tilts = mesh.tilts_out_view().copy()
        k_tilt = float(mesh.global_parameters.get("tilt_modulus_out") or 0.0)
    else:
        tilts = mesh.tilts_in_view().copy()
        k_tilt = float(mesh.global_parameters.get("tilt_modulus_in") or 0.0)
    gp = mesh.global_parameters
    if str(leaflet) == "out":
        exclude_shared_rim = bool(
            gp.get("tilt_out_exclude_shared_rim_outer_rows")
            or gp.get("tilt_out_exclude_shared_rim_rows")
        )
        target_groups = {"outer"}
    else:
        target_groups: set[str] = set()
        if bool(
            gp.get("tilt_in_exclude_shared_rim_rows")
            or gp.get("tilt_exclude_shared_rim_rows_in")
        ):
            target_groups.add("rim")
        if bool(
            gp.get("tilt_in_exclude_shared_rim_outer_rows")
            or gp.get("tilt_exclude_shared_rim_outer_rows_in")
        ):
            target_groups.add("outer")
        exclude_shared_rim = bool(target_groups)
    match_mode = str(gp.get("rim_slope_match_mode") or "").strip().lower()
    if exclude_shared_rim and match_mode == "shared_rim_staggered_v1":
        for row, vid in enumerate(mesh.vertex_ids):
            opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
            if str(opts.get("rim_slope_match_group") or "") in target_groups:
                tilts[row] = 0.0
    if str(leaflet) == "in" and match_mode == "shared_rim_staggered_v1":
        outer_row_energy_weight = gp.get("tilt_in_shared_rim_outer_row_energy_weight")
        if outer_row_energy_weight is not None:
            outer_row_energy_weight = float(outer_row_energy_weight)
            if outer_row_energy_weight < 0.0 or not np.isfinite(
                outer_row_energy_weight
            ):
                raise ValueError(
                    "tilt_in_shared_rim_outer_row_energy_weight must be a finite nonnegative float."
                )
            outer_row_scale = float(np.sqrt(outer_row_energy_weight))
            for row, vid in enumerate(mesh.vertex_ids):
                opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
                if str(opts.get("rim_slope_match_group") or "") == "outer":
                    tilts[row] *= outer_row_scale
    tri_tilt_sq_sum = np.sum(
        np.einsum("...i,...i->...", tilts[tri_rows_eff], tilts[tri_rows_eff]),
        axis=1,
    )
    if str(leaflet) == "in":
        mode = (
            str(gp.get("tilt_mass_mode_in") or gp.get("tilt_mass_mode") or "lumped")
            .strip()
            .lower()
        )
        shell_mode = gp.get("tilt_in_shared_rim_outer_shell_mass_mode")
        shell_mode = None if shell_mode is None else str(shell_mode).strip().lower()
    else:
        mode = (
            str(gp.get("tilt_mass_mode_out") or gp.get("tilt_mass_mode") or "lumped")
            .strip()
            .lower()
        )
        shell_mode = None
    if mode not in {"lumped", "consistent"}:
        raise ValueError("Leaflet tilt mass mode must be 'lumped' or 'consistent'.")
    if shell_mode not in {None, "lumped", "consistent"}:
        raise ValueError(
            "tilt_in_shared_rim_outer_shell_mass_mode must be 'lumped' or 'consistent'."
        )

    consistent_s = (
        tri_tilt_sq_sum
        + np.einsum("ij,ij->i", tilts[tri_rows_eff[:, 0]], tilts[tri_rows_eff[:, 1]])
        + np.einsum("ij,ij->i", tilts[tri_rows_eff[:, 1]], tilts[tri_rows_eff[:, 2]])
        + np.einsum("ij,ij->i", tilts[tri_rows_eff[:, 2]], tilts[tri_rows_eff[:, 0]])
    )
    region_masks = _triangle_region_masks(mesh, tri_rows_eff)
    outer_support_mask = region_masks["outer_support_band"]
    use_consistent = np.full(len(tri_rows_eff), mode == "consistent", dtype=bool)
    if str(leaflet) == "in" and shell_mode is not None:
        use_consistent[outer_support_mask] = shell_mode == "consistent"
    coeff = np.empty(len(tri_rows_eff), dtype=float)
    use_lumped = ~use_consistent
    coeff[use_lumped] = 0.5 * k_tilt * (tri_tilt_sq_sum[use_lumped] / 3.0)
    coeff[use_consistent] = (k_tilt / 12.0) * consistent_s[use_consistent]
    tri_energy = coeff * areas

    return {key: float(np.sum(tri_energy[mask])) for key, mask in region_masks.items()}


def _tilt_in_region_split(mesh) -> dict[str, float]:
    """Return a regional split of the inner-leaflet tilt energy."""
    return _tilt_leaflet_region_split(mesh, leaflet="in")


def _tilt_out_region_split(mesh) -> dict[str, float]:
    """Return a regional split of the outer-leaflet tilt energy."""
    return _tilt_leaflet_region_split(mesh, leaflet="out")


def _shared_rim_group_rows(mesh, group: str) -> np.ndarray:
    """Return row indices carrying a given shared-rim group tag."""
    rows: list[int] = []
    for row, vid in enumerate(mesh.vertex_ids):
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if str(opts.get("rim_slope_match_group") or "") == str(group):
            rows.append(int(row))
    return np.asarray(rows, dtype=int)


def _shared_rim_inner_control_volume_audit(mesh) -> dict[str, float]:
    """Return barycentric inner-leaflet area carried by shared-rim support groups."""
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return {
            "outer_control_area": 0.0,
            "rim_control_area": 0.0,
            "total_control_area": 0.0,
        }
    absent_mask = leaflet_absent_vertex_mask(mesh, mesh.global_parameters, leaflet="in")
    tri_keep = leaflet_present_triangle_mask(
        mesh, tri_rows, absent_vertex_mask=absent_mask
    )
    tri_rows_eff = tri_rows[tri_keep] if tri_keep.size else tri_rows
    if tri_rows_eff.size == 0:
        return {
            "outer_control_area": 0.0,
            "rim_control_area": 0.0,
            "total_control_area": 0.0,
        }
    tri_pos = positions[tri_rows_eff]
    n = np.cross(
        tri_pos[:, 1, :] - tri_pos[:, 0, :], tri_pos[:, 2, :] - tri_pos[:, 0, :]
    )
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm >= 1.0e-12
    if not np.any(mask):
        return {
            "outer_control_area": 0.0,
            "rim_control_area": 0.0,
            "total_control_area": 0.0,
        }
    areas = 0.5 * n_norm[mask]
    tri_rows_eff = tri_rows_eff[mask]
    vertex_areas = mesh.barycentric_vertex_areas(
        positions,
        tri_rows=tri_rows_eff,
        areas=areas,
        mask=np.ones(len(tri_rows_eff), dtype=bool),
        cache=False,
    )
    outer_rows = _shared_rim_group_rows(mesh, "outer")
    rim_rows = _shared_rim_group_rows(mesh, "rim")
    return {
        "outer_control_area": float(np.sum(vertex_areas[outer_rows]))
        if outer_rows.size
        else 0.0,
        "rim_control_area": float(np.sum(vertex_areas[rim_rows]))
        if rim_rows.size
        else 0.0,
        "total_control_area": float(np.sum(vertex_areas)),
    }


def shared_rim_continuum_annulus_audit(mesh) -> dict[str, float]:
    """Return simple continuum annulus targets associated with the shared-rim rows.

    The audit uses:
    - the physical rim radius from the current ``rim``-tagged band
    - the activated ``outer`` support-ring radius
    - the midpoint annulus split between those two radii

    This is not a correction rule; it is a diagnostic comparison target.
    """
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    rim_rows = _shared_rim_group_rows(mesh, "rim")
    outer_rows = _shared_rim_group_rows(mesh, "outer")
    if rim_rows.size == 0 or outer_rows.size == 0:
        raise AssertionError(
            "Shared-rim annulus audit requires both rim and outer groups"
        )
    rim_r_min = float(np.min(r[rim_rows]))
    rim_r_max = float(np.max(r[rim_rows]))
    outer_r = float(np.median(r[outer_rows]))
    split_r = 0.5 * (rim_r_max + outer_r)
    rim_annulus_area = float(
        np.pi * max(split_r * split_r - rim_r_max * rim_r_max, 0.0)
    )
    outer_annulus_area = float(np.pi * max(outer_r * outer_r - split_r * split_r, 0.0))
    return {
        "rim_r_min": rim_r_min,
        "rim_r_max": rim_r_max,
        "outer_r": outer_r,
        "split_r": split_r,
        "rim_annulus_area": rim_annulus_area,
        "outer_annulus_area": outer_annulus_area,
    }


def shared_rim_shell_area_audit(mesh) -> dict[str, float]:
    """Return adjacent-ring shell areas represented by the shared-rim rows.

    This treats the current ``rim`` and activated ``outer`` rows as discrete ring
    samples whose natural control volumes extend to midpoints with neighboring
    rings, rather than only across the narrow gap between ``R`` and ``R+``.
    """
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    rim_rows = _shared_rim_group_rows(mesh, "rim")
    outer_rows = _shared_rim_group_rows(mesh, "outer")
    if rim_rows.size == 0 or outer_rows.size == 0:
        raise AssertionError(
            "Shared-rim shell audit requires both rim and outer groups"
        )

    disk_rows: list[int] = []
    for row, vid in enumerate(mesh.vertex_ids):
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("preset") == "disk":
            disk_rows.append(int(row))
    disk_rows_arr = np.asarray(disk_rows, dtype=int)
    if disk_rows_arr.size == 0:
        raise AssertionError("Shared-rim shell audit requires disk preset rows")

    rim_r_min = float(np.min(r[rim_rows]))
    rim_r_max = float(np.max(r[rim_rows]))
    outer_r = float(np.median(r[outer_rows]))
    disk_unique = sorted(
        {float(rr) for rr in r[disk_rows_arr] if rr < rim_r_min - 1.0e-6}
    )
    if not disk_unique:
        raise AssertionError("No disk ring found inside the shared rim")
    disk_prev_r = float(disk_unique[-1])
    outer_unique = sorted({float(rr) for rr in r if rr > outer_r + 1.0e-6})
    if not outer_unique:
        raise AssertionError("No outer ring found beyond the activated support ring")
    next_outer_r = float(outer_unique[0])

    rim_inner_boundary = 0.5 * (disk_prev_r + rim_r_min)
    rim_outer_boundary = 0.5 * (rim_r_max + outer_r)
    outer_outer_boundary = 0.5 * (outer_r + next_outer_r)

    rim_shell_area = float(
        np.pi
        * max(
            rim_outer_boundary * rim_outer_boundary
            - rim_inner_boundary * rim_inner_boundary,
            0.0,
        )
    )
    outer_shell_area = float(
        np.pi
        * max(
            outer_outer_boundary * outer_outer_boundary
            - rim_outer_boundary * rim_outer_boundary,
            0.0,
        )
    )
    return {
        "disk_prev_r": disk_prev_r,
        "next_outer_r": next_outer_r,
        "rim_shell_inner_boundary": rim_inner_boundary,
        "rim_shell_outer_boundary": rim_outer_boundary,
        "outer_shell_outer_boundary": outer_outer_boundary,
        "rim_shell_area": rim_shell_area,
        "outer_shell_area": outer_shell_area,
    }


def _bending_tilt_leaflet_region_split(mesh, *, leaflet: str) -> dict[str, float]:
    """Return a regional split of leaflet bending-tilt coupling energy."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    gp = mesh.global_parameters
    cache_tag = "out" if str(leaflet) == "out" else "in"
    kappa_key = "bending_modulus_out" if cache_tag == "out" else "bending_modulus_in"
    tilts = mesh.tilts_out_view() if cache_tag == "out" else mesh.tilts_in_view()

    k_vecs, vertex_areas_vor, weights, tri_rows = compute_curvature_data(
        mesh, positions, index_map
    )
    if tri_rows.size == 0:
        return {
            "disk_core": 0.0,
            "disk_rim": 0.0,
            "rim_outer": 0.0,
            "outer_membrane": 0.0,
        }

    absent_mask = leaflet_absent_vertex_mask(mesh, gp, leaflet=cache_tag)
    tri_keep = leaflet_present_triangle_mask(
        mesh, tri_rows, absent_vertex_mask=absent_mask
    )
    if tri_keep.size:
        tri_rows = tri_rows[tri_keep]
        weights = weights[tri_keep]
        if tri_rows.size == 0:
            return {
                "disk_core": 0.0,
                "disk_rim": 0.0,
                "rim_outer": 0.0,
                "outer_membrane": 0.0,
            }

    transport_model = _resolve_transport_model(
        gp.get("tilt_transport_model", "ambient_v1") if gp is not None else "ambient_v1"
    )
    div_tri, _, _, _, _ = p1_triangle_divergence(
        mesh=mesh,
        positions=positions,
        tilts=tilts,
        tri_rows=tri_rows,
        transport_model=transport_model,
    )
    div_term = _bending_tilt_leaflet._apply_inner_divergence_update_mode(
        mesh,
        gp,
        positions=positions,
        tri_rows=tri_rows,
        cache_tag=cache_tag,
        div_term=div_tri,
    )
    _, va0_eff, va1_eff, va2_eff = _bending_tilt_leaflet._compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token=f"diagnostic_bending_tilt_{cache_tag}",
    )
    safe_areas_vor = np.maximum(vertex_areas_vor, 1.0e-12)
    kappa_arr, c0_arr = _bending_tilt_leaflet._per_vertex_params_leaflet(
        mesh,
        gp,
        model="helfrich",
        kappa_key=kappa_key,
        cache_tag=cache_tag,
    )
    k_mag = np.linalg.norm(k_vecs, axis=1)
    h_vor = k_mag / (2.0 * safe_areas_vor)
    is_interior = _bending_tilt_leaflet._interior_mask_leaflet(
        mesh, gp, cache_tag=cache_tag, index_map=index_map
    )
    base_term = (2.0 * h_vor) - c0_arr
    base_term[~is_interior] = 0.0
    presets = _bending_tilt_leaflet._assume_J0_presets(gp, cache_tag=cache_tag)
    if presets:
        radius_max = _bending_tilt_leaflet._assume_J0_radius_max(
            gp, cache_tag=cache_tag
        )
        center_xy = _bending_tilt_leaflet._assume_J0_center_xy(gp)
        rows = _bending_tilt_leaflet._collect_preset_rows(
            mesh,
            presets=presets,
            cache_tag=cache_tag,
            index_map=index_map,
            radius_max=radius_max,
            center_xy=center_xy,
        )
        if rows.size:
            base_term[rows] = 0.0
    region_rows = _bending_tilt_leaflet._base_term_region_zero_rows(
        mesh,
        gp,
        cache_tag=cache_tag,
        index_map=index_map,
    )
    if region_rows.size:
        base_term[region_rows] = 0.0

    base_tri = base_term[tri_rows]
    term_tri = base_tri + div_term[:, None]
    kappa_tri = kappa_arr[tri_rows]
    tri_energy = 0.5 * (
        (kappa_tri[:, 0] * term_tri[:, 0] ** 2 * va0_eff)
        + (kappa_tri[:, 1] * term_tri[:, 1] ** 2 * va1_eff)
        + (kappa_tri[:, 2] * term_tri[:, 2] ** 2 * va2_eff)
    )
    region_masks = _triangle_region_masks(mesh, tri_rows)
    return {key: float(np.sum(tri_energy[mask])) for key, mask in region_masks.items()}


def load_free_disk_theory_mesh(path: str | Path | None = None):
    """Return the canonical free-disk theory fixture as a parsed mesh."""
    mesh_path = Path(path) if path is not None else DEFAULT_FREE_DISK_FIXTURE
    return parse_geometry(load_data(str(mesh_path)))


def load_free_disk_curved_bilayer_mesh(path: str | Path | None = None):
    """Return the refined curved free-disk mesh configured for bilayer parity.

    This reuses the refined free-disk geometry with the physical rim at
    ``R = 7/15`` and a resolved first free ring just outside the disk. The
    returned mesh is adjusted in-memory so the outer leaflet is present on the
    disk and the truncated far ring can slide collectively along ``z`` while
    keeping its radius fixed.
    """
    mesh_path = (
        Path(path) if path is not None else DEFAULT_FREE_DISK_CURVED_BILAYER_SOURCE
    )
    mesh = parse_geometry(load_data(str(mesh_path)))
    gp = mesh.global_parameters
    gp.set("leaflet_out_absent_presets", [])

    defs = getattr(mesh, "definitions", None)
    if isinstance(defs, dict):
        outer_def = defs.get("outer_rim")
        if isinstance(outer_def, dict):
            outer_def["pin_to_plane_mode"] = "slide"
            outer_def["pin_to_plane_group"] = "outer_height_gauge"
            outer_def["pin_to_circle_mode"] = "slide"
            outer_def["pin_to_circle_group"] = "outer"
            outer_def["pin_to_circle_normal"] = [0.0, 0.0, 1.0]
            outer_def["pin_to_circle_point"] = [0.0, 0.0, 0.0]

    for vid in mesh.vertex_ids:
        vertex = mesh.vertices[int(vid)]
        opts = getattr(vertex, "options", None) or {}
        if (
            opts.get("preset") == "outer_rim"
            or opts.get("pin_to_circle_group") == "outer"
        ):
            opts["pin_to_plane_mode"] = "slide"
            opts["pin_to_plane_group"] = "outer_height_gauge"
            opts["pin_to_circle_mode"] = "slide"
            opts["pin_to_circle_group"] = "outer"
            opts["pin_to_circle_normal"] = [0.0, 0.0, 1.0]
            opts["pin_to_circle_point"] = [0.0, 0.0, 0.0]
        vertex.options = opts
    return mesh


def optimize_free_disk_theta_b(mesh, *, scans: int = 4) -> float:
    """Optimize the theory scalar drive on the flat free-disk lane."""
    minim = _configure_theta_scan(mesh)
    tilt_mode = str(mesh.global_parameters.get("tilt_solve_mode") or "coupled")
    for i in range(int(scans)):
        minim._relax_leaflet_tilts(
            positions=mesh.positions_view(),
            mode=tilt_mode,
        )
        minim._optimize_thetaB_scalar(tilt_mode=tilt_mode, iteration=i)
    return float(mesh.global_parameters.get("tilt_thetaB_value") or 0.0)


def _physical_rim_and_first_shell_radius(mesh) -> tuple[float, float]:
    """Return the physical rim radius and the first free-shell radius."""
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    rim_rows: list[int] = []
    for vid in mesh.vertex_ids:
        opts = getattr(mesh.vertices[int(vid)], "options", None) or {}
        if opts.get("rim_slope_match_group") == "rim":
            rim_rows.append(mesh.vertex_index_to_row[int(vid)])
    if not rim_rows:
        raise AssertionError("No rim_slope_match_group='rim' rows found")

    # Refined meshes can carry rim-tagged midpoint rows slightly inside the
    # physical shared rim. Use the outer edge of that tagged band so the first
    # selected shell is truly outside the rim.
    rim_radius = float(np.max(r[np.asarray(rim_rows, dtype=int)]))
    shell_radii = np.unique(np.round(r[r > rim_radius + 1.0e-3], 3))
    if shell_radii.size == 0:
        raise AssertionError("No free shell found outside the rim")
    return rim_radius, float(shell_radii[0])


def activate_local_outer_shell(mesh, *, z_bump: float = 1.5e-4) -> float:
    """Tag the first shell outside the rim as the local outer slope ring.

    Returns the activated shell radius.
    """
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    rim_radius, shell_radius = _physical_rim_and_first_shell_radius(mesh)

    shell_rows = np.where(np.isclose(r, shell_radius, atol=1.0e-3))[0]
    if shell_rows.size == 0:
        raise AssertionError("No rows found on the first shell outside the rim")

    for row in shell_rows:
        vid = int(mesh.vertex_ids[int(row)])
        mesh.vertices[vid].options["rim_slope_match_group"] = "outer"
        mesh.vertices[vid].position[2] = float(z_bump)

    mesh.build_position_cache()
    return shell_radius


def configure_free_disk_curved_bilayer_stage2(
    mesh,
    *,
    theta_b: float,
    z_bump: float | None = None,
) -> float:
    """Configure the refined curved bilayer mesh for stage-2 parity checks.

    The physical disk edge remains the ``rim`` group at ``R = 7/15``. The first
    free shell outside that shared rim is tagged as the local ``outer`` ring so
    near-rim parity can be measured at ``R+epsilon``.
    """
    gp = mesh.global_parameters
    gp.set("tilt_thetaB_group_in", "rim")
    gp.set("rim_slope_match_group", "rim")
    gp.set("rim_slope_match_outer_group", "outer")
    gp.set("rim_slope_match_disk_group", "disk")
    gp.set("rim_slope_match_mode", "shared_rim_staggered_v1")
    if gp.get("tilt_in_exclude_shared_rim_rows") is None:
        gp.set("tilt_in_exclude_shared_rim_rows", True)
    if gp.get("tilt_in_shared_rim_outer_shell_mass_mode") is None:
        gp.set("tilt_in_shared_rim_outer_shell_mass_mode", "consistent")
    if gp.get("tilt_out_exclude_shared_rim_outer_rows") is None:
        gp.set("tilt_out_exclude_shared_rim_outer_rows", True)
    # In the shared-rim discretization the inner leaflet is first free on the
    # outer support ring, so its base-term boundary should exclude that ring.
    gp.set("bending_tilt_base_term_boundary_group_in", "outer")
    gp.set("bending_tilt_base_term_boundary_group_out", "rim")
    gp.set("rim_slope_match_strength", 0.0)
    gp.set("tilt_thetaB_optimize", False)
    gp.set("tilt_thetaB_value", float(theta_b))
    for vid in mesh.vertex_ids:
        vertex = mesh.vertices[int(vid)]
        opts = getattr(vertex, "options", None) or {}
        if opts.get("rim_slope_match_group") == "outer":
            opts.pop("rim_slope_match_group", None)
        vertex.options = opts
    if z_bump is None:
        rim_radius, shell_radius = _physical_rim_and_first_shell_radius(mesh)
        z_bump = 0.5 * float(theta_b) * float(shell_radius - rim_radius)
    return activate_local_outer_shell(mesh, z_bump=float(z_bump))


def run_free_disk_two_stage_profile_protocol(
    *,
    path: str | Path | None = None,
    theta_scans: int = 4,
    shape_steps: int = 40,
    z_bump: float = 1.5e-4,
):
    """Return ``(mesh, theta_b)`` after the approved two-stage profile protocol."""
    theta_mesh = load_free_disk_theory_mesh(path)
    theta_b = optimize_free_disk_theta_b(theta_mesh, scans=theta_scans)
    if theta_b <= 0.0:
        raise AssertionError("thetaB optimization did not produce a positive drive")

    mesh = load_free_disk_theory_mesh(path)
    activate_local_outer_shell(mesh, z_bump=z_bump)
    minim = _configure_shape_relax(mesh, theta_b=theta_b)
    minim.minimize(n_steps=int(shape_steps))
    return mesh, theta_b


def run_free_disk_curved_bilayer_protocol(
    *,
    theta_scans: int = 4,
    shape_steps: int = 60,
    z_bump: float | None = None,
    theta_mode: str = "curved_local_scan",
    theta_path: str | Path | None = None,
    curved_path: str | Path | None = None,
    global_parameter_overrides: dict[str, object] | None = None,
):
    """Return ``(mesh, theta_b)`` for the refined curved bilayer stage-2 lane."""
    theta_mesh = load_free_disk_theory_mesh(theta_path)
    theta_seed = optimize_free_disk_theta_b(theta_mesh, scans=theta_scans)
    if theta_seed <= 0.0:
        raise AssertionError("thetaB optimization did not produce a positive drive")

    if theta_mode == "flat_stage1":
        theta_b = float(theta_seed)
    elif theta_mode == "curved_local_scan":
        result = optimize_free_disk_curved_theta_b(
            theta_b_seed=float(theta_seed),
            shape_steps=shape_steps,
            curved_path=curved_path,
            global_parameter_overrides=global_parameter_overrides,
        )
        theta_b = float(result["best_theta_b"])
    else:
        raise ValueError(f"Unsupported theta_mode={theta_mode!r}")

    mesh = load_free_disk_curved_bilayer_mesh(curved_path)
    _apply_global_parameter_overrides(mesh, global_parameter_overrides)
    configure_free_disk_curved_bilayer_stage2(mesh, theta_b=theta_b, z_bump=z_bump)
    minim = _configure_shape_relax(mesh, theta_b=theta_b)
    minim.minimize(n_steps=int(shape_steps))
    return mesh, theta_b


def measure_free_disk_curved_bilayer_near_rim(
    mesh,
    *,
    theta_b: float,
    rim_radius: float = 7.0 / 15.0,
) -> dict[str, float]:
    """Return near-rim shared-rim observables for a curved bilayer run."""
    positions = mesh.positions_view()
    r = np.linalg.norm(positions[:, :2], axis=1)
    r_hat = np.zeros_like(positions)
    good = r > 1.0e-12
    r_hat[good, 0] = positions[good, 0] / r[good]
    r_hat[good, 1] = positions[good, 1] / r[good]

    theta_in = np.einsum("ij,ij->i", mesh.tilts_in_view(), r_hat)
    theta_out = np.einsum("ij,ij->i", mesh.tilts_out_view(), r_hat)

    free_radii = sorted(
        {
            round(float(rr), 6)
            for rr in r
            if rr > float(rim_radius) + 1.0e-6 and rr < 12.0 - 1.0e-6
        }
    )
    if not free_radii:
        raise AssertionError("No free ring found outside the physical disk edge")
    ring_r = float(free_radii[0])

    disk_rows = np.where(np.isclose(r, float(rim_radius), atol=1.0e-6))[0]
    outer_rows = np.where(np.isclose(r, ring_r, atol=1.0e-6))[0]
    if disk_rows.size == 0:
        raise AssertionError(
            f"No rows found near the physical disk edge r={rim_radius}"
        )
    if outer_rows.size == 0:
        raise AssertionError(f"No rows found near the first free ring r={ring_r}")

    disk_z = float(np.median(positions[disk_rows, 2]))
    outer_z = float(np.median(positions[outer_rows, 2]))
    dr = float(np.median(r[outer_rows]) - np.median(r[disk_rows]))
    if abs(dr) <= 1.0e-12:
        raise AssertionError("Near-rim slope estimate has zero radial spacing")

    theta_disk = float(np.median(theta_in[disk_rows]))
    theta_outer_in = float(np.median(theta_in[outer_rows]))
    theta_outer_out = float(np.median(theta_out[outer_rows]))
    phi = float((outer_z - disk_z) / dr)
    phi_abs = abs(phi)
    closure = float(theta_outer_in + theta_outer_out)

    return {
        "theta_b": float(theta_b),
        "rim_radius": float(rim_radius),
        "ring_r": ring_r,
        "theta_disk": theta_disk,
        "theta_outer_in": theta_outer_in,
        "theta_outer_out": theta_outer_out,
        "phi": phi,
        "phi_abs": phi_abs,
        "target_half_theta": 0.5 * float(theta_b),
        "closure": closure,
        "closure_error": float(closure - theta_b),
        "theta_out_phi_gap": float(theta_outer_out - phi),
        "phi_deficit": float(theta_b - (2.0 * phi_abs)),
        "z_span": float(np.ptp(positions[:, 2])),
    }


def run_free_disk_curved_bilayer_theta_sweep(
    theta_values: list[float] | tuple[float, ...] | np.ndarray,
    *,
    shape_steps: int = 60,
    z_bump: float | None = None,
    curved_path: str | Path | None = None,
    global_parameter_overrides: dict[str, object] | None = None,
) -> list[dict[str, float]]:
    """Run the curved bilayer stage-2 lane for imposed ``thetaB`` values."""
    rows: list[dict[str, float]] = []
    for theta_b in np.asarray(theta_values, dtype=float):
        mesh = load_free_disk_curved_bilayer_mesh(curved_path)
        _apply_global_parameter_overrides(mesh, global_parameter_overrides)
        configure_free_disk_curved_bilayer_stage2(
            mesh,
            theta_b=float(theta_b),
            z_bump=z_bump,
        )
        minim = _configure_shape_relax(mesh, theta_b=float(theta_b))
        minim.minimize(n_steps=int(shape_steps))
        rows.append(
            measure_free_disk_curved_bilayer_near_rim(mesh, theta_b=float(theta_b))
        )
    return rows


def run_free_disk_curved_bilayer_energy_sweep(
    theta_values: list[float] | tuple[float, ...] | np.ndarray,
    *,
    shape_steps: int = 60,
    z_bump: float | None = None,
    curved_path: str | Path | None = None,
    global_parameter_overrides: dict[str, object] | None = None,
) -> list[dict[str, float]]:
    """Run imposed-``thetaB`` curved states and report near-rim metrics + energy."""
    rows: list[dict[str, float]] = []
    for theta_b in np.asarray(theta_values, dtype=float):
        mesh = load_free_disk_curved_bilayer_mesh(curved_path)
        _apply_global_parameter_overrides(mesh, global_parameter_overrides)
        configure_free_disk_curved_bilayer_stage2(
            mesh,
            theta_b=float(theta_b),
            z_bump=z_bump,
        )
        minim = _configure_shape_relax(mesh, theta_b=float(theta_b))
        minim.minimize(n_steps=int(shape_steps))
        breakdown = minim.compute_energy_breakdown()
        row = measure_free_disk_curved_bilayer_near_rim(mesh, theta_b=float(theta_b))
        contact_energy = float(breakdown.get("tilt_thetaB_contact_in") or 0.0)
        tilt_in_energy = float(breakdown.get("tilt_in") or 0.0)
        tilt_out_energy = float(breakdown.get("tilt_out") or 0.0)
        bending_tilt_in_energy = float(breakdown.get("bending_tilt_in") or 0.0)
        bending_tilt_out_energy = float(breakdown.get("bending_tilt_out") or 0.0)
        elastic_energy = float(
            tilt_in_energy
            + tilt_out_energy
            + bending_tilt_in_energy
            + bending_tilt_out_energy
        )
        row["total_energy"] = _energy_total(breakdown)
        row["contact_energy"] = contact_energy
        row["elastic_energy"] = elastic_energy
        row["tilt_in_energy"] = tilt_in_energy
        row["tilt_out_energy"] = tilt_out_energy
        row["bending_tilt_in_energy"] = bending_tilt_in_energy
        row["bending_tilt_out_energy"] = bending_tilt_out_energy
        row.update({f"tilt_in_{k}": v for k, v in _tilt_in_region_split(mesh).items()})
        row.update(
            {f"tilt_out_{k}": v for k, v in _tilt_out_region_split(mesh).items()}
        )
        row.update(_shared_rim_inner_control_volume_audit(mesh))
        row.update(
            {
                f"bending_tilt_out_{k}": v
                for k, v in _bending_tilt_leaflet_region_split(
                    mesh, leaflet="out"
                ).items()
            }
        )
        rows.append(row)
    return rows


def summarize_free_disk_curved_elastic_growth(
    rows: list[dict[str, float]] | tuple[dict[str, float], ...],
) -> list[dict[str, object]]:
    """Return incremental elastic-growth attribution across a curved theta sweep."""
    out: list[dict[str, object]] = []
    keys = (
        "tilt_in_energy",
        "tilt_out_energy",
        "bending_tilt_in_energy",
        "bending_tilt_out_energy",
    )
    for prev, cur in zip(rows, rows[1:]):
        deltas = {key: float(cur[key]) - float(prev[key]) for key in keys}
        dominant = max(deltas, key=deltas.get)
        out.append(
            {
                "theta_b_lo": float(prev["theta_b"]),
                "theta_b_hi": float(cur["theta_b"]),
                "term_deltas": deltas,
                "dominant_term": dominant,
                "dominant_delta": float(deltas[dominant]),
            }
        )
    return out


def run_free_disk_curved_bilayer_refinement_sweep(
    theta_values: list[float] | tuple[float, ...] | np.ndarray,
    *,
    refine_steps: int = 0,
    shape_steps: int = 60,
    z_bump: float | None = None,
    curved_path: str | Path | None = None,
    global_parameter_overrides: dict[str, object] | None = None,
) -> list[dict[str, float]]:
    """Run imposed-``thetaB`` curved states after a fixed number of refinements."""
    rows: list[dict[str, float]] = []
    for theta_b in np.asarray(theta_values, dtype=float):
        mesh = load_free_disk_curved_bilayer_mesh(curved_path)
        for _ in range(int(refine_steps)):
            mesh = refine_triangle_mesh(mesh)
        _apply_global_parameter_overrides(mesh, global_parameter_overrides)
        configure_free_disk_curved_bilayer_stage2(
            mesh,
            theta_b=float(theta_b),
            z_bump=z_bump,
        )
        minim = _configure_shape_relax(mesh, theta_b=float(theta_b))
        minim.minimize(n_steps=int(shape_steps))
        breakdown = minim.compute_energy_breakdown()
        row = measure_free_disk_curved_bilayer_near_rim(mesh, theta_b=float(theta_b))
        row["total_energy"] = _energy_total(breakdown)
        row["tilt_in_energy"] = float(breakdown.get("tilt_in") or 0.0)
        row.update({f"tilt_in_{k}": v for k, v in _tilt_in_region_split(mesh).items()})
        row.update(_shared_rim_inner_control_volume_audit(mesh))
        row.update(shared_rim_continuum_annulus_audit(mesh))
        row.update(shared_rim_shell_area_audit(mesh))
        row["refine_steps"] = float(refine_steps)
        rows.append(row)
    return rows


def optimize_free_disk_curved_theta_b(
    *,
    theta_b_seed: float,
    theta_offsets: tuple[float, ...] = (
        -0.02,
        0.0,
        0.01,
        0.02,
        0.04,
        0.06,
        0.08,
        0.10,
        0.12,
        0.14,
    ),
    shape_steps: int = 60,
    z_bump: float | None = None,
    curved_path: str | Path | None = None,
    global_parameter_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    """Return the best curved-branch ``thetaB`` from a local imposed scan."""
    theta_values = sorted(
        {
            round(max(0.0, float(theta_b_seed) + float(delta)), 8)
            for delta in theta_offsets
        }
    )
    rows = run_free_disk_curved_bilayer_energy_sweep(
        theta_values,
        shape_steps=shape_steps,
        z_bump=z_bump,
        curved_path=curved_path,
        global_parameter_overrides=global_parameter_overrides,
    )
    if not rows:
        raise AssertionError("Curved theta sweep produced no samples")
    best = min(rows, key=lambda row: float(row["total_energy"]))
    return {
        "theta_b_seed": float(theta_b_seed),
        "theta_values": theta_values,
        "rows": rows,
        "best_theta_b": float(best["theta_b"]),
        "best_total_energy": float(best["total_energy"]),
    }


__all__ = [
    "DEFAULT_FREE_DISK_FIXTURE",
    "DEFAULT_FREE_DISK_CURVED_BILAYER_SOURCE",
    "activate_local_outer_shell",
    "configure_free_disk_curved_bilayer_stage2",
    "load_free_disk_curved_bilayer_mesh",
    "load_free_disk_theory_mesh",
    "measure_free_disk_curved_bilayer_near_rim",
    "optimize_free_disk_theta_b",
    "optimize_free_disk_curved_theta_b",
    "run_free_disk_curved_bilayer_energy_sweep",
    "run_free_disk_curved_bilayer_refinement_sweep",
    "run_free_disk_curved_bilayer_protocol",
    "run_free_disk_curved_bilayer_theta_sweep",
    "run_free_disk_two_stage_profile_protocol",
    "shared_rim_continuum_annulus_audit",
    "shared_rim_shell_area_audit",
    "summarize_free_disk_curved_elastic_growth",
]

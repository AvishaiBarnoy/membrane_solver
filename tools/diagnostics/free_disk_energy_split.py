#!/usr/bin/env python3
"""Report disk vs outer energy contributions for the free-disk mesh/output.

This diagnostic mirrors the TeX decomposition:
  - disk lipid patch elastic energy (inner leaflet, disk triangles)
  - outer membrane elastic energy (outer leaflet, non-disk triangles)
  - contact work term

It loads a base mesh YAML (for definitions/global parameters) and an output
mesh (for the relaxed vertex positions/tilts), then computes approximate
energy splits using the same formulas as the runtime energy modules.
"""

from __future__ import annotations

import argparse
from typing import Tuple

import numpy as np

from core.parameters.resolver import ParameterResolver
from geometry.curvature import compute_curvature_data
from geometry.geom_io import load_data, parse_geometry
from geometry.tilt_operators import p1_triangle_divergence_from_shape_gradients
from modules.energy.bending import _compute_effective_areas, _energy_model
from modules.energy.bending_tilt_leaflet import (
    _assume_J0_presets,
    _base_term_boundary_group,
    _collect_group_rows,
    _collect_preset_rows,
    _per_vertex_params_leaflet,
)
from modules.energy.leaflet_presence import (
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)
from modules.energy.tilt_thetaB_contact_in import (
    compute_energy_and_gradient_array as contact_energy,
)
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.minimizer import Minimizer
from runtime.steppers.gradient_descent import GradientDescent


def _merge_base_config(base: dict, out: dict) -> dict:
    merged = dict(out)
    for key in (
        "definitions",
        "global_parameters",
        "constraint_modules",
        "energy_modules",
    ):
        if key not in merged and key in base:
            merged[key] = base[key]
    return merged


def _tilt_energy(
    *, positions: np.ndarray, tri_rows: np.ndarray, tilts: np.ndarray, k_tilt: float
) -> float:
    if tri_rows.size == 0 or k_tilt == 0.0:
        return 0.0

    tilt_sq = np.einsum("ij,ij->i", tilts, tilts)
    tri_pos = positions[tri_rows]
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]
    n = np.cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm >= 1e-12
    if not np.any(mask):
        return 0.0

    areas = 0.5 * n_norm[mask]
    tri_tilt_sq_sum = tilt_sq[tri_rows[mask]].sum(axis=1)
    coeff = 0.5 * k_tilt * (tri_tilt_sq_sum / 3.0)
    return float(np.dot(coeff, areas))


def _bending_tilt_energy(
    *,
    mesh,
    global_params,
    positions: np.ndarray,
    tri_rows_full: np.ndarray,
    weights_full: np.ndarray,
    tri_mask: np.ndarray,
    tilts: np.ndarray,
    kappa_key: str,
    cache_tag: str,
) -> float:
    if tri_rows_full.size == 0 or not np.any(tri_mask):
        return 0.0

    tri_rows = tri_rows_full[tri_mask]
    weights = weights_full[tri_mask]

    area_cache, g0_cache, g1_cache, g2_cache, tri_rows_cache = (
        mesh.p1_triangle_shape_gradient_cache(positions)
    )
    if tri_rows_cache.shape[0] != tri_rows_full.shape[0]:
        raise RuntimeError("Triangle cache shape mismatch; cannot compute divergence.")

    g0 = g0_cache[tri_mask]
    g1 = g1_cache[tri_mask]
    g2 = g2_cache[tri_mask]
    div_tri = p1_triangle_divergence_from_shape_gradients(
        tilts=tilts, tri_rows=tri_rows, g0=g0, g1=g1, g2=g2
    )

    vertex_areas_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh, positions, tri_rows, weights, mesh.vertex_index_to_row
    )

    model = _energy_model(global_params)
    if model != "helfrich":
        model = "helfrich"
    kappa_arr, c0_arr = _per_vertex_params_leaflet(
        mesh, global_params, model=model, kappa_key=kappa_key, cache_tag=cache_tag
    )

    k_vecs, vertex_areas_vor, _, tri_rows_curv = compute_curvature_data(
        mesh, positions, mesh.vertex_index_to_row
    )
    if tri_rows_curv.shape[0] != tri_rows_full.shape[0]:
        raise RuntimeError("Curvature tri_rows mismatch; cannot split energies.")

    k_mag = np.linalg.norm(k_vecs, axis=1)
    safe_areas_vor = np.maximum(vertex_areas_vor, 1e-12)
    H_vor = k_mag / (2.0 * safe_areas_vor)

    boundary_vids = mesh.boundary_vertex_ids
    is_interior = np.ones(len(mesh.vertex_ids), dtype=bool)
    if boundary_vids:
        boundary_rows = [
            mesh.vertex_index_to_row[vid]
            for vid in boundary_vids
            if vid in mesh.vertex_index_to_row
        ]
        is_interior[boundary_rows] = False
    group = _base_term_boundary_group(global_params, cache_tag=cache_tag)
    if group:
        rows = _collect_group_rows(
            mesh, group=group, index_map=mesh.vertex_index_to_row
        )
        if rows.size:
            is_interior[rows] = False

    base_term = (2.0 * H_vor) - c0_arr
    base_term[~is_interior] = 0.0

    presets = _assume_J0_presets(global_params, cache_tag=cache_tag)
    if presets:
        rows = _collect_preset_rows(
            mesh,
            presets=presets,
            cache_tag=cache_tag,
            index_map=mesh.vertex_index_to_row,
        )
        if rows.size:
            base_term[rows] = 0.0

    term_tri = base_term[tri_rows] + div_tri[:, None]
    va_eff = np.stack([va0_eff, va1_eff, va2_eff], axis=1)
    kappa_tri = kappa_arr[tri_rows]
    return float(0.5 * np.sum(kappa_tri * term_tri**2 * va_eff))


def _split_masks(mesh, tri_rows_full: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vertices = list(mesh.vertices.values())
    presets = np.array(
        [str(v.options.get("preset") or "") for v in vertices], dtype=object
    )
    is_disk = presets == "disk"
    # Disk patch: any triangle that touches a disk vertex.
    tri_mask_disk = np.any(is_disk[tri_rows_full], axis=1)

    absent_mask = leaflet_absent_vertex_mask(
        mesh, mesh.global_parameters, leaflet="out"
    )
    tri_mask_outer = leaflet_present_triangle_mask(
        mesh, tri_rows_full, absent_vertex_mask=absent_mask
    )
    tri_mask_outer &= ~tri_mask_disk
    return tri_mask_disk, tri_mask_outer


def _energy_breakdown(mesh) -> dict[str, float]:
    minim = Minimizer(
        mesh,
        mesh.global_parameters,
        GradientDescent(),
        EnergyModuleManager(mesh.energy_modules),
        ConstraintModuleManager(mesh.constraint_modules),
        quiet=True,
        tol=1e-6,
    )
    return minim.compute_energy_breakdown()


def _inner_leaflet_vertex_split(
    *, mesh, positions: np.ndarray, tri_rows_full: np.ndarray, weights_full: np.ndarray
) -> dict[str, float]:
    presets = np.array(
        [str(v.options.get("preset") or "") for v in mesh.vertices.values()],
        dtype=object,
    )
    disk_mask = presets == "disk"

    tri_pos = positions[tri_rows_full]
    v0 = tri_pos[:, 0, :]
    v1 = tri_pos[:, 1, :]
    v2 = tri_pos[:, 2, :]
    n = np.cross(v1 - v0, v2 - v0)
    n_norm = np.linalg.norm(n, axis=1)
    mask = n_norm >= 1e-12
    areas = 0.5 * n_norm[mask]

    vertex_areas = mesh.barycentric_vertex_areas(
        positions,
        tri_rows=tri_rows_full,
        areas=areas,
        mask=mask,
        cache=True,
    )
    tilt_sq = np.einsum("ij,ij->i", mesh.tilts_in_view(), mesh.tilts_in_view())
    k_tilt_in = float(mesh.global_parameters.get("tilt_modulus_in") or 0.0)
    tilt_energy_vertex = 0.5 * k_tilt_in * tilt_sq * vertex_areas

    tilt_in_disk = float(tilt_energy_vertex[disk_mask].sum())
    tilt_in_outer = float(tilt_energy_vertex[~disk_mask].sum())

    k_vecs, vertex_areas_vor, _, tri_rows_curv = compute_curvature_data(
        mesh, positions, mesh.vertex_index_to_row
    )
    if tri_rows_curv.shape[0] != tri_rows_full.shape[0]:
        raise ValueError("Triangle rows mismatch between caches.")

    area_cache, g0_cache, g1_cache, g2_cache, tri_rows_cache = (
        mesh.p1_triangle_shape_gradient_cache(positions)
    )
    if tri_rows_cache.shape[0] != tri_rows_full.shape[0]:
        raise ValueError("Triangle cache mismatch; cannot split inner leaflet.")

    div_tri = p1_triangle_divergence_from_shape_gradients(
        tilts=mesh.tilts_in_view(),
        tri_rows=tri_rows_full,
        g0=g0_cache,
        g1=g1_cache,
        g2=g2_cache,
    )

    vertex_areas_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh, positions, tri_rows_full, weights_full, mesh.vertex_index_to_row
    )
    safe_areas_vor = np.maximum(vertex_areas_vor, 1e-12)

    model = _energy_model(mesh.global_parameters)
    if model != "helfrich":
        model = "helfrich"
    kappa_arr, c0_arr = _per_vertex_params_leaflet(
        mesh,
        mesh.global_parameters,
        model=model,
        kappa_key="bending_modulus_in",
        cache_tag="in",
    )

    k_mag = np.linalg.norm(k_vecs, axis=1)
    H_vor = k_mag / (2.0 * safe_areas_vor)

    boundary_vids = mesh.boundary_vertex_ids
    is_interior = np.ones(len(mesh.vertex_ids), dtype=bool)
    if boundary_vids:
        boundary_rows = [
            mesh.vertex_index_to_row[vid]
            for vid in boundary_vids
            if vid in mesh.vertex_index_to_row
        ]
        is_interior[boundary_rows] = False
    group = _base_term_boundary_group(mesh.global_parameters, cache_tag="in")
    if group:
        rows = _collect_group_rows(
            mesh, group=group, index_map=mesh.vertex_index_to_row
        )
        if rows.size:
            is_interior[rows] = False

    base_term = (2.0 * H_vor) - c0_arr
    base_term[~is_interior] = 0.0

    presets_in = _assume_J0_presets(mesh.global_parameters, cache_tag="in")
    if presets_in:
        rows = _collect_preset_rows(
            mesh,
            presets=presets_in,
            cache_tag="in",
            index_map=mesh.vertex_index_to_row,
        )
        if rows.size:
            base_term[rows] = 0.0

    term_tri = base_term[tri_rows_full] + (-1.0) * div_tri[:, None]
    va_eff = np.stack([va0_eff, va1_eff, va2_eff], axis=1)
    kappa_tri = kappa_arr[tri_rows_full]
    energy_tri_vertex = 0.5 * kappa_tri * term_tri**2 * va_eff

    bend_energy_vertex = np.zeros_like(base_term)
    np.add.at(bend_energy_vertex, tri_rows_full[:, 0], energy_tri_vertex[:, 0])
    np.add.at(bend_energy_vertex, tri_rows_full[:, 1], energy_tri_vertex[:, 1])
    np.add.at(bend_energy_vertex, tri_rows_full[:, 2], energy_tri_vertex[:, 2])

    bend_in_disk = float(bend_energy_vertex[disk_mask].sum())
    bend_in_outer = float(bend_energy_vertex[~disk_mask].sum())

    return {
        "tilt_in_disk": tilt_in_disk,
        "tilt_in_outer": tilt_in_outer,
        "bending_tilt_in_disk": bend_in_disk,
        "bending_tilt_in_outer": bend_in_outer,
    }


def compute_energy_split(base: dict, out: dict) -> dict[str, float]:
    merged = _merge_base_config(base, out)

    mesh = parse_geometry(merged)
    positions = mesh.positions_view()

    tri_rows_full, _ = mesh.triangle_row_cache()
    if tri_rows_full is None or tri_rows_full.size == 0:
        raise ValueError("No triangles found in mesh.")

    _, _, weights_full, tri_rows_curv = compute_curvature_data(
        mesh, positions, mesh.vertex_index_to_row
    )
    if tri_rows_curv.shape[0] != tri_rows_full.shape[0]:
        raise ValueError("Triangle rows mismatch between caches.")

    tri_mask_disk, tri_mask_outer = _split_masks(mesh, tri_rows_full)

    inner_split = _inner_leaflet_vertex_split(
        mesh=mesh,
        positions=positions,
        tri_rows_full=tri_rows_full,
        weights_full=weights_full,
    )

    tilts_out = mesh.tilts_out_view()
    k_tilt_out = float(mesh.global_parameters.get("tilt_modulus_out") or 0.0)

    tilt_out_outer = _tilt_energy(
        positions=positions,
        tri_rows=tri_rows_full[tri_mask_outer],
        tilts=tilts_out,
        k_tilt=k_tilt_out,
    )
    bend_out_outer = _bending_tilt_energy(
        mesh=mesh,
        global_params=mesh.global_parameters,
        positions=positions,
        tri_rows_full=tri_rows_full,
        weights_full=weights_full,
        tri_mask=tri_mask_outer,
        tilts=tilts_out,
        kappa_key="bending_modulus_out",
        cache_tag="out",
    )

    resolver = ParameterResolver(mesh.global_parameters)
    contact = contact_energy(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=None,
    )

    breakdown = _energy_breakdown(mesh)
    total = float(sum(breakdown.values()))

    disk_total = inner_split["tilt_in_disk"] + inner_split["bending_tilt_in_disk"]
    outer_total = (
        inner_split["tilt_in_outer"]
        + inner_split["bending_tilt_in_outer"]
        + tilt_out_outer
        + bend_out_outer
    )
    split_total = disk_total + outer_total + float(contact)
    return {
        "tilt_in_disk": float(inner_split["tilt_in_disk"]),
        "tilt_in_outer": float(inner_split["tilt_in_outer"]),
        "tilt_out_outer": float(tilt_out_outer),
        "bending_tilt_in_disk": float(inner_split["bending_tilt_in_disk"]),
        "bending_tilt_in_outer": float(inner_split["bending_tilt_in_outer"]),
        "bending_tilt_out_outer": float(bend_out_outer),
        "contact": float(contact),
        "disk_total": float(disk_total),
        "outer_total": float(outer_total),
        "split_total": float(split_total),
        "global_total": float(total),
        "global_breakdown": breakdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base", required=True, help="Base YAML with definitions/global parameters."
    )
    parser.add_argument(
        "--out", required=True, help="Output YAML with relaxed positions/tilts."
    )
    args = parser.parse_args()

    base = load_data(args.base)
    out = load_data(args.out)
    split = compute_energy_split(base, out)

    print("Disk patch energy (inner leaflet, disk triangles):")
    print(f"  tilt_in: {split['tilt_in_disk']:.6g}")
    print(f"  bending_tilt_in: {split['bending_tilt_in_disk']:.6g}")
    print(f"  total: {split['disk_total']:.6g}")
    print("Outer membrane energy (inner non-disk + outer leaflet):")
    print(f"  tilt_in (non-disk): {split['tilt_in_outer']:.6g}")
    print(f"  bending_tilt_in (non-disk): {split['bending_tilt_in_outer']:.6g}")
    print(f"  tilt_out (outer leaflet): {split['tilt_out_outer']:.6g}")
    print(f"  bending_tilt_out (outer leaflet): {split['bending_tilt_out_outer']:.6g}")
    print(f"  total: {split['outer_total']:.6g}")
    print("Contact energy:")
    print(f"  tilt_thetaB_contact_in: {split['contact']:.6g}")
    print("Global energy check:")
    print(f"  global_total: {split['global_total']:.6g}")
    print(f"  split_total: {split['split_total']:.6g}")
    print(f"  delta: {split['split_total'] - split['global_total']:.6g}")


if __name__ == "__main__":
    main()

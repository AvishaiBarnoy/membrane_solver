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
    merged = _merge_base_config(base, out)

    mesh = parse_geometry(merged)
    positions = mesh.positions_view()

    tri_rows_full, _ = mesh.triangle_row_cache()
    if tri_rows_full is None or tri_rows_full.size == 0:
        raise SystemExit("No triangles found in mesh.")

    k_vecs, vertex_areas_vor, weights_full, tri_rows_curv = compute_curvature_data(
        mesh, positions, mesh.vertex_index_to_row
    )
    if tri_rows_curv.shape[0] != tri_rows_full.shape[0]:
        raise SystemExit("Triangle rows mismatch between caches.")

    tri_mask_disk, tri_mask_outer = _split_masks(mesh, tri_rows_full)

    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    k_tilt_in = float(mesh.global_parameters.get("tilt_modulus_in") or 0.0)
    k_tilt_out = float(mesh.global_parameters.get("tilt_modulus_out") or 0.0)

    tilt_in_disk = _tilt_energy(
        positions=positions,
        tri_rows=tri_rows_full[tri_mask_disk],
        tilts=tilts_in,
        k_tilt=k_tilt_in,
    )
    tilt_out_outer = _tilt_energy(
        positions=positions,
        tri_rows=tri_rows_full[tri_mask_outer],
        tilts=tilts_out,
        k_tilt=k_tilt_out,
    )

    bend_in_disk = _bending_tilt_energy(
        mesh=mesh,
        global_params=mesh.global_parameters,
        positions=positions,
        tri_rows_full=tri_rows_full,
        weights_full=weights_full,
        tri_mask=tri_mask_disk,
        tilts=tilts_in,
        kappa_key="bending_modulus_in",
        cache_tag="in",
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

    print("Disk patch energy (inner leaflet, disk triangles):")
    print(f"  tilt_in: {tilt_in_disk:.6g}")
    print(f"  bending_tilt_in: {bend_in_disk:.6g}")
    print(f"  total: {tilt_in_disk + bend_in_disk:.6g}")
    print("Outer membrane energy (outer leaflet, non-disk triangles):")
    print(f"  tilt_out: {tilt_out_outer:.6g}")
    print(f"  bending_tilt_out: {bend_out_outer:.6g}")
    print(f"  total: {tilt_out_outer + bend_out_outer:.6g}")
    print("Contact energy:")
    print(f"  tilt_thetaB_contact_in: {float(contact):.6g}")


if __name__ == "__main__":
    main()

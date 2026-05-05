"""Shared helpers for leaflet-specific bending-tilt coupling."""

from __future__ import annotations

from typing import Dict

import numpy as np

from geometry.bending_derivatives import grad_triangle_area
from geometry.curvature import compute_curvature_data
from geometry.entities import Mesh
from geometry.tilt_operators import (
    _resolve_transport_model,
    compute_divergence_from_basis,
    p1_triangle_divergence,
)
from modules.energy.bending import (  # noqa: PLC0415
    _apply_beltrami_laplacian,
    _cached_cotan_gradients,
    _compute_effective_areas,
    _energy_model,
    _grad_cotan,
    _gradient_mode,
    _spontaneous_curvature,
    _vertex_normals,
)
from modules.energy.leaflet_presence import (
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)
from modules.energy.scatter import scatter_triangle_scalar_to_vertices

from .bt_params import (
    _assume_J0_center_xy,
    _assume_J0_presets,
    _assume_J0_radius_max,
    _base_term_boundary_group,
    _base_term_region_mode,
    _base_term_region_radius,
    _bending_tilt_in_update_mode,
    _per_vertex_params_leaflet,
    _resolve_bending_modulus,
    _use_inner_recovered_divergence,
    _use_stage_a_inner_shape_cross_suppression,
    _use_stage_a_outer_grad_linear_transition_operator,
)
from .bt_selection import (
    _apply_inner_divergence_update_mode,
    _base_term_region_zero_rows,
    _collect_preset_rows,
    _interior_mask_leaflet,
    _shared_rim_support_transition_triangle_mask,
)
from .bt_utils import _accumulate_leaflet_tilt_gradient, _mean_reconstructed_field

_OUTER_TRANSITION_OPERATOR_SHELLS = (
    0.837822,
    0.842474,
    0.853008,
    0.866643,
    0.965910,
    0.974541,
    0.999984,
)


def _outer_transition_operator_payload(
    mesh: Mesh,
    *,
    tri_rows_full: np.ndarray,
    tri_rows: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build cached transition-patch payload for the outer grad_linear operator."""
    cache_attr = "_bending_tilt_out_transition_operator_payload"
    cache_key = (
        int(mesh._facet_loops_version),
        int(mesh._vertex_ids_version),
        int(tri_rows_full.shape[0]),
        int(tri_rows.shape[0]),
    )
    cached = getattr(mesh, cache_attr, None)
    if cached is not None and cached.get("key") == cache_key:
        return cached["value"]

    n_vertices = len(mesh.vertex_ids)
    full_counts = np.zeros(n_vertices, dtype=np.int32)
    keep_counts = np.zeros(n_vertices, dtype=np.int32)
    for col in range(3):
        np.add.at(full_counts, tri_rows_full[:, col], 1)
        np.add.at(keep_counts, tri_rows[:, col], 1)

    participation = np.zeros(n_vertices, dtype=float)
    mask_full = full_counts > 0
    participation[mask_full] = keep_counts[mask_full] / full_counts[mask_full]

    domain = np.zeros(n_vertices, dtype=np.int8)
    domain[(participation > 1.0e-12) & (participation < 1.0 - 1.0e-12)] = 1
    domain[participation >= 1.0 - 1.0e-12] = 2

    n_tri = tri_rows.shape[0]
    edge_endpoint_a = np.empty((n_tri, 3), dtype=np.int32)
    edge_endpoint_b = np.empty((n_tri, 3), dtype=np.int32)
    edge_endpoint_a[:, 0] = tri_rows[:, 1]
    edge_endpoint_b[:, 0] = tri_rows[:, 2]
    edge_endpoint_a[:, 1] = tri_rows[:, 2]
    edge_endpoint_b[:, 1] = tri_rows[:, 0]
    edge_endpoint_a[:, 2] = tri_rows[:, 0]
    edge_endpoint_b[:, 2] = tri_rows[:, 1]

    edge_third = np.empty((n_tri, 3), dtype=np.int32)
    edge_third[:, 0] = tri_rows[:, 0]
    edge_third[:, 1] = tri_rows[:, 1]
    edge_third[:, 2] = tri_rows[:, 2]

    edge_to_thirds: dict[tuple[int, int], list[int]] = {}
    for tri_idx in range(n_tri):
        v0, v1, v2 = tri_rows[tri_idx]
        edge_to_thirds.setdefault(tuple(sorted((int(v1), int(v2)))), []).append(int(v0))
        edge_to_thirds.setdefault(tuple(sorted((int(v2), int(v0)))), []).append(int(v1))
        edge_to_thirds.setdefault(tuple(sorted((int(v0), int(v1)))), []).append(int(v2))

    recon_a_idx = np.full((n_tri, 3, 2), -1, dtype=np.int32)
    recon_b_idx = np.full((n_tri, 3, 2), -1, dtype=np.int32)
    recon_a_count = np.zeros((n_tri, 3), dtype=np.int8)
    recon_b_count = np.zeros((n_tri, 3), dtype=np.int8)

    for tri_idx in range(n_tri):
        for edge_idx in range(3):
            a = int(edge_endpoint_a[tri_idx, edge_idx])
            b = int(edge_endpoint_b[tri_idx, edge_idx])
            thirds = edge_to_thirds.get(tuple(sorted((a, b))), [])
            a_candidates = [idx for idx in thirds if domain[idx] == domain[a]]
            b_candidates = [idx for idx in thirds if domain[idx] == domain[b]]
            if a_candidates:
                recon_a_count[tri_idx, edge_idx] = min(len(a_candidates), 2)
                recon_a_idx[tri_idx, edge_idx, : recon_a_count[tri_idx, edge_idx]] = (
                    np.asarray(a_candidates[:2], dtype=np.int32)
                )
            if b_candidates:
                recon_b_count[tri_idx, edge_idx] = min(len(b_candidates), 2)
                recon_b_idx[tri_idx, edge_idx, : recon_b_count[tri_idx, edge_idx]] = (
                    np.asarray(b_candidates[:2], dtype=np.int32)
                )

    edge_domain_a = domain[edge_endpoint_a]
    edge_domain_b = domain[edge_endpoint_b]
    transition_mask = ((edge_domain_a == 1) & (edge_domain_b == 2)) | (
        (edge_domain_a == 2) & (edge_domain_b == 1)
    )
    same_mask = ~transition_mask

    transition_full_vertex_mask = np.zeros(n_vertices, dtype=bool)
    transition_full_a = transition_mask & (edge_domain_a == 2)
    transition_full_b = transition_mask & (edge_domain_b == 2)
    if np.any(transition_full_a):
        transition_full_vertex_mask[np.unique(edge_endpoint_a[transition_full_a])] = (
            True
        )
    if np.any(transition_full_b):
        transition_full_vertex_mask[np.unique(edge_endpoint_b[transition_full_b])] = (
            True
        )
    partial_vertex_mask = domain == 1
    halo_full_vertex_mask = transition_full_vertex_mask & (domain == 2)
    patch_vertex_mask = partial_vertex_mask | halo_full_vertex_mask

    edge_unique_idx = np.empty((n_tri, 3), dtype=np.int32)
    unique_edge_u: list[int] = []
    unique_edge_v: list[int] = []
    edge_to_unique: dict[tuple[int, int], int] = {}
    for tri_idx in range(n_tri):
        for edge_idx in range(3):
            a = int(edge_endpoint_a[tri_idx, edge_idx])
            b = int(edge_endpoint_b[tri_idx, edge_idx])
            key = (a, b) if a < b else (b, a)
            unique_idx = edge_to_unique.get(key)
            if unique_idx is None:
                unique_idx = len(unique_edge_u)
                edge_to_unique[key] = unique_idx
                unique_edge_u.append(key[0])
                unique_edge_v.append(key[1])
            edge_unique_idx[tri_idx, edge_idx] = unique_idx

    unique_edge_u_arr = np.asarray(unique_edge_u, dtype=np.int32)
    unique_edge_v_arr = np.asarray(unique_edge_v, dtype=np.int32)
    patch_internal_edge_mask = (
        patch_vertex_mask[unique_edge_u_arr] & patch_vertex_mask[unique_edge_v_arr]
    )
    patch_boundary_edge_mask = (
        patch_vertex_mask[unique_edge_u_arr] ^ patch_vertex_mask[unique_edge_v_arr]
    )
    exterior_edge_mask = ~(patch_internal_edge_mask | patch_boundary_edge_mask)

    adjacency: dict[int, list[int]] = {
        int(row): [] for row in np.flatnonzero(patch_vertex_mask)
    }
    for edge_idx, is_internal in enumerate(patch_internal_edge_mask):
        if not is_internal:
            continue
        u = int(unique_edge_u_arr[edge_idx])
        v = int(unique_edge_v_arr[edge_idx])
        adjacency[u].append(v)
        adjacency[v].append(u)

    patch_component_rows: list[np.ndarray] = []
    patch_boundary_rows: list[np.ndarray] = []
    seen = np.zeros(n_vertices, dtype=bool)
    for start in np.flatnonzero(patch_vertex_mask):
        if seen[start]:
            continue
        stack = [int(start)]
        component: list[int] = []
        seen[start] = True
        while stack:
            row = stack.pop()
            component.append(row)
            for neighbor in adjacency.get(row, ()):
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        component_rows = np.asarray(sorted(component), dtype=np.int32)
        patch_component_rows.append(component_rows)

        component_mask = np.zeros(n_vertices, dtype=bool)
        component_mask[component_rows] = True
        boundary_edge_mask = (
            component_mask[unique_edge_u_arr] ^ component_mask[unique_edge_v_arr]
        )
        boundary_rows = np.unique(
            np.concatenate(
                (
                    unique_edge_u_arr[
                        boundary_edge_mask & ~component_mask[unique_edge_u_arr]
                    ],
                    unique_edge_v_arr[
                        boundary_edge_mask & ~component_mask[unique_edge_v_arr]
                    ],
                )
            )
        )
        patch_boundary_rows.append(boundary_rows.astype(np.int32, copy=False))

    value = {
        "domain": domain,
        "participation": participation,
        "edge_endpoint_a": edge_endpoint_a,
        "edge_endpoint_b": edge_endpoint_b,
        "edge_domain_a": edge_domain_a,
        "edge_domain_b": edge_domain_b,
        "edge_unique_idx": edge_unique_idx,
        "unique_edge_u": unique_edge_u_arr,
        "unique_edge_v": unique_edge_v_arr,
        "recon_a_idx": recon_a_idx,
        "recon_b_idx": recon_b_idx,
        "recon_a_count": recon_a_count,
        "recon_b_count": recon_b_count,
        "transition_mask": transition_mask,
        "same_mask": same_mask,
        "partial_vertex_mask": partial_vertex_mask,
        "halo_full_vertex_mask": halo_full_vertex_mask,
        "transition_full_vertex_mask": transition_full_vertex_mask,
        "patch_vertex_mask": patch_vertex_mask,
        "patch_internal_edge_mask": patch_internal_edge_mask,
        "patch_boundary_edge_mask": patch_boundary_edge_mask,
        "exterior_edge_mask": exterior_edge_mask,
        "patch_component_rows": patch_component_rows,
        "patch_boundary_rows": patch_boundary_rows,
    }
    setattr(mesh, cache_attr, {"key": cache_key, "value": value})
    return value


def _apply_edge_reconstructed_beltrami_laplacian(
    mesh: Mesh,
    *,
    positions: np.ndarray,
    weights: np.ndarray,
    tri_rows: np.ndarray,
    tri_rows_full: np.ndarray,
    field: np.ndarray,
    cache_tag: str,
    payload: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply the merged PR #496 transition-edge reconstruction operator."""
    if payload is None:
        payload = _outer_transition_operator_payload(
            mesh, tri_rows_full=tri_rows_full, tri_rows=tri_rows
        )
    edge_endpoint_a = np.asarray(payload["edge_endpoint_a"], dtype=np.int32)
    edge_endpoint_b = np.asarray(payload["edge_endpoint_b"], dtype=np.int32)
    recon_a_idx = np.asarray(payload["recon_a_idx"], dtype=np.int32)
    recon_b_idx = np.asarray(payload["recon_b_idx"], dtype=np.int32)
    recon_a_count = np.asarray(payload["recon_a_count"], dtype=np.int8)
    recon_b_count = np.asarray(payload["recon_b_count"], dtype=np.int8)
    transition_mask = np.asarray(payload["transition_mask"], dtype=bool)
    same_mask = np.asarray(payload["same_mask"], dtype=bool)
    participation = np.asarray(payload["participation"], dtype=float)
    domain = np.asarray(payload["domain"], dtype=np.int8)

    state_a = field[edge_endpoint_a]
    state_b = field[edge_endpoint_b]
    recon_a = _mean_reconstructed_field(
        field, edge_endpoint_a, recon_a_idx, recon_a_count
    )
    recon_b = _mean_reconstructed_field(
        field, edge_endpoint_b, recon_b_idx, recon_b_count
    )

    use_a = np.where(transition_mask[..., None], recon_a, state_a)
    use_b = np.where(transition_mask[..., None], recon_b, state_b)
    coeff = 0.5 * weights[:, :, None]
    flux = coeff * (use_a - use_b)
    transition_flux = np.where(transition_mask[..., None], flux, 0.0)
    same_flux = np.where(same_mask[..., None], flux, 0.0)

    out = np.zeros_like(field)
    out_transition = np.zeros_like(field)
    out_same = np.zeros_like(field)
    for edge_idx in range(3):
        a = edge_endpoint_a[:, edge_idx]
        b = edge_endpoint_b[:, edge_idx]
        f = flux[:, edge_idx, :]
        ft = transition_flux[:, edge_idx, :]
        fs = same_flux[:, edge_idx, :]
        np.add.at(out, a, f)
        np.add.at(out, b, -f)
        np.add.at(out_transition, a, ft)
        np.add.at(out_transition, b, -ft)
        np.add.at(out_same, a, fs)
        np.add.at(out_same, b, -fs)

    radii = np.linalg.norm(positions[:, :2], axis=1)
    shell_summary: list[dict[str, float | int]] = []
    for target in _OUTER_TRANSITION_OPERATOR_SHELLS:
        rows = np.where(np.isclose(radii, target, atol=1.0e-5))[0]
        if rows.size == 0:
            continue
        shell_summary.append(
            {
                "radius": float(np.mean(radii[rows])),
                "vertex_count": int(rows.size),
                "transition_z_mean": float(np.mean(out_transition[rows, 2])),
                "same_z_mean": float(np.mean(out_same[rows, 2])),
                "total_z_mean": float(np.mean(out[rows, 2])),
            }
        )

    transition_examples = np.argwhere(transition_mask)
    example = None
    if transition_examples.size:
        tri_idx, edge_idx = transition_examples[0]
        a = int(edge_endpoint_a[tri_idx, edge_idx])
        b = int(edge_endpoint_b[tri_idx, edge_idx])
        example = {
            "triangle_index": int(tri_idx),
            "edge_index": int(edge_idx),
            "endpoint_rows": [a, b],
            "endpoint_domains": [
                "partial" if int(domain[a]) == 1 else "full",
                "partial" if int(domain[b]) == 1 else "full",
            ],
            "endpoint_participation": [
                float(participation[a]),
                float(participation[b]),
            ],
            "raw_endpoint_factor_K_vec": [
                field[a].tolist(),
                field[b].tolist(),
            ],
            "reconstructed_side_states": [
                recon_a[tri_idx, edge_idx].tolist(),
                recon_b[tri_idx, edge_idx].tolist(),
            ],
        }

    stats = {
        "enabled": True,
        "mode": "outer_grad_linear_transition_operator_v1",
        "cache_tag": str(cache_tag),
        "lane": str(mesh.global_parameters.get("theory_parity_lane") or ""),
        "transition_edge_count": int(np.sum(transition_mask)),
        "same_domain_edge_count": int(np.sum(same_mask)),
        "nontransition_uses_raw_state": True,
        "reconstructed_state_variable": "factor_K_vec",
        "reconstruction_rule": "same-domain edge-adjacent third-vertex averaging",
        "transition_example": example,
        "shell_summary": shell_summary,
    }
    return out, stats


def _apply_transition_aware_beltrami_laplacian(
    mesh: Mesh,
    *,
    positions: np.ndarray,
    weights: np.ndarray,
    tri_rows: np.ndarray,
    tri_rows_full: np.ndarray,
    field: np.ndarray,
    cache_tag: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Apply a patch-local outer grad_linear operator on the transition region."""
    payload = _outer_transition_operator_payload(
        mesh, tri_rows_full=tri_rows_full, tri_rows=tri_rows
    )
    patch_vertex_mask = np.asarray(payload["patch_vertex_mask"], dtype=bool)
    if not np.any(patch_vertex_mask):
        return _apply_edge_reconstructed_beltrami_laplacian(
            mesh,
            positions=positions,
            weights=weights,
            tri_rows=tri_rows,
            tri_rows_full=tri_rows_full,
            field=field,
            cache_tag=cache_tag,
            payload=payload,
        )

    edge_endpoint_a = np.asarray(payload["edge_endpoint_a"], dtype=np.int32)
    edge_endpoint_b = np.asarray(payload["edge_endpoint_b"], dtype=np.int32)
    edge_unique_idx = np.asarray(payload["edge_unique_idx"], dtype=np.int32)
    unique_edge_u = np.asarray(payload["unique_edge_u"], dtype=np.int32)
    unique_edge_v = np.asarray(payload["unique_edge_v"], dtype=np.int32)
    transition_mask = np.asarray(payload["transition_mask"], dtype=bool)
    participation = np.asarray(payload["participation"], dtype=float)
    partial_vertex_mask = np.asarray(payload["partial_vertex_mask"], dtype=bool)
    halo_full_vertex_mask = np.asarray(payload["halo_full_vertex_mask"], dtype=bool)
    transition_full_vertex_mask = np.asarray(
        payload["transition_full_vertex_mask"], dtype=bool
    )
    patch_internal_edge_mask = np.asarray(
        payload["patch_internal_edge_mask"], dtype=bool
    )
    patch_boundary_edge_mask = np.asarray(
        payload["patch_boundary_edge_mask"], dtype=bool
    )
    exterior_edge_mask = np.asarray(payload["exterior_edge_mask"], dtype=bool)
    patch_component_rows = payload["patch_component_rows"]
    patch_boundary_rows = payload["patch_boundary_rows"]

    coeff_scalar = 0.5 * np.asarray(weights, dtype=float)
    coeff_unique = np.bincount(
        edge_unique_idx.ravel(),
        weights=coeff_scalar.ravel(),
        minlength=int(len(unique_edge_u)),
    )

    effective_field = field.copy()
    fallback_component_count = 0
    fallback_vertex_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
    fallback = _apply_edge_reconstructed_beltrami_laplacian(
        mesh,
        positions=positions,
        weights=weights,
        tri_rows=tri_rows,
        tri_rows_full=tri_rows_full,
        field=field,
        cache_tag=cache_tag,
        payload=payload,
    )
    fallback_out = fallback[0]

    for component_index, rows in enumerate(patch_component_rows):
        rows = np.asarray(rows, dtype=np.int32)
        if rows.size == 0:
            continue
        boundary_rows = np.asarray(patch_boundary_rows[component_index], dtype=np.int32)
        if boundary_rows.size == 0:
            fallback_component_count += 1
            fallback_vertex_mask[rows] = True
            continue

        local_index = {int(row): idx for idx, row in enumerate(rows)}
        size = int(rows.size)
        lhs = np.zeros((size, size), dtype=float)
        rhs = np.zeros((size, 3), dtype=float)
        component_mask = np.zeros(len(mesh.vertex_ids), dtype=bool)
        component_mask[rows] = True

        touching_edges = np.where(
            component_mask[unique_edge_u] | component_mask[unique_edge_v]
        )[0]
        for edge_idx in touching_edges:
            weight = float(coeff_unique[edge_idx])
            if weight == 0.0:
                continue
            u = int(unique_edge_u[edge_idx])
            v = int(unique_edge_v[edge_idx])
            iu = local_index.get(u, -1)
            iv = local_index.get(v, -1)
            if iu >= 0 and iv >= 0:
                lhs[iu, iu] += weight
                lhs[iv, iv] += weight
                lhs[iu, iv] -= weight
                lhs[iv, iu] -= weight
            elif iu >= 0:
                lhs[iu, iu] += weight
                rhs[iu] += weight * field[v]
            elif iv >= 0:
                lhs[iv, iv] += weight
                rhs[iv] += weight * field[u]

        try:
            solved = np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            fallback_component_count += 1
            fallback_vertex_mask[rows] = True
            if boundary_rows.size:
                fallback_vertex_mask[boundary_rows] = True
            continue
        effective_field[rows] = solved

    out = _apply_beltrami_laplacian(weights, tri_rows, effective_field)
    if np.any(fallback_vertex_mask):
        out[fallback_vertex_mask] = fallback_out[fallback_vertex_mask]

    eff_a = effective_field[edge_endpoint_a]
    eff_b = effective_field[edge_endpoint_b]
    coeff = coeff_scalar[:, :, None]
    flux = coeff * (eff_a - eff_b)
    occ_internal_mask = patch_internal_edge_mask[edge_unique_idx]
    occ_boundary_mask = patch_boundary_edge_mask[edge_unique_idx]
    occ_exterior_mask = exterior_edge_mask[edge_unique_idx]

    patch_internal_flux = np.where(occ_internal_mask[..., None], flux, 0.0)
    patch_boundary_flux = np.where(occ_boundary_mask[..., None], flux, 0.0)
    exterior_flux = np.where(occ_exterior_mask[..., None], flux, 0.0)

    out_patch_internal = np.zeros_like(field)
    out_patch_boundary = np.zeros_like(field)
    out_exterior = np.zeros_like(field)
    for edge_idx in range(3):
        a = edge_endpoint_a[:, edge_idx]
        b = edge_endpoint_b[:, edge_idx]
        fi = patch_internal_flux[:, edge_idx, :]
        fb = patch_boundary_flux[:, edge_idx, :]
        fe = exterior_flux[:, edge_idx, :]
        np.add.at(out_patch_internal, a, fi)
        np.add.at(out_patch_internal, b, -fi)
        np.add.at(out_patch_boundary, a, fb)
        np.add.at(out_patch_boundary, b, -fb)
        np.add.at(out_exterior, a, fe)
        np.add.at(out_exterior, b, -fe)

    radii = np.linalg.norm(positions[:, :2], axis=1)
    patch_shell_summary: list[dict[str, float | int]] = []
    for radius in np.unique(np.round(radii[patch_vertex_mask], 6)):
        rows = np.where(patch_vertex_mask & np.isclose(radii, radius, atol=1.0e-5))[0]
        if rows.size == 0:
            continue
        patch_shell_summary.append(
            {
                "radius": float(np.mean(radii[rows])),
                "vertex_count": int(rows.size),
                "participation_mean": float(np.mean(participation[rows])),
            }
        )

    shell_summary: list[dict[str, float | int]] = []
    for target in _OUTER_TRANSITION_OPERATOR_SHELLS:
        rows = np.where(np.isclose(radii, target, atol=1.0e-5))[0]
        if rows.size == 0:
            continue
        shell_summary.append(
            {
                "radius": float(np.mean(radii[rows])),
                "vertex_count": int(rows.size),
                "patch_internal_z_mean": float(np.mean(out_patch_internal[rows, 2])),
                "patch_boundary_z_mean": float(np.mean(out_patch_boundary[rows, 2])),
                "exterior_z_mean": float(np.mean(out_exterior[rows, 2])),
                "total_z_mean": float(np.mean(out[rows, 2])),
            }
        )

    patch_example = None
    if patch_component_rows:
        rows = np.asarray(patch_component_rows[0], dtype=np.int32)
        boundary_rows = np.asarray(patch_boundary_rows[0], dtype=np.int32)
        sample_rows = boundary_rows[: min(3, boundary_rows.size)]
        patch_example = {
            "component_index": 0,
            "patch_vertex_count": int(rows.size),
            "boundary_vertex_count": int(boundary_rows.size),
            "component_radii": np.unique(np.round(radii[rows], 6)).tolist(),
            "boundary_factor_K_vec_sample": effective_field[sample_rows].tolist(),
        }

    stats = {
        "enabled": True,
        "mode": "outer_grad_linear_transition_patch_operator_v1",
        "cache_tag": str(cache_tag),
        "lane": str(mesh.global_parameters.get("theory_parity_lane") or ""),
        "transition_edge_count": int(np.sum(transition_mask)),
        "same_domain_edge_count": int(np.sum(~transition_mask)),
        "patch_component_count": int(len(patch_component_rows)),
        "patch_vertex_count": int(np.sum(patch_vertex_mask)),
        "partial_vertex_count": int(np.sum(partial_vertex_mask)),
        "halo_full_vertex_count": int(np.sum(halo_full_vertex_mask)),
        "halo_vertices_are_direct_transition_full_endpoints": bool(
            np.array_equal(halo_full_vertex_mask, transition_full_vertex_mask)
        ),
        "fallback_component_count": int(fallback_component_count),
        "exterior_uses_raw_state": True,
        "reconstructed_state_variable": "factor_K_vec",
        "reconstruction_rule": "patch-local Dirichlet harmonic solve",
        "patch_example": patch_example,
        "patch_shell_summary": patch_shell_summary,
        "shell_summary": shell_summary,
    }
    return out, stats


def _inner_bending_tilt_dE_ddiv(
    *,
    mesh: Mesh,
    global_params,
    cache_tag: str,
    kappa_tri: np.ndarray,
    base_tri: np.ndarray,
    div_term: np.ndarray,
    va0_eff: np.ndarray,
    va1_eff: np.ndarray,
    va2_eff: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int | bool | str]]:
    """Return inner divergence gradient contribution under benchmark modes."""
    mode = _bending_tilt_in_update_mode(global_params)
    stats = {
        "enabled": bool(mode != "off"),
        "mode": str(mode),
        "candidate_tri_count": 0,
        "capped_tri_count": 0,
        "rim_tri_count": 0,
        "cap_magnitude": 0.0,
        "cross_term_removed": False,
    }
    if str(cache_tag) != "in":
        return (
            (kappa_tri[:, 0] * (base_tri[:, 0] + div_term) * va0_eff)
            + (kappa_tri[:, 1] * (base_tri[:, 1] + div_term) * va1_eff)
            + (kappa_tri[:, 2] * (base_tri[:, 2] + div_term) * va2_eff),
            stats,
        )
    if mode == "radial_cross_term_off_v1":
        stats["cross_term_removed"] = True
        setattr(mesh, "_last_bending_tilt_in_update_mode_stats", stats)
        return (
            (kappa_tri[:, 0] * div_term * va0_eff)
            + (kappa_tri[:, 1] * div_term * va1_eff)
            + (kappa_tri[:, 2] * div_term * va2_eff)
        ), stats
    return (
        (kappa_tri[:, 0] * (base_tri[:, 0] + div_term) * va0_eff)
        + (kappa_tri[:, 1] * (base_tri[:, 1] + div_term) * va1_eff)
        + (kappa_tri[:, 2] * (base_tri[:, 2] + div_term) * va2_eff)
    ), stats


def _inner_recovered_divergence(
    *,
    global_params,
    cache_tag: str,
    tri_rows: np.ndarray,
    tri_area: np.ndarray,
    div_tri: np.ndarray,
    n_vertices: int,
    ctx=None,
    scratch_tag: str,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Return divergence used for inner-leaflet evaluation.

    For the inner leaflet, recover a per-vertex divergence from surrounding
    triangle values using barycentric area weights, then average it back to
    triangles. Other leaflets keep the raw constant-per-triangle divergence.
    """
    div_tri = np.asarray(div_tri, dtype=float)
    if str(cache_tag) != "in" or div_tri.size == 0:
        return div_tri, None, None
    if not _use_inner_recovered_divergence(global_params, cache_tag=cache_tag):
        return div_tri, None, None

    tri_area = np.asarray(tri_area, dtype=float)
    w = tri_area / 3.0
    if ctx is not None:
        v_area = ctx.scratch_array(
            f"{scratch_tag}_v_area", shape=(n_vertices,), dtype=float
        )
        v_div_num = ctx.scratch_array(
            f"{scratch_tag}_v_div_num", shape=(n_vertices,), dtype=float
        )
        v_div = ctx.scratch_array(
            f"{scratch_tag}_v_div", shape=(n_vertices,), dtype=float
        )
        div_eval = ctx.scratch_array(
            f"{scratch_tag}_div_eval", shape=div_tri.shape, dtype=float
        )
        v_area.fill(0.0)
        v_div_num.fill(0.0)
        v_div.fill(0.0)
    else:
        v_area = np.zeros(n_vertices, dtype=float)
        v_div_num = np.zeros(n_vertices, dtype=float)
        v_div = np.zeros(n_vertices, dtype=float)
        div_eval = np.zeros_like(div_tri)

    np.add.at(v_area, tri_rows[:, 0], w)
    np.add.at(v_area, tri_rows[:, 1], w)
    np.add.at(v_area, tri_rows[:, 2], w)
    np.add.at(v_div_num, tri_rows[:, 0], w * div_tri)
    np.add.at(v_div_num, tri_rows[:, 1], w * div_tri)
    np.add.at(v_div_num, tri_rows[:, 2], w * div_tri)

    good_v = v_area > 1.0e-20
    v_div[good_v] = v_div_num[good_v] / v_area[good_v]
    div_eval[:] = (
        v_div[tri_rows[:, 0]] + v_div[tri_rows[:, 1]] + v_div[tri_rows[:, 2]]
    ) / 3.0
    return div_eval, v_div, v_area


def _inner_recovered_divergence_pullback(
    *,
    global_params,
    cache_tag: str,
    tri_rows: np.ndarray,
    tri_area: np.ndarray,
    coeff_div_eval: np.ndarray,
    v_area: np.ndarray | None,
    ctx=None,
    scratch_tag: str,
) -> np.ndarray:
    """Map dE/d(div_eval) back to raw triangle-divergence coefficients."""
    coeff_div_eval = np.asarray(coeff_div_eval, dtype=float)
    if str(cache_tag) != "in" or coeff_div_eval.size == 0:
        return coeff_div_eval
    if not _use_inner_recovered_divergence(global_params, cache_tag=cache_tag):
        return coeff_div_eval
    if v_area is None:
        raise ValueError("Recovered inner divergence requires vertex areas.")

    n_vertices = int(v_area.shape[0])
    if ctx is not None:
        v_grad = ctx.scratch_array(
            f"{scratch_tag}_v_grad", shape=(n_vertices,), dtype=float
        )
        inv_v_area = ctx.scratch_array(
            f"{scratch_tag}_inv_v_area", shape=(n_vertices,), dtype=float
        )
        coeff_div = ctx.scratch_array(
            f"{scratch_tag}_coeff_div", shape=coeff_div_eval.shape, dtype=float
        )
        v_grad.fill(0.0)
        inv_v_area.fill(0.0)
    else:
        v_grad = np.zeros(n_vertices, dtype=float)
        inv_v_area = np.zeros_like(v_area)
        coeff_div = np.zeros_like(coeff_div_eval)

    np.add.at(v_grad, tri_rows[:, 0], coeff_div_eval / 3.0)
    np.add.at(v_grad, tri_rows[:, 1], coeff_div_eval / 3.0)
    np.add.at(v_grad, tri_rows[:, 2], coeff_div_eval / 3.0)
    good_v = v_area > 1.0e-20
    inv_v_area[good_v] = 1.0 / v_area[good_v]
    coeff_div[:] = (tri_area / 3.0) * (
        v_grad[tri_rows[:, 0]] * inv_v_area[tri_rows[:, 0]]
        + v_grad[tri_rows[:, 1]] * inv_v_area[tri_rows[:, 1]]
        + v_grad[tri_rows[:, 2]] * inv_v_area[tri_rows[:, 2]]
    )
    return coeff_div


def _leaflet_triangle_payload(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    cache_tag: str,
    ctx=None,
) -> dict[str, np.ndarray | None]:
    """Return cached leaflet-masked triangle geometry for fixed positions."""
    mesh.build_position_cache()
    absent_mask = None
    absent_key = None
    if cache_tag in {"in", "out"}:
        absent_mask = leaflet_absent_vertex_mask(mesh, global_params, leaflet=cache_tag)
        absent_key = id(absent_mask)

    use_cache = mesh._geometry_cache_active(positions)
    cache_attr = f"_bending_tilt_leaflet_triangle_payload_cache_{cache_tag}"
    cache_key = (
        int(mesh._version),
        int(mesh._facet_loops_version),
        int(mesh._vertex_ids_version),
        id(positions),
        absent_key,
        str(global_params.get("rim_slope_match_mode") or "")
        if global_params is not None
        else "",
        str(global_params.get("rim_slope_match_outer_group") or "")
        if global_params is not None
        else "",
        str(global_params.get("rim_slope_match_group") or "")
        if global_params is not None
        else "",
        str(global_params.get("rim_slope_match_disk_group") or "")
        if global_params is not None
        else "",
    )
    if use_cache:
        cached = getattr(mesh, cache_attr, None)
        if cached is not None and cached.get("key") == cache_key:
            payload = cached["value"]
            if isinstance(payload, dict) and "tri_area" not in payload:
                payload = dict(payload)
                payload.setdefault("tri_area", None)
                cached["value"] = payload
            return payload

    k_vecs, vertex_areas_vor, weights_full, tri_rows_full = compute_curvature_data(
        mesh, positions, index_map
    )
    if tri_rows_full.size == 0:
        payload = {
            "k_vecs": k_vecs,
            "vertex_areas_vor": vertex_areas_vor,
            "weights_full": weights_full,
            "tri_rows_full": tri_rows_full,
            "weights": weights_full,
            "tri_rows": tri_rows_full,
            "tri_keep": np.zeros(0, dtype=bool),
            "tri_area": None,
            "g0": None,
            "g1": None,
            "g2": None,
        }
        if use_cache:
            setattr(mesh, cache_attr, {"key": cache_key, "value": payload})
        return payload

    tri_keep = np.zeros(0, dtype=bool)
    weights = weights_full
    tri_rows = tri_rows_full
    if absent_mask is not None:
        tri_keep = leaflet_present_triangle_mask(
            mesh, tri_rows_full, absent_vertex_mask=absent_mask
        )
        if tri_keep.size:
            tri_rows = tri_rows_full[tri_keep]
            weights = weights_full[tri_keep]

    transition_mask = _shared_rim_support_transition_triangle_mask(
        mesh,
        global_params,
        tri_rows_full,
        keep_physical_outer_edge=str(cache_tag) == "out",
    )
    if transition_mask is not None:
        keep_non_transition = ~transition_mask
        if tri_keep.size:
            tri_keep = tri_keep & keep_non_transition
        else:
            tri_keep = keep_non_transition
        tri_rows = tri_rows_full[tri_keep]
        weights = weights_full[tri_keep]

    g0_use = None
    g1_use = None
    g2_use = None
    area_use = None
    if ctx is not None:
        _area_cache, g0_cache, g1_cache, g2_cache, tri_rows_cache = (
            ctx.geometry.p1_triangle_shape_gradients(mesh, positions)
        )
    else:
        _area_cache, g0_cache, g1_cache, g2_cache, tri_rows_cache = (
            mesh.p1_triangle_shape_gradient_cache(positions)
        )
    if tri_rows_cache.size and tri_rows_cache.shape[0] == tri_rows_full.shape[0]:
        if tri_keep.size:
            area_use = _area_cache[tri_keep]
            g0_use = g0_cache[tri_keep]
            g1_use = g1_cache[tri_keep]
            g2_use = g2_cache[tri_keep]
        else:
            area_use = _area_cache
            g0_use = g0_cache
            g1_use = g1_cache
            g2_use = g2_cache

    payload = {
        "k_vecs": k_vecs,
        "vertex_areas_vor": vertex_areas_vor,
        "weights_full": weights_full,
        "tri_rows_full": tri_rows_full,
        "weights": weights,
        "tri_rows": tri_rows,
        "tri_keep": tri_keep,
        "tri_area": area_use,
        "g0": g0_use,
        "g1": g1_use,
        "g2": g2_use,
    }
    if use_cache:
        setattr(mesh, cache_attr, {"key": cache_key, "value": payload})
    return payload


def _leaflet_static_tilt_payload(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    k_vecs: np.ndarray,
    vertex_areas_vor: np.ndarray,
    tri_rows: np.ndarray,
    kappa_key: str,
    cache_tag: str,
) -> dict[str, np.ndarray]:
    """Return cached fixed-geometry arrays used by tilt-only leaflet coupling."""
    model = _energy_model(global_params)
    if model != "helfrich":
        model = "helfrich"
    kappa_default = _resolve_bending_modulus(global_params, kappa_key)
    c0_key = f"spontaneous_curvature_{cache_tag}"
    c0_default = global_params.get(c0_key)
    if c0_default is None:
        c0_default = (
            _spontaneous_curvature(global_params) if model == "helfrich" else 0.0
        )
    c0_default = float(c0_default or 0.0)

    presets = _assume_J0_presets(global_params, cache_tag=cache_tag)
    radius_max = _assume_J0_radius_max(global_params, cache_tag=cache_tag)
    center_xy = _assume_J0_center_xy(global_params)
    center_key = (float(center_xy[0]), float(center_xy[1]))
    region_mode = _base_term_region_mode(global_params)
    region_radius = _base_term_region_radius(global_params)
    boundary_group = _base_term_boundary_group(global_params, cache_tag=cache_tag)

    use_cache = mesh._geometry_cache_active(positions)
    cache_attr = f"_bending_tilt_leaflet_static_cache_{cache_tag}"
    cache_key = (
        int(mesh._version),
        int(mesh._facet_loops_version),
        int(mesh._vertex_ids_version),
        int(mesh._topology_version),
        id(positions),
        id(k_vecs),
        id(vertex_areas_vor),
        id(tri_rows),
        str(model),
        float(kappa_default),
        float(c0_default),
        None if boundary_group is None else str(boundary_group),
        presets,
        None if radius_max is None else float(radius_max),
        center_key,
        str(region_mode),
        None if region_radius is None else float(region_radius),
    )
    if use_cache:
        cached = getattr(mesh, cache_attr, None)
        if cached is not None and cached.get("key") == cache_key:
            return cached["value"]

    kappa_arr, c0_arr = _per_vertex_params_leaflet(
        mesh, global_params, model=model, kappa_key=kappa_key, cache_tag=cache_tag
    )
    safe_areas_vor = np.maximum(vertex_areas_vor, 1e-12)
    k_mag = np.linalg.norm(k_vecs, axis=1)
    h_vor = k_mag / (2.0 * safe_areas_vor)
    is_interior = _interior_mask_leaflet(
        mesh, global_params, cache_tag=cache_tag, index_map=index_map
    )

    base_term = (2.0 * h_vor) - c0_arr
    base_term[~is_interior] = 0.0
    if presets:
        rows = _collect_preset_rows(
            mesh,
            presets=presets,
            cache_tag=cache_tag,
            index_map=index_map,
            radius_max=radius_max,
            center_xy=center_xy,
        )
        if rows.size:
            base_term[rows] = 0.0
    region_rows = _base_term_region_zero_rows(
        mesh,
        global_params,
        cache_tag=cache_tag,
        index_map=index_map,
    )
    if region_rows.size:
        base_term[region_rows] = 0.0

    value = {
        "base_tri": base_term[tri_rows],
        "kappa_tri": kappa_arr[tri_rows],
    }
    if use_cache:
        setattr(mesh, cache_attr, {"key": cache_key, "value": value})
    return value


def _total_energy_leaflet(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts: np.ndarray,
    kappa_key: str,
    cache_tag: str,
    div_sign: float,
) -> float:
    """Energy-only helper for finite-difference debugging."""
    payload = _leaflet_triangle_payload(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
        cache_tag=cache_tag,
    )
    k_vecs = np.asarray(payload["k_vecs"], dtype=float)
    vertex_areas_vor = np.asarray(payload["vertex_areas_vor"], dtype=float)
    weights = np.asarray(payload["weights"], dtype=float)
    tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
    tri_area = payload.get("tri_area")
    if tri_rows.size == 0:
        return 0.0

    transport_model = _resolve_transport_model(
        global_params.get("tilt_transport_model", "ambient_v1")
        if global_params is not None
        else "ambient_v1"
    )
    g0 = payload["g0"]
    g1 = payload["g1"]
    g2 = payload["g2"]
    if g0 is not None and g1 is not None and g2 is not None:
        div_tri = compute_divergence_from_basis(
            mesh=mesh,
            tilts=tilts,
            tri_rows=tri_rows,
            g0=np.asarray(g0, dtype=float),
            g1=np.asarray(g1, dtype=float),
            g2=np.asarray(g2, dtype=float),
            positions=positions,
            transport_model=transport_model,
        )
    else:
        div_tri, _, _, _, _ = p1_triangle_divergence(
            mesh=mesh,
            positions=positions,
            tilts=tilts,
            tri_rows=tri_rows,
            transport_model=transport_model,
        )
    div_term = float(div_sign) * div_tri
    div_term = _apply_inner_divergence_update_mode(
        mesh,
        global_params,
        positions=positions,
        tri_rows=tri_rows,
        cache_tag=cache_tag,
        div_term=div_term,
    )
    use_recovered_div = _use_inner_recovered_divergence(
        global_params, cache_tag=cache_tag
    )
    if use_recovered_div:
        if tri_area is None:
            tri_area = mesh.p1_triangle_shape_gradient_cache(positions)[0]
        tri_area = np.asarray(tri_area, dtype=float)
        div_eval_tri, _, _ = _inner_recovered_divergence(
            global_params=global_params,
            cache_tag=cache_tag,
            tri_rows=tri_rows,
            tri_area=tri_area,
            div_tri=div_term,
            n_vertices=len(mesh.vertex_ids),
            scratch_tag=f"btl_{cache_tag}",
        )
    else:
        div_eval_tri = div_term

    _, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        compute_vertex_areas=False,
    )
    static_payload = _leaflet_static_tilt_payload(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
        k_vecs=k_vecs,
        vertex_areas_vor=vertex_areas_vor,
        tri_rows=tri_rows,
        kappa_key=kappa_key,
        cache_tag=cache_tag,
    )
    base_tri = np.asarray(static_payload["base_tri"], dtype=float)
    term_tri = base_tri + div_eval_tri[:, None]
    va_eff = np.stack([va0_eff, va1_eff, va2_eff], axis=1)
    kappa_tri = np.asarray(static_payload["kappa_tri"], dtype=float)

    return float(0.5 * np.sum(kappa_tri * term_tri**2 * va_eff))


def _finite_difference_gradient_shape_leaflet(
    mesh: Mesh,
    global_params,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    tilts: np.ndarray,
    kappa_key: str,
    cache_tag: str,
    div_sign: float,
    eps: float,
) -> np.ndarray:
    """Energy-consistent shape gradient by central differences (slow)."""
    grad = np.zeros_like(positions)
    base = positions.copy()
    for row, vid in enumerate(mesh.vertex_ids):
        if getattr(mesh.vertices[int(vid)], "fixed", False):
            continue
        for d in range(3):
            pos_plus = base.copy()
            pos_minus = base.copy()
            pos_plus[row, d] += eps
            pos_minus[row, d] -= eps
            e_plus = _total_energy_leaflet(
                mesh,
                global_params,
                positions=pos_plus,
                index_map=index_map,
                tilts=tilts,
                kappa_key=kappa_key,
                cache_tag=cache_tag,
                div_sign=div_sign,
            )
            e_minus = _total_energy_leaflet(
                mesh,
                global_params,
                positions=pos_minus,
                index_map=index_map,
                tilts=tilts,
                kappa_key=kappa_key,
                cache_tag=cache_tag,
                div_sign=div_sign,
            )
            grad[row, d] = (e_plus - e_minus) / (2.0 * eps)
    return grad


def compute_energy_and_gradient_array_leaflet(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    positions: np.ndarray,
    index_map: Dict[int, int],
    grad_arr: np.ndarray | None,
    ctx=None,
    tilts: np.ndarray,
    tilt_grad_arr: np.ndarray | None,
    kappa_key: str,
    cache_tag: str,
    div_sign: float = 1.0,
) -> float:
    """Compute coupled bending+tilt energy and accumulate gradients."""
    _ = param_resolver
    payload = _leaflet_triangle_payload(
        mesh,
        global_params,
        positions=positions,
        index_map=index_map,
        cache_tag=cache_tag,
        ctx=ctx,
    )
    k_vecs = np.asarray(payload["k_vecs"], dtype=float)
    vertex_areas_vor = np.asarray(payload["vertex_areas_vor"], dtype=float)
    tri_rows_full = np.asarray(payload["tri_rows_full"], dtype=np.int32)
    weights = np.asarray(payload["weights"], dtype=float)
    tri_rows = np.asarray(payload["tri_rows"], dtype=np.int32)
    tri_keep = np.asarray(payload["tri_keep"], dtype=bool)
    tri_area = payload.get("tri_area")

    if tri_rows_full.size == 0 or tri_rows.size == 0:
        return 0.0

    tilts = np.asarray(tilts, dtype=float)
    if tilts.shape != (len(mesh.vertex_ids), 3):
        raise ValueError("tilts must have shape (N_vertices, 3)")

    transport_model = _resolve_transport_model(
        global_params.get("tilt_transport_model", "ambient_v1")
        if global_params is not None
        else "ambient_v1"
    )

    g0_use = payload["g0"]
    g1_use = payload["g1"]
    g2_use = payload["g2"]
    if g0_use is not None and g1_use is not None and g2_use is not None:
        div_tri = compute_divergence_from_basis(
            mesh=mesh,
            tilts=tilts,
            tri_rows=tri_rows,
            g0=np.asarray(g0_use, dtype=float),
            g1=np.asarray(g1_use, dtype=float),
            g2=np.asarray(g2_use, dtype=float),
            positions=positions,
            transport_model=transport_model,
        )
        g0 = np.asarray(g0_use, dtype=float)
        g1 = np.asarray(g1_use, dtype=float)
        g2 = np.asarray(g2_use, dtype=float)
    else:
        div_tri, _, g0, g1, g2 = p1_triangle_divergence(
            mesh=mesh,
            positions=positions,
            tilts=tilts,
            tri_rows=tri_rows,
            transport_model=transport_model,
        )
    div_term = float(div_sign) * div_tri
    div_term = _apply_inner_divergence_update_mode(
        mesh,
        global_params,
        positions=positions,
        tri_rows=tri_rows,
        cache_tag=cache_tag,
        div_term=div_term,
    )
    use_recovered_div = _use_inner_recovered_divergence(
        global_params, cache_tag=cache_tag
    )
    if use_recovered_div:
        if tri_area is None:
            tri_area = mesh.p1_triangle_shape_gradient_cache(positions)[0]
        tri_area = np.asarray(tri_area, dtype=float)
        div_eval_tri, _, div_eval_vertex_area = _inner_recovered_divergence(
            global_params=global_params,
            cache_tag=cache_tag,
            tri_rows=tri_rows,
            tri_area=tri_area,
            div_tri=div_term,
            n_vertices=len(mesh.vertex_ids),
            ctx=ctx,
            scratch_tag=f"btl_{cache_tag}",
        )
    else:
        div_eval_tri = div_term
        div_eval_vertex_area = None

    vertex_areas_eff, va0_eff, va1_eff, va2_eff = _compute_effective_areas(
        mesh,
        positions,
        tri_rows,
        weights,
        index_map,
        cache_token=f"bending_tilt_leaflet_{cache_tag}",
        compute_vertex_areas=grad_arr is not None,
    )
    if grad_arr is None:
        static_payload = _leaflet_static_tilt_payload(
            mesh,
            global_params,
            positions=positions,
            index_map=index_map,
            k_vecs=k_vecs,
            vertex_areas_vor=vertex_areas_vor,
            tri_rows=tri_rows,
            kappa_key=kappa_key,
            cache_tag=cache_tag,
        )
        base_tri = np.asarray(static_payload["base_tri"], dtype=float)
        kappa_tri = np.asarray(static_payload["kappa_tri"], dtype=float)
        if ctx is not None:
            term_tri = ctx.scratch_array(
                f"btl_{cache_tag}_tilt_only_term_tri",
                shape=base_tri.shape,
                dtype=base_tri.dtype,
            )
            np.copyto(term_tri, base_tri)
            term_tri += div_eval_tri[:, None]
        else:
            term_tri = base_tri + div_eval_tri[:, None]
        total_energy = float(
            0.5
            * np.sum(
                (kappa_tri[:, 0] * term_tri[:, 0] ** 2 * va0_eff)
                + (kappa_tri[:, 1] * term_tri[:, 1] ** 2 * va1_eff)
                + (kappa_tri[:, 2] * term_tri[:, 2] ** 2 * va2_eff)
            )
        )

        if tilt_grad_arr is not None:
            tilt_grad_arr = np.asarray(tilt_grad_arr, dtype=float)
            if tilt_grad_arr.shape != (len(mesh.vertex_ids), 3):
                raise ValueError("tilt_grad_arr must have shape (N_vertices, 3)")

            dE_ddiv_base, mode_stats = _inner_bending_tilt_dE_ddiv(
                mesh=mesh,
                global_params=global_params,
                cache_tag=cache_tag,
                kappa_tri=kappa_tri,
                base_tri=base_tri,
                div_term=div_eval_tri,
                va0_eff=va0_eff,
                va1_eff=va1_eff,
                va2_eff=va2_eff,
            )
            if str(cache_tag) == "in":
                setattr(mesh, "_last_bending_tilt_in_update_mode_stats", mode_stats)
            if use_recovered_div:
                dE_ddiv = _inner_recovered_divergence_pullback(
                    global_params=global_params,
                    cache_tag=cache_tag,
                    tri_rows=tri_rows,
                    tri_area=tri_area,
                    coeff_div_eval=float(div_sign) * dE_ddiv_base,
                    v_area=div_eval_vertex_area,
                    ctx=ctx,
                    scratch_tag=f"btl_{cache_tag}",
                )
            else:
                dE_ddiv = float(div_sign) * dE_ddiv_base
            factor = dE_ddiv[:, None]
            _accumulate_leaflet_tilt_gradient(
                tilt_grad_arr,
                tri_rows,
                factor,
                g0,
                g1,
                g2,
                ctx=ctx,
                scratch_tag=f"btl_{cache_tag}_tilt_only_scaled",
            )
        return float(total_energy)

    safe_areas_vor = np.maximum(vertex_areas_vor, 1e-12)

    model = _energy_model(global_params)
    if model != "helfrich":
        model = "helfrich"
    kappa_arr, c0_arr = _per_vertex_params_leaflet(
        mesh, global_params, model=model, kappa_key=kappa_key, cache_tag=cache_tag
    )

    k_mag = np.linalg.norm(k_vecs, axis=1)
    H_vor = k_mag / (2.0 * safe_areas_vor)

    is_interior = _interior_mask_leaflet(
        mesh, global_params, cache_tag=cache_tag, index_map=index_map
    )

    base_term = (2.0 * H_vor) - c0_arr
    base_term[~is_interior] = 0.0
    presets = _assume_J0_presets(global_params, cache_tag=cache_tag)
    if presets:
        radius_max = _assume_J0_radius_max(global_params, cache_tag=cache_tag)
        center_xy = _assume_J0_center_xy(global_params)
        rows = _collect_preset_rows(
            mesh,
            presets=presets,
            cache_tag=cache_tag,
            index_map=index_map,
            radius_max=radius_max,
            center_xy=center_xy,
        )
        if rows.size:
            base_term[rows] = 0.0
    region_rows = _base_term_region_zero_rows(
        mesh,
        global_params,
        cache_tag=cache_tag,
        index_map=index_map,
    )
    if region_rows.size:
        base_term[region_rows] = 0.0

    base_tri = base_term[tri_rows]
    term_tri = base_tri + div_eval_tri[:, None]
    kappa_tri = kappa_arr[tri_rows]
    total_energy = float(
        0.5
        * np.sum(
            (kappa_tri[:, 0] * term_tri[:, 0] ** 2 * va0_eff)
            + (kappa_tri[:, 1] * term_tri[:, 1] ** 2 * va1_eff)
            + (kappa_tri[:, 2] * term_tri[:, 2] ** 2 * va2_eff)
        )
    )

    if ctx is not None:
        ratio = ctx.scratch_array(
            f"btl_{cache_tag}_ratio",
            shape=vertex_areas_eff.shape,
            dtype=vertex_areas_eff.dtype,
        )
    else:
        ratio = np.zeros_like(vertex_areas_eff)
    mask_vor = safe_areas_vor > 1e-15
    ratio[mask_vor] = vertex_areas_eff[mask_vor] / safe_areas_vor[mask_vor]

    if ctx is not None:
        div_eff_num = ctx.scratch_array(
            f"btl_{cache_tag}_div_eff_num",
            shape=base_term.shape,
            dtype=base_term.dtype,
        )
    else:
        div_eff_num = np.zeros_like(base_term)
    div_eff_source = div_eval_tri if use_recovered_div else div_term
    div_eff_num = scatter_triangle_scalar_to_vertices(
        tri_rows=tri_rows,
        w0=va0_eff * div_eff_source,
        w1=va1_eff * div_eff_source,
        w2=va2_eff * div_eff_source,
        n_vertices=base_term.shape[0],
        out=div_eff_num,
    )
    if ctx is not None:
        div_eff = ctx.scratch_array(
            f"btl_{cache_tag}_div_eff",
            shape=base_term.shape,
            dtype=base_term.dtype,
        )
    else:
        div_eff = np.zeros_like(base_term)
    mask_eff = vertex_areas_eff > 1e-20
    div_eff[mask_eff] = div_eff_num[mask_eff] / vertex_areas_eff[mask_eff]

    term = base_term + div_eff
    term[~is_interior] = 0.0

    suppress_shape_cross = _use_stage_a_inner_shape_cross_suppression(
        mesh, global_params, cache_tag=cache_tag
    )
    setattr(
        mesh,
        f"_last_bending_tilt_{cache_tag}_shape_cross_stats",
        {
            "enabled": bool(suppress_shape_cross),
            "cache_tag": str(cache_tag),
            "lane": str(global_params.get("theory_parity_lane") or "")
            if global_params is not None
            else "",
        },
    )

    mode = _gradient_mode(global_params)
    normals = _vertex_normals(mesh, positions, tri_rows)
    if ctx is not None:
        K_dir = ctx.scratch_array(
            f"btl_{cache_tag}_K_dir", shape=k_vecs.shape, dtype=k_vecs.dtype
        )
    else:
        K_dir = np.zeros_like(k_vecs)
    mask_k = k_mag > 1e-15
    K_dir[mask_k] = k_vecs[mask_k] / k_mag[mask_k][:, None]
    K_dir[~mask_k] = normals[~mask_k]

    shape_term = base_term if suppress_shape_cross else term
    scale_K = (kappa_arr * shape_term * ratio).astype(float, copy=False)
    if ctx is not None:
        factor_K_vec = ctx.scratch_array(
            f"btl_{cache_tag}_factor_K_vec", shape=K_dir.shape, dtype=K_dir.dtype
        )
    else:
        factor_K_vec = np.empty_like(K_dir, order="F")
    np.multiply(K_dir, scale_K[:, None], out=factor_K_vec)

    transition_operator_enabled = _use_stage_a_outer_grad_linear_transition_operator(
        global_params, cache_tag=cache_tag
    )
    transition_operator_stats = {
        "enabled": False,
        "mode": "off",
        "cache_tag": str(cache_tag),
        "lane": str(global_params.get("theory_parity_lane") or "")
        if global_params is not None
        else "",
    }

    if suppress_shape_cross:
        fA_eff = 0.5 * kappa_arr * (base_term**2 + div_eff**2)
        fA_vor = -2.0 * kappa_arr * base_term * ratio * H_vor
    else:
        fA_eff = 0.5 * kappa_arr * term**2
        fA_vor = -2.0 * kappa_arr * term * ratio * H_vor

    if mode == "finite_difference":  # pragma: no cover - slow debugging path
        eps = float(global_params.get("bending_fd_eps", 1e-6))
        grad_arr[:] += _finite_difference_gradient_shape_leaflet(
            mesh,
            global_params,
            positions=positions,
            index_map=index_map,
            tilts=tilts,
            kappa_key=kappa_key,
            cache_tag=cache_tag,
            div_sign=div_sign,
            eps=eps,
        )
    elif mode == "approx":
        if transition_operator_enabled:
            grad_linear_raw, transition_operator_stats = (
                _apply_transition_aware_beltrami_laplacian(
                    mesh,
                    positions=positions,
                    weights=weights,
                    tri_rows=tri_rows,
                    tri_rows_full=tri_rows_full,
                    field=factor_K_vec,
                    cache_tag=cache_tag,
                )
            )
            grad_arr[:] -= grad_linear_raw
        else:
            grad_arr[:] -= _apply_beltrami_laplacian(weights, tri_rows, factor_K_vec)
        grad_arr[~is_interior] = 0.0
    else:
        # --- Analytic gradient backpropagation (copied from bending.py) ---
        v0_idxs, v1_idxs, v2_idxs = tri_rows[:, 0], tri_rows[:, 1], tri_rows[:, 2]
        v0, v1, v2 = positions[v0_idxs], positions[v1_idxs], positions[v2_idxs]
        e0, e1, e2 = v2 - v1, v0 - v2, v1 - v0
        c0, c1, c2 = weights[:, 0], weights[:, 1], weights[:, 2]

        cached = _cached_cotan_gradients(
            mesh,
            positions=positions,
            tri_rows=tri_rows_full if tri_keep.size else tri_rows,
        )
        if cached is not None and tri_keep.size:
            cached = tuple(arr[tri_keep] for arr in cached)

        # Term 1: Variation assuming cotans constant (L constant)
        # factor_K_vec already zeroed at boundaries
        if transition_operator_enabled:
            grad_linear_raw, transition_operator_stats = (
                _apply_transition_aware_beltrami_laplacian(
                    mesh,
                    positions=positions,
                    weights=weights,
                    tri_rows=tri_rows,
                    tri_rows_full=tri_rows_full,
                    field=factor_K_vec,
                    cache_tag=cache_tag,
                )
            )
            grad_linear = -grad_linear_raw
        else:
            grad_linear = -_apply_beltrami_laplacian(weights, tri_rows, factor_K_vec)

        # Term 2: Variation of L (cotangents)
        fK = factor_K_vec
        dE_dc0 = -0.5 * np.einsum("ij,ij->i", fK[v1_idxs] - fK[v2_idxs], v1 - v2)
        dE_dc1 = -0.5 * np.einsum("ij,ij->i", fK[v2_idxs] - fK[v0_idxs], v2 - v0)
        dE_dc2 = -0.5 * np.einsum("ij,ij->i", fK[v0_idxs] - fK[v1_idxs], v0 - v1)

        if cached is not None:
            (
                g_c0_u,
                g_c0_v,
                g_c1_u,
                g_c1_v,
                g_c2_u,
                g_c2_v,
                gc0u,
                gc0v,
                gc1u,
                gc1v,
                gc2u,
                gc2v,
            ) = cached
        else:
            u0, v0_vec = v1 - v0, v2 - v0
            g_c0_u, g_c0_v = _grad_cotan(u0, v0_vec)
            u1, v1_vec = v2 - v1, v0 - v1
            g_c1_u, g_c1_v = _grad_cotan(u1, v1_vec)
            u2, v2_vec = v0 - v2, v1 - v2
            g_c2_u, g_c2_v = _grad_cotan(u2, v2_vec)

        if ctx is not None:
            grad_cot = ctx.scratch_array(
                f"btl_{cache_tag}_grad_cot",
                shape=positions.shape,
                dtype=positions.dtype,
            )
        else:
            grad_cot = np.zeros_like(positions)
        val0, val1, val2 = dE_dc0[:, None], dE_dc1[:, None], dE_dc2[:, None]
        np.add.at(grad_cot, v1_idxs, val0 * g_c0_u)
        np.add.at(grad_cot, v2_idxs, val0 * g_c0_v)
        np.add.at(grad_cot, v0_idxs, val0 * -(g_c0_u + g_c0_v))
        np.add.at(grad_cot, v2_idxs, val1 * g_c1_u)
        np.add.at(grad_cot, v0_idxs, val1 * g_c1_v)
        np.add.at(grad_cot, v1_idxs, val1 * -(g_c1_u + g_c1_v))
        np.add.at(grad_cot, v0_idxs, val2 * g_c2_u)
        np.add.at(grad_cot, v1_idxs, val2 * g_c2_v)
        np.add.at(grad_cot, v2_idxs, val2 * -(g_c2_u + g_c2_v))

        # --- Area Gradients (Step 2: Propagate area reassignment) ---

        tri_is_int = is_interior[tri_rows]
        interior_counts = np.sum(tri_is_int, axis=1)

        tri_fA_eff = fA_eff[tri_rows]
        sum_fA_eff_int = np.sum(tri_fA_eff * tri_is_int, axis=1)

        if ctx is not None:
            avg_fA_eff = ctx.scratch_array(
                f"btl_{cache_tag}_avg_fA_eff",
                shape=(len(tri_rows),),
                dtype=float,
            )
        else:
            avg_fA_eff = np.zeros(len(tri_rows), dtype=float)
        mask_has_int = interior_counts > 0
        avg_fA_eff[mask_has_int] = (
            sum_fA_eff_int[mask_has_int] / interior_counts[mask_has_int]
        )

        C_eff = np.where(tri_is_int, tri_fA_eff, avg_fA_eff[:, None])
        tri_fA_vor = fA_vor[tri_rows]
        C = C_eff + tri_fA_vor

        if ctx is not None:
            grad_area = ctx.scratch_array(
                f"btl_{cache_tag}_grad_area",
                shape=positions.shape,
                dtype=positions.dtype,
            )
        else:
            grad_area = np.zeros_like(positions)

        is_obtuse = (c0 < 0) | (c1 < 0) | (c2 < 0)
        m_std = ~is_obtuse

        if np.any(m_std):
            c0s, c1s, c2s = c0[m_std], c1[m_std], c2[m_std]
            C0s, C1s, C2s = C[m_std, 0], C[m_std, 1], C[m_std, 2]
            e0s, e1s, e2s = e0[m_std], e1[m_std], e2[m_std]
            v0s, v1s, v2s = v0_idxs[m_std], v1_idxs[m_std], v2_idxs[m_std]

            coeff = 0.25 * c1s * C0s
            np.add.at(grad_area, v0s, coeff[:, None] * e1s)
            np.add.at(grad_area, v2s, -coeff[:, None] * e1s)
            coeff = 0.25 * c2s * C0s
            np.add.at(grad_area, v1s, coeff[:, None] * e2s)
            np.add.at(grad_area, v0s, -coeff[:, None] * e2s)
            coeff = 0.25 * c2s * C1s
            np.add.at(grad_area, v1s, coeff[:, None] * e2s)
            np.add.at(grad_area, v0s, -coeff[:, None] * e2s)
            coeff = 0.25 * c0s * C1s
            np.add.at(grad_area, v2s, coeff[:, None] * e0s)
            np.add.at(grad_area, v1s, -coeff[:, None] * e0s)
            coeff = 0.25 * c0s * C2s
            np.add.at(grad_area, v2s, coeff[:, None] * e0s)
            np.add.at(grad_area, v1s, -coeff[:, None] * e0s)
            coeff = 0.25 * c1s * C2s
            np.add.at(grad_area, v0s, coeff[:, None] * e1s)
            np.add.at(grad_area, v2s, -coeff[:, None] * e1s)

            l0sq = np.einsum("ij,ij->i", e0s, e0s)
            l1sq = np.einsum("ij,ij->i", e1s, e1s)
            l2sq = np.einsum("ij,ij->i", e2s, e2s)

            coeff_c0 = 0.125 * l0sq * (C1s + C2s)
            coeff_c1 = 0.125 * l1sq * (C0s + C2s)
            coeff_c2 = 0.125 * l2sq * (C0s + C1s)

            if cached is None:
                gc0u, gc0v = _grad_cotan(e2s, -e1s)
                gc1u, gc1v = _grad_cotan(e0s, -e2s)
                gc2u, gc2v = _grad_cotan(e1s, -e0s)
            else:
                gc0u, gc0v = gc0u[m_std], gc0v[m_std]
                gc1u, gc1v = gc1u[m_std], gc1v[m_std]
                gc2u, gc2v = gc2u[m_std], gc2v[m_std]

            v_c0, v_c1, v_c2 = (
                coeff_c0[:, None],
                coeff_c1[:, None],
                coeff_c2[:, None],
            )
            np.add.at(grad_area, v1s, v_c0 * gc0u)
            np.add.at(grad_area, v2s, v_c0 * gc0v)
            np.add.at(grad_area, v0s, v_c0 * -(gc0u + gc0v))
            np.add.at(grad_area, v2s, v_c1 * gc1u)
            np.add.at(grad_area, v0s, v_c1 * gc1v)
            np.add.at(grad_area, v1s, v_c1 * -(gc1u + gc1v))
            np.add.at(grad_area, v0s, v_c2 * gc2u)
            np.add.at(grad_area, v1s, v_c2 * gc2v)
            np.add.at(grad_area, v2s, v_c2 * -(gc2u + gc2v))

        if np.any(is_obtuse):
            for i, m_sub in enumerate([(c0 < 0), (c1 < 0), (c2 < 0)]):
                m_do = m_sub & is_obtuse
                if np.any(m_do):
                    v0o, v1o, v2o = v0_idxs[m_do], v1_idxs[m_do], v2_idxs[m_do]
                    gT_u, gT_v = grad_triangle_area(
                        positions[v1o] - positions[v0o],
                        positions[v2o] - positions[v0o],
                    )

                    C0o, C1o, C2o = C[m_do, 0], C[m_do, 1], C[m_do, 2]
                    if i == 0:
                        factor = (0.5 * C0o + 0.25 * C1o + 0.25 * C2o)[:, None]
                    elif i == 1:
                        factor = (0.5 * C1o + 0.25 * C0o + 0.25 * C2o)[:, None]
                    else:
                        factor = (0.5 * C2o + 0.25 * C0o + 0.25 * C1o)[:, None]

                    np.add.at(grad_area, v1o, factor * gT_u)
                    np.add.at(grad_area, v2o, factor * gT_v)
                    np.add.at(grad_area, v0o, factor * -(gT_u + gT_v))

        grad_arr[:] += grad_linear
        grad_arr[:] += grad_cot
        grad_arr[:] += grad_area

    setattr(
        mesh,
        f"_last_bending_tilt_{cache_tag}_grad_linear_transition_stats",
        transition_operator_stats,
    )

    if tilt_grad_arr is not None:
        tilt_grad_arr = np.asarray(tilt_grad_arr, dtype=float)
        if tilt_grad_arr.shape != (len(mesh.vertex_ids), 3):
            raise ValueError("tilt_grad_arr must have shape (N_vertices, 3)")

        dE_ddiv_base, mode_stats = _inner_bending_tilt_dE_ddiv(
            mesh=mesh,
            global_params=global_params,
            cache_tag=cache_tag,
            kappa_tri=kappa_tri,
            base_tri=base_tri,
            div_term=div_eval_tri,
            va0_eff=va0_eff,
            va1_eff=va1_eff,
            va2_eff=va2_eff,
        )
        if str(cache_tag) == "in":
            setattr(mesh, "_last_bending_tilt_in_update_mode_stats", mode_stats)
        if use_recovered_div:
            dE_ddiv = _inner_recovered_divergence_pullback(
                global_params=global_params,
                cache_tag=cache_tag,
                tri_rows=tri_rows,
                tri_area=tri_area,
                coeff_div_eval=float(div_sign) * dE_ddiv_base,
                v_area=div_eval_vertex_area,
                ctx=ctx,
                scratch_tag=f"btl_{cache_tag}",
            )
        else:
            dE_ddiv = float(div_sign) * dE_ddiv_base
        factor = dE_ddiv[:, None]

        np.add.at(tilt_grad_arr, tri_rows[:, 0], factor * g0)
        np.add.at(tilt_grad_arr, tri_rows[:, 1], factor * g1)
        np.add.at(tilt_grad_arr, tri_rows[:, 2], factor * g2)

    return total_energy


def compute_energy_and_gradient_leaflet(
    mesh: Mesh,
    global_params,
    param_resolver,
    *,
    tilts: np.ndarray,
    kappa_key: str,
    cache_tag: str,
    div_sign: float = 1.0,
    compute_gradient: bool = True,
) -> (
    tuple[float, Dict[int, np.ndarray]]
    | tuple[float, Dict[int, np.ndarray], Dict[int, np.ndarray]]
):
    """Dict-based wrapper returning (E, shape_grad[, tilt_grad])."""
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_arr = np.zeros_like(positions)
    tilt_grad_arr = np.zeros_like(positions) if compute_gradient else None
    energy = compute_energy_and_gradient_array_leaflet(
        mesh,
        global_params,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_arr,
        tilts=tilts,
        tilt_grad_arr=tilt_grad_arr,
        kappa_key=kappa_key,
        cache_tag=cache_tag,
        div_sign=div_sign,
    )
    if not compute_gradient:
        return float(energy), {}

    shape_grad = {
        int(vid): grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if np.any(grad_arr[row])
    }
    tilt_grad = {
        int(vid): tilt_grad_arr[row].copy()
        for row, vid in enumerate(mesh.vertex_ids)
        if tilt_grad_arr is not None and np.any(tilt_grad_arr[row])
    }
    return float(energy), shape_grad, tilt_grad


__all__ = [
    "compute_energy_and_gradient_array_leaflet",
    "compute_energy_and_gradient_leaflet",
]

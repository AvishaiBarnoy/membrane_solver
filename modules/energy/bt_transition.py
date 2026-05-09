"""Transition-region curvature operators for leaflet-specific bending-tilt coupling."""

from __future__ import annotations

import numpy as np

from geometry.entities import Mesh

from .bt_utils import _mean_reconstructed_field

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
    from modules.energy.bending import _apply_beltrami_laplacian

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

    edge_unique_idx = np.asarray(payload["edge_unique_idx"], dtype=np.int32)
    unique_edge_u = np.asarray(payload["unique_edge_u"], dtype=np.int32)
    unique_edge_v = np.asarray(payload["unique_edge_v"], dtype=np.int32)
    transition_mask = np.asarray(payload["transition_mask"], dtype=bool)
    partial_vertex_mask = np.asarray(payload["partial_vertex_mask"], dtype=bool)
    halo_full_vertex_mask = np.asarray(payload["halo_full_vertex_mask"], dtype=bool)
    transition_full_vertex_mask = np.asarray(
        payload["transition_full_vertex_mask"], dtype=bool
    )
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
                "is_fallback": bool(np.any(fallback_vertex_mask[rows])),
                "total_z_mean": float(np.mean(out[rows, 2])),
            }
        )

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
        "shell_summary": shell_summary,
    }
    return out, stats

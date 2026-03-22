"""Topology checking and mesh consistency functions."""

import logging
from typing import Dict, List, Tuple

import numpy as np

from geometry.entities import Mesh, _fast_cross

logger = logging.getLogger("membrane_solver")


def check_max_normal_change_positions(
    mesh: Mesh,
    original_positions: np.ndarray,
    new_positions: np.ndarray,
    limit_radians: float = 0.5,
) -> bool:
    """Check triangle normal rotation between two dense position arrays."""
    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or tri_rows.size == 0:
        return True

    v0_old = original_positions[tri_rows[:, 0]]
    v1_old = original_positions[tri_rows[:, 1]]
    v2_old = original_positions[tri_rows[:, 2]]
    n_old = _fast_cross(v1_old - v0_old, v2_old - v0_old)

    norms_old = np.linalg.norm(n_old, axis=1)
    good_old = norms_old > 1e-12
    if not np.any(good_old):
        return True
    n_old = n_old[good_old] / norms_old[good_old][:, None]

    tri_good = tri_rows[good_old]
    v0_new = new_positions[tri_good[:, 0]]
    v1_new = new_positions[tri_good[:, 1]]
    v2_new = new_positions[tri_good[:, 2]]
    n_new = _fast_cross(v1_new - v0_new, v2_new - v0_new)

    norms_new = np.linalg.norm(n_new, axis=1)
    if np.any(norms_new < 1e-12):
        return False
    n_new = n_new / norms_new[:, None]

    dots = np.sum(n_old * n_new, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return bool(np.all(np.arccos(dots) <= limit_radians))


def check_max_normal_change(
    mesh: Mesh, original_positions: Dict[int, np.ndarray], limit_radians: float = 0.5
) -> bool:
    """
    Check if any facet rotates more than `limit_radians` from original_positions.

    Returns:
        True if the step is safe (max rotation < limit).
        False if the step is unsafe (rotation too large or facet collapsed).
    """
    # 1. Get current (new) positions
    new_pos_view = mesh.positions_view()

    # 2. Reconstruct old positions array
    # Note: This relies on mesh.vertex_index_to_row being stable, which it is
    # during a line search (no topology changes).
    old_pos_view = new_pos_view.copy()
    idx_map = mesh.vertex_index_to_row

    # We only need to update the rows corresponding to vertices that moved.
    # original_positions only contains non-fixed vertices.
    for vidx, pos in original_positions.items():
        if vidx in idx_map:
            old_pos_view[idx_map[vidx]] = pos

    return check_max_normal_change_positions(
        mesh,
        original_positions=old_pos_view,
        new_positions=new_pos_view,
        limit_radians=limit_radians,
    )


def detect_vertex_edge_collisions(
    mesh: Mesh, threshold: float = 1e-3
) -> List[Tuple[int, int]]:
    """
    Finds (vertex_id, edge_id) pairs where the vertex is dangerously close to the edge.
    Returns a list of collisions to be handled (e.g., by refining/popping).

    Naive O(V*E) implementation.
    """
    collisions = []

    # Ensure cached vertex order/index map exists before accessing mesh.vertex_ids.
    if getattr(mesh, "vertex_ids", None) is None:
        mesh.build_position_cache()

    # Pre-cache positions to avoid dictionary lookups in inner loop
    # We can use mesh.positions_view() but we need a mapping back to IDs
    # mesh.vertex_ids contains the IDs in order
    ids = mesh.vertex_ids
    positions = mesh.positions_view()

    # Build edge array: [ [tail_pos_x, ...], [head_pos_x, ...] ]
    edge_indices = []
    edge_ids = []

    # Filter valid edges and store their vertex ROW indices
    idx_map = mesh.vertex_index_to_row

    for eid, edge in mesh.edges.items():
        if edge.tail_index in idx_map and edge.head_index in idx_map:
            t_row = idx_map[edge.tail_index]
            h_row = idx_map[edge.head_index]
            edge_indices.append((t_row, h_row))
            edge_ids.append(eid)

    if not edge_indices:
        return []

    edge_indices = np.array(edge_indices)
    edge_tails = positions[edge_indices[:, 0]]
    edge_heads = positions[edge_indices[:, 1]]
    edge_vecs = edge_heads - edge_tails
    edge_lens_sq = np.sum(edge_vecs**2, axis=1)

    # Avoid degenerate edges
    valid_edges = edge_lens_sq > 1e-12
    edge_tails = edge_tails[valid_edges]
    edge_vecs = edge_vecs[valid_edges]
    edge_lens_sq = edge_lens_sq[valid_edges]
    # Keep track of original IDs
    valid_edge_ids = np.array(edge_ids)[valid_edges]

    # Iterate over vertices
    for i, p in enumerate(positions):
        v_id = ids[i]

        # Vector from edge tail to point
        # Shape (N_edges, 3)
        ap = p - edge_tails

        # Project p onto line segment
        # t = dot(ap, ab) / dot(ab, ab)
        t = np.sum(ap * edge_vecs, axis=1) / edge_lens_sq

        # Check if projection falls strictly within the edge (not endpoints)
        # Using a buffer (0.05 to 0.95) to avoid detecting the vertex itself
        # if it's one of the edge endpoints (or very close neighbors)
        mask = (t > 0.05) & (t < 0.95)

        if np.any(mask):
            # Calculate actual distances for candidates
            # closest = a + t * ab
            closest = edge_tails[mask] + t[mask, None] * edge_vecs[mask]
            dists = np.linalg.norm(p - closest, axis=1)

            collisions_mask = dists < threshold

            if np.any(collisions_mask):
                # Map back to edge IDs
                colliding_edge_indices = np.where(mask)[0][collisions_mask]
                for idx in colliding_edge_indices:
                    eid = valid_edge_ids[idx]

                    # Double check we aren't detecting the vertex itself if topology is weird
                    # (Though the t > 0.05 check usually handles this)
                    collisions.append((v_id, int(eid)))

    return collisions


def get_min_edge_length(mesh: Mesh) -> float:
    """Return the minimum edge length in the mesh."""
    # Pre-cache positions
    if not mesh.edges:
        return 0.0

    positions = mesh.positions_view()
    idx_map = mesh.vertex_index_to_row

    # Collect valid edge indices
    edge_indices = []
    for edge in mesh.edges.values():
        if edge.tail_index in idx_map and edge.head_index in idx_map:
            edge_indices.append((idx_map[edge.tail_index], idx_map[edge.head_index]))

    if not edge_indices:
        return 0.0

    edge_indices = np.array(edge_indices)
    edge_vecs = positions[edge_indices[:, 1]] - positions[edge_indices[:, 0]]
    edge_lens = np.linalg.norm(edge_vecs, axis=1)

    if len(edge_lens) == 0:
        return 0.0

    return float(np.min(edge_lens))

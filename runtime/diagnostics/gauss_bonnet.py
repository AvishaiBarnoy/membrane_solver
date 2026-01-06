"""Discrete Gauss-Bonnet diagnostics for triangulated meshes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from geometry.entities import Edge, Facet, Mesh, Vertex

logger = logging.getLogger("membrane_solver")


def _default_facet_filter(facet: Facet) -> bool:
    return not bool(facet.options.get("gauss_bonnet_exclude", False))


def _included_facets(
    mesh: Mesh, facet_filter: Callable[[Facet], bool] | None
) -> set[int]:
    if facet_filter is None:
        facet_filter = _default_facet_filter
    return {fid for fid, facet in mesh.facets.items() if facet_filter(facet)}


def find_boundary_edges(
    mesh: Mesh, *, facet_filter: Callable[[Facet], bool] | None = None
) -> List[Edge]:
    """Return edges with exactly one incident included facet."""
    mesh.build_connectivity_maps()
    included_facets = _included_facets(mesh, facet_filter)

    boundary = []
    for eid, facets in mesh.edge_to_facets.items():
        included = sum(1 for fid in facets if fid in included_facets)
        if included == 1:
            boundary.append(mesh.edges[eid])
    return boundary


def _vertex_index(vertex: Vertex | int) -> int:
    if isinstance(vertex, Vertex):
        return int(vertex.index)
    return int(vertex)


def extract_boundary_loops(
    mesh: Mesh, boundary_edges: List[Edge]
) -> List[List[Vertex]]:
    """Extract ordered boundary loops from boundary edges."""
    if not boundary_edges:
        return []

    edge_by_id = {edge.index: edge for edge in boundary_edges}
    vertex_to_edges: Dict[int, List[int]] = {}
    for edge in boundary_edges:
        for vid in (edge.tail_index, edge.head_index):
            vertex_to_edges.setdefault(vid, []).append(edge.index)

    # Deterministic traversal: sort incident edge IDs so loop reconstruction
    # is stable across Python hash seeds and runs.
    for eids in vertex_to_edges.values():
        eids.sort()

    for vid, eids in vertex_to_edges.items():
        if len(eids) != 2:
            logger.warning(
                "Boundary vertex %d has degree %d (expected 2).", vid, len(eids)
            )

    remaining = set(edge_by_id.keys())
    loops: List[List[Vertex]] = []

    while remaining:
        start_eid = min(remaining)
        start_edge = edge_by_id[start_eid]
        start_vid = start_edge.tail_index
        current_vid = start_edge.head_index
        loop_vids = [start_vid, current_vid]
        remaining.remove(start_eid)

        safety = 0
        while True:
            safety += 1
            if safety > len(boundary_edges) + 5:
                logger.warning("Boundary loop walk exceeded safety limit; aborting.")
                break

            if current_vid == start_vid:
                break

            next_eid = None
            for eid in vertex_to_edges.get(current_vid, []):
                if eid in remaining:
                    next_eid = eid
                    break

            if next_eid is None:
                logger.warning(
                    "Boundary loop ended prematurely at vertex %d.", current_vid
                )
                break

            remaining.remove(next_eid)
            edge = edge_by_id[next_eid]
            next_vid = (
                edge.head_index if edge.tail_index == current_vid else edge.tail_index
            )

            if next_vid == start_vid:
                break

            loop_vids.append(next_vid)
            current_vid = next_vid

        loop_vids = _canonicalize_boundary_loop(mesh, loop_vids)
        loops.append([mesh.vertices[vid] for vid in loop_vids])

    return loops


def _rotate_loop_to_min_vertex(loop_vids: List[int]) -> List[int]:
    if not loop_vids:
        return loop_vids
    min_vid = min(loop_vids)
    start = loop_vids.index(min_vid)
    return loop_vids[start:] + loop_vids[:start]


def _canonicalize_boundary_loop(mesh: Mesh, loop_vids: List[int]) -> List[int]:
    """Return a deterministically oriented boundary loop.

    The output is rotated so that the smallest vertex ID is first, and the loop
    orientation is chosen so the dominant component of the polygon area-vector
    is non-negative. This keeps boundary-loop orientation stable across runs
    and mesh operations (e.g. refinement) without relying on hash iteration.
    """
    if len(loop_vids) > 1 and loop_vids[0] == loop_vids[-1]:
        loop_vids = loop_vids[:-1]

    loop_vids = _rotate_loop_to_min_vertex(loop_vids)
    if len(loop_vids) < 3:
        return loop_vids

    coords = np.array([mesh.vertices[vid].position for vid in loop_vids], dtype=float)
    area_vec = np.sum(np.cross(coords, np.roll(coords, -1, axis=0)), axis=0)
    if float(np.max(np.abs(area_vec))) < 1e-14:
        return loop_vids

    axis = int(np.argmax(np.abs(area_vec)))
    if float(area_vec[axis]) < 0.0:
        loop_vids = list(reversed(loop_vids))
        loop_vids = _rotate_loop_to_min_vertex(loop_vids)

    return loop_vids


def _facet_vertex_loop(mesh: Mesh, face: Facet) -> List[int]:
    if not getattr(mesh, "facet_vertex_loops", None):
        mesh.build_facet_vertex_loops()

    loop = mesh.facet_vertex_loops.get(face.index)
    if loop is not None:
        return [int(v) for v in loop.tolist()]

    v_ids: List[int] = []
    for signed_ei in face.edge_indices:
        edge = mesh.edges[abs(signed_ei)]
        tail = edge.tail_index if signed_ei > 0 else edge.head_index
        if not v_ids or v_ids[-1] != tail:
            v_ids.append(tail)
    return v_ids


def corner_angle(mesh: Mesh, face: Facet, vertex: Vertex | int) -> float:
    """Return the interior angle at ``vertex`` inside ``face``."""
    vid = _vertex_index(vertex)
    loop = _facet_vertex_loop(mesh, face)
    if vid not in loop:
        raise ValueError(f"Vertex {vid} not found in facet {face.index}.")
    if len(loop) < 3:
        return 0.0

    idx = loop.index(vid)
    prev_vid = loop[idx - 1]
    next_vid = loop[(idx + 1) % len(loop)]

    v = mesh.vertices[vid].position
    u = mesh.vertices[prev_vid].position - v
    w = mesh.vertices[next_vid].position - v

    nu = np.linalg.norm(u)
    nw = np.linalg.norm(w)
    if nu < 1e-15 or nw < 1e-15:
        return 0.0

    cos_theta = float(np.dot(u, w) / (nu * nw))
    cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
    return float(np.arccos(cos_theta))


def _vertex_angle_sums(
    mesh: Mesh, *, facet_filter: Callable[[Facet], bool] | None = None
) -> np.ndarray:
    """Return per-vertex sums of incident triangle corner angles."""
    if not getattr(mesh, "facet_vertex_loops", None):
        mesh.build_facet_vertex_loops()
    positions = mesh.positions_view()
    tri_rows, _ = mesh.triangle_row_cache()
    n_verts = len(mesh.vertex_ids)
    angle_sums = np.zeros(n_verts, dtype=float)

    if tri_rows is None or len(tri_rows) == 0:
        return angle_sums

    included_facets = _included_facets(mesh, facet_filter)
    if len(included_facets) != len(mesh.facets):
        if not included_facets:
            return angle_sums
        _, tri_facets = mesh.triangle_row_cache()
        mask = [fid in included_facets for fid in tri_facets]
        if not any(mask):
            return angle_sums
        tri_rows = tri_rows[np.array(mask, dtype=bool)]

    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]

    a = np.linalg.norm(v2 - v1, axis=1)
    b = np.linalg.norm(v0 - v2, axis=1)
    c = np.linalg.norm(v1 - v0, axis=1)

    eps = 1e-15
    a = np.maximum(a, eps)
    b = np.maximum(b, eps)
    c = np.maximum(c, eps)

    cos0 = (b * b + c * c - a * a) / (2.0 * b * c)
    cos1 = (c * c + a * a - b * b) / (2.0 * c * a)
    cos2 = (a * a + b * b - c * c) / (2.0 * a * b)

    cos0 = np.clip(cos0, -1.0, 1.0)
    cos1 = np.clip(cos1, -1.0, 1.0)
    cos2 = np.clip(cos2, -1.0, 1.0)

    ang0 = np.arccos(cos0)
    ang1 = np.arccos(cos1)
    ang2 = np.arccos(cos2)

    np.add.at(angle_sums, tri_rows[:, 0], ang0)
    np.add.at(angle_sums, tri_rows[:, 1], ang1)
    np.add.at(angle_sums, tri_rows[:, 2], ang2)

    return angle_sums


def interior_angle_deficit(
    mesh: Mesh,
    interior_vertices: Iterable[Vertex | int],
    *,
    facet_filter: Callable[[Facet], bool] | None = None,
) -> Dict[int, float]:
    """Return angle-deficit values for interior vertices."""
    mesh.build_position_cache()
    angle_sums = _vertex_angle_sums(mesh, facet_filter=facet_filter)
    index_map = mesh.vertex_index_to_row

    deficits: Dict[int, float] = {}
    for vertex in interior_vertices:
        vid = _vertex_index(vertex)
        row = index_map.get(vid)
        if row is None:
            continue
        deficits[vid] = float(2.0 * np.pi - angle_sums[row])
    return deficits


def boundary_geodesic_sum(
    mesh: Mesh,
    boundary_loops: List[List[Vertex | int]],
    *,
    facet_filter: Callable[[Facet], bool] | None = None,
) -> Dict[int, float]:
    """Return discrete boundary geodesic-curvature sums per loop."""
    mesh.build_position_cache()
    angle_sums = _vertex_angle_sums(mesh, facet_filter=facet_filter)
    index_map = mesh.vertex_index_to_row

    per_loop: Dict[int, float] = {}
    for loop_idx, loop in enumerate(boundary_loops):
        total = 0.0
        for vertex in loop:
            vid = _vertex_index(vertex)
            row = index_map.get(vid)
            if row is None:
                continue
            total += float(np.pi - angle_sums[row])
        per_loop[loop_idx] = total
    return per_loop


def gauss_bonnet_invariant(
    mesh: Mesh, *, facet_filter: Callable[[Facet], bool] | None = None
) -> Tuple[float, float, float, Dict[int, float]]:
    """Return G, K_int_total, B_total, and per-loop boundary sums."""
    included_facets = _included_facets(mesh, facet_filter)
    boundary_edges = find_boundary_edges(mesh, facet_filter=facet_filter)
    boundary_loops = extract_boundary_loops(mesh, boundary_edges)

    boundary_vids = {_vertex_index(v) for loop in boundary_loops for v in loop}
    included_vids: set[int] = set()
    if included_facets:
        if not getattr(mesh, "facet_vertex_loops", None):
            mesh.build_facet_vertex_loops()
        for fid in included_facets:
            loop = mesh.facet_vertex_loops.get(fid)
            if loop is not None:
                included_vids.update(int(v) for v in loop.tolist())
            else:
                facet = mesh.facets[fid]
                for signed_ei in facet.edge_indices:
                    edge = mesh.edges[abs(signed_ei)]
                    included_vids.add(edge.tail_index)
                    included_vids.add(edge.head_index)
    else:
        included_vids = set(mesh.vertices.keys())

    interior_vids = [vid for vid in included_vids if vid not in boundary_vids]

    deficits = interior_angle_deficit(mesh, interior_vids, facet_filter=facet_filter)
    k_int_total = float(sum(deficits.values()))

    per_loop_b = boundary_geodesic_sum(mesh, boundary_loops, facet_filter=facet_filter)
    b_total = float(sum(per_loop_b.values()))
    g_total = float(k_int_total + b_total)

    return g_total, k_int_total, b_total, per_loop_b


@dataclass
class GaussBonnetMonitor:
    """Track Gauss-Bonnet invariants and report drift."""

    baseline_g: float
    baseline_per_loop_b: Dict[int, float]
    boundary_vertex_count: int
    loop_sizes: Dict[int, int]
    facet_filter: Callable[[Facet], bool] | None = None
    eps_angle: float = 1e-4
    c1: float = 1.0
    c2: float = 1.0

    @classmethod
    def from_mesh(
        cls,
        mesh: Mesh,
        *,
        facet_filter: Callable[[Facet], bool] | None = None,
        eps_angle: float = 1e-4,
        c1: float = 1.0,
        c2: float = 1.0,
    ) -> "GaussBonnetMonitor":
        g_total, _, _, per_loop_b = gauss_bonnet_invariant(
            mesh, facet_filter=facet_filter
        )
        boundary_edges = find_boundary_edges(mesh, facet_filter=facet_filter)
        loops = extract_boundary_loops(mesh, boundary_edges)
        boundary_vertex_count = len({v.index for loop in loops for v in loop})
        loop_sizes = {idx: len(loop) for idx, loop in enumerate(loops)}
        return cls(
            baseline_g=g_total,
            baseline_per_loop_b=per_loop_b,
            boundary_vertex_count=boundary_vertex_count,
            loop_sizes=loop_sizes,
            facet_filter=facet_filter,
            eps_angle=float(eps_angle),
            c1=float(c1),
            c2=float(c2),
        )

    def evaluate(self, mesh: Mesh) -> Dict[str, float | Dict[int, float] | bool]:
        g_total, k_int_total, b_total, per_loop_b = gauss_bonnet_invariant(
            mesh, facet_filter=self.facet_filter
        )
        tol_g = self.c1 * np.sqrt(max(self.boundary_vertex_count, 1)) * self.eps_angle

        drift_g = abs(g_total - self.baseline_g)
        loop_drifts: Dict[int, float] = {}
        loop_tols: Dict[int, float] = {}
        ok = drift_g <= tol_g

        for loop_idx, b_val in per_loop_b.items():
            base = self.baseline_per_loop_b.get(loop_idx, b_val)
            drift = abs(b_val - base)
            loop_drifts[loop_idx] = drift
            loop_tols[loop_idx] = (
                self.c2
                * np.sqrt(max(self.loop_sizes.get(loop_idx, 1), 1))
                * self.eps_angle
            )
            if drift > loop_tols[loop_idx]:
                ok = False

        logger.debug(
            "Gauss-Bonnet: G=%.6e K_int=%.6e B_total=%.6e |ΔG|=%.3e tol=%.3e",
            g_total,
            k_int_total,
            b_total,
            drift_g,
            tol_g,
        )
        for loop_idx, b_val in per_loop_b.items():
            logger.debug(
                "Gauss-Bonnet loop %d: B=%.6e |ΔB|=%.3e tol=%.3e",
                loop_idx,
                b_val,
                loop_drifts[loop_idx],
                loop_tols[loop_idx],
            )

        return {
            "ok": ok,
            "G": g_total,
            "K_int_total": k_int_total,
            "B_total": b_total,
            "per_loop_B": per_loop_b,
            "drift_G": drift_g,
            "drift_B": loop_drifts,
            "tol_G": float(tol_g),
            "tol_B": loop_tols,
        }


__all__ = [
    "find_boundary_edges",
    "extract_boundary_loops",
    "corner_angle",
    "interior_angle_deficit",
    "boundary_geodesic_sum",
    "gauss_bonnet_invariant",
    "GaussBonnetMonitor",
]

from typing import Dict, Iterable

import numpy as np


def compute_total_surface_area(mesh) -> float:
    if (
        mesh._cached_total_area_version == mesh._version
        and mesh._cached_total_area is not None
    ):
        return mesh._cached_total_area
    area = sum(facet.compute_area(mesh) for facet in mesh.facets.values())
    mesh._cached_total_area = area
    mesh._cached_total_area_version = mesh._version
    return area


def compute_total_area_and_gradient(
    mesh,
    positions: np.ndarray | None = None,
    index_map: Dict[int, int] | None = None,
) -> tuple[float, Dict[int, np.ndarray]]:
    is_cached_pos = positions is None or positions is getattr(
        mesh, "_positions_cache", None
    )
    if (
        is_cached_pos
        and mesh._cached_total_area_version == mesh._version
        and mesh._cached_total_area is not None
        and mesh._cached_total_area_grad is not None
    ):
        return mesh._cached_total_area, mesh._cached_total_area_grad

    total_area = 0.0
    total_grad: Dict[int, np.ndarray] = {}
    for facet in mesh.facets.values():
        area, grad = facet.compute_area_and_gradient(
            mesh, positions=positions, index_map=index_map
        )
        total_area += area
        for vid, gvec in grad.items():
            if vid not in total_grad:
                total_grad[vid] = gvec.copy()
            else:
                total_grad[vid] += gvec

    if is_cached_pos:
        mesh._cached_total_area = total_area
        mesh._cached_total_area_grad = total_grad
        mesh._cached_total_area_version = mesh._version

    return total_area, total_grad


def compute_total_volume(mesh) -> float:
    return sum(body.compute_volume(mesh) for body in mesh.bodies.values())


def compute_surface_radius_of_gyration(
    mesh, facet_indices: Iterable[int] | None = None
) -> float:
    """Return the surface-area-weighted radius of gyration for the mesh."""
    if facet_indices is None:
        facet_indices = mesh.facets.keys()

    total_area = 0.0
    centroid_sum = np.zeros(3, dtype=float)
    mean_r2_sum = 0.0

    for facet_idx in facet_indices:
        facet = mesh.facets.get(facet_idx)
        if facet is None:
            continue

        if (
            getattr(mesh, "facet_vertex_loops", None)
            and facet_idx in mesh.facet_vertex_loops
        ):
            v_ids_array = mesh.facet_vertex_loops[facet_idx]
            v_ids = v_ids_array.tolist()
        else:
            v_ids = []
            for signed_ei in facet.edge_indices:
                edge = mesh.edges[abs(signed_ei)]
                tail = edge.tail_index if signed_ei > 0 else edge.head_index
                if not v_ids or v_ids[-1] != tail:
                    v_ids.append(tail)

        if len(v_ids) < 3:
            continue

        v_pos = np.array([mesh.vertices[i].position for i in v_ids], dtype=float)
        v0 = v_pos[0]
        for i in range(1, len(v_pos) - 1):
            a = v0
            b = v_pos[i]
            c = v_pos[i + 1]
            cross = np.cross(b - a, c - a)
            area = 0.5 * float(np.linalg.norm(cross))
            if area == 0.0:
                continue
            centroid = (a + b + c) / 3.0
            mean_r2 = (
                np.dot(a, a)
                + np.dot(b, b)
                + np.dot(c, c)
                + np.dot(a, b)
                + np.dot(b, c)
                + np.dot(c, a)
            ) / 6.0

            total_area += area
            centroid_sum += area * centroid
            mean_r2_sum += area * mean_r2

    if total_area == 0.0:
        return 0.0

    centroid = centroid_sum / total_area
    mean_r2 = mean_r2_sum / total_area
    rg2 = float(mean_r2 - np.dot(centroid, centroid))
    if rg2 < 0.0 and rg2 > -1e-12:
        rg2 = 0.0
    return float(np.sqrt(max(rg2, 0.0)))

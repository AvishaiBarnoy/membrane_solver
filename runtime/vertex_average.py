# runtime/vertex_average.py

import logging

import numpy as np

logger = logging.getLogger("membrane_solver")

def compute_facet_centroid(mesh,facet, vertices):
    v_ids = set()
    for signed_ei in facet.edge_indices:
        edge = mesh.get_edge(signed_ei)
        v_ids.update([edge.tail_index, edge.head_index])
    coords = np.array([vertices[i].position for i in v_ids])
    return np.mean(coords, axis=0)

def compute_facet_normal(mesh, facet, vertices):
    # Use the first 3 vertices to compute a normal
    v_ids = []
    for signed_ei in facet.edge_indices:
        edge = mesh.get_edge(signed_ei)
        v_ids.append(edge.tail_index)
        if len(set(v_ids)) >= 3:
            break
    a = vertices[v_ids[1]].position - vertices[v_ids[0]].position
    b = vertices[v_ids[2]].position - vertices[v_ids[0]].position
    return 0.5 * np.cross(a, b)  # area-weighted normal

def vertex_average(mesh):
    """
    Perform area/volume aware vertex averaging on all non-fixed vertices,
    using the connectivity map mesh.vertex_to_facets.

    For open patches (no volume constraint) this smooths vertices toward
    area-weighted centroids but then rescales each incident facet in-plane
    to preserve its original (or target) area, avoiding collapse.
    """
    # Cache original facet areas to enable area restoration on open patches.
    facet_orig_area = {
        f_id: facet.compute_area(mesh) for f_id, facet in mesh.facets.items()
    }

    new_positions = {}

    for v_id, vertex in mesh.vertices.items():
        if vertex.fixed:
            continue

        facet_ids = mesh.vertex_to_facets.get(v_id, [])
        if not facet_ids or len(facet_ids) <= 1:
            continue

        total_area = 0.0
        weighted_sum = np.zeros(3)
        total_normal = np.zeros(3)

        for f_id in facet_ids:
            facet = mesh.facets[f_id]
            centroid = compute_facet_centroid(mesh, facet, mesh.vertices)
            normal = compute_facet_normal(mesh, facet, mesh.vertices)
            area = np.linalg.norm(normal)

            weighted_sum += area * centroid
            total_normal += normal
            total_area += area

        if total_area < 1e-12 or np.linalg.norm(total_normal) < 1e-12:
            continue

        v_avg = weighted_sum / total_area
        v = vertex.position
        lambda_ = (np.dot(v_avg, total_normal) - np.dot(v, total_normal)) / np.dot(total_normal, total_normal)
        v_new = v_avg - lambda_ * total_normal

        new_positions[v_id] = v_new

    for v_id, pos in new_positions.items():
        mesh.vertices[v_id].position = pos

    logger.info("Vertex averaging completed with volume conservation.")

    # Area restoration for cases with explicit targets; skip for unconstrained open patches
    # to retain smoothing behavior.
    any_area_target = any(f.options.get("target_area") is not None for f in mesh.facets.values()) or any(
        b.options.get("target_area") is not None for b in mesh.bodies.values()
    )
    if any_area_target:
        accum = {}
        counts = {}
        for f_id, facet in mesh.facets.items():
            v_ids = []
            for signed_ei in facet.edge_indices:
                edge = mesh.get_edge(signed_ei)
                tail = edge.tail_index if signed_ei > 0 else edge.head_index
                if not v_ids or v_ids[-1] != tail:
                    v_ids.append(tail)
            if len(v_ids) < 3:
                continue

            target_area = facet.options.get("target_area", facet_orig_area.get(f_id, None))
            orig_area = facet_orig_area.get(f_id, None)
            if target_area is None and orig_area is None:
                continue
            desired_area = target_area if target_area is not None else orig_area

            pts = np.array([mesh.vertices[i].position for i in v_ids])
            centroid = pts.mean(axis=0)
            n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
            area_now = 0.5 * np.linalg.norm(n)
            if area_now < 1e-12 or desired_area is None or desired_area < 1e-12:
                continue
            n_hat = n / (np.linalg.norm(n) + 1e-18)

            scale = np.sqrt(desired_area / area_now)
            proposed = []
            for p in pts:
                offset = p - centroid
                normal_comp = np.dot(offset, n_hat) * n_hat
                in_plane = offset - normal_comp
                new_p = centroid + scale * in_plane + normal_comp
                proposed.append(new_p)

            for vid, p_new in zip(v_ids, proposed):
                accum.setdefault(vid, np.zeros(3))
                counts[vid] = counts.get(vid, 0) + 1
                accum[vid] += p_new

        for vid, pos_sum in accum.items():
            mesh.vertices[vid].position = pos_sum / counts[vid]

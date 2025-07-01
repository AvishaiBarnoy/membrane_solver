# runtime/vertex_average.py

import numpy as np
import logging

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
    Perform volume-conserving vertex averaging on all non-fixed vertices,
    using the connectivity map mesh.vertex_to_facets.
    """
    new_positions = {}

    for v_id, vertex in mesh.vertices.items():
        if vertex.fixed:
            continue

        facet_ids = mesh.vertex_to_facets.get(v_id, [])
        if not facet_ids:
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

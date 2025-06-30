# runtime/vertex_average.py

import numpy as np

def compute_facet_centroid(facet, vertices):
    v_ids = facet.vertex_ids
    coords = np.array([vertices[i].position for i in v_ids])
    return np.mean(coords, axis=0)

def compute_facet_normal(facet, vertices):
    v_ids = facet.vertex_ids
    a = vertices[v_ids[1]].position - vertices[v_ids[0]].position
    b = vertices[v_ids[2]].position - vertices[v_ids[0]].position
    return 0.5 * np.cross(a, b)  # area-weighted normal

def vertex_average(mesh):
    new_positions = {}
    for v_id, vertex in mesh.vertices.items():
        if vertex.fixed or len(vertex.adjacent_facets) == 0:
            continue

        total_area = 0.0
        weighted_sum = np.zeros(3)
        N_total = np.zeros(3)
        for f_id in vertex.adjacent_facets:
            facet = mesh.facets[f_id]
            centroid = compute_facet_centroid(facet, mesh.vertices)
            normal = compute_facet_normal(facet, mesh.vertices)
            area = np.linalg.norm(normal)
            weighted_sum += area * centroid
            N_total += normal
            total_area += area

        if total_area < 1e-12 or np.linalg.norm(N_total) < 1e-12:
            continue

        v_avg = weighted_sum / total_area
        v = vertex.position
        lambda_ = (np.dot(v_avg, N_total) - np.dot(v, N_total)) / np.dot(N_total, N_total)
        v_new = v_avg - lambda_ * N_total
        new_positions[v_id] = v_new

    for v_id, pos in new_positions.items():
        mesh.vertices[v_id].position = pos

# runtime/vertex_average.py

import logging

import numpy as np

logger = logging.getLogger("membrane_solver")


def _has_pin_to_circle(options: dict | None) -> bool:
    if not options:
        return False
    constraints = options.get("constraints")
    if constraints == "pin_to_circle":
        return True
    if isinstance(constraints, list):
        return "pin_to_circle" in constraints
    return False


def _pin_to_circle_group(options: dict | None) -> str | None:
    if not _has_pin_to_circle(options):
        return None
    group = (options or {}).get("pin_to_circle_group")
    return "default" if group is None else str(group)


def vertex_average(mesh):
    """
    Perform Evolver-style vertex averaging on movable interior vertices.

    This follows the "soapfilm" averaging behavior in Surface Evolver
    (`vertex_average` / `find_vertex_average` in `src/veravg.c`):

    - For each vertex, iterate its incident edges.
    - For each edge, compute a weight equal to the sum of incident facet areas
      on that edge.
    - Move the vertex along the weighted average of incident edge vectors:

        x_new = x_old + 0.25 * Σ(w_e^2 * (x_neighbor - x_old)) / Σ(w_e^2)

    Constraint compatibility:
    - Surface Evolver skips edges that do not carry all constraints required by
      the vertex. In this codebase, most geometric constraints are stored on
      vertices (e.g. `pin_to_circle`), not edges, so we approximate this rule
      by requiring both endpoints to share the same pin-to-circle group before
      using that edge for averaging.
    """
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()

    # Cache facet areas once so averaging is not order-dependent.
    facet_orig_area = {
        f_id: facet.compute_area(mesh) for f_id, facet in mesh.facets.items()
    }

    edge_weights: dict[int, float] = {}
    for e_id, facet_ids in mesh.edge_to_facets.items():
        w = 0.0
        for f_id in facet_ids:
            w += float(facet_orig_area.get(f_id, 0.0) or 0.0)
        edge_weights[int(e_id)] = float(w)

    new_positions = {}

    for v_id, vertex in mesh.vertices.items():
        # Keep pin-to-circle vertices anchored during smoothing. Repeated
        # averaging can otherwise collapse constrained disk/outer rings before
        # projection is re-applied.
        if vertex.fixed or _has_pin_to_circle(getattr(vertex, "options", None)):
            continue

        edge_ids = mesh.vertex_to_edges.get(v_id, set())
        if not edge_ids or len(edge_ids) <= 1:
            continue

        group = _pin_to_circle_group(getattr(vertex, "options", None))

        total_weight = 0.0
        xsum = np.zeros(3, dtype=float)
        used = 0

        for e_id in edge_ids:
            e_id_int = int(e_id)
            edge = mesh.edges.get(e_id_int)
            if edge is None:
                continue
            other = edge.head_index if edge.tail_index == v_id else edge.tail_index

            if group is not None:
                other_group = _pin_to_circle_group(mesh.vertices[int(other)].options)
                if other_group != group:
                    continue

            weight = float(edge_weights.get(e_id_int, 0.0) or 0.0)
            if weight <= 0.0:
                continue
            w2 = weight * weight

            side = np.asarray(
                mesh.vertices[int(other)].position, dtype=float
            ) - np.asarray(vertex.position, dtype=float)
            xsum += w2 * side
            total_weight += w2
            used += 1

        if used <= 1 or total_weight < 1e-15:
            continue

        new_positions[v_id] = np.asarray(vertex.position, dtype=float) + 0.25 * (
            xsum / total_weight
        )

    for v_id, pos in new_positions.items():
        mesh.vertices[v_id].position = pos

    logger.info("Vertex averaging completed.")

    # Area restoration for cases with explicit targets; skip for unconstrained open patches.
    any_area_target = any(
        f.options.get("target_area") is not None for f in mesh.facets.values()
    ) or any(b.options.get("target_area") is not None for b in mesh.bodies.values())
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

            target_area = facet.options.get(
                "target_area", facet_orig_area.get(f_id, None)
            )
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

    mesh.increment_version()

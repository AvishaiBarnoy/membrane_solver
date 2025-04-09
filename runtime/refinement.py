import numpy as np
import logging
from geometry.geometry_entities import Vertex, Edge, Facet
import sys
logger = logging.getLogger("membrane_solver")

def refine_polygonal_facets(vertices, edges, facets, bodies, global_params):
    # TODO: option for loop "r2" will refine twice
    # TODO: why is global_params passes through here?
    """
    Refines all non-triangular facets by subdividing them into triangles using
    centroid-based fan triangulation. Triangles remain unchanged.

    Returns:
        (updated_vertices, updated_facets)
    """
    new_vertices = vertices[:]
    new_edges = edges[:]
    new_facets = []

    for facet in facets:
        if len(facet.edges) == 3:
            new_facets.append(facet)
            continue

        # --- Step 1: Reconstruct vertex loop from ordered edges ---
        vertex_loop = [facet.edges[0].tail]
        for edge in facet.edges:
            if vertex_loop[-1] != edge.tail:
                raise ValueError(f"Edge loop is not continuous in facet {facet.index}")
            vertex_loop.append(edge.head)

        if vertex_loop[0] == vertex_loop[-1]:
            vertex_loop = vertex_loop[:-1]

        if len(vertex_loop) < 3:
            logger.warning(f"Facet {facet.index} has <3 vertices after reconstruction.")
            continue

        # --- Step 2: Create centroid vertex ---
        centroid_pos = np.mean([v.position for v in vertex_loop], axis=0)
        centroid_vertex = Vertex(position=centroid_pos, index=len(new_vertices))
        new_vertices.append(centroid_vertex)

        # --- Step 3: Create triangle facets using fan triangulation ---
        n = len(vertex_loop)
        for i in range(n):
            a = vertex_loop[i]
            b = vertex_loop[(i + 1) % n]

            e1 = Edge(a, b, index=len(new_edges), options=facet.options.copy())
            e2 = Edge(b, centroid_vertex, index=len(new_edges)+1, options=facet.options.copy())
            e3 = Edge(centroid_vertex, a, index=len(new_edges)+2, options=facet.options.copy())

            child_edges = [e1, e2, e3]

            new_edges.extend(child_edges)
            child_options = facet.options.copy()
            child_options.pop("refine", None)   # TODO: Why? to not use edge options on facet?

            new_facet = Facet(child_edges, index=len(new_facets), options=child_options)
            new_facets.append(new_facet)

    return new_vertices, new_edges, new_facets, bodies

def refine_triangle_mesh(vertices, edges, facets, bodies):
    """
    Refines a triangle mesh by splitting each triangle into 4 new triangles.
    A vertex is added at the midpoint of each edge.

    Args:
        vertices (list of Vertex)
        facets (list of Facet) – all must be triangles

    Returns:
        (new_vertices, new_facets)
    """
    # TODO: Don't refie no_refine objects!
    # TODO: check refined instances inherit options, how do new edges and new
    #           vertices behave. E.g, vertex in mid-point of fixed edge inherts
    #           some behavior -> check SE manual
    new_vertices = vertices[:]
    new_edges = edges[:]
    new_facets = []
    old_to_new_map = {}
    midpoint_cache = {}  # maps (min_i, max_j) -> midpoint vertex

    for facet in facets:
        if len(facet.edges) != 3:
            raise ValueError(f"Facet {facet.index} is not a triangle.")

        # Step 1: Identify the original triangle vertices
        v0 = facet.edges[0].tail
        v1 = facet.edges[0].head
        v2 = facet.edges[1].head  # assuming correct orientation

        # Step 2: Compute midpoints and create new vertices
        def get_midpoint(a, b):
            key = tuple(sorted((a.index, b.index)))
            if key in midpoint_cache:
                return midpoint_cache[key]
            pos = 0.5 * (a.position + b.position)
            # TODO: inherit constraints from edge?
            v = Vertex(pos, index=len(new_vertices))
            if a.options.get("fixed") and b.options.get("fixed"):
                v.options["fixed"] = True
            v.options["parent_edge"] = key
            new_vertices.append(v)
            midpoint_cache[key] = v
            return v

        m01 = get_midpoint(v0, v1)
        m12 = get_midpoint(v1, v2)
        m20 = get_midpoint(v2, v0)

        # Step 3: Create 4 new triangles (as Facets)

        def make_facet(a, b, c, parent_opts):
            # TODO: inherit from constraint from facet?
            e1 = Edge(a, b, index=len(new_edges), options=parent_opts.copy())
            e2 = Edge(b, c, index=len(new_edges)+1, options=parent_opts.copy())
            e3 = Edge(c, a, index=len(new_edges)+2, options=parent_opts.copy())
            new_edges.extend([e1, e2, e3])
            child_opts = parent_opts.copy()
            child_opts["parent_facet"] = facet.index    # for parent tracking
            return Facet([e1, e2, e3], index=len(new_facets), options=child_opts)

        child_facets = [
            make_facet(v0, m01, m20, facet.options),  # outer
            make_facet(v1, m12, m01, facet.options),  # outer
            make_facet(v2, m20, m12, facet.options),  # outer
            make_facet(m01, m12, m20, facet.options), # center triangle
        ]

        old_to_new_map[facet.index] = child_facets
        new_facets.extend(child_facets)

    # Update body objects in place
    update_bodies_after_refinement(bodies, old_to_new_map)

    return new_vertices, new_edges, new_facets, bodies

def update_bodies_after_refinement(bodies, old_to_new_map):
    """
    Updates body facet lists after mesh refinement.

    Args:
        bodies (list of Body)
        old_to_new_map (dict): old facet index → list of new Facet instances
    """
    for body in bodies:
        new_facet_list = []
        for facet in body.facets:
            if facet.index in old_to_new_map:
                new_facet_list.extend(old_to_new_map[facet.index])
            else:
                new_facet_list.append(facet)
        body.facets = new_facet_list

import numpy as np
import logging
from geometry.geometry_entities import Vertex, Edge, Facet
import sys
logger = logging.getLogger("membrane_solver")

def get_or_create_edge(u, v, options):
    nonlocal next_edge_idx, new_edges, edge_lookup

    key = (u, v) if u < v else (v, u)
    if key in edge_lookup:
        existing = edge_lookup[key]
        # if that existing edge already goes exactly u → v, reuse it:
        if existing.tail == u and existing.head == v:
            return existing
        # otherwise, we found it reversed v → u, so return an alias in the right direction
        return Edge(u, v, index=existing.index, options=existing.options.copy())
    # if we get here, no edge at all exists between u and v, so create one:
    e = Edge(u, v, index=next_edge_idx, options=options.copy())
    next_edge_idx += 1
    new_edges.append(e)
    edge_lookup[key] = e
    return e

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
    old_to_new_map = {}

    for facet in facets:
        if len(facet.edges) == 3:
            new_facets.append(facet)
            continue

        # Reconstruct vertex loop
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

        # Created centroid
        centroid_pos = np.mean([v.position for v in vertex_loop], axis=0)
        centroid_vertex = Vertex(position=centroid_pos, index=len(new_vertices))
        new_vertices.append(centroid_vertex)

        # Create child facets
        child_facets = []
        n = len(vertex_loop)
        for i in range(n):
            a = vertex_loop[i]
            b = vertex_loop[(i + 1) % n]

            # Create new edges 
            e1 = Edge(a, b, index=len(new_edges), options=facet.options.copy())
            e2 = Edge(b, centroid_vertex, index=len(new_edges) + 1, options=facet.options.copy())
            e3 = Edge(centroid_vertex, a, index=len(new_edges) + 2, options=facet.options.copy())
            child_edges = [e1, e2, e3]

            new_edges.extend(child_edges)

            # Create child facet
            child_options = facet.options.copy()
            child_options.pop("refine", None)   # TODO: Why? to not use edge options on facet?
            child_options["parent_facet"] = facet.index

            child_facet = Facet(child_edges, index=len(new_facets), options=child_options)
            new_facets.append(child_facet)

    assert all(isinstance(f, Facet) for f in new_facets), "new_facets still nested!"
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
    facet_index = len(facets) # Start index at 0 or len(facets) if you prefer continuity
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

        # Create child facets
        child_facets = []
        triangle_vertices = [
            (v0, m01, m20),
            (v1, m12, m01),
            (v2, m20, m12),
            (m01, m12, m20)
        ]

        # Step 3: Create 4 new triangles (as Facets)
        # def make_facet(a, b, c, parent_opts):
        for a, b, c in triangle_vertices:
            # TODO: inherit from constraint from facet?
            e1 = Edge(a, b, index=len(new_edges), options=facet.options.copy())
            e2 = Edge(b, c, index=len(new_edges)+1, options=facet.options.copy())
            e3 = Edge(c, a, index=len(new_edges)+2, options=facet.options.copy())

            new_edges.extend([e1, e2, e3])

            child_options = facet.options.copy()
            child_options["parent_facet"] = facet.index    # for parent tracking

            child_facet = Facet([e1, e2, e3], index=len(new_facets), options=child_options)
            child_facets.append(child_facet)
            new_facets.extend(child_facets)
            # facet_obj = Facet([e1, e2, e3], index=facet_index, options=child_opts)
            # return facet_obj

        # facet_index += 1    # Increment for next facet
        """child_facets = [
            make_facet(v0, m01, m20, facet.options),  # outer
            make_facet(v1, m12, m01, facet.options),  # outer
            make_facet(v2, m20, m12, facet.options),  # outer
            make_facet(m01, m12, m20, facet.options), # center triangle
        ]"""

        old_to_new_map[facet.index] = child_facets
        # new_facets.extend(child_facets)

    # Update body objects in place
    update_bodies_after_refinement(bodies, old_to_new_map)

    assert all(isinstance(f, Facet) for f in new_facets), "new_facets still nested!"
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
    for body in bodies:
        print(f"Updated body {body.index}: {[f.index for f in body.facets]}")


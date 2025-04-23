import numpy as np
import logging
from geometry.geometry_entities import Vertex, Edge, Facet, Body
import sys
logger = logging.getLogger("membrane_solver")

def refine_polygonal_facets(vertices, edges, facets, bodies):
    # TODO: option for loop "r2" will refine twice
    """
    Refines all non-triangular facets by subdividing them into triangles using
    centroid-based fan triangulation. Triangles remain unchanged.

    Returns:
        (updated_vertices, updated_facets)
    """
    new_vertices = vertices[:]
    new_edges = edges[:]
    new_facets = []
    next_edge_idx = max(e.index for e in new_edges) + 1

    # Prepare a map from old facet idx → list of new child facet idxs:
    children_map = {facet.index: [] for facet in facets}

    for facet in facets:
        # 1. Leave triangles alone
        if len(facet.edges) == 3:
            new_facets.append(facet)
            continue

        # 2. Reconstruct the boundary loop of vertex‐indice
        vertex_loop = [facet.edges[0].tail]
        for edge in facet.edges:
            if vertex_loop[-1] != edge.tail:
                raise ValueError(f"Edge loop is not continuous in facet {facet.index}")
            vertex_loop.append(edge.head)

        if vertex_loop[0] == vertex_loop[-1]:
            vertex_loop.pop()

        if len(vertex_loop) < 3:
            logger.warning(f"Facet {facet.index} has <3 vertices after reconstruction.")
            continue

        # 3. Create centroid
        centroid_pos = np.mean([v.position for v in vertex_loop], axis=0)
        centroid_idx = len(new_vertices)
        centroid_vertex = Vertex(position=centroid_pos, index=centroid_idx)
        new_vertices.append(centroid_vertex)

        # 4. build exactly one spoke edge per vertex in that loop
        spokes = {} # maps vertex_idx -> the Edge( vertex -> centroid )
        for vi in vertex_loop:
            # TODO: deal with how edges inherit options from facets
            e = Edge(vi, centroid_vertex, index=next_edge_idx,
                    options=facet.options.copy())
            next_edge_idx += 1
            new_edges.append(e)
            spokes[vi] = e

        # 5. now fan‐triangulate: each triangle uses
        #    - the old boundary edge
        #    - the spoke from b -> centroid
        #    - the spoke from centroid -> a  (just flip the first spoke)
        n = len(vertex_loop)
        for i in range(n):
            a = vertex_loop[i]
            b = vertex_loop[(i + 1) % n]

            # find the original boundary edge object
            # (it’s in facet.edges[i], by construction)
            boundary_edge = facet.edges[i]

            # the two spokes: b→centroid and centroid→a
            spoke_b = spokes[b]
            spoke_a = spokes[a]
            # make sure the second spoke is oriented centroid→a:
            # TODO: POSSIBLE PROBLEM WITH USING ABSOLUTE VALUE FOR INDEX AND NOT REVERSED
            rev_spoke_a = Edge(new_vertices[centroid_idx], a,
                           index=spoke_a.index,
                           options=spoke_a.options.copy())

            # build the new facet’s edge‐list **in the correct orientation**:
            child_edges = [boundary_edge, spoke_b, rev_spoke_a]
            child_idx = len(new_facets)
            child_options = facet.options.copy()
            child_options["parent_facet"] = facet.index

            child_facet = Facet(child_edges,
                                    index=child_idx,
                                    options=child_options)
            new_facets.append(child_facet)
            # Record that this child belongs to the same bodies
            children_map[facet.index].append(child_idx)

    # TODO: Associate facets with bodies! If no body exists skip
    new_bodies = []
    for body in bodies:
        new_list = []
        for facet in body.facets:
            #print(f"f_idx: {f_idx}")
            if facet.index in children_map:
                # replaced by its child facets
                new_list.extend(children_map[facet.index])
            else:
                # the facet stayed the same (triangles)
                new_list.append(facet)

        # ideally your Body ctor also carries over volume, constraints, etc.
        new_facet_list = [new_facets[i] for i in new_list]
        new_bodies.append( Body(new_facet_list, index=body.index,
                                options=body.options.copy()) )

    assert all(isinstance(f, Facet) for f in new_facets), "new_facets still nested!"
    return new_vertices, new_edges, new_facets, new_bodies

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
    # TODO: Don't refine no_refine objects!
    # TODO: check refined instances inherit options, how do new edges and new
    #           vertices behave. E.g, vertex in mid-point of fixed edge inherts
    #           some behavior -> check SE manual

    new_vertices = vertices[:]
    new_edges = edges[:]
    new_facets = []
    facet_index = len(facets) # Start index at 0 or len(facets) if you prefer continuity
    next_edge_idx = max(e.index for e in new_edges) + 1

    # 1. map old facet → list of new child facets
    children_map = {f.index: [] for f in facets}

    midpoint_cache = {}  # maps (min_i, max_j) -> midpoint vertex

    # 2. subdivide each facet
    for facet in facets:
        assert len(facet.edges) == 3, "refine_triangle_mesh only handles triangles"

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
    #for body in bodies:
    #    print(f"Updated body {body.index}: {[f.index for f in body.facets]}")


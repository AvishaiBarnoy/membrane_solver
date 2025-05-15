import numpy as np
import logging
from geometry.entities import Vertex, Edge, Facet, Body, Mesh
from parameters.global_parameters import GlobalParameters
import sys
logger = logging.getLogger("membrane_solver")

def orient_edges_cycle(edge_indices: list[int], mesh: Mesh) -> list[int]:
    """
    Given a raw list of signed edge indices for an N-gon,
    reorder + re-sign them into a proper cycle of length N.
    """
    # Make a working copy
    remaining = edge_indices.copy()
    if not remaining:
        return []

    # Start with the first edge, force it to positive orientation (tail→head)
    first = remaining.pop(0)
    idx0 = abs(first)
    # We always start by traversing tail->head, so sign is +idx0:
    cycle = [idx0]
    prev_head = mesh.get_edge(idx0).head_index

    # Now greedily pick the next edge that hooks onto prev_head
    while remaining:
        for i, raw in enumerate(remaining):
            idx = abs(raw)
            E   = mesh.get_edge(idx)
            # Case A: we traverse E as tail->head
            if E.tail_index == prev_head:
                cycle.append( idx )
                prev_head = E.head_index
                remaining.pop(i)
                break

            # Case B: we traverse E as head->tail  (so sign it negative)
            if E.head_index == prev_head:
                cycle.append( -idx )
                prev_head = E.tail_index
                remaining.pop(i)
                break
        else:
            raise ValueError(f"Could not complete cycle: stuck at vertex {prev_head}, remaining edges {remaining}")

    # Sanity
    if len(cycle) != len(edge_indices):
        raise AssertionError("orient_edges_cycle() returned wrong length")

    return cycle

def refine_polygonal_facets(mesh):
    """
    Refines all non-triangular facets by subdividing them into triangles using
    centroid-based fan triangulation. Triangles remain unchanged.

    Returns:
        (updated_vertices, updated_facets)
    """
    new_mesh = Mesh()
    new_vertices = mesh.vertices.copy()
    new_edges = mesh.edges.copy()
    new_mesh.vertices = new_vertices.copy()
    new_facets = {}
    next_edge_idx = max(e for e in new_edges.keys()) + 1

    new_mesh.edges = new_edges.copy()

    # Prepare a map from old facet idx → list of new child facet idxs:
    children_map = {mesh.facets[facet_idx].index: [] for facet_idx in mesh.facets.keys()}

    for f_idx, facet in mesh.facets.items():
        # 1. Leave triangles alone
        if len(facet.edge_indices) == 3:
            new_facets[f_idx] = facet
            continue

        # 2. Reconstruct the boundary loop of vertex‐indice
        tail_index = mesh.get_edge(facet.edge_indices[0]).tail_index
        head_index = mesh.get_edge(facet.edge_indices[0]).head_index
        # TODO: The -1 is a patchwork and not a good solution, maybe change to dictionary in the Mesh class 
        vertex_loop = [mesh.get_edge(facet.edge_indices[0]).tail_index]
        for edge_idx in facet.edge_indices:
            # TODO: The -1 is a patchwork and not a good solution, maybe change to dictionary in the Mesh class 
            edge = mesh.get_edge(edge_idx)
            if vertex_loop[-1] != edge.tail_index:
                raise ValueError(f"Edge loop is not continuous in facet {facet.index}")
            vertex_loop.append(edge.head_index)

        if vertex_loop[0] == vertex_loop[-1]:
            vertex_loop.pop()

        if len(vertex_loop) < 3:
            logger.warning(f"Facet {facet.index} has <3 vertices after reconstruction.")
            continue

        # 3. Create centroid
        centroid_pos = np.mean([mesh.vertices[v].position for v in vertex_loop], axis=0)
        centroid_idx = len(new_vertices)
        centroid_vertex = Vertex(index=centroid_idx,
                                 position=np.asarray(centroid_pos,
                                                     dtype=float))
        #print(midpoint_position, type(midpoint_position))
        new_vertices[centroid_idx] = centroid_vertex

        new_mesh.vertices = new_vertices.copy()

        # 4. build exactly one spoke edge per vertex in that loop
        spokes = {} # maps vertex_idx -> the Edge( vertex -> centroid )
        for vi in vertex_loop:
            # TODO: deal with how edges inherit options from facets
            e = Edge(next_edge_idx, vi, centroid_vertex.index,
                    options=facet.options.copy())
            new_edges[next_edge_idx] = e
            next_edge_idx += 1
            spokes[vi] = e

        new_mesh.edges = new_edges.copy()

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
            boundary_edge = mesh.get_edge(facet.edge_indices[i])
            # the two spokes: b→centroid and centroid→a
            spoke_b = spokes[b]
            spoke_a = spokes[a]
            # make sure the second spoke is oriented centroid→a:
            # TODO: POSSIBLE PROBLEM WITH USING ABSOLUTE VALUE FOR INDEX AND NOT REVERSED

            # build the new facet’s edge‐list **in the correct orientation**:
            child_edges = [boundary_edge.index, spoke_b.index, -spoke_a.index]

            child_idx = len(new_facets)
            child_options = facet.options.copy()
            child_options["parent_facet"] = facet.index

            cycled_edges = orient_edges_cycle(child_edges, new_mesh)

            child_facet = Facet(child_idx, cycled_edges,
                                    options=child_options)

            # After creating child_facet:
            # Get the parent normal
            parent_normal = facet.normal(mesh)
            # Get the child normal
            child_normal = child_facet.normal(new_mesh)
            # If the child normal is not aligned with the parent, flip the child facet
            if np.dot(child_normal, parent_normal) < 0:
                child_facet.edge_indices = [-idx for idx in reversed(child_facet.edge_indices)]
            new_facets[child_idx] = child_facet
            # Record that this child belongs to the same bodies
            children_map[facet.index].append(child_idx)

    # TODO: Associate facets with bodies! If no body exists skip
    # Step 3: Build updated bodies
    new_bodies = {}
    for body_idx, body in mesh.bodies.items():
        #body = mesh.bodies[body_idx]
        new_body_facets = []
        for old_facet_idx in body.facet_indices:
            # Instead of checking "if mesh.facets[old_facet_idx].index in facet_to_new_facets",
            # use children_map directly.
            if old_facet_idx in children_map and len(children_map[old_facet_idx]) > 0:
                new_body_facets.extend(children_map[old_facet_idx])
            else:
                new_body_facets.append(old_facet_idx)
        new_bodies[len(new_bodies)] = Body(len(new_bodies), new_body_facets,
                                options=body.options.copy(),
                                target_volume=body.target_volume)
    new_mesh.bodies = new_bodies

    new_mesh.facets = new_facets
    new_mesh.bodies = new_bodies
    new_mesh.global_parameters = mesh.global_parameters
    new_mesh.energy_modules = mesh.energy_modules
    new_mesh.instructions = mesh.instructions

    return new_mesh

def refine_triangle_mesh(mesh):
    # TODO: option for loop "r2" will refine twice

    new_mesh = Mesh()
    new_vertices = mesh.vertices.copy()
    new_edges = {}
    new_facets = {}
    edge_midpoints = {}  # (min_idx, max_idx) → midpoint Vertex
    edge_lookup = {}     # (min_idx, max_idx) → Edge
    facet_to_new_facets = {}  # facet.index → [Facet, ...]

    def get_or_create_edge(v_from, v_to):
        key = (min(v_from, v_to), max(v_from, v_to))
        if key in edge_lookup:
            return edge_lookup[key]
        new_edge_idx = len(new_edges) + 1
        edge = Edge(new_edge_idx, v_from, v_to)
        new_edges[new_edge_idx] = edge
        edge_lookup[key] = edge
        return edge

    # Step 1: Compute midpoint vertices and split each edge
    for edge_idx in mesh.edges.keys():
        edge = mesh.get_edge(edge_idx)
        v1, v2 = edge.tail_index, edge.head_index
        key = (min(v1, v2), max(v1, v2))
        if key not in edge_midpoints:
            midpoint_position = 0.5 * (mesh.vertices[v1].position + mesh.vertices[v2].position)
            midpoint_idx = len(new_vertices)
            print(midpoint_position, type(midpoint_position))
            midpoint = Vertex(midpoint_idx,
                              np.asarray(midpoint_position, dtype=float))
            new_vertices[midpoint_idx] = midpoint
            edge_midpoints[key] = midpoint

    new_mesh.vertices = new_vertices

    # Step 2: Subdivide each triangle into four smaller triangles
    for facet in mesh.facets.values():
        e0, e1, e2 = orient_edges_cycle(facet.edge_indices, mesh)
        E0 = mesh.get_edge(e0)
        v0, v1 = E0.tail_index, E0.head_index
        E1 = mesh.get_edge(e1)
        _, v2 = E1.tail_index, E1.head_index

        # simple sanity-check
        if v0 == v1 or v1 == v2 or v2 == v0:
            raise ValueError(f"Degenerate triangle: verts {v0},{v1},{v2}")

        m01 = edge_midpoints[(min(mesh.vertices[v0].index, mesh.vertices[v1].index),
                              max(mesh.vertices[v0].index,
                                  mesh.vertices[v1].index))].index
        m12 = edge_midpoints[(min(mesh.vertices[v1].index, mesh.vertices[v2].index),
                              max(mesh.vertices[v1].index,
                                  mesh.vertices[v2].index))].index
        m20 = edge_midpoints[(min(mesh.vertices[v2].index, mesh.vertices[v0].index),
                              max(mesh.vertices[v2].index,
                                  mesh.vertices[v0].index))].index
        child_facets = []

        # compute normal to make sure orientation of edges in child  facet is correct
        parent_normal = facet.normal(mesh)

        # Triangle 1: v0, m01, m20
        e1 = get_or_create_edge(v0, m01)
        e2 = get_or_create_edge(m01, m20)
        e3 = get_or_create_edge(m20, v0)
        new_mesh.edges.update(new_edges)
        #f1 = Facet(len(new_facets), [e1.index, e2.index, e3.index])
        raw1 = [e1.index, e2.index, e3.index]
        cyc1 = orient_edges_cycle(raw1, new_mesh)
        f1 = Facet(len(new_facets), cyc1)
        new_facets[len(new_facets)] = f1

        # Triangle 2: v1, m12, m01
        e1 = get_or_create_edge(v1, m12)
        e2 = get_or_create_edge(m12, m01)
        e3 = get_or_create_edge(m01, v1)
        new_mesh.edges.update(new_edges)
        #f2 = Facet(len(new_facets), [e1.index, e2.index, e3.index])
        raw2 = [e1.index, e2.index, e3.index]
        cyc2 = orient_edges_cycle(raw2, new_mesh)
        f2 = Facet(len(new_facets), cyc2)
        new_facets[len(new_facets)] = f2

        # Triangle 3: v2, m20, m12
        e1 = get_or_create_edge(v2, m20)
        e2 = get_or_create_edge(m20, m12)
        e3 = get_or_create_edge(m12, v2)
        new_mesh.edges.update(new_edges)
        #f3 = Facet(len(new_facets), [e1.index, e2.index, e3.index])
        raw3 = [e1.index, e2.index, e3.index]
        cyc3 = orient_edges_cycle(raw3, new_mesh)
        f3 = Facet(len(new_facets), cyc3)
        new_facets[len(new_facets)] = f3

        # Triangle 4 (center): m01, m12, m20
        e1 = get_or_create_edge(m01, m12)
        e2m = get_or_create_edge(m12, m20)
        e3 = get_or_create_edge(m20, m01)
        new_mesh.edges.update(new_edges)
        raw4 = [e1.index, e2.index, e3.index]
        cyc4 = orient_edges_cycle(raw4, new_mesh)
        f4 = Facet(len(new_facets), cyc4)
        #f4 = Facet(len(new_facets), [e1.index, e3.index, e2.index])
        new_facets[len(new_facets)] = f4

        #new_mesh.edges = new_edges

        f1_norm = f1.normal(new_mesh)
        if np.dot(f1_norm, parent_normal) < 0:
            f1.edge_indices = [-idx for idx in reversed(f1.edge_indices)]
        new_facets[f1.index] = f1
        child_facets.append(f1.index)
        facet_to_new_facets[facet.index] = child_facets

        if np.dot(f2.normal(new_mesh), parent_normal) < 0:
            f2.edge_indices = [-idx for idx in reversed(f2.edge_indices)]
        new_facets[f2.index] = f2
        child_facets.append(f2.index)

        if np.dot(f3.normal(new_mesh), parent_normal) < 0:
            f3.edge_indices = [-idx for idx in reversed(f3.edge_indices)]
        new_facets[f3.index] = f3
        child_facets.append(f3.index)

        if np.dot(f4.normal(new_mesh), parent_normal) < 0:
            f4.edge_indices = [-idx for idx in reversed(f4.edge_indices)]
        new_facets[f4.index] = f4
        child_facets.append(f4.index)

    # Step 3: Build updated bodies
    new_bodies = {}
    for body_idx, body in mesh.bodies.items():
        new_body_facets = []
        for old_facet_idx in body.facet_indices:
            if mesh.facets[old_facet_idx].index in facet_to_new_facets:
                new_body_facets.extend(facet_to_new_facets[old_facet_idx])
        new_bodies[len(new_bodies)] = Body(len(new_bodies), new_body_facets,
                                           target_volume=body.target_volume)
    new_mesh.vertices = new_vertices
    new_mesh.facets = new_facets
    new_mesh.bodies = new_bodies
    new_mesh.global_parameters = mesh.global_parameters
    new_mesh.energy_modules = mesh.energy_modules
    new_mesh.instructions = mesh.instructions

    return new_mesh


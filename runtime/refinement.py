import numpy as np
import logging
from geometry.entities import Vertex, Edge, Facet, Body, Mesh
from parameters.global_parameters import GlobalParameters
import sys
logger = logging.getLogger("membrane_solver")

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
    new_facets = {}
    next_edge_idx = max(e for e in new_edges.keys()) + 1

    # Prepare a map from old facet idx → list of new child facet idxs:
    children_map = {mesh.facets[facet_idx].index: [] for facet_idx in mesh.facets.keys()}

    for facet_idx in mesh.facets.keys():
        facet = mesh.facets[facet_idx]
        # 1. Leave triangles alone
        if len(facet.edge_indices) == 3:
            new_facets[facet_idx] = mesh.facets[facet_idx]
            continue

        # 2. Reconstruct the boundary loop of vertex‐indice
        tail_index = mesh.edges[facet.edge_indices[0]].tail_index
        head_index = mesh.edges[facet.edge_indices[0]].head_index
        # TODO: The -1 is a patchwork and not a good solution, maybe change to dictionary in the Mesh class 
        vertex_loop = [mesh.edges[facet.edge_indices[0]].tail_index]
        for edge_idx in facet.edge_indices:
            # TODO: The -1 is a patchwork and not a good solution, maybe change to dictionary in the Mesh class 
            edge = mesh.edges[edge_idx]
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
        centroid_vertex = Vertex(position=centroid_pos, index=centroid_idx)
        new_vertices[centroid_idx] = centroid_vertex

        # 4. build exactly one spoke edge per vertex in that loop
        spokes = {} # maps vertex_idx -> the Edge( vertex -> centroid )
        for vi in vertex_loop:
            # TODO: deal with how edges inherit options from facets
            e = Edge(next_edge_idx, vi, centroid_vertex.index,
                    options=facet.options.copy())
            new_edges[next_edge_idx] = e
            next_edge_idx += 1
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

            child_facet = Facet(child_idx, child_edges,
                                    options=child_options)
            new_facets[child_idx] = child_facet
            # Record that this child belongs to the same bodies
            children_map[facet.index].append(child_idx)

    # TODO: Associate facets with bodies! If no body exists skip
    new_bodies = {}
    for body_idx in mesh.bodies.keys():
        body = mesh.bodies[body_idx]
        new_list = []
        for facet_idx in body.facet_indices:
            if facet_idx in children_map[body_idx]:
                # replaced by its child facets
                new_list.extend(children_map[facet_idx])
            else:
                # the facet stayed the same (triangles)
                new_list.append(facet_idx)
        # ideally your Body ctor also carries over volume, constraints, etc.
        new_facet_list = [new_facets[idx].index for idx in new_list]
        new_bodies[body.index] = Body(body.index, new_facet_list,
                                options=body.options.copy())

    # TODO: is this assert even necessary?
    assert all(isinstance(f, Facet) for f in new_facets.values()), "new_facets still nested!"

    new_mesh.vertices = new_vertices
    new_mesh.edges = new_edges
    new_mesh.facets = new_facets
    new_mesh.bodies = new_bodies
    new_mesh.global_parameters = mesh.global_parameters
    #sys.exit(1)
    return new_mesh # new_vertices, new_edges, new_facets, new_bodies

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
            midpoint = Vertex(midpoint_idx, midpoint_position)
            new_vertices[midpoint_idx] = midpoint
            edge_midpoints[key] = midpoint

    new_mesh.vertices = new_vertices
    new_mesh.edges = new_edges

    # Step 2: Subdivide each triangle into four smaller triangles
    for f_idx in mesh.facets.keys():
        facet = mesh.facets[f_idx]
        v0, v1, v2 = [mesh.get_edge(e_idx).tail_index for e_idx in facet.edge_indices]
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
        f1 = Facet(len(new_facets), [e1.index, e2.index, e3.index])
        if np.dot(f1.normal(new_mesh), parent_normal) < 0.99:
            f1.edge_indices = [-e1.index, -e2.index, -e3.index]
        else:
            continue
        new_facets[f1.index] = f1
        child_facets.append(f1.index)

        # Triangle 2: v1, m12, m01
        e1 = get_or_create_edge(v1, m12)
        e2 = get_or_create_edge(m12, m01)
        e3 = get_or_create_edge(m01, v1)
        if np.dot(f2.normal(new_mesh), parent_normal) < 0.99:
            f2.edge_indices = [-e1.index, -e2.index, -e3.index]
        else:
            continue
        f2 = Facet(len(new_facets), [e1.index, e2.index, e3.index])
        new_facets[f2.index] = f2
        child_facets.append(f2.index)

        # Triangle 3: v2, m20, m12
        e1 = get_or_create_edge(v2, m20)
        e2 = get_or_create_edge(m20, m12)
        e3 = get_or_create_edge(m12, v2)
        f3 = Facet(len(new_facets), [e1.index, e2.index, e3.index])
        if np.dot(f3.normal(new_mesh), parent_normal) < 0.99:
            f3.edge_indices = [-e1.index, -e2.index, -e3.index]
        else:
            continue
        print(f"f3 {f3}")
        new_facets[f3.index] = f3
        child_facets.append(f3.index)

        # Triangle 4 (center): m01, m12, m20
        e1 = get_or_create_edge(m01, m12)
        e2 = get_or_create_edge(m12, m20)
        e3 = get_or_create_edge(m20, m01)
        f4 = Facet(len(new_facets), [-e1.index, -e2.index, -e3.index])
        if f4.normal(mesh) == parent_normal:
            continue
        else:
            f4.edge_indices = [-e1.index, -e2.index, -e3.index]
        new_facets[f4.index] = f4
        child_facets.append(f4.index)

        facet_to_new_facets[facet.index] = child_facets

    # Step 3: Build updated bodies
    new_bodies = {}
    for body in mesh.bodies.keys():
        new_body_facets = []
        for old_facet_idx in mesh.bodies[body].facet_indices:
            if mesh.facets[old_facet_idx].index in facet_to_new_facets:
                new_body_facets.extend(facet_to_new_facets[old_facet_idx])
        new_bodies[len(new_bodies)] = Body(len(new_bodies), new_body_facets)

    new_mesh.vertices = new_vertices
    new_mesh.edges = new_edges
    new_mesh.facets = new_facets
    new_mesh.bodies = new_bodies
    new_mesh.global_parameters = mesh.global_parameters

    return new_mesh

def compute_normal(v0, v1, v2):
    u = v1 - v0
    v = v2 - v0
    return np.cross(u, v)


import numpy as np
import logging
from geometry.entities import Vertex, Edge, Facet, Body, Mesh
from parameters.global_parameters import GlobalParameters
import sys
logger = logging.getLogger("membrane_solver")

def orient_edges_cycle(edge_idxs: list[int], mesh: Mesh) -> list[int]:
    """
    Given three edge indices (signed or unsigned), return a reordered list
    of _signed_ indices that form a proper tail→head cycle.
    """
    # build a map: vertex → list of (orig_idx, tail, head)
    conn = {}
    for orig in edge_idxs:
        ed = mesh.get_edge(abs(orig))
        t, h = (ed.tail_index, ed.head_index) if orig > 0 else (ed.head_index, ed.tail_index)
        conn.setdefault(t, []).append((orig, t, h))
        # also record the reversed orientation as a possibility,
        # so we can flip edges if necessary
        conn.setdefault(h, []).append((-orig, h, t))

    # pick one of the three edges to start
    start_raw = edge_idxs[0]
    # determine its tail→head in signed form
    first = None
    for signed, t, h in conn.get(mesh.get_edge(abs(start_raw)).tail_index, []):
        if abs(signed) == abs(start_raw):
            first = (signed, t, h)
            break
    if first is None:
        # fallback: just use raw as‐is
        first = (start_raw, *(
            (mesh.get_edge(start_raw).tail_index,
             mesh.get_edge(start_raw).head_index)
            if start_raw>0 else
            (mesh.get_edge(-start_raw).head_index,
             mesh.get_edge(-start_raw).tail_index)
        ))
    ordered = [first]
    used = {abs(first[0])}

    # now stitch the other two
    while len(ordered) < 3:
        _, _, cur_head = ordered[-1]
        for signed, t, h in sum(conn.get(cur_head, []), []):
            if abs(signed) in used:
                continue
            # this signed edge starts where we ended
            ordered.append((signed, t, h))
            used.add(abs(signed))
            break

    # return just the signed indices
    return [signed for signed,_,_ in ordered]

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
            midpoint = Vertex(midpoint_idx, midpoint_position)
            new_vertices[midpoint_idx] = midpoint
            edge_midpoints[key] = midpoint

    new_mesh.vertices = new_vertices
    #new_mesh.edges = new_edges

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
        #print(f"parent normal {parent_normal}")

        #print(f"new_facets {new_facets}, len {len(new_facets)}")
        # Triangle 1: v0, m01, m20
        e1 = get_or_create_edge(v0, m01)
        e2 = get_or_create_edge(m01, m20)
        e3 = get_or_create_edge(m20, v0)
        new_mesh.edges.update(new_edges)
        f1 = Facet(len(new_facets), [e1.index, e2.index, e3.index])
        new_facets[len(new_facets)] = f1

        # Triangle 2: v1, m12, m01
        e1 = get_or_create_edge(v1, m12)
        e2 = get_or_create_edge(m12, m01)
        e3 = get_or_create_edge(m01, v1)
        new_mesh.edges.update(new_edges)
        f2 = Facet(len(new_facets), [e1.index, e2.index, e3.index])
        new_facets[len(new_facets)] = f2

        # Triangle 3: v2, m20, m12
        e1 = get_or_create_edge(v2, m20)
        e2 = get_or_create_edge(m20, m12)
        e3 = get_or_create_edge(m12, v2)
        new_mesh.edges.update(new_edges)
        f3 = Facet(len(new_facets), [e1.index, e2.index, e3.index])
        new_facets[len(new_facets)] = f3

        # Triangle 4 (center): m01, m12, m20
        e1 = get_or_create_edge(m01, m12)
        e2m = get_or_create_edge(m12, m20)
        e3 = get_or_create_edge(m20, m01)
        new_mesh.edges.update(new_edges)
        f4 = Facet(len(new_facets), [e1.index, e3.index, e2.index])
        new_facets[len(new_facets)] = f4

        #–– debug interior edges ––
        # 1) create each interior edge and give it a clear name
        edge_mid01_mid12 = get_or_create_edge(m01, m12)
        edge_mid12_mid20 = get_or_create_edge(m12, m20)
        edge_mid20_mid01 = get_or_create_edge(m20, m01)

        # 3) pull out the numeric indices
        mid01_idx = edge_mid01_mid12.index
        mid12_idx = edge_mid12_mid20.index
        mid20_idx = edge_mid20_mid01.index

        for name, idx in [("mid01", mid01_idx),
                          ("mid12", mid12_idx),
                          ("mid20", mid20_idx)]:
            ed = new_mesh.get_edge(idx)
            print(f"Interior {name}: edge {idx} goes "
                  f"{ed.tail_index}→{ed.head_index}")
        print("About to build central facet f4 with edges:",
              [mid01_idx, mid12_idx, mid20_idx])
        #–– end debug ––

        #new_mesh.edges = new_edges

        # Debugging block: dump edge and vertex info for each new facet
        for name, f in zip(("f1","f2","f3","f4"), (f1, f2, f3, f4)):
            print(f"--- Debug {name} (index={f.index}) ---")
            print("  edge_indices:", f.edge_indices)
            # fetch the three vertices used in normal computation
            e0 = new_mesh.get_edge(f.edge_indices[0])
            e1 = new_mesh.get_edge(f.edge_indices[1])
            a_idx, b_idx, c_idx = e0.tail_index, e0.head_index, e1.head_index
            a = new_mesh.vertices[a_idx].position
            b = new_mesh.vertices[b_idx].position
            c = new_mesh.vertices[c_idx].position
            print(f"  vertex indices: a={a_idx}, b={b_idx}, c={c_idx}")
            print(f"  positions:\n    a={a}\n    b={b}\n    c={c}")
            n_unnormed = f.compute_normal(new_mesh)
            norm = np.linalg.norm(n_unnormed)
            print(f"  unnormalized normal = {n_unnormed},  |n| = {norm}")
            print()
        # end debug block



        f1_norm = f1.normal(new_mesh)
        if np.dot(f1_norm, parent_normal) < 0:
            f1.edge_indices = [-idx for idx in reversed(f1.edge_indices)]
            #print(f"f1_flipped: {f1.normal(new_mesh)}")
        new_facets[f1.index] = f1
        child_facets.append(f1.index)
        facet_to_new_facets[facet.index] = child_facets

        if np.dot(f2.normal(new_mesh), parent_normal) < 0:
            f2.edge_indices = [-idx for idx in reversed(f2.edge_indices)]
            #print(f"f2_flipped: {f2.normal(new_mesh)}")
        new_facets[f2.index] = f2
        child_facets.append(f2.index)

        if np.dot(f3.normal(new_mesh), parent_normal) < 0:
            f3.edge_indices = [-idx for idx in reversed(f3.edge_indices)]
            #print(f"f3_flipped: {f3.normal(new_mesh)}")
        new_facets[f3.index] = f3
        child_facets.append(f3.index)

        if np.dot(f4.normal(new_mesh), parent_normal) < 0:
            f4.edge_indices = [-idx for idx in reversed(f4.edge_indices)]
            #print(f"f4_flipped: {f4.normal(new_mesh)}")
        new_facets[f4.index] = f4
        child_facets.append(f4.index)

    # Step 3: Build updated bodies
    new_bodies = {}
    for body in mesh.bodies.keys():
        new_body_facets = []
        for old_facet_idx in mesh.bodies[body].facet_indices:
            if mesh.facets[old_facet_idx].index in facet_to_new_facets:
                new_body_facets.extend(facet_to_new_facets[old_facet_idx])
        new_bodies[len(new_bodies)] = Body(len(new_bodies), new_body_facets)
    new_mesh.vertices = new_vertices
    new_mesh.facets = new_facets
    new_mesh.bodies = new_bodies
    new_mesh.global_parameters = mesh.global_parameters

    return new_mesh


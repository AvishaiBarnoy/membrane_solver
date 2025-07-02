import numpy as np
import logging
from geometry.entities import Vertex, Edge, Facet, Body, Mesh
from parameters.global_parameters import GlobalParameters
import sys

logger = logging.getLogger("membrane_solver")

def orient_edges_cycle(edge_indices: list[int], mesh: Mesh, edge_dict: dict = None) -> list[int]:
    """
    Given a list of signed edge indices, return a properly oriented cycle using edge connectivity.
    If `edge_dict` is provided, use it instead of mesh.edges for lookup.
    """
    remaining = edge_indices.copy()
    if not remaining:
        return []

    edge_source = edge_dict if edge_dict is not None else mesh.edges

    first = remaining.pop(0)
    idx0 = abs(first)
    cycle = [idx0]
    prev_head = edge_source[idx0].head_index

    while remaining:
        for i, raw in enumerate(remaining):
            idx = abs(raw)
            E = edge_source[idx]
            if E.tail_index == prev_head:
                cycle.append(idx)
                prev_head = E.head_index
                remaining.pop(i)
                break
            if E.head_index == prev_head:
                cycle.append(-idx)
                prev_head = E.tail_index
                remaining.pop(i)
                break
        else:
            raise ValueError(f"Could not complete cycle: stuck at vertex {prev_head}, remaining edges {remaining}")

    if len(cycle) != len(edge_indices):
        raise AssertionError("orient_edges_cycle() returned wrong length")

    return cycle

def refine_polygonal_facets(mesh):
    """
    Triangulate all non-triangular facets in the mesh, regardless of 'no_refine'.
    Used when a facet needs to be broken into triangles due to adjacent edge refinement,
    even if it wasn't actively marked for refinement.

    This preserves topological consistency and is similar to how Surface Evolver maintains mesh watertightness.
    """
    import numpy as np
    from geometry.entities import Vertex, Edge, Facet, Body, Mesh
    import logging
    logger = logging.getLogger("membrane_solver")

    new_mesh = Mesh()
    new_vertices = mesh.vertices.copy()
    new_edges = mesh.edges.copy()
    new_facets = {}
    facet_children = {}

    next_edge_idx = max(new_edges.keys(), default=-1) + 1
    next_facet_idx = 0

    def add_edge(tail, head):
        nonlocal next_edge_idx
        e = Edge(next_edge_idx, tail, head)
        new_edges[next_edge_idx] = e
        idx = next_edge_idx
        next_edge_idx += 1
        return idx

    for f in mesh.facets.values():
        if len(f.edge_indices) == 3:
            new_facets[next_facet_idx] = Facet(
                index=next_facet_idx,
                edge_indices=f.edge_indices,
                fixed=f.fixed,
                options=f.options.copy()
            )
            facet_children[f.index] = [next_facet_idx]
            next_facet_idx += 1
            continue

        try:
            oriented = orient_edges_cycle(f.edge_indices, mesh)
        except Exception as e:
            logger.warning(f"Skipping facet {f.index} — bad edge loop: {e}")
            continue

        vertex_loop = []
        for ei in oriented:
            edge = mesh.get_edge(abs(ei))
            vi = edge.tail_index if ei > 0 else edge.head_index
            vertex_loop.append(vi)

        # Close the loop if needed
        if vertex_loop[0] == vertex_loop[-1]:
            vertex_loop.pop()

        if len(vertex_loop) < 3:
            logger.warning(f"Facet {f.index} has <3 vertices after loop reconstruction.")
            continue

        # Create centroid
        centroid_pos = np.mean([new_vertices[v].position for v in vertex_loop], axis=0)
        centroid_idx = max(new_vertices.keys()) + 1
        new_vertices[centroid_idx] = Vertex(centroid_idx, centroid_pos)

        # Create spokes and triangles
        child_ids = []
        for i in range(len(vertex_loop)):
            a = vertex_loop[i]
            b = vertex_loop[(i + 1) % len(vertex_loop)]

            # Existing boundary edge (or recreate it locally)
            e0 = add_edge(a, b)
            e1 = add_edge(b, centroid_idx)
            e2 = add_edge(centroid_idx, a)

            raw_edges = [e0, e1, e2]
            try:
                cyc = orient_edges_cycle(raw_edges, mesh, edge_dict=new_edges)
            except Exception as e:
                logger.warning(f"Skipping triangle from facet {f.index}: {e}")
                continue

            child_facet = Facet(
                index=next_facet_idx,
                edge_indices=cyc,
                fixed=f.fixed,
                options=f.options.copy()
            )

            new_facets[next_facet_idx] = child_facet
            child_ids.append(next_facet_idx)
            next_facet_idx += 1

        facet_children[f.index] = child_ids

    # Rebuild bodies
    new_bodies = {}
    for b in mesh.bodies.values():
        new_facet_list = []
        for fidx in b.facet_indices:
            new_facet_list.extend(facet_children.get(fidx, [fidx]))
        new_bodies[b.index] = Body(
            index=b.index,
            facet_indices=new_facet_list,
            options=b.options.copy(),
            target_volume=b.target_volume
        )

    new_mesh.vertices = new_vertices
    new_mesh.edges = new_edges
    new_mesh.facets = new_facets
    new_mesh.bodies = new_bodies
    new_mesh.global_parameters = mesh.global_parameters
    new_mesh.energy_modules = mesh.energy_modules
    new_mesh.constraint_modules = mesh.constraint_modules
    new_mesh.instructions = mesh.instructions
    new_mesh.build_connectivity_maps()

    return new_mesh

def refine_triangle_mesh(mesh):
    """
    Refines each triangular facet individually by splitting its edges and creating 4 new triangles.
    Partial refinement is allowed: neighboring facets may be refined or not independently.
    """
    import numpy as np
    from geometry.entities import Vertex, Edge, Facet, Body, Mesh

    def orient_edges_cycle(edge_indices: list[int], mesh: Mesh, edge_dict: dict = None) -> list[int]:
        """
        Given a list of signed edge indices, return a properly oriented cycle using edge connectivity.
        If `edge_dict` is provided, use it instead of mesh.edges for lookup.
        """
        remaining = edge_indices.copy()
        if not remaining:
            return []

        edge_source = edge_dict if edge_dict is not None else mesh.edges

        first = remaining.pop(0)
        idx0 = abs(first)
        cycle = [idx0]
        prev_head = edge_source[idx0].head_index

        while remaining:
            for i, raw in enumerate(remaining):
                idx = abs(raw)
                E = edge_source[idx]
                if E.tail_index == prev_head:
                    cycle.append(idx)
                    prev_head = E.head_index
                    remaining.pop(i)
                    break
                if E.head_index == prev_head:
                    cycle.append(-idx)
                    prev_head = E.tail_index
                    remaining.pop(i)
                    break
            else:
                raise ValueError(f"Could not complete cycle: stuck at vertex {prev_head}, remaining edges {remaining}")

        if len(cycle) != len(edge_indices):
            raise AssertionError("orient_edges_cycle() returned wrong length")

        return cycle

    new_mesh = Mesh()
    new_vertices = mesh.vertices.copy()
    new_edges = mesh.edges.copy()
    new_facets = {}
    facet_to_new_facets = {}

    next_vertex_idx = max(new_vertices.keys(), default=-1) + 1
    next_edge_idx = max(new_edges.keys(), default=-1) + 1

    def midpoint(a, b):
        return 0.5 * (a + b)

    def add_midpoint(v1_idx, v2_idx):
        pos1 = new_vertices[v1_idx].position
        pos2 = new_vertices[v2_idx].position
        mid_pos = midpoint(pos1, pos2)

        nonlocal next_vertex_idx
        vm = Vertex(
            index=next_vertex_idx,
            position=np.asarray(mid_pos, dtype=float),
            fixed=False
        )
        new_vertices[next_vertex_idx] = vm
        idx = next_vertex_idx
        next_vertex_idx += 1
        return idx

    def add_edge(tail, head, fixed=False, options=None):
        nonlocal next_edge_idx
        e = Edge(next_edge_idx, tail, head, fixed=fixed, options=options or {})
        new_edges[next_edge_idx] = e
        idx = next_edge_idx
        next_edge_idx += 1
        return idx

    for f in mesh.facets.values():
        if f.options.get("no_refine", False):
            new_facets[f.index] = f
            facet_to_new_facets[f.index] = [f.index]
            continue

        try:
            oriented = orient_edges_cycle(f.edge_indices, mesh)
        except Exception as e:
            print(f"[refine_triangle_mesh] Failed to orient facet {f.index}: {e}")
            continue

        v = []
        for ei in oriented:
            edge = mesh.get_edge(abs(ei))
            v.append(edge.tail_index if ei > 0 else edge.head_index)
        v0, v1, v2 = v

        vm01 = add_midpoint(v0, v1)
        vm12 = add_midpoint(v1, v2)
        vm20 = add_midpoint(v2, v0)

        tri_vertices = [
            (v0, vm01, vm20),
            (v1, vm12, vm01),
            (v2, vm20, vm12),
            (vm01, vm12, vm20)
        ]

        child_ids = []
        parent_normal = f.normal(mesh)

        for verts in tri_vertices:
            e1 = add_edge(verts[0], verts[1])
            e2 = add_edge(verts[1], verts[2])
            e3 = add_edge(verts[2], verts[0])

            raw_edges = [e1, e2, e3]
            try:
                cyc = orient_edges_cycle(raw_edges, mesh, edge_dict=new_edges)
            except Exception as e:
                print(f"[refine_triangle_mesh] Failed to orient child triangle: {e}")
                continue

            child_idx = len(new_facets)
            child_facet = Facet(child_idx, cyc, fixed=f.fixed, options=f.options.copy())

            if np.dot(child_facet.normal(Mesh(vertices=new_vertices, edges=new_edges, facets={}, bodies={})), parent_normal) < 0:
                child_facet.edge_indices = [-idx for idx in reversed(child_facet.edge_indices)]

            new_facets[child_idx] = child_facet
            child_ids.append(child_idx)

        facet_to_new_facets[f.index] = child_ids

    new_bodies = {}
    for b in mesh.bodies.values():
        new_facet_indices = []
        for fidx in b.facet_indices:
            new_facet_indices.extend(facet_to_new_facets.get(fidx, [fidx]))
        new_bodies[b.index] = Body(
            index=b.index,
            facet_indices=new_facet_indices,
            options=b.options.copy(),
            target_volume=b.target_volume
        )

    new_mesh.vertices = new_vertices
    new_mesh.edges = new_edges
    new_mesh.facets = new_facets
    new_mesh.bodies = new_bodies
    new_mesh.global_parameters = mesh.global_parameters
    new_mesh.energy_modules = mesh.energy_modules
    new_mesh.constraint_modules = mesh.constraint_modules
    new_mesh.instructions = mesh.instructions
    new_mesh.build_connectivity_maps()

    return new_mesh

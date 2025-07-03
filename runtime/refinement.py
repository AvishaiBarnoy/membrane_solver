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

    def add_edge(tail, head, fixed=False, options=None):
        nonlocal next_edge_idx
        e = Edge(next_edge_idx, tail, head, fixed=fixed, options=options or {})
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
        new_vertices[centroid_idx] = Vertex(
            centroid_idx,
            centroid_pos,
            fixed=f.fixed,
            options=f.options.copy(),
        )

        parent_normal = f.normal(mesh)
        # Create spokes and triangles
        child_ids = []
        for i in range(len(vertex_loop)):
            a = vertex_loop[i]
            b = vertex_loop[(i + 1) % len(vertex_loop)]

            boundary_edge = mesh.get_edge(f.edge_indices[i])
            e0 = add_edge(a, b, fixed=boundary_edge.fixed, options=boundary_edge.options.copy())
            e1 = add_edge(b, centroid_idx, fixed=f.fixed, options=f.options.copy())
            e2 = add_edge(centroid_idx, a, fixed=f.fixed, options=f.options.copy())

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

            if np.dot(
                child_facet.normal(Mesh(vertices=new_vertices, edges=new_edges, facets={}, bodies={})),
                parent_normal,
            ) < 0:
                child_facet.edge_indices = [-idx for idx in reversed(child_facet.edge_indices)]

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
    new_facets: dict[int, Facet] = {}
    facet_children: dict[int, list[int]] = {}

    next_vertex_idx = max(new_vertices.keys(), default=-1) + 1
    next_edge_idx = max(new_edges.keys(), default=-1) + 1

    def add_edge(tail: int, head: int, *, fixed: bool = False, options: dict | None = None) -> int:
        """Create an edge in ``new_edges`` and return its index."""
        nonlocal next_edge_idx
        e = Edge(next_edge_idx, tail, head, fixed=fixed, options=options or {})
        new_edges[next_edge_idx] = e
        idx = next_edge_idx
        next_edge_idx += 1
        return idx

    def add_midpoint(v1_idx: int, v2_idx: int, parent_edge: Edge) -> int:
        """Create a midpoint vertex inheriting options from ``parent_edge``."""
        nonlocal next_vertex_idx
        pos1 = new_vertices[v1_idx].position
        pos2 = new_vertices[v2_idx].position
        mid_pos = 0.5 * (pos1 + pos2)

        vm = Vertex(
            index=next_vertex_idx,
            position=np.asarray(mid_pos, dtype=float),
            fixed=parent_edge.fixed,
            options=parent_edge.options.copy(),
        )
        new_vertices[next_vertex_idx] = vm
        idx = next_vertex_idx
        next_vertex_idx += 1
        return idx

    # --- Step 1: gather edges that need splitting ---
    facets_to_refine = [f for f in mesh.facets.values() if not f.options.get("no_refine", False)]
    edges_to_split: set[int] = set()
    for facet in facets_to_refine:
        try:
            oriented = orient_edges_cycle(facet.edge_indices, mesh)
        except Exception:
            continue
        for ei in oriented:
            edges_to_split.add(abs(ei))

    # Create midpoint vertices and split edges
    split_info: dict[int, tuple[int, int, int]] = {}
    for ei in edges_to_split:
        edge = mesh.edges[ei]
        mid_idx = add_midpoint(edge.tail_index, edge.head_index, edge)
        e1 = add_edge(edge.tail_index, mid_idx, fixed=edge.fixed, options=edge.options.copy())
        e2 = add_edge(mid_idx, edge.head_index, fixed=edge.fixed, options=edge.options.copy())
        split_info[ei] = (mid_idx, e1, e2)

    # --- Step 2: rebuild facets ---
    temp_mesh = Mesh(vertices=new_vertices, edges=new_edges, facets={}, bodies={})

    for facet in mesh.facets.values():
        oriented = orient_edges_cycle(facet.edge_indices, mesh)

        expanded_edges: list[int] = []
        midpoints: list[int] = []
        for ei in oriented:
            abs_ei = abs(ei)
            if abs_ei in split_info:
                mid, e1, e2 = split_info[abs_ei]
                if ei > 0:
                    expanded_edges.extend([e1, e2])
                else:
                    expanded_edges.extend([-e2, -e1])
                midpoints.append(mid)
            else:
                expanded_edges.append(ei)
        if facet.options.get("no_refine", False):
            cyc = orient_edges_cycle(expanded_edges, temp_mesh, edge_dict=new_edges)
            new_facets[facet.index] = Facet(facet.index, cyc, fixed=facet.fixed, options=facet.options.copy())
            facet_children[facet.index] = [facet.index]
            continue

        if len(midpoints) != 3:
            # Should not happen but guard against malformed input
            cyc = orient_edges_cycle(expanded_edges, temp_mesh, edge_dict=new_edges)
            new_facets[facet.index] = Facet(facet.index, cyc, fixed=facet.fixed, options=facet.options.copy())
            facet_children[facet.index] = [facet.index]
            continue

        v = []
        for ei in oriented:
            edge = mesh.get_edge(abs(ei))
            v.append(edge.tail_index if ei > 0 else edge.head_index)
        v0, v1, v2 = v
        m01, m12, m20 = (
            split_info[abs(oriented[0])][0],
            split_info[abs(oriented[1])][0],
            split_info[abs(oriented[2])][0],
        )

        tri_vertices = [
            (v0, m01, m20),
            (v1, m12, m01),
            (v2, m20, m12),
            (m01, m12, m20),
        ]

        child_ids: list[int] = []
        parent_normal = facet.normal(mesh)

        for verts in tri_vertices:
            # Reuse boundary split edges when possible
            raw_edges: list[int] = []
            pairs = [(verts[0], verts[1]), (verts[1], verts[2]), (verts[2], verts[0])]
            for tail, head in pairs:
                reused = False
                # Check if a split edge already exists
                for ei, (mid, e1, e2) in split_info.items():
                    if (mesh.edges[ei].tail_index, mid) == (tail, head):
                        raw_edges.append(e1)
                        reused = True
                        break
                    if (mid, mesh.edges[ei].head_index) == (tail, head):
                        raw_edges.append(e2)
                        reused = True
                        break
                    if (mesh.edges[ei].head_index, mid) == (tail, head):
                        raw_edges.append(-e2)
                        reused = True
                        break
                    if (mid, mesh.edges[ei].tail_index) == (tail, head):
                        raw_edges.append(-e1)
                        reused = True
                        break
                if not reused:
                    raw_edges.append(add_edge(tail, head, options=facet.options.copy(), fixed=facet.fixed))
            cyc = orient_edges_cycle(raw_edges, temp_mesh, edge_dict=new_edges)

            child_idx = len(new_facets)
            child_facet = Facet(child_idx, cyc, fixed=facet.fixed, options=facet.options.copy())
            if np.dot(child_facet.normal(Mesh(vertices=new_vertices, edges=new_edges, facets={}, bodies={})), parent_normal) < 0:
                child_facet.edge_indices = [-idx for idx in reversed(child_facet.edge_indices)]
            new_facets[child_idx] = child_facet
            child_ids.append(child_idx)

        facet_children[facet.index] = child_ids

    # --- Step 3: rebuild bodies ---
    new_bodies: dict[int, Body] = {}
    for b in mesh.bodies.values():
        new_facet_indices: list[int] = []
        for fidx in b.facet_indices:
            new_facet_indices.extend(facet_children.get(fidx, [fidx]))
        new_bodies[b.index] = Body(
            index=b.index,
            facet_indices=new_facet_indices,
            options=b.options.copy(),
            target_volume=b.target_volume,
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

    # Final step – triangulate any polygonal facets created during edge splitting
    new_mesh = refine_polygonal_facets(new_mesh)
    return new_mesh

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
            E = mesh.get_edge(idx)
            # Case A: we traverse E as tail->head
            if E.tail_index == prev_head:
                cycle.append(idx)
                prev_head = E.head_index
                remaining.pop(i)
                break

            # Case B: we traverse E as head->tail  (so sign it negative)
            if E.head_index == prev_head:
                cycle.append(-idx)
                prev_head = E.tail_index
                remaining.pop(i)
                break
        else:
            raise ValueError(
                f"Could not complete cycle: stuck at vertex {prev_head}, remaining edges {remaining}"
            )

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
    next_edge_idx = max(e for e in new_edges.keys()) + 1 if new_edges else 0

    new_mesh.edges = new_edges.copy()

    # Prepare a map from old facet idx → list of new child facet idxs:
    children_map = {
        mesh.facets[facet_idx].index: [] for facet_idx in mesh.facets.keys()
    }

    for f_idx, facet in mesh.facets.items():
        # 1. Leave triangles alone
        if len(facet.edge_indices) == 3:
            if "surface_tension" not in facet.options:
                facet.options["surface_tension"] = mesh.global_parameters.get(
                    "surface_tension", 1.0
                )
            new_facets[f_idx] = facet
            continue

        # 2. Reconstruct the boundary loop of vertex‐indice
        vertex_loop = [mesh.get_edge(facet.edge_indices[0]).tail_index]
        for edge_idx in facet.edge_indices:
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
        centroid_options = {}
        if "constraints" in facet.options:
            centroid_options["constraints"] = facet.options["constraints"]
        centroid_vertex = Vertex(
            index=centroid_idx,
            position=np.asarray(centroid_pos, dtype=float),
            fixed=facet.fixed,
            options=centroid_options,
        )
        new_vertices[centroid_idx] = centroid_vertex

        new_mesh.vertices = new_vertices.copy()

        # 4. build exactly one spoke edge per vertex in that loop
        spokes = {}  # maps vertex_idx -> the Edge( vertex -> centroid )
        for vi in vertex_loop:
            e = Edge(
                next_edge_idx,
                vi,
                centroid_vertex.index,
                fixed=facet.fixed,
                options=facet.options.copy(),
            )
            # If parent facet has no_refine, mark the new edge as non-refinable
            if facet.options.get("no_refine", False):
                e.options["no_refine"] = True
            new_edges[next_edge_idx] = e
            spokes[vi] = e
            next_edge_idx += 1
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
            boundary_edge = mesh.get_edge(facet.edge_indices[i])
            spoke_b = spokes[b]
            spoke_a = spokes[a]

            child_options = facet.options
            child_options["surface_tension"] = facet.options.get("surface_tension", 1.0)
            child_options["parent_facet"] = facet.index
            child_options["constraints"] = facet.options.get("constraints", [])

            # build the new facet's edge‐list **in the correct orientation**:
            child_edges = [boundary_edge.index, spoke_b.index, -spoke_a.index]

            child_idx = len(new_facets)

            cycled_edges = orient_edges_cycle(child_edges, new_mesh)

            child_facet = Facet(
                child_idx, cycled_edges, fixed=facet.fixed, options=child_options
            )

            # After creating child_facet:
            # Get the parent normal
            parent_normal = facet.normal(mesh)
            # Get the child normal
            child_normal = child_facet.normal(new_mesh)
            # If the child normal is not aligned with the parent, flip the child facet
            if np.dot(child_normal, parent_normal) < 0:
                child_facet.edge_indices = [
                    -idx for idx in reversed(child_facet.edge_indices)
                ]
            new_facets[child_idx] = child_facet
            # Record that this child belongs to the same bodies
            children_map[facet.index].append(child_idx)

    # TODO: Associate facets with bodies! If no body exists skip
    # Step 3: Build updated bodies
    new_bodies = {}
    for body_idx, body in mesh.bodies.items():
        # body = mesh.bodies[body_idx]
        new_body_facets = []
        for old_facet_idx in body.facet_indices:
            # Instead of checking "if mesh.facets[old_facet_idx].index in facet_to_new_facets",
            # use children_map directly.
            if old_facet_idx in children_map and len(children_map[old_facet_idx]) > 0:
                new_body_facets.extend(children_map[old_facet_idx])
            else:
                new_body_facets.append(old_facet_idx)
        new_bodies[len(new_bodies)] = Body(
            len(new_bodies),
            new_body_facets,
            options=body.options.copy(),
            target_volume=body.target_volume,
        )
    new_mesh.bodies = new_bodies

    new_mesh.facets = new_facets
    new_mesh.bodies = new_bodies
    new_mesh.global_parameters = mesh.global_parameters
    new_mesh.energy_modules = mesh.energy_modules
    new_mesh.constraint_modules = mesh.constraint_modules
    new_mesh.instructions = mesh.instructions
    mesh.build_connectivity_maps()

    return new_mesh


def refine_triangle_mesh(mesh):
    new_mesh = Mesh()
    new_vertices = mesh.vertices.copy()
    new_edges = {}
    new_facets = {}
    edge_midpoints = {}  # (min_idx, max_idx) → midpoint Vertex
    edge_lookup = {}  # (min_idx, max_idx) → Edge
    facet_to_new_facets = {}  # facet.index → [Facet, ...]
    next_facet_idx = max(mesh.facets.keys()) + 1 if mesh.facets else 0

    def get_or_create_edge(v_from, v_to, parent_edge=None, parent_facet=None):
        key = (min(v_from, v_to), max(v_from, v_to))
        if key in edge_lookup:
            return edge_lookup[key]
        new_edge_idx = len(new_edges) + 1
        edge = Edge(new_edge_idx, v_from, v_to)
        
        # Inherit properties from parent edge if available
        if parent_edge:
            edge.fixed = parent_edge.fixed
            edge.options = parent_edge.options.copy()
        elif parent_facet:
            # For new edges created within a facet, inherit facet properties
            edge.fixed = parent_facet.fixed
            edge.options = parent_facet.options.copy()
            # If parent facet has no_refine, mark the new edge as non-refinable
            if parent_facet.options.get("no_refine", False):
                edge.options["no_refine"] = True
        
        new_edges[new_edge_idx] = edge
        edge_lookup[key] = edge
        return edge

    # Collect all edges that should be refined
    # An edge should be refined if it belongs to at least one facet that will be refined
    # AND the edge itself is not marked with no_refine
    # AND none of the facets it belongs to have no_refine=True
    edges_to_refine = set()
    edges_in_no_refine_facets = set()
    
    # First, collect all edges that are in facets marked with no_refine
    for facet in mesh.facets.values():
        if facet.options.get("no_refine", False):
            for ei in facet.edge_indices:
                edges_in_no_refine_facets.add(abs(ei))
    
    # Then, collect edges that should be refined
    for facet in mesh.facets.values():
        if not facet.options.get("no_refine", False):  # Only consider refinable facets
            for ei in facet.edge_indices:
                edge_idx = abs(ei)
                edge = mesh.get_edge(edge_idx)
                # Edge should be refined if:
                # 1. It's not marked no_refine itself
                # 2. It's not in a facet marked no_refine
                if (not edge.options.get("no_refine", False) and 
                    edge_idx not in edges_in_no_refine_facets):
                    edges_to_refine.add(edge_idx)

    # Step 1: Compute midpoint vertices only for edges that will be refined
    for edge_idx in edges_to_refine:
        edge = mesh.get_edge(edge_idx)
        v1, v2 = edge.tail_index, edge.head_index
        key = (min(v1, v2), max(v1, v2))
        if key not in edge_midpoints:
            midpoint_position = 0.5 * (
                mesh.vertices[v1].position + mesh.vertices[v2].position
            )
            midpoint_idx = len(new_vertices)
            midpoint = Vertex(
                midpoint_idx,
                np.asarray(midpoint_position, dtype=float),
                fixed=edge.fixed,
                options=edge.options.copy(),
            )
            new_vertices[midpoint_idx] = midpoint
            edge_midpoints[key] = midpoint

    new_mesh.vertices = new_vertices

    # Step 2: Subdivide each triangle
    for facet in mesh.facets.values():
        oriented = orient_edges_cycle(facet.edge_indices, mesh)
        e0parent, e1parent, e2parent = oriented
        E0 = mesh.get_edge(e0parent)
        v0, v1 = E0.tail_index, E0.head_index
        E1 = mesh.get_edge(e1parent)
        _, v2 = E1.tail_index, E1.head_index

        # Check if any of the facet's edges can be refined
        parent_edges = [mesh.get_edge(abs(ei)) for ei in oriented]
        refinable_edges = [abs(ei) in edges_to_refine for ei in oriented]
        
        # If no edges can be refined, just copy the facet
        if not any(refinable_edges):
            raw_edges = []
            for ei in oriented:
                edge = mesh.get_edge(ei)
                if ei > 0:
                    e = get_or_create_edge(edge.tail_index, edge.head_index, parent_edge=edge)
                    raw_edges.append(e.index)
                else:
                    e = get_or_create_edge(edge.head_index, edge.tail_index, parent_edge=edge)
                    raw_edges.append(-e.index)
            new_mesh.edges.update(new_edges)
            cyc = orient_edges_cycle(raw_edges, new_mesh)
            nf = Facet(facet.index, cyc, fixed=facet.fixed, options=facet.options.copy())
            new_facets[facet.index] = nf
            facet_to_new_facets[facet.index] = [facet.index]
            continue

        # simple sanity-check
        if v0 == v1 or v1 == v2 or v2 == v0:
            raise ValueError(f"Degenerate triangle: verts {v0},{v1},{v2}")

        # Get midpoints for refinable edges, or use original vertices for non-refinable edges
        m01 = edge_midpoints[(min(v0, v1), max(v0, v1))].index if refinable_edges[0] else None
        m12 = edge_midpoints[(min(v1, v2), max(v1, v2))].index if refinable_edges[1] else None
        m20 = edge_midpoints[(min(v2, v0), max(v2, v0))].index if refinable_edges[2] else None
        
        child_facets = []
        parent_normal = facet.normal(mesh)

        # Create child triangles based on which edges are refinable
        if all(refinable_edges):
            # All edges refinable - standard 1-to-4 refinement
            # Triangle 1: v0, m01, m20
            e1 = get_or_create_edge(v0, m01, parent_edge=parent_edges[0])
            e2 = get_or_create_edge(m01, m20, parent_facet=facet)
            e3 = get_or_create_edge(m20, v0, parent_edge=parent_edges[2])

            new_mesh.edges.update(new_edges)
            raw1 = [e1.index, e2.index, e3.index]
            cyc1 = orient_edges_cycle(raw1, new_mesh)
            f1 = Facet(next_facet_idx, cyc1, fixed=facet.fixed, options=facet.options.copy())
            new_facets[next_facet_idx] = f1
            next_facet_idx += 1

            # Triangle 2: v1, m12, m01
            e1 = get_or_create_edge(v1, m12, parent_edge=parent_edges[1])
            e2 = get_or_create_edge(m12, m01, parent_facet=facet)
            e3 = get_or_create_edge(m01, v1, parent_edge=parent_edges[0])

            new_mesh.edges.update(new_edges)
            raw2 = [e1.index, e2.index, e3.index]
            cyc2 = orient_edges_cycle(raw2, new_mesh)
            f2 = Facet(next_facet_idx, cyc2, fixed=facet.fixed, options=facet.options.copy())
            new_facets[next_facet_idx] = f2
            next_facet_idx += 1

            # Triangle 3: v2, m20, m12
            e1 = get_or_create_edge(v2, m20, parent_edge=parent_edges[2])
            e2 = get_or_create_edge(m20, m12, parent_facet=facet)
            e3 = get_or_create_edge(m12, v2, parent_edge=parent_edges[1])

            new_mesh.edges.update(new_edges)
            raw3 = [e1.index, e2.index, e3.index]
            cyc3 = orient_edges_cycle(raw3, new_mesh)
            f3 = Facet(next_facet_idx, cyc3, fixed=facet.fixed, options=facet.options.copy())
            new_facets[next_facet_idx] = f3
            next_facet_idx += 1

            # Triangle 4 (center): m01, m12, m20
            e1 = get_or_create_edge(m01, m12, parent_facet=facet)
            e2 = get_or_create_edge(m12, m20, parent_facet=facet)
            e3 = get_or_create_edge(m20, m01, parent_facet=facet)
            new_mesh.edges.update(new_edges)
            raw4 = [e1.index, e2.index, e3.index]
            cyc4 = orient_edges_cycle(raw4, new_mesh)
            f4 = Facet(next_facet_idx, cyc4, fixed=facet.fixed, options=facet.options.copy())
            new_facets[next_facet_idx] = f4
            next_facet_idx += 1

            child_facets = [f1, f2, f3, f4]
        else:
            # Partial refinement - handle cases where only some edges are refinable
            # This is more complex and requires careful handling of the subdivision
            # For now, implement the most common cases
            
            if sum(refinable_edges) == 1:
                # Only one edge is refinable - split into 2 triangles
                if refinable_edges[0]:  # edge v0-v1 is refinable
                    # Triangle 1: v0, m01, v2
                    e1 = get_or_create_edge(v0, m01, parent_edge=parent_edges[0])
                    e2 = get_or_create_edge(m01, v2, parent_facet=facet)
                    e3 = get_or_create_edge(v2, v0, parent_edge=parent_edges[2])
                    
                    # Triangle 2: m01, v1, v2
                    e4 = get_or_create_edge(m01, v1, parent_edge=parent_edges[0])
                    e5 = get_or_create_edge(v1, v2, parent_edge=parent_edges[1])
                    e6 = get_or_create_edge(v2, m01, parent_facet=facet)
                    
                elif refinable_edges[1]:  # edge v1-v2 is refinable
                    # Triangle 1: v1, m12, v0
                    e1 = get_or_create_edge(v1, m12, parent_edge=parent_edges[1])
                    e2 = get_or_create_edge(m12, v0, parent_facet=facet)
                    e3 = get_or_create_edge(v0, v1, parent_edge=parent_edges[0])
                    
                    # Triangle 2: m12, v2, v0
                    e4 = get_or_create_edge(m12, v2, parent_edge=parent_edges[1])
                    e5 = get_or_create_edge(v2, v0, parent_edge=parent_edges[2])
                    e6 = get_or_create_edge(v0, m12, parent_facet=facet)
                    
                else:  # edge v2-v0 is refinable
                    # Triangle 1: v2, m20, v1
                    e1 = get_or_create_edge(v2, m20, parent_edge=parent_edges[2])
                    e2 = get_or_create_edge(m20, v1, parent_facet=facet)
                    e3 = get_or_create_edge(v1, v2, parent_edge=parent_edges[1])
                    
                    # Triangle 2: m20, v0, v1
                    e4 = get_or_create_edge(m20, v0, parent_edge=parent_edges[2])
                    e5 = get_or_create_edge(v0, v1, parent_edge=parent_edges[0])
                    e6 = get_or_create_edge(v1, m20, parent_facet=facet)
                
                new_mesh.edges.update(new_edges)
                raw1 = [e1.index, e2.index, e3.index]
                raw2 = [e4.index, e5.index, e6.index]
                cyc1 = orient_edges_cycle(raw1, new_mesh)
                cyc2 = orient_edges_cycle(raw2, new_mesh)
                f1 = Facet(next_facet_idx, cyc1, fixed=facet.fixed, options=facet.options.copy())
                f2 = Facet(next_facet_idx + 1, cyc2, fixed=facet.fixed, options=facet.options.copy())
                new_facets[next_facet_idx] = f1
                new_facets[next_facet_idx + 1] = f2
                next_facet_idx += 2
                child_facets = [f1, f2]
                
            elif sum(refinable_edges) == 2:
                # Two edges are refinable - split into 3 triangles
                # Implementation would go here for the 2-edge case
                # For now, fall back to copying the original facet
                raw_edges = []
                for ei in oriented:
                    edge = mesh.get_edge(ei)
                    if ei > 0:
                        e = get_or_create_edge(edge.tail_index, edge.head_index, parent_edge=edge)
                        raw_edges.append(e.index)
                    else:
                        e = get_or_create_edge(edge.head_index, edge.tail_index, parent_edge=edge)
                        raw_edges.append(-e.index)
                new_mesh.edges.update(new_edges)
                cyc = orient_edges_cycle(raw_edges, new_mesh)
                nf = Facet(next_facet_idx, cyc, fixed=facet.fixed, options=facet.options.copy())
                new_facets[next_facet_idx] = nf
                next_facet_idx += 1
                child_facets = [nf]

        # Check if the child facets are oriented correctly and preserve parent normal
        for child_facet in child_facets:
            child_normal = child_facet.normal(new_mesh)
            if np.dot(child_normal, parent_normal) < 0:
                child_facet.edge_indices = [-idx for idx in reversed(child_facet.edge_indices)]
            new_facets[child_facet.index] = child_facet

        facet_to_new_facets[facet.index] = [f.index for f in child_facets]

    # Step 3: Build updated bodies
    new_bodies = {}
    for body_idx, body in mesh.bodies.items():
        new_body_facets = []
        for old_facet_idx in body.facet_indices:
            if mesh.facets[old_facet_idx].index in facet_to_new_facets:
                new_body_facets.extend(facet_to_new_facets[old_facet_idx])
        new_bodies[len(new_bodies)] = Body(
            len(new_bodies), new_body_facets, target_volume=body.target_volume
        )
    new_mesh.vertices = new_vertices
    new_mesh.facets = new_facets
    new_mesh.bodies = new_bodies
    new_mesh.global_parameters = mesh.global_parameters
    new_mesh.energy_modules = mesh.energy_modules
    new_mesh.instructions = mesh.instructions

    new_mesh.build_connectivity_maps()

    return new_mesh

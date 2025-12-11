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
        parent_target_area = facet.options.get("target_area")
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
            # Spoke edges created within no_refine facets should be marked non-refinable
            # This is correct behavior - new edges within no_refine facets inherit no_refine
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

            child_options = facet.options.copy()
            child_options.pop("target_area", None)
            child_options["surface_tension"] = facet.options.get("surface_tension", mesh.global_parameters.get("surface_tension", 1.0))
            child_options["parent_facet"] = facet.index
            child_options["constraints"] = facet.options.get("constraints", [])
            # Ensure child facets have energy module set
            if "energy" not in child_options:
                child_options["energy"] = ["surface"]

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

        # Distribute facet target area across children if needed
        child_ids = children_map.get(facet.index, [])
        if parent_target_area is not None and child_ids:
            child_areas = [(cid, new_facets[cid].compute_area(new_mesh)) for cid in child_ids]
            total = sum(area for _, area in child_areas)
            if total > 1e-12:
                for cid, area in child_areas:
                    new_facets[cid].options["target_area"] = parent_target_area * (area / total)

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
    new_mesh.build_connectivity_maps()
    new_mesh.build_facet_vertex_loops()

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
    # An edge should be refined if:
    # 1. The edge itself is not marked with no_refine
    # 2. The edge belongs to at least one refinable facet (not marked no_refine)
    # This follows Evolver behavior: original boundary edges are refinable unless explicitly marked no_refine
    edges_to_refine = set()
    
    # Collect edges that should be refined
    for facet in mesh.facets.values():
        for ei in facet.edge_indices:
            edge_idx = abs(ei)
            edge = mesh.get_edge(edge_idx)
            # Edge should be refined if:
            # 1. It's not marked no_refine itself
            # 2. At least one facet containing this edge is refinable
            if not edge.options.get("no_refine", False):
                # Check if this edge belongs to at least one refinable facet
                belongs_to_refinable_facet = False
                for other_facet in mesh.facets.values():
                    if edge_idx in [abs(e) for e in other_facet.edge_indices]:
                        if not other_facet.options.get("no_refine", False):
                            belongs_to_refinable_facet = True
                            break
                
                if belongs_to_refinable_facet:
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
        parent_target_area = facet.options.get("target_area")
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
            child_opts = facet.options.copy()
            child_opts.pop("target_area", None)
            f1 = Facet(next_facet_idx, cyc1, fixed=facet.fixed, options=child_opts)
            new_facets[next_facet_idx] = f1
            next_facet_idx += 1

            # Triangle 2: v1, m12, m01
            e1 = get_or_create_edge(v1, m12, parent_edge=parent_edges[1])
            e2 = get_or_create_edge(m12, m01, parent_facet=facet)
            e3 = get_or_create_edge(m01, v1, parent_edge=parent_edges[0])

            new_mesh.edges.update(new_edges)
            raw2 = [e1.index, e2.index, e3.index]
            cyc2 = orient_edges_cycle(raw2, new_mesh)
            child_opts = facet.options.copy()
            child_opts.pop("target_area", None)
            f2 = Facet(next_facet_idx, cyc2, fixed=facet.fixed, options=child_opts)
            new_facets[next_facet_idx] = f2
            next_facet_idx += 1

            # Triangle 3: v2, m20, m12
            e1 = get_or_create_edge(v2, m20, parent_edge=parent_edges[2])
            e2 = get_or_create_edge(m20, m12, parent_facet=facet)
            e3 = get_or_create_edge(m12, v2, parent_edge=parent_edges[1])

            new_mesh.edges.update(new_edges)
            raw3 = [e1.index, e2.index, e3.index]
            cyc3 = orient_edges_cycle(raw3, new_mesh)
            child_opts = facet.options.copy()
            child_opts.pop("target_area", None)
            f3 = Facet(next_facet_idx, cyc3, fixed=facet.fixed, options=child_opts)
            new_facets[next_facet_idx] = f3
            next_facet_idx += 1

            # Triangle 4 (center): m01, m12, m20
            e1 = get_or_create_edge(m01, m12, parent_facet=facet)
            e2 = get_or_create_edge(m12, m20, parent_facet=facet)
            e3 = get_or_create_edge(m20, m01, parent_facet=facet)
            new_mesh.edges.update(new_edges)
            raw4 = [e1.index, e2.index, e3.index]
            cyc4 = orient_edges_cycle(raw4, new_mesh)
            child_opts = facet.options.copy()
            child_opts.pop("target_area", None)
            f4 = Facet(next_facet_idx, cyc4, fixed=facet.fixed, options=child_opts)
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
                child_opts = facet.options.copy()
                child_opts.pop("target_area", None)
                f1 = Facet(next_facet_idx, cyc1, fixed=facet.fixed, options=child_opts)
                child_opts = facet.options.copy()
                child_opts.pop("target_area", None)
                f2 = Facet(next_facet_idx + 1, cyc2, fixed=facet.fixed, options=child_opts)
                new_facets[next_facet_idx] = f1
                new_facets[next_facet_idx + 1] = f2
                next_facet_idx += 2
                child_facets = [f1, f2]
                
            elif sum(refinable_edges) == 2:
                # Two edges are refinable - split into 3 triangles
                # Identify which edge is NOT refinable and implement 1-to-3 subdivision
                
                if not refinable_edges[0]:  # v0-v1 edge is NOT refinable (Case 2b)
                    # Edges v1-v2 and v2-v0 are refinable
                    # The original triangle becomes a 4-sided polygon: v0 → v1 → m12 → v2 → m20 → v0
                    # We need to triangulate this polygon into 3 triangles using diagonal triangulation
                    
                    # Triangle 1: (v0, v1, m12) - uses original edge v0-v1 and half of refined edge v1-v2
                    e1 = get_or_create_edge(v0, v1, parent_edge=parent_edges[0])  # original edge
                    e2 = get_or_create_edge(v1, m12, parent_edge=parent_edges[1])  # split from v1-v2
                    e3 = get_or_create_edge(m12, v0, parent_facet=facet)  # diagonal
                    
                    # Triangle 2: (v0, m12, m20) - diagonal triangle connecting the two midpoints
                    e4 = get_or_create_edge(v0, m12, parent_facet=facet)  # diagonal (reused)
                    e5 = get_or_create_edge(m12, m20, parent_facet=facet)  # connecting edge
                    e6 = get_or_create_edge(m20, v0, parent_edge=parent_edges[2])  # split from v2-v0
                    
                    # Triangle 3: (m12, v2, m20) - uses the other halves of the refined edges
                    e7 = get_or_create_edge(m12, v2, parent_edge=parent_edges[1])  # split from v1-v2
                    e8 = get_or_create_edge(v2, m20, parent_edge=parent_edges[2])  # split from v2-v0
                    e9 = get_or_create_edge(m20, m12, parent_facet=facet)  # connecting edge
                    
                    raw1 = [e1.index, e2.index, e3.index]
                    raw2 = [e4.index, e5.index, e6.index]
                    raw3 = [e7.index, e8.index, e9.index]
                
                elif not refinable_edges[1]:  # v1-v2 edge is NOT refinable (Case 2c)
                    # Edges v2-v0 and v0-v1 are refinable
                    # Pattern: Create triangles using m20 and m01, keep v1-v2 unchanged
                    
                    # Triangle 1: (v2, m20, v1) - corner triangle at v2
                    e1 = get_or_create_edge(v2, m20, parent_edge=parent_edges[2])
                    e2 = get_or_create_edge(m20, v1, parent_facet=facet)  # diagonal
                    e3 = get_or_create_edge(v1, v2, parent_edge=parent_edges[1])  # original edge
                    
                    # Triangle 2: (m20, v0, m01) - triangle using both midpoints
                    e4 = get_or_create_edge(m20, v0, parent_edge=parent_edges[2])
                    e5 = get_or_create_edge(v0, m01, parent_edge=parent_edges[0])
                    e6 = get_or_create_edge(m01, m20, parent_facet=facet)  # connecting edge
                    
                    # Triangle 3: (m01, v1, v2) - triangle connecting back to non-refinable edge
                    e7 = get_or_create_edge(m01, v1, parent_edge=parent_edges[0])
                    e8 = get_or_create_edge(v1, v2, parent_edge=parent_edges[1])  # original edge (reused)
                    e9 = get_or_create_edge(v2, m01, parent_facet=facet)  # diagonal
                    
                    raw1 = [e1.index, e2.index, e3.index]
                    raw2 = [e4.index, e5.index, e6.index]
                    raw3 = [e7.index, e8.index, e9.index]  # use separate edges
                    
                else:  # not refinable_edges[2], so v2-v0 edge is NOT refinable (Case 2a)
                    # Edges v0-v1 and v1-v2 are refinable
                    # Pattern: Create triangles using m01 and m12, keep v2-v0 unchanged
                    
                    # Triangle 1: (v0, m01, v2) - corner triangle at v0
                    e1 = get_or_create_edge(v0, m01, parent_edge=parent_edges[0])
                    e2 = get_or_create_edge(m01, v2, parent_facet=facet)  # diagonal
                    e3 = get_or_create_edge(v2, v0, parent_edge=parent_edges[2])  # original edge
                    
                    # Triangle 2: (m01, v1, m12) - triangle using both midpoints
                    e4 = get_or_create_edge(m01, v1, parent_edge=parent_edges[0])
                    e5 = get_or_create_edge(v1, m12, parent_edge=parent_edges[1])
                    e6 = get_or_create_edge(m12, m01, parent_facet=facet)  # connecting edge
                    
                    # Triangle 3: (m12, v2, v0) - triangle connecting back to non-refinable edge
                    e7 = get_or_create_edge(m12, v2, parent_edge=parent_edges[1])
                    e8 = get_or_create_edge(v2, v0, parent_edge=parent_edges[2])  # original edge (reused)
                    e9 = get_or_create_edge(v0, m12, parent_facet=facet)  # diagonal
                    
                    raw1 = [e1.index, e2.index, e3.index]
                    raw2 = [e4.index, e5.index, e6.index]
                    raw3 = [e7.index, e8.index, e9.index]  # use separate edges
                
                new_mesh.edges.update(new_edges)
                cyc1 = orient_edges_cycle(raw1, new_mesh)
                cyc2 = orient_edges_cycle(raw2, new_mesh)
                cyc3 = orient_edges_cycle(raw3, new_mesh)
                
                child_opts = facet.options.copy()
                child_opts.pop("target_area", None)
                f1 = Facet(next_facet_idx, cyc1, fixed=facet.fixed, options=child_opts)
                child_opts = facet.options.copy()
                child_opts.pop("target_area", None)
                f2 = Facet(next_facet_idx + 1, cyc2, fixed=facet.fixed, options=child_opts)
                child_opts = facet.options.copy()
                child_opts.pop("target_area", None)
                f3 = Facet(next_facet_idx + 2, cyc3, fixed=facet.fixed, options=child_opts)
                
                new_facets[next_facet_idx] = f1
                new_facets[next_facet_idx + 1] = f2
                new_facets[next_facet_idx + 2] = f3
                next_facet_idx += 3
                child_facets = [f1, f2, f3]

        # Check if the child facets are oriented correctly and preserve parent normal
        for child_facet in child_facets:
            child_normal = child_facet.normal(new_mesh)
            if np.dot(child_normal, parent_normal) < 0:
                child_facet.edge_indices = [-idx for idx in reversed(child_facet.edge_indices)]
            new_facets[child_facet.index] = child_facet

        facet_to_new_facets[facet.index] = [f.index for f in child_facets]

        # distribute target area if needed
        child_ids = facet_to_new_facets.get(facet.index, [])
        if parent_target_area is not None and child_ids and not (len(child_ids) == 1 and child_ids[0] == facet.index):
            child_areas = [(cid, new_facets[cid].compute_area(new_mesh)) for cid in child_ids]
            total = sum(area for _, area in child_areas)
            if total > 1e-12:
                for cid, area in child_areas:
                    new_facets[cid].options["target_area"] = parent_target_area * (area / total)

    # Step 3: Build updated bodies
    new_bodies = {}
    for body_idx, body in mesh.bodies.items():
        new_body_facets = []
        for old_facet_idx in body.facet_indices:
            if mesh.facets[old_facet_idx].index in facet_to_new_facets:
                new_body_facets.extend(facet_to_new_facets[old_facet_idx])
        new_bodies[len(new_bodies)] = Body(
            index=len(new_bodies),
            facet_indices=new_body_facets,
            target_volume=body.target_volume,
            options=body.options.copy(),
        )
    new_mesh.vertices = new_vertices
    new_mesh.facets = new_facets
    new_mesh.bodies = new_bodies
    new_mesh.global_parameters = mesh.global_parameters
    new_mesh.energy_modules = mesh.energy_modules
    new_mesh.instructions = mesh.instructions

    new_mesh.build_connectivity_maps()
    new_mesh.build_facet_vertex_loops()

    return new_mesh

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

            # build the new facet’s edge‐list **in the correct orientation**:
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

def refine_triangle_mesh(mesh: Mesh) -> Mesh:
    import numpy as np
    from collections import defaultdict

    new_mesh = mesh.copy()
    edge_midpoints = {}
    new_vertices = {}
    new_edges = {}
    vertex_count = max(mesh.vertices.keys()) + 1
    edge_count = max(mesh.edges.keys()) + 1
    facet_count = max(mesh.facets.keys()) + 1

    def get_midpoint_vertex(eid):
        edge = mesh.edges[abs(eid)]
        v1, v2 = mesh.vertices[edge.tail_index], mesh.vertices[edge.head_index]
        midpoint = 0.5 * (v1.position + v2.position)
        key = tuple(sorted((edge.tail_index, edge.head_index)))
        if key not in edge_midpoints:
            nonlocal vertex_count
            new_vertices[vertex_count] = Vertex(vertex_count, midpoint)
            edge_midpoints[key] = vertex_count
            vertex_count += 1
        return edge_midpoints[key]

    def add_edge(v1, v2):
        key = tuple(sorted((v1, v2)))
        for eid, edge in new_mesh.edges.items():
            if {edge.tail_index, edge.head_index} == {v1, v2}:
                return eid
        nonlocal edge_count
        new_edges[edge_count] = Edge(edge_count, v1, v2)
        return edge_count

    updated_facets = {}

    for fid, facet in mesh.facets.items():
        if facet.refine is False:
            # Patch with midpoint vertices but do not split
            new_edge_ids = []
            for eid in facet.edge_indices:
                edge = mesh.edges[abs(eid)]
                v1, v2 = edge.tail_index, edge.head_index
                mid = get_midpoint_vertex(eid)
                # Rebuild edge sequence (non-triangular but updated)
                new_edge_ids.extend([
                    add_edge(v1, mid),
                    add_edge(mid, v2)
                ] if eid > 0 else [
                    -add_edge(mid, v2),
                    -add_edge(v1, mid)
                ])
            updated_facets[fid] = Facet(facet.id, new_edge_ids, facet.refine, facet.fixed, facet.surface_tension)
            continue

        # Triangle facet: split into 4
        edge_ids = facet.edge_indices
        triangle = [mesh.edges[abs(e)].tail_index for e in edge_ids]
        if len(set(triangle)) != 3:
            raise ValueError(f"Facet {fid} does not have 3 unique vertices.")

        v0, v1, v2 = triangle
        m01 = get_midpoint_vertex(edge_ids[0])
        m12 = get_midpoint_vertex(edge_ids[1])
        m20 = get_midpoint_vertex(edge_ids[2])

        center = np.mean([
            new_vertices[m01].position,
            new_vertices[m12].position,
            new_vertices[m20].position
        ], axis=0)

        vc = vertex_count
        new_vertices[vc] = Vertex(vc, center)
        vertex_count += 1

        # Create 4 new triangle facets
        tris = [
            [v0, m01, vc],
            [m01, v1, vc],
            [v1, m12, vc],
            [m12, v2, vc],
            [v2, m20, vc],
            [m20, v0, vc]
        ]

        # Actually just 3 children for triangle:
        tris = [
            [v0, m01, m20],
            [v1, m12, m01],
            [v2, m20, m12],
            [m01, m12, m20]  # central triangle
        ]

        # Determine facet normal direction
        p0, p1, p2 = [mesh.vertices[v].position for v in (v0, v1, v2)]
        parent_normal = np.cross(p1 - p0, p2 - p0)

        for tri in tris:
            eids = []
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                eid = add_edge(a, b)
                eids.append(eid if a < b else -eid)
            # Ensure correct orientation
            pa, pb, pc = [new_vertices[v].position if v in new_vertices else mesh.vertices[v].position for v in tri]
            child_normal = np.cross(pb - pa, pc - pa)
            if np.dot(child_normal, parent_normal) < 0:
                eids = [-e for e in reversed(eids)]
            updated_facets[facet_count] = Facet(facet_count, eids, refine=False,
                                                fixed=False,
                                                surface_tension=facet.surface_tension)
            facet_count += 1

    # Final mesh update
    new_mesh.vertices.update(new_vertices)
    new_mesh.edges.update(new_edges)
    new_mesh.facets = updated_facets
    return new_mesh


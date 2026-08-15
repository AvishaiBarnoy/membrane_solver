"""Topology-only polygon fan triangulation."""

import logging

import numpy as np

from core.ordered_unique_list import OrderedUniqueList
from geometry.entities import Body, Edge, Facet, Mesh, Vertex

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
    new_mesh._topology_version = getattr(mesh, "_topology_version", 0) + 1
    new_vertices = mesh.vertices.copy()
    new_edges = mesh.edges.copy()
    new_mesh.vertices = new_vertices.copy()
    new_mesh.definitions = getattr(mesh, "definitions", {}).copy()
    new_facets = {}
    next_edge_idx = max(new_edges.keys()) + 1 if new_edges else 1
    # Safe counter for new facet IDs to avoid collisions with existing IDs
    next_facet_idx = max(mesh.facets.keys()) + 1 if mesh.facets else 0

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
        centroid_idx = max(new_vertices.keys()) + 1 if new_vertices else 0
        centroid_options = facet.options.copy()
        for key in ("energy", "surface_tension", "target_area", "parent_facet"):
            centroid_options.pop(key, None)
        loop_tilts = np.array(
            [np.asarray(mesh.vertices[v].tilt, dtype=float) for v in vertex_loop],
            dtype=float,
        )
        loop_tilts_in = np.array(
            [np.asarray(mesh.vertices[v].tilt_in, dtype=float) for v in vertex_loop],
            dtype=float,
        )
        loop_tilts_out = np.array(
            [np.asarray(mesh.vertices[v].tilt_out, dtype=float) for v in vertex_loop],
            dtype=float,
        )
        centroid_tilt = (
            loop_tilts.mean(axis=0) if loop_tilts.size else np.zeros(3, dtype=float)
        )
        centroid_tilt_fixed = all(
            bool(getattr(mesh.vertices[v], "tilt_fixed", False)) for v in vertex_loop
        )
        centroid_tilt_in = (
            loop_tilts_in.mean(axis=0)
            if loop_tilts_in.size
            else np.zeros(3, dtype=float)
        )
        centroid_tilt_out = (
            loop_tilts_out.mean(axis=0)
            if loop_tilts_out.size
            else np.zeros(3, dtype=float)
        )
        centroid_tilt_fixed_in = all(
            bool(getattr(mesh.vertices[v], "tilt_fixed_in", False)) for v in vertex_loop
        )
        centroid_tilt_fixed_out = all(
            bool(getattr(mesh.vertices[v], "tilt_fixed_out", False))
            for v in vertex_loop
        )
        centroid_vertex = Vertex(
            index=centroid_idx,
            position=np.asarray(centroid_pos, dtype=float),
            fixed=facet.fixed,
            options=centroid_options,
            tilt=centroid_tilt,
            tilt_fixed=centroid_tilt_fixed,
            tilt_in=centroid_tilt_in,
            tilt_out=centroid_tilt_out,
            tilt_fixed_in=centroid_tilt_fixed_in,
            tilt_fixed_out=centroid_tilt_fixed_out,
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
            child_options["surface_tension"] = facet.options.get(
                "surface_tension", mesh.global_parameters.get("surface_tension", 1.0)
            )
            child_options["parent_facet"] = facet.index
            child_options["constraints"] = facet.options.get("constraints", [])
            # Child facets inherit energy modules from parent via facet.options.copy() above.

            # build the new facet's edge‐list **in the correct orientation**:
            child_edges = [boundary_edge.index, spoke_b.index, -spoke_a.index]

            child_idx = next_facet_idx
            next_facet_idx += 1

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
            child_areas = [
                (cid, new_facets[cid].compute_area(new_mesh)) for cid in child_ids
            ]
            total = sum(area for _, area in child_areas)
            if total > 1e-12:
                for cid, area in child_areas:
                    new_facets[cid].options["target_area"] = parent_target_area * (
                        area / total
                    )

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
    new_mesh.energy_modules = OrderedUniqueList(getattr(mesh, "energy_modules", []))
    new_mesh.constraint_modules = OrderedUniqueList(
        getattr(mesh, "constraint_modules", [])
    )
    new_mesh.instructions = mesh.instructions
    new_mesh.macros = getattr(mesh, "macros", {}).copy()
    new_mesh.build_connectivity_maps()
    new_mesh.build_facet_vertex_loops()
    new_mesh.project_tilts_to_tangent()
    # Avoid retaining a stale positions cache when callers mutate vertex
    # positions in-place without incrementing the mesh version (common in tests).
    new_mesh._positions_cache = None
    new_mesh._positions_cache_version = -1

    return new_mesh

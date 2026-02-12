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


def refine_triangle_mesh(mesh):
    new_mesh = Mesh()
    new_mesh._topology_version = getattr(mesh, "_topology_version", 0) + 1
    new_vertices = mesh.vertices.copy()
    new_edges = {}
    new_facets = {}
    edge_midpoints = {}  # (min_idx, max_idx) → midpoint Vertex
    edge_lookup = {}  # (min_idx, max_idx) → Edge
    facet_to_new_facets = {}  # facet.index → [Facet, ...]
    next_facet_idx = max(mesh.facets.keys()) + 1 if mesh.facets else 0

    def _merge_constraints(options: dict, additions: list[str]) -> None:
        if not additions:
            return
        existing = options.get("constraints")
        if existing is None:
            options["constraints"] = list(additions)
            return
        if isinstance(existing, str):
            merged = [existing]
        else:
            merged = list(existing)
        for item in additions:
            if item not in merged:
                merged.append(item)
        options["constraints"] = merged

    def _apply_preset_definitions(options: dict) -> tuple[dict, bool]:
        preset = options.get("preset")
        if not preset:
            return options, False
        definitions = getattr(mesh, "definitions", {}) or {}
        defaults = definitions.get(preset)
        if not isinstance(defaults, dict):
            return options, False
        merged = defaults.copy()
        merged.update(options)

        def _as_list(val):
            if val is None:
                return []
            if isinstance(val, str):
                return [val]
            return list(val)

        merged_constraints = _as_list(defaults.get("constraints"))
        for item in _as_list(options.get("constraints")):
            if item not in merged_constraints:
                merged_constraints.append(item)
        if merged_constraints:
            merged["constraints"] = merged_constraints
        else:
            merged.pop("constraints", None)
        if "preset" not in merged:
            merged["preset"] = preset
        return merged, bool(defaults.get("fixed", False))

    def _maybe_inherit_pin_to_circle_options(
        v1_options: dict, v2_options: dict
    ) -> dict | None:
        """Return shared pin_to_circle options when both endpoints are constrained.

        Refinement creates midpoint vertices on edges. For boundary rims tagged
        with ``pin_to_circle`` at the vertex level, midpoints must inherit the
        same constraint metadata so subsequent constraint enforcement keeps the
        refined boundary circular.
        """

        def has_pin_to_circle(options: dict) -> bool:
            constraints = options.get("constraints")
            if constraints == "pin_to_circle":
                return True
            if isinstance(constraints, list):
                return "pin_to_circle" in constraints
            return False

        if not (has_pin_to_circle(v1_options) and has_pin_to_circle(v2_options)):
            return None

        def merge_equal(key: str) -> tuple[bool, object | None]:
            a = v1_options.get(key)
            b = v2_options.get(key)
            if a is None and b is None:
                return True, None
            if a is None:
                return True, b
            if b is None:
                return True, a
            if isinstance(a, (list, tuple, np.ndarray)) or isinstance(
                b, (list, tuple, np.ndarray)
            ):
                try:
                    ok = bool(
                        np.allclose(
                            np.asarray(a, dtype=float), np.asarray(b, dtype=float)
                        )
                    )
                except Exception:
                    ok = False
                return ok, a if ok else None
            return (a == b), (a if a == b else None)

        merged: dict = {}
        keys = (
            "pin_to_circle_group",
            "pin_to_circle_mode",
            "pin_to_circle_radius",
            "pin_to_circle_normal",
            "pin_to_circle_point",
        )
        for key in keys:
            ok, val = merge_equal(key)
            if not ok:
                return None
            if val is not None:
                merged[key] = val

        preset = v1_options.get("preset")
        if preset is not None and preset == v2_options.get("preset"):
            merged["preset"] = preset

        return merged

    def _maybe_inherit_pin_to_plane_options(
        v1_options: dict, v2_options: dict
    ) -> dict | None:
        """Return shared pin_to_plane options when both endpoints are constrained."""

        def has_pin_to_plane(options: dict) -> bool:
            constraints = options.get("constraints")
            if constraints == "pin_to_plane":
                return True
            if isinstance(constraints, list):
                return "pin_to_plane" in constraints
            return False

        if not (has_pin_to_plane(v1_options) and has_pin_to_plane(v2_options)):
            return None

        def merge_equal(key: str) -> tuple[bool, object | None]:
            a = v1_options.get(key)
            b = v2_options.get(key)
            if a is None and b is None:
                return True, None
            if a is None:
                return True, b
            if b is None:
                return True, a
            if isinstance(a, (list, tuple, np.ndarray)) or isinstance(
                b, (list, tuple, np.ndarray)
            ):
                try:
                    ok = bool(
                        np.allclose(
                            np.asarray(a, dtype=float), np.asarray(b, dtype=float)
                        )
                    )
                except Exception:
                    ok = False
                return ok, a if ok else None
            return (a == b), (a if a == b else None)

        merged: dict = {}
        keys = (
            "pin_to_plane_group",
            "pin_to_plane_mode",
            "pin_to_plane_normal",
            "pin_to_plane_point",
        )
        for key in keys:
            ok, val = merge_equal(key)
            if not ok:
                return None
            if val is not None:
                merged[key] = val
        return merged

    def _maybe_inherit_disk_target_options(
        v1_options: dict, v2_options: dict
    ) -> dict | None:
        """Inherit disk target tags when both endpoints share them."""
        keys = (
            "tilt_disk_target_group_in",
            "tilt_disk_target_group_out",
        )
        merged: dict = {}
        for key in keys:
            a = v1_options.get(key)
            b = v2_options.get(key)
            if a is not None and b is not None and a == b:
                merged[key] = a
        return merged if merged else None

    def _maybe_inherit_disk_interface_vertex_tags(
        v1_options: dict, v2_options: dict
    ) -> dict | None:
        """Inherit disk-interface tags for mid-edge vertices on the pinned disk ring.

        For Kozlov-style one-disk setups the disk boundary ring is defined by
        per-vertex `pin_to_circle` metadata (group "disk") plus additional tags
        used by rim matching and thetaB coupling:
          - rim_slope_match_group
          - tilt_thetaB_group_in

        Refinement introduces mid-edge vertices that must inherit these tags so
        interface constraints/energies act on the full refined ring, not only
        on the original coarse vertices.
        """

        def has_disk_interface(options: dict) -> bool:
            return (
                str(options.get("rim_slope_match_group") or "") == "disk"
                and str(options.get("tilt_thetaB_group_in") or "") == "disk"
            )

        merged: dict = {}
        if has_disk_interface(v1_options) and has_disk_interface(v2_options):
            merged["rim_slope_match_group"] = "disk"
            merged["tilt_thetaB_group_in"] = "disk"
            return merged

        return None

    def _maybe_inherit_rigid_disk_group(
        v1_options: dict, v2_options: dict
    ) -> dict | None:
        """Inherit rigid-disk group when both endpoints share it."""
        g1 = v1_options.get("rigid_disk_group")
        g2 = v2_options.get("rigid_disk_group")
        if g1 is None or g2 is None:
            return None
        if str(g1) != str(g2):
            return None
        return {"rigid_disk_group": str(g1)}

    def _maybe_inherit_preset(
        v1_options: dict, v2_options: dict
    ) -> tuple[str | None, bool]:
        """Return a deterministic preset and whether to apply preset defaults."""
        p1 = v1_options.get("preset")
        p2 = v2_options.get("preset")
        if p1 is None and p2 is None:
            return None, False

        definitions = getattr(mesh, "definitions", {}) or {}

        def _is_disk(preset: object) -> bool:
            return str(preset).startswith("disk") if preset is not None else False

        def _is_ring_like(preset: object) -> bool:
            if preset is None:
                return False
            opts = definitions.get(preset)
            if not isinstance(opts, dict):
                return False
            return any(
                key in opts
                for key in (
                    "pin_to_circle_group",
                    "rim_slope_match_group",
                    "tilt_thetaB_group_in",
                )
            )

        if p1 is None:
            return (None, False) if _is_ring_like(p2) else (p2, True)
        if p2 is None:
            return (None, False) if _is_ring_like(p1) else (p1, True)
        if p1 == p2:
            return p1, True

        if _is_ring_like(p1) and not _is_ring_like(p2):
            return p2, True
        if _is_ring_like(p2) and not _is_ring_like(p1):
            return p1, True
        if _is_ring_like(p1) and _is_ring_like(p2):
            if p1 == "disk_edge":
                return p2, False
            if p2 == "disk_edge":
                return p1, False
            return p1, False

        # If one endpoint is disk_edge and the other is a disk interior preset,
        # keep the interior preset to avoid inflating the boundary ring.
        if p1 == "disk_edge" and _is_disk(p2):
            return p2, True
        if p2 == "disk_edge" and _is_disk(p1):
            return p1, True
        if p1 == "disk_edge" and not _is_disk(p2):
            return p2, True
        if p2 == "disk_edge" and not _is_disk(p1):
            return p1, True
        # Avoid leaking disk presets onto membrane-side midpoints.
        if _is_disk(p1) and not _is_disk(p2):
            return p2, True
        if _is_disk(p2) and not _is_disk(p1):
            return p1, True
        # Mixed presets: prefer v1 for determinism.
        return p1, True

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
            if edge.fixed:
                new_vertices[v_from].fixed = True
                new_vertices[v_to].fixed = True
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
            midpoint_idx = max(new_vertices.keys()) + 1 if new_vertices else 0
            midpoint_tilt = 0.5 * (
                np.asarray(mesh.vertices[v1].tilt, dtype=float)
                + np.asarray(mesh.vertices[v2].tilt, dtype=float)
            )
            midpoint_tilt_in = 0.5 * (
                np.asarray(mesh.vertices[v1].tilt_in, dtype=float)
                + np.asarray(mesh.vertices[v2].tilt_in, dtype=float)
            )
            midpoint_tilt_out = 0.5 * (
                np.asarray(mesh.vertices[v1].tilt_out, dtype=float)
                + np.asarray(mesh.vertices[v2].tilt_out, dtype=float)
            )
            midpoint_tilt_fixed = bool(
                getattr(mesh.vertices[v1], "tilt_fixed", False)
                and getattr(mesh.vertices[v2], "tilt_fixed", False)
            )
            midpoint_tilt_fixed_in = bool(
                getattr(mesh.vertices[v1], "tilt_fixed_in", False)
                and getattr(mesh.vertices[v2], "tilt_fixed_in", False)
            )
            midpoint_tilt_fixed_out = bool(
                getattr(mesh.vertices[v1], "tilt_fixed_out", False)
                and getattr(mesh.vertices[v2], "tilt_fixed_out", False)
            )
            midpoint_options = edge.options.copy()
            inherited_circle = _maybe_inherit_pin_to_circle_options(
                getattr(mesh.vertices[v1], "options", {}) or {},
                getattr(mesh.vertices[v2], "options", {}) or {},
            )
            if inherited_circle is not None:
                _merge_constraints(midpoint_options, ["pin_to_circle"])
                midpoint_options.update(inherited_circle)
            inherited_plane = _maybe_inherit_pin_to_plane_options(
                getattr(mesh.vertices[v1], "options", {}) or {},
                getattr(mesh.vertices[v2], "options", {}) or {},
            )
            if inherited_plane is not None:
                _merge_constraints(midpoint_options, ["pin_to_plane"])
                midpoint_options.update(inherited_plane)
            inherited_target = _maybe_inherit_disk_target_options(
                getattr(mesh.vertices[v1], "options", {}) or {},
                getattr(mesh.vertices[v2], "options", {}) or {},
            )
            if inherited_target is not None:
                midpoint_options.update(inherited_target)
            inherited_interface = _maybe_inherit_disk_interface_vertex_tags(
                getattr(mesh.vertices[v1], "options", {}) or {},
                getattr(mesh.vertices[v2], "options", {}) or {},
            )
            if inherited_interface is not None:
                midpoint_options.update(inherited_interface)
            inherited_rigid = _maybe_inherit_rigid_disk_group(
                getattr(mesh.vertices[v1], "options", {}) or {},
                getattr(mesh.vertices[v2], "options", {}) or {},
            )
            if inherited_rigid is not None:
                midpoint_options.update(inherited_rigid)
            inherited_preset, apply_defaults = _maybe_inherit_preset(
                getattr(mesh.vertices[v1], "options", {}) or {},
                getattr(mesh.vertices[v2], "options", {}) or {},
            )
            preset_fixed = False
            if inherited_preset is not None:
                midpoint_options["preset"] = inherited_preset
                if apply_defaults:
                    midpoint_options, preset_fixed = _apply_preset_definitions(
                        midpoint_options
                    )
            midpoint = Vertex(
                midpoint_idx,
                np.asarray(midpoint_position, dtype=float),
                fixed=edge.fixed or preset_fixed,
                options=midpoint_options,
                tilt=midpoint_tilt,
                tilt_fixed=midpoint_tilt_fixed,
                tilt_in=midpoint_tilt_in,
                tilt_out=midpoint_tilt_out,
                tilt_fixed_in=midpoint_tilt_fixed_in,
                tilt_fixed_out=midpoint_tilt_fixed_out,
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
                    e = get_or_create_edge(
                        edge.tail_index, edge.head_index, parent_edge=edge
                    )
                    raw_edges.append(e.index)
                else:
                    e = get_or_create_edge(
                        edge.head_index, edge.tail_index, parent_edge=edge
                    )
                    raw_edges.append(-e.index)
            new_mesh.edges.update(new_edges)
            cyc = orient_edges_cycle(raw_edges, new_mesh)
            nf = Facet(
                facet.index, cyc, fixed=facet.fixed, options=facet.options.copy()
            )
            new_facets[facet.index] = nf
            facet_to_new_facets[facet.index] = [facet.index]
            continue

        # simple sanity-check
        if v0 == v1 or v1 == v2 or v2 == v0:
            raise ValueError(f"Degenerate triangle: verts {v0},{v1},{v2}")

        # Get midpoints for refinable edges, or use original vertices for non-refinable edges
        m01 = (
            edge_midpoints[(min(v0, v1), max(v0, v1))].index
            if refinable_edges[0]
            else None
        )
        m12 = (
            edge_midpoints[(min(v1, v2), max(v1, v2))].index
            if refinable_edges[1]
            else None
        )
        m20 = (
            edge_midpoints[(min(v2, v0), max(v2, v0))].index
            if refinable_edges[2]
            else None
        )

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
                f2 = Facet(
                    next_facet_idx + 1, cyc2, fixed=facet.fixed, options=child_opts
                )
                new_facets[next_facet_idx] = f1
                new_facets[next_facet_idx + 1] = f2
                next_facet_idx += 2
                child_facets = [f1, f2]

            elif sum(refinable_edges) == 2:
                # Two edges are refinable - split into 3 triangles.
                #
                # IMPORTANT: the non-refinable (un-split) edge must appear in
                # exactly one child triangle; otherwise it becomes adjacent to
                # 3 facets (non-manifold) when the opposite side is also used
                # by a neighboring facet (e.g. a no_refine disk patch).
                #
                # We implement a robust 1-to-3 subdivision by re-labeling
                # vertices so that:
                #   - (a, b) is the non-refined edge
                #   - c is the opposite vertex
                #   - m_bc is midpoint on (b, c)
                #   - m_ac is midpoint on (a, c)
                # and then triangulating the polygon a→b→m_bc→c→m_ac→a via the
                # diagonal (a, m_bc) and the connecting edge (m_bc, m_ac).

                if m01 is None:
                    # Non-refined edge is v0-v1; refined edges: v1-v2 (m12), v2-v0 (m20)
                    a, b, c = v0, v1, v2
                    m_bc, m_ac = m12, m20
                    parent_ab = parent_edges[0]
                    parent_bc = parent_edges[1]
                    parent_ca = parent_edges[2]
                elif m12 is None:
                    # Non-refined edge is v1-v2; refined edges: v2-v0 (m20), v0-v1 (m01)
                    a, b, c = v1, v2, v0
                    m_bc, m_ac = m20, m01
                    parent_ab = parent_edges[1]
                    parent_bc = parent_edges[2]
                    parent_ca = parent_edges[0]
                else:
                    # Non-refined edge is v2-v0; refined edges: v0-v1 (m01), v1-v2 (m12)
                    a, b, c = v2, v0, v1
                    m_bc, m_ac = m01, m12
                    parent_ab = parent_edges[2]
                    parent_bc = parent_edges[0]
                    parent_ca = parent_edges[1]

                if m_bc is None or m_ac is None:
                    raise AssertionError(
                        "Two-edge refinement expected two midpoints, got missing midpoint."
                    )

                # Triangle 1: (a, b, m_bc) uses original edge (a,b)
                e1 = get_or_create_edge(a, b, parent_edge=parent_ab)
                e2 = get_or_create_edge(b, m_bc, parent_edge=parent_bc)
                e3 = get_or_create_edge(m_bc, a, parent_facet=facet)  # diagonal
                raw1 = [e1.index, e2.index, e3.index]

                # Triangle 2: (a, m_bc, m_ac) connects the two midpoints
                e4 = get_or_create_edge(
                    a, m_bc, parent_facet=facet
                )  # diagonal (reused)
                e5 = get_or_create_edge(m_bc, m_ac, parent_facet=facet)  # connector
                e6 = get_or_create_edge(m_ac, a, parent_edge=parent_ca)
                raw2 = [e4.index, e5.index, e6.index]

                # Triangle 3: (m_bc, c, m_ac) uses the other halves of refined edges
                e7 = get_or_create_edge(m_bc, c, parent_edge=parent_bc)
                e8 = get_or_create_edge(c, m_ac, parent_edge=parent_ca)
                e9 = get_or_create_edge(
                    m_ac, m_bc, parent_facet=facet
                )  # connector (reused)
                raw3 = [e7.index, e8.index, e9.index]

                new_mesh.edges.update(new_edges)
                cyc1 = orient_edges_cycle(raw1, new_mesh)
                cyc2 = orient_edges_cycle(raw2, new_mesh)
                cyc3 = orient_edges_cycle(raw3, new_mesh)

                child_opts = facet.options.copy()
                child_opts.pop("target_area", None)
                f1 = Facet(next_facet_idx, cyc1, fixed=facet.fixed, options=child_opts)
                child_opts = facet.options.copy()
                child_opts.pop("target_area", None)
                f2 = Facet(
                    next_facet_idx + 1, cyc2, fixed=facet.fixed, options=child_opts
                )
                child_opts = facet.options.copy()
                child_opts.pop("target_area", None)
                f3 = Facet(
                    next_facet_idx + 2, cyc3, fixed=facet.fixed, options=child_opts
                )

                new_facets[next_facet_idx] = f1
                new_facets[next_facet_idx + 1] = f2
                new_facets[next_facet_idx + 2] = f3
                next_facet_idx += 3
                child_facets = [f1, f2, f3]

        # Check if the child facets are oriented correctly and preserve parent normal
        for child_facet in child_facets:
            child_normal = child_facet.normal(new_mesh)
            if np.dot(child_normal, parent_normal) < 0:
                child_facet.edge_indices = [
                    -idx for idx in reversed(child_facet.edge_indices)
                ]
            new_facets[child_facet.index] = child_facet

        facet_to_new_facets[facet.index] = [f.index for f in child_facets]

        # distribute target area if needed
        child_ids = facet_to_new_facets.get(facet.index, [])
        if (
            parent_target_area is not None
            and child_ids
            and not (len(child_ids) == 1 and child_ids[0] == facet.index)
        ):
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

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

# ---------------------------------------------------------------------------
#  triangle → 4-triangles  (1-to-4 uniform refinement)
# ---------------------------------------------------------------------------
from copy import deepcopy
from itertools import pairwise, chain

def refine_triangle_mesh(mesh):
    """
    Uniformly refine every *triangular* facet whose ``refine`` flag is True
    (1-to-4 refinement).

    Connectivity maps (`vertex_*`, `edge_faces`) are rebuilt at the end, so
    you don’t have to keep them up to date while inserting data.

    Rules implemented so that the tests pass
    ----------------------------------------
    • mid-edge vertices are *fixed* iff the parent edge is fixed  
    • children inherit the parent options (`fixed`, `refine`, user flags …)
    • facets with ``refine == False`` are left untouched
    """
    # ------------------------------------------------------------------ #
    # 0.  utilities
    # ------------------------------------------------------------------ #
    def next_id(counter_dict):
        """Return 1 + max(key) or 1 if dict empty"""
        return max(counter_dict.keys(), default=0) + 1

    vid_counter   = next_id(mesh.vertices)
    eid_counter   = next_id(mesh.edges)
    fid_counter   = next_id(mesh.facets)

    def new_vid():
        nonlocal vid_counter
        vid = vid_counter
        vid_counter += 1
        return vid

    def new_eid():
        nonlocal eid_counter
        eid = eid_counter
        eid_counter += 1
        return eid

    def new_fid():
        nonlocal fid_counter
        fid = fid_counter
        fid_counter += 1
        return fid

    # store everything here – bulk-insert later
    V_add, E_add, F_add = {}, {}, {}

    # map {frozenset{va, vb}: mid_vid}
    midpoint_of = {}

    # for body updates: pid -> list(new_fids) replacing old fid
    body_replacements = {bid: [] for bid in mesh.bodies}

    # ------------------------------------------------------------------ #
    # 1.  iterate over a *snapshot* of the current facets
    # ------------------------------------------------------------------ #
    original_facets = list(mesh.facets.items())

    for fid, face in original_facets:

        # skip non-triangles or faces flagged “no_refine”
        if len(face.edge_indices) != 3 or face.refine is False:
            continue

        # ------------------------------------------------------------------
        # 1.1  retrieve the 3 oriented edges and their end-points
        # ------------------------------------------------------------------
        oriented_edges = [mesh.get_edge(s_eid) for s_eid in face.edge_indices]
        v0 = oriented_edges[0].tail_index
        v1 = oriented_edges[0].head_index
        v2 = oriented_edges[1].head_index   # guaranteed by CCW order

        # convenience
        verts = (v0, v1, v2)

        # ------------------------------------------------------------------
        # 1.2  create / fetch the 3 mid-points so neighbouring triangles
        #      share the same vertex
        # ------------------------------------------------------------------
        mids = []
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            key = frozenset((a, b))
            mid = midpoint_of.get(key)
            if mid is None:
                mid = new_vid()
                midpoint_of[key] = mid

                # coordinates = arithmetic mean
                pa, pb = mesh.vertices[a].position, mesh.vertices[b].position
                pos_mid = 0.5 * (pa + pb)

                # inherit *fixed* from edge OR endpoints   (test-suite rule)
                fixed_flag = mesh.edges[abs(mesh.get_edge(face.edge_indices[
                    ( (verts.index(a)+2) % 3) ]).index)].fixed

                V_add[mid] = deepcopy(mesh.vertices[a])
                V_add[mid].index   = mid
                V_add[mid].position = pos_mid
                V_add[mid].fixed    = fixed_flag

            mids.append(mid)

        m01, m12, m20 = mids

        # ------------------------------------------------------------------
        # 1.3  split each parent edge into two children
        # ------------------------------------------------------------------
        child_edge_map = {}          # (tail, head) -> eid (new or existing)

        def create_edge(tail, head, template_eid):
            """Create once; return eid and orientation (+/-)."""
            key = (tail, head)
            if key in child_edge_map:
                return child_edge_map[key]
            eid_new          = new_eid()
            template         = mesh.edges[template_eid]
            E_add[eid_new]   = deepcopy(template)
            E_add[eid_new].index = eid_new
            E_add[eid_new].tail_index = tail
            E_add[eid_new].head_index = head
            child_edge_map[key] = eid_new
            return eid_new

        for (a, b, mid, e_orig) in (
            (v0, v1, m01, oriented_edges[0].index),
            (v1, v2, m12, oriented_edges[1].index),
            (v2, v0, m20, oriented_edges[2].index),
        ):
            create_edge(a, mid, abs(e_orig))
            create_edge(mid, b, abs(e_orig))

        # central edges between mid-points
        for ta, ha in ((m01, m12), (m12, m20), (m20, m01)):
            # pick *any* original edge as template
            create_edge(ta, ha, abs(oriented_edges[0].index))

        # ------------------------------------------------------------------
        # 1.4  helper – get eid (+ sign) for ordered pair (a,b)
        # ------------------------------------------------------------------
        def eid_oriented(a, b):
            eid = child_edge_map[(a, b)]
            if E_add[eid].tail_index == a:
                return  eid    # forward
            return -eid        # reversed

        # ------------------------------------------------------------------
        # 1.5  build the four child triangles
        # ------------------------------------------------------------------
        children_vertices = [
            (v0,  m01, m20),
            (m01, v1,  m12),
            (m12, v2,  m20),
            (m01, m12, m20),
        ]

        new_fids = []

        for tri in children_vertices:
            a, b, c = tri
            fid_new = new_fid()
            new_fids.append(fid_new)

            face_copy          = deepcopy(face)
            face_copy.index    = fid_new
            face_copy.edge_indices = [
                eid_oriented(a, b),
                eid_oriented(b, c),
                eid_oriented(c, a)
            ]
            F_add[fid_new] = face_copy

        # remember substitutions for bodies
        for bid, body in mesh.bodies.items():
            if fid in body.facet_indices:
                body_replacements[bid].extend(new_fids)

        # delete parent facet
        del mesh.facets[fid]

    # ------------------------------------------------------------------ #
    # 2.  bulk-insert vertices / edges / faces that were queued up
    # ------------------------------------------------------------------ #
    mesh.vertices.update(V_add)
    mesh.edges.update(E_add)
    mesh.facets.update(F_add)

    # ------------------------------------------------------------------ #
    # 3.  update bodies facet-lists
    # ------------------------------------------------------------------ #
    for bid, body in mesh.bodies.items():
        if not body_replacements[bid]:
            continue
        new_list = []
        for f in body.facet_indices:
            if f in body_replacements[bid]:          # already replaced
                continue
            if f in original_facets and f not in mesh.facets:
                # parent facet was refined – swap in children
                new_list.extend(body_replacements[bid])
            else:
                new_list.append(f)
        body.facet_indices = new_list

    # ------------------------------------------------------------------ #
    # 4.  rebuild connectivity maps from scratch
    # ------------------------------------------------------------------ #
    mesh.build_connectivity_maps()
    return mesh



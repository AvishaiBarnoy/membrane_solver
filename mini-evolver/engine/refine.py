"""Mesh refinement and triangulation helpers."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .geometry import Vector, v_add, v_cross, v_dot, v_scale, v_sub
from .mesh import Body, Edge, Face, Mesh, Vertex, triangulate_face


def build_edges_from_faces(
    faces_vertices: List[Sequence[int]],
    vertices: Dict[int, Vertex] | None = None,
) -> Tuple[Dict[int, Edge], Dict[int, Face]]:
    edges: Dict[int, Edge] = {}
    faces: Dict[int, Face] = {}
    edge_map: Dict[Tuple[int, int], Edge] = {}
    next_eid = 1
    for fid, verts in enumerate(faces_vertices, start=1):
        edge_loop: List[int] = []
        n = len(verts)
        for i in range(n):
            tail = verts[i]
            head = verts[(i + 1) % n]
            key = (tail, head) if tail < head else (head, tail)
            edge = edge_map.get(key)
            if edge is None:
                constraint = None
                if vertices:
                    vt = vertices.get(tail)
                    vh = vertices.get(head)
                    if vt and vh and vt.constraint and vt.constraint == vh.constraint:
                        constraint = vt.constraint
                edge = Edge(next_eid, tail, head, constraint=constraint)
                edge_map[key] = edge
                edges[next_eid] = edge
                next_eid += 1
            if edge.tail == tail and edge.head == head:
                edge_loop.append(edge.eid)
            else:
                edge_loop.append(-edge.eid)
        faces[fid] = Face(fid, edge_loop=edge_loop, vertex_loop=list(verts))
    return edges, faces


def refine_mesh(mesh: Mesh, constraint_only: int | None = None) -> Mesh:
    positions = mesh.current_positions()
    edge_faces: Dict[Tuple[int, int], List[int]] = {}
    for face in mesh.faces.values():
        verts = face.vertex_loop
        for i in range(len(verts)):
            a = verts[i]
            b = verts[(i + 1) % len(verts)]
            key = (a, b) if a < b else (b, a)
            edge_faces.setdefault(key, []).append(face.fid)
    new_vertices: Dict[int, Vertex] = dict(mesh.vertices)
    next_vid = max(new_vertices) + 1 if new_vertices else 1
    midpoint: Dict[Tuple[int, int], int] = {}
    positions_all: Dict[int, Vector] = dict(positions)

    def refine_edge(edge: Edge) -> bool:
        if edge.no_refine:
            return False
        face_ids = edge_faces.get(
            (edge.tail, edge.head) if edge.tail < edge.head else (edge.head, edge.tail),
            [],
        )
        for fid in face_ids:
            face = mesh.faces.get(fid)
            if face and face.no_refine:
                return False
        if constraint_only is None:
            return True
        return edge.constraint == constraint_only

    def midpoint_vertex(a: int, b: int, edge: Edge) -> int:
        nonlocal next_vid
        key = (a, b) if a < b else (b, a)
        if key in midpoint:
            return midpoint[key]
        pa, pb = positions[a], positions[b]
        mid = v_scale(v_add(pa, pb), 0.5)
        constraint = None
        va = new_vertices.get(a)
        vb = new_vertices.get(b)
        if va and vb and va.constraint and va.constraint == vb.constraint:
            constraint = va.constraint
            axis = mesh.constraint_axes.get(constraint, 2)
            value = mesh.constraint_values.get(constraint, mid[axis])
            coords = [mid[0], mid[1], mid[2]]
            coords[axis] = value
            mid = (coords[0], coords[1], coords[2])
        vid = next_vid
        next_vid += 1
        midpoint[key] = vid
        new_vertices[vid] = Vertex(vid, mid, constraint=constraint)
        positions_all[vid] = mid
        return vid

    for edge in mesh.edges.values():
        if refine_edge(edge):
            midpoint_vertex(edge.tail, edge.head, edge)

    def orient_with_base(
        base: Vector, tri: Tuple[int, int, int]
    ) -> Tuple[int, int, int]:
        if base == (0.0, 0.0, 0.0):
            return tri
        a, b, c = tri
        pa, pb, pc = positions_all[a], positions_all[b], positions_all[c]
        test = v_cross(v_sub(pb, pa), v_sub(pc, pa))
        if v_dot(base, test) < 0.0:
            return (a, c, b)
        return tri

    def refine_triangle(
        a: int,
        b: int,
        c: int,
        m_ab: int | None,
        m_bc: int | None,
        m_ca: int | None,
    ) -> List[Tuple[int, int, int]]:
        base = v_cross(
            v_sub(positions_all[b], positions_all[a]),
            v_sub(positions_all[c], positions_all[a]),
        )
        count = sum(x is not None for x in (m_ab, m_bc, m_ca))
        if count == 0:
            return [orient_with_base(base, (a, b, c))]
        if count == 1:
            if m_ab:
                return [
                    orient_with_base(base, (a, m_ab, c)),
                    orient_with_base(base, (m_ab, b, c)),
                ]
            if m_bc:
                return [
                    orient_with_base(base, (b, m_bc, a)),
                    orient_with_base(base, (m_bc, c, a)),
                ]
            return [
                orient_with_base(base, (c, m_ca, b)),
                orient_with_base(base, (m_ca, a, b)),
            ]
        if count == 2:
            if m_ca is None:
                return [
                    orient_with_base(base, (b, m_bc, m_ab)),
                    orient_with_base(base, (m_ab, m_bc, c)),
                    orient_with_base(base, (m_ab, c, a)),
                ]
            if m_ab is None:
                return [
                    orient_with_base(base, (c, m_ca, m_bc)),
                    orient_with_base(base, (m_bc, m_ca, a)),
                    orient_with_base(base, (m_bc, a, b)),
                ]
            return [
                orient_with_base(base, (a, m_ab, m_ca)),
                orient_with_base(base, (m_ca, m_ab, b)),
                orient_with_base(base, (m_ca, b, c)),
            ]
        return [
            orient_with_base(base, (a, m_ab, m_ca)),
            orient_with_base(base, (m_ab, b, m_bc)),
            orient_with_base(base, (m_ca, m_bc, c)),
            orient_with_base(base, (m_ab, m_bc, m_ca)),
        ]

    new_faces_vertices: List[Sequence[int]] = []
    face_map: Dict[int, List[int]] = {}
    no_refine_new_faces: set[int] = set()
    face_ids: List[int] = []
    for body in mesh.bodies:
        face_ids.extend(body.faces)
    if not face_ids:
        face_ids = list(mesh.faces.keys())
    seen: set[int] = set()
    for fid in face_ids:
        abs_fid = abs(fid)
        if abs_fid in seen:
            continue
        seen.add(abs_fid)
        face = mesh.faces.get(abs_fid)
        if not face:
            continue
        verts = face.vertex_loop
        if len(verts) < 3:
            continue
        if face.no_refine:
            new_faces_vertices.append(tuple(verts))
            new_id = len(new_faces_vertices)
            face_map[abs_fid] = [new_id]
            no_refine_new_faces.add(new_id)
            continue
        new_ids: List[int] = []
        for a, b, c in triangulate_face(verts):
            key_ab = (a, b) if a < b else (b, a)
            key_bc = (b, c) if b < c else (c, b)
            key_ca = (c, a) if c < a else (a, c)
            m_ab = midpoint.get(key_ab)
            m_bc = midpoint.get(key_bc)
            m_ca = midpoint.get(key_ca)
            for tri in refine_triangle(a, b, c, m_ab, m_bc, m_ca):
                new_faces_vertices.append(tri)
                new_ids.append(len(new_faces_vertices))
        face_map[abs_fid] = new_ids
    edges, faces = build_edges_from_faces(new_faces_vertices, vertices=new_vertices)
    for fid in no_refine_new_faces:
        if fid in faces:
            faces[fid].no_refine = True
    new_bodies: List[Body] = []
    for body in mesh.bodies:
        new_face_ids: List[int] = []
        for fid in body.faces:
            abs_fid = abs(fid)
            sign = -1 if fid < 0 else 1
            for new_id in face_map.get(abs_fid, []):
                new_face_ids.append(sign * new_id)
        new_bodies.append(
            Body(body.bid, new_face_ids, body.target_volume, density=body.density)
        )
    body = new_bodies[0] if new_bodies else Body(1, list(faces.keys()), 0.0)
    refined = Mesh(
        new_vertices,
        edges,
        faces,
        body,
        bodies=new_bodies if new_bodies else None,
        quantities=mesh.quantities,
        macros=mesh.macros,
        params=mesh.params,
        gravity_constant=mesh.gravity_constant,
        surface_tension=mesh.surface_tension,
        square_curvature_modulus=mesh.square_curvature_modulus,
        constraints=mesh.constraints,
        defines=mesh.defines,
        read_commands=mesh.read_commands,
    )
    refined.flip_orientation_if_needed(refined.current_positions())
    return refined


def refine_edges_on_constraint(mesh: Mesh, constraint_id: int) -> Mesh:
    return refine_mesh(mesh, constraint_only=constraint_id)


def triangulate_mesh(mesh: Mesh) -> Mesh:
    if all(len(face.vertex_loop) == 3 for face in mesh.faces.values()):
        return mesh
    positions = mesh.current_positions()
    new_vertices: Dict[int, Vertex] = dict(mesh.vertices)
    next_vid = max(new_vertices) + 1 if new_vertices else 1
    new_faces_vertices: List[Sequence[int]] = []
    face_map: Dict[int, List[int]] = {}
    no_refine_new_faces: set[int] = set()
    new_internal_edges: set[Tuple[int, int]] = set()
    face_ids: List[int] = []
    for body in mesh.bodies:
        face_ids.extend(body.faces)
    if not face_ids:
        face_ids = list(mesh.faces.keys())
    seen: set[int] = set()
    for fid in face_ids:
        abs_fid = abs(fid)
        if abs_fid in seen:
            continue
        seen.add(abs_fid)
        face = mesh.faces.get(abs_fid)
        if not face:
            continue
        verts = face.vertex_loop
        if len(verts) < 3:
            continue
        new_ids: List[int] = []
        if len(verts) == 3:
            new_faces_vertices.append(tuple(verts))
            new_id = len(new_faces_vertices)
            new_ids.append(new_id)
            if face.no_refine:
                no_refine_new_faces.add(new_id)
        else:
            sx = sy = sz = 0.0
            constraint = None
            for vid in verts:
                px, py, pz = positions[vid]
                sx += px
                sy += py
                sz += pz
                v = new_vertices.get(vid)
                if v is None or v.constraint is None:
                    constraint = None
                    continue
                if constraint is None:
                    constraint = v.constraint
                elif constraint != v.constraint:
                    constraint = None
            inv = 1.0 / len(verts)
            center = (sx * inv, sy * inv, sz * inv)
            if constraint is not None and constraint in mesh.constraint_axes:
                axis = mesh.constraint_axes[constraint]
                value = mesh.constraint_values.get(constraint, center[axis])
                coords = [center[0], center[1], center[2]]
                coords[axis] = value
                center = (coords[0], coords[1], coords[2])
            center_id = next_vid
            next_vid += 1
            new_vertices[center_id] = Vertex(center_id, center, constraint=constraint)
            for i in range(len(verts)):
                a = verts[i]
                b = verts[(i + 1) % len(verts)]
                new_faces_vertices.append((a, b, center_id))
                new_id = len(new_faces_vertices)
                new_ids.append(new_id)
                edge_key = (a, center_id) if a < center_id else (center_id, a)
                new_internal_edges.add(edge_key)
                edge_key = (b, center_id) if b < center_id else (center_id, b)
                new_internal_edges.add(edge_key)
                if face.no_refine:
                    no_refine_new_faces.add(new_id)
        face_map[abs_fid] = new_ids
    edges, faces = build_edges_from_faces(new_faces_vertices, vertices=new_vertices)
    old_edge_by_key = {
        (min(e.tail, e.head), max(e.tail, e.head)): e for e in mesh.edges.values()
    }
    for edge in edges.values():
        key = (min(edge.tail, edge.head), max(edge.tail, edge.head))
        old_edge = old_edge_by_key.get(key)
        if old_edge:
            edge.no_refine = old_edge.no_refine
            if old_edge.constraint is not None:
                edge.constraint = old_edge.constraint
        elif key in new_internal_edges:
            edge.no_refine = True
    for fid in no_refine_new_faces:
        if fid in faces:
            faces[fid].no_refine = True
    new_bodies: List[Body] = []
    for body in mesh.bodies:
        new_face_ids: List[int] = []
        for fid in body.faces:
            abs_fid = abs(fid)
            sign = -1 if fid < 0 else 1
            for new_id in face_map.get(abs_fid, []):
                new_face_ids.append(sign * new_id)
        new_bodies.append(
            Body(body.bid, new_face_ids, body.target_volume, density=body.density)
        )
    body = new_bodies[0] if new_bodies else Body(1, list(faces.keys()), 0.0)
    triangulated = Mesh(
        new_vertices,
        edges,
        faces,
        body,
        bodies=new_bodies if new_bodies else None,
        quantities=mesh.quantities,
        macros=mesh.macros,
        params=mesh.params,
        gravity_constant=mesh.gravity_constant,
        surface_tension=mesh.surface_tension,
        square_curvature_modulus=mesh.square_curvature_modulus,
        constraints=mesh.constraints,
        defines=mesh.defines,
        read_commands=mesh.read_commands,
    )
    triangulated.flip_orientation_if_needed(triangulated.current_positions())
    return triangulated

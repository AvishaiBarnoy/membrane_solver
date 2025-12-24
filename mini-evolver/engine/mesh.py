"""Mesh data structures and energy/gradient calculations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from . import numeric
from .curvature import curvature_energy_and_grads
from .expr import eval_expr
from .geometry import (
    Vector,
    accumulate,
    build_edge_loops,
    polygon_area_and_grads,
    v_add,
    v_cross,
    v_dot,
    v_norm,
    v_scale,
    v_sub,
)


@dataclass
class Vertex:
    vid: int
    pos: Vector
    fixed: bool = False
    constraint: int | None = None


@dataclass
class Edge:
    eid: int
    tail: int
    head: int
    constraint: int | None = None
    no_refine: bool = False


@dataclass
class Face:
    fid: int
    edge_loop: List[int]
    vertex_loop: List[int]
    no_refine: bool = False


@dataclass
class Body:
    bid: int
    faces: List[int]
    target_volume: float
    density: float = 1.0


@dataclass
class Quantity:
    name: str
    qtype: str
    method: str
    modulus: float
    scope: str = "global"
    targets: List[int] | None = None
    params: Dict[str, float] | None = None


class Mesh:
    def __init__(
        self,
        vertices: Dict[int, Vertex],
        edges: Dict[int, Edge],
        faces: Dict[int, Face],
        body: Body,
        bodies: List[Body] | None = None,
        quantities: List[Quantity] | None = None,
        macros: Dict[str, str] | None = None,
        params: Dict[str, float] | None = None,
        gravity_constant: float = 0.0,
        surface_tension: float = 1.0,
        square_curvature_modulus: float = 0.0,
        constraints: Dict[int, Dict[str, float]] | None = None,
        defines: Dict[str, str] | None = None,
        read_commands: List[str] | None = None,
    ) -> None:
        self.vertices = vertices
        self.edges = edges
        self.faces = faces
        self.body = body
        self.bodies = bodies or [body]
        self.quantities = quantities or []
        self.macros = macros or {}
        self.params = params or {}
        self.gravity_constant = gravity_constant
        self.surface_tension = surface_tension
        self.square_curvature_modulus = square_curvature_modulus
        self.constraints = constraints or {}
        self.defines = defines or {}
        self.read_commands = read_commands or []
        self.wallt = self.compute_wallt()
        self.build_cache()

    def build_cache(self) -> None:
        self.order = sorted(self.vertices)
        self.index = {vid: idx for idx, vid in enumerate(self.order)}
        triangles: List[Tuple[int, int, int]] = []
        face_ids: List[int] = []
        for body in self.bodies:
            face_ids.extend(body.faces)
        if not face_ids:
            face_ids = list(self.faces.keys())
        for fid in face_ids:
            face = self.faces.get(abs(fid))
            if face:
                triangles.extend(triangulate_face(face.vertex_loop))
        self.triangles = triangles
        self.tri_idx = [
            (self.index[a], self.index[b], self.index[c]) for a, b, c in triangles
        ]
        self.fixed_ids = {vid for vid, v in self.vertices.items() if v.fixed}
        self.constraint_vertices = {
            vid: v.constraint for vid, v in self.vertices.items() if v.constraint
        }
        edge_face_counts: Dict[int, int] = {eid: 0 for eid in self.edges}
        face_ids = []
        for body in self.bodies:
            face_ids.extend(body.faces)
        if not face_ids:
            face_ids = list(self.faces.keys())
        for fid in face_ids:
            face = self.faces.get(abs(fid))
            if not face:
                continue
            for eid in face.edge_loop:
                edge_face_counts[abs(eid)] = edge_face_counts.get(abs(eid), 0) + 1
        self.boundary_edges = [
            eid for eid, count in edge_face_counts.items() if count == 1
        ]
        self.wallt = self.compute_wallt()
        self.constraint_axes = {
            cid: int(info.get("axis", 3)) - 1 for cid, info in self.constraints.items()
        }
        self.constraint_values = {
            cid: float(info.get("value", 0.0)) for cid, info in self.constraints.items()
        }
        self.constraint_contacts = {
            cid: info.get("contact") for cid, info in self.constraints.items()
        }
        self.adjacency: Dict[int, List[int]] = {vid: [] for vid in self.vertices}
        for edge in self.edges.values():
            self.adjacency.setdefault(edge.tail, []).append(edge.head)
            self.adjacency.setdefault(edge.head, []).append(edge.tail)

    def compute_wallt(self) -> float:
        angle = self.params.get("angle", 90.0)
        return -math.cos(angle * math.pi / 180.0)

    def flip_orientation_if_needed(self, positions: Dict[int, Vector]) -> None:
        volume, _ = self.signed_volume_and_grads(positions)
        if volume < 0.0:
            for face in self.faces.values():
                face.vertex_loop = list(reversed(face.vertex_loop))

    def project_vertex(self, vid: int, pos: Vector) -> Vector:
        cid = self.vertices[vid].constraint
        if cid is None:
            return pos
        axis = self.constraint_axes.get(cid, 2)
        value = self.constraint_values.get(cid, 0.0)
        coords = [pos[0], pos[1], pos[2]]
        coords[axis] = value
        return (coords[0], coords[1], coords[2])

    def current_positions(self) -> Dict[int, Vector]:
        return {vid: v.pos for vid, v in self.vertices.items()}

    def face_triangles(
        self, face: Face, positions: Dict[int, Vector]
    ) -> Iterable[Tuple[Vector, Vector, Vector, Tuple[int, int, int]]]:
        verts = face.vertex_loop
        if len(verts) < 3:
            return []
        v0 = positions[verts[0]]
        for i in range(1, len(verts) - 1):
            vi = positions[verts[i]]
            vj = positions[verts[i + 1]]
            yield v0, vi, vj, (verts[0], verts[i], verts[i + 1])

    def signed_volume_and_grads(
        self, positions: Dict[int, Vector]
    ) -> Tuple[float, Dict[int, Vector]]:
        total_volume = 0.0
        grads: Dict[int, Vector] = {}
        face_ids = []
        for body in self.bodies:
            face_ids.extend(body.faces)
        if not face_ids:
            face_ids = list(self.faces.keys())
        for fid in face_ids:
            face = self.faces.get(abs(fid))
            if not face:
                continue
            for a, b, c, vids in self.face_triangles(face, positions):
                vol = v_dot(a, v_cross(b, c)) / 6.0
                total_volume += vol
                grad_a = v_cross(b, c)
                grad_b = v_cross(c, a)
                grad_c = v_cross(a, b)
                accumulate(grads, vids[0], v_scale(grad_a, 1.0 / 6.0))
                accumulate(grads, vids[1], v_scale(grad_b, 1.0 / 6.0))
                accumulate(grads, vids[2], v_scale(grad_c, 1.0 / 6.0))
        return total_volume, grads

    def body_volume_and_grads(
        self, body: Body, positions: Dict[int, Vector]
    ) -> Tuple[float, Dict[int, Vector]]:
        total_volume = 0.0
        grads: Dict[int, Vector] = {}
        face_ids = body.faces or list(self.faces.keys())
        for fid in face_ids:
            face = self.faces.get(abs(fid))
            if not face:
                continue
            for a, b, c, vids in self.face_triangles(face, positions):
                vol = v_dot(a, v_cross(b, c)) / 6.0
                total_volume += vol
                grad_a = v_cross(b, c)
                grad_b = v_cross(c, a)
                grad_c = v_cross(a, b)
                accumulate(grads, vids[0], v_scale(grad_a, 1.0 / 6.0))
                accumulate(grads, vids[1], v_scale(grad_b, 1.0 / 6.0))
                accumulate(grads, vids[2], v_scale(grad_c, 1.0 / 6.0))
        return total_volume, grads

    def area_and_grads(
        self, positions: Dict[int, Vector]
    ) -> Tuple[float, Dict[int, Vector]]:
        if numeric.USE_NUMPY and len(self.bodies) == 1:
            return self.area_and_grads_np(positions)
        total_area = 0.0
        grads: Dict[int, Vector] = {}
        for face in self.faces.values():
            for a, b, c, vids in self.face_triangles(face, positions):
                ab = v_sub(b, a)
                ac = v_sub(c, a)
                normal = v_cross(ab, ac)
                norm = v_norm(normal)
                area = 0.5 * norm
                total_area += area
                if norm < 1e-12:
                    continue
                n_hat = v_scale(normal, 1.0 / norm)
                grad_a = v_scale(v_cross(v_sub(b, c), n_hat), 0.5)
                grad_b = v_scale(v_cross(v_sub(c, a), n_hat), 0.5)
                grad_c = v_scale(v_cross(v_sub(a, b), n_hat), 0.5)
                accumulate(grads, vids[0], grad_a)
                accumulate(grads, vids[1], grad_b)
                accumulate(grads, vids[2], grad_c)
        return total_area, grads

    def area_and_grads_np(
        self, positions: Dict[int, Vector]
    ) -> Tuple[float, Dict[int, Vector]]:
        if not self.tri_idx:
            return 0.0, {}
        pos = numeric.np.array([positions[vid] for vid in self.order], dtype=float)
        tri = numeric.np.array(self.tri_idx, dtype=int)
        a = pos[tri[:, 0]]
        b = pos[tri[:, 1]]
        c = pos[tri[:, 2]]
        ab = b - a
        ac = c - a
        normal = numeric.np.cross(ab, ac)
        norm = numeric.np.linalg.norm(normal, axis=1)
        area = 0.5 * numeric.np.sum(norm)
        grads = numeric.np.zeros_like(pos)
        mask = norm > 1e-12
        if numeric.np.any(mask):
            n_hat = normal[mask] / norm[mask][:, None]
            bc = (b - c)[mask]
            ca = (c - a)[mask]
            ab = (a - b)[mask]
            ga = 0.5 * numeric.np.cross(bc, n_hat)
            gb = 0.5 * numeric.np.cross(ca, n_hat)
            gc = 0.5 * numeric.np.cross(ab, n_hat)
            tri_mask = tri[mask]
            numeric.np.add.at(grads, tri_mask[:, 0], ga)
            numeric.np.add.at(grads, tri_mask[:, 1], gb)
            numeric.np.add.at(grads, tri_mask[:, 2], gc)
        grad_dict = {vid: tuple(grads[i]) for i, vid in enumerate(self.order)}
        return float(area), grad_dict

    def energy(
        self, positions: Dict[int, Vector], penalty: float
    ) -> Tuple[float, float, float, Dict[int, Vector], float]:
        if (
            numeric.USE_NUMPY
            and len(self.bodies) == 1
            and self.square_curvature_modulus == 0.0
            and not self.quantities
        ):
            return self.energy_np(positions, penalty)
        area, area_grads = self.area_and_grads(positions)
        if self.surface_tension != 1.0:
            area_grads = {
                vid: v_scale(g, self.surface_tension) for vid, g in area_grads.items()
            }
        volume = 0.0
        penalty_grads: Dict[int, Vector] = {}
        contact_energy, contact_grads = self.contact_energy_and_grads(positions)
        gravity_energy, gravity_grads = self.gravity_energy_and_grads(positions)
        energy = self.surface_tension * area + contact_energy + gravity_energy
        quantity_grads: Dict[int, Vector] = {}
        quantity_energy = 0.0
        curvature_energy = 0.0
        curvature_grads: Dict[int, Vector] = {}
        curvature_methods = {"sq_mean_curvature", "square_curvature"}
        for qty in self.quantities:
            if qty.qtype != "energy" or qty.modulus == 0.0:
                continue
            method = qty.method
            scope = qty.scope or "global"
            targets = qty.targets or []
            h0 = 0.0
            if qty.params:
                h0 = float(qty.params.get("h0", 0.0))
            if method == "area":
                if scope != "global":
                    continue
                quantity_energy += qty.modulus * area
                for vid, g in area_grads.items():
                    accumulate(quantity_grads, vid, v_scale(g, qty.modulus))
            elif method in curvature_methods:
                if scope == "global":
                    if not curvature_grads:
                        curvature_energy, curvature_grads = curvature_energy_and_grads(
                            self, positions, h0=h0
                        )
                    quantity_energy += qty.modulus * curvature_energy
                    for vid, g in curvature_grads.items():
                        accumulate(quantity_grads, vid, v_scale(g, qty.modulus))
                elif scope == "body":
                    for bid in targets:
                        body = next((b for b in self.bodies if b.bid == bid), None)
                        if not body:
                            continue
                        e, g = curvature_energy_and_grads(
                            self, positions, face_ids=body.faces, h0=h0
                        )
                        quantity_energy += qty.modulus * e
                        for vid, gv in g.items():
                            accumulate(quantity_grads, vid, v_scale(gv, qty.modulus))
                elif scope == "facet":
                    for fid in targets:
                        e, g = curvature_energy_and_grads(
                            self, positions, face_ids=[fid], h0=h0
                        )
                        quantity_energy += qty.modulus * e
                        for vid, gv in g.items():
                            accumulate(quantity_grads, vid, v_scale(gv, qty.modulus))
        if self.square_curvature_modulus != 0.0 and not any(
            q.method in curvature_methods for q in self.quantities
        ):
            if not curvature_grads:
                curvature_energy, curvature_grads = curvature_energy_and_grads(
                    self, positions
                )
            quantity_energy += self.square_curvature_modulus * curvature_energy
            for vid, g in curvature_grads.items():
                accumulate(
                    quantity_grads, vid, v_scale(g, self.square_curvature_modulus)
                )
        energy += quantity_energy
        for body in self.bodies:
            body_vol, body_grads = self.body_volume_and_grads(body, positions)
            volume += body_vol
            volume_error = body_vol - body.target_volume
            energy += penalty * volume_error * volume_error
            scale = 2.0 * penalty * volume_error
            for vid, g in body_grads.items():
                accumulate(penalty_grads, vid, v_scale(g, scale))
        grads: Dict[int, Vector] = {}
        for vid in positions:
            ag = area_grads.get(vid, (0.0, 0.0, 0.0))
            pg = penalty_grads.get(vid, (0.0, 0.0, 0.0))
            cg = contact_grads.get(vid, (0.0, 0.0, 0.0))
            gg = gravity_grads.get(vid, (0.0, 0.0, 0.0))
            qg = quantity_grads.get(vid, (0.0, 0.0, 0.0))
            total = v_add(ag, pg)
            total = v_add(total, cg)
            total = v_add(total, gg)
            total = v_add(total, qg)
            if vid in self.fixed_ids:
                grads[vid] = (0.0, 0.0, 0.0)
            elif vid in self.constraint_vertices:
                axis = self.constraint_axes.get(self.vertices[vid].constraint, 2)
                coords = [total[0], total[1], total[2]]
                coords[axis] = 0.0
                grads[vid] = (coords[0], coords[1], coords[2])
            else:
                grads[vid] = total
        grad_norm = math.sqrt(sum(v_dot(g, g) for g in grads.values()))
        return energy, area, volume, grads, grad_norm

    def energy_np(
        self, positions: Dict[int, Vector], penalty: float
    ) -> Tuple[float, float, float, Dict[int, Vector], float]:
        if not self.tri_idx:
            return 0.0, 0.0, 0.0, {}, 0.0
        pos = numeric.np.array([positions[vid] for vid in self.order], dtype=float)
        tri = numeric.np.array(self.tri_idx, dtype=int)
        a = pos[tri[:, 0]]
        b = pos[tri[:, 1]]
        c = pos[tri[:, 2]]
        ab = b - a
        ac = c - a
        normal = numeric.np.cross(ab, ac)
        norm = numeric.np.linalg.norm(normal, axis=1)
        area = 0.5 * numeric.np.sum(norm)
        cross_bc = numeric.np.cross(b, c)
        cross_ca = numeric.np.cross(c, a)
        cross_ab = numeric.np.cross(a, b)
        vol_terms = numeric.np.einsum("ij,ij->i", a, cross_bc) / 6.0
        volume = numeric.np.sum(vol_terms)
        area_grads = numeric.np.zeros_like(pos)
        mask = norm > 1e-12
        if numeric.np.any(mask):
            n_hat = normal[mask] / norm[mask][:, None]
            bc = (b - c)[mask]
            ca = (c - a)[mask]
            ab = (a - b)[mask]
            ga = 0.5 * numeric.np.cross(bc, n_hat)
            gb = 0.5 * numeric.np.cross(ca, n_hat)
            gc = 0.5 * numeric.np.cross(ab, n_hat)
            tri_mask = tri[mask]
            numeric.np.add.at(area_grads, tri_mask[:, 0], ga)
            numeric.np.add.at(area_grads, tri_mask[:, 1], gb)
            numeric.np.add.at(area_grads, tri_mask[:, 2], gc)
        vol_grads = numeric.np.zeros_like(pos)
        numeric.np.add.at(vol_grads, tri[:, 0], cross_bc / 6.0)
        numeric.np.add.at(vol_grads, tri[:, 1], cross_ca / 6.0)
        numeric.np.add.at(vol_grads, tri[:, 2], cross_ab / 6.0)
        volume_error = volume - self.body.target_volume
        area_energy = self.surface_tension * area
        area_grads = area_grads * self.surface_tension
        grads = area_grads + (2.0 * penalty * volume_error) * vol_grads
        contact_energy, contact_grads = self.contact_energy_and_grads_np(positions, pos)
        gravity_energy, gravity_grads = self.gravity_energy_and_grads_np(
            positions, pos, a, b, c, vol_terms, cross_bc, cross_ca, cross_ab
        )
        grads = grads + contact_grads + gravity_grads
        for vid in self.fixed_ids:
            grads[self.index[vid], :] = 0.0
        for vid in self.constraint_vertices:
            axis = self.constraint_axes.get(self.vertices[vid].constraint, 2)
            grads[self.index[vid], axis] = 0.0
        grad_norm = float(numeric.np.sqrt(numeric.np.sum(grads * grads)))
        grad_dict = {vid: tuple(grads[i]) for i, vid in enumerate(self.order)}
        energy = float(
            area_energy
            + penalty * volume_error * volume_error
            + contact_energy
            + gravity_energy
        )
        return energy, float(area), float(volume), grad_dict, grad_norm

    def contact_energy_and_grads(
        self, positions: Dict[int, Vector]
    ) -> Tuple[float, Dict[int, Vector]]:
        if not self.boundary_edges or abs(self.wallt) < 1e-12:
            return 0.0, {}
        energy = 0.0
        grads: Dict[int, Vector] = {}
        for cid, info in self.constraints.items():
            exprs = info.get("energy")
            if exprs:
                e, g = self.constraint_energy_and_grads(positions, cid, exprs)
                energy += e
                for vid, gv in g.items():
                    accumulate(grads, vid, gv)
                continue
            contact = self.constraint_contacts.get(cid)
            if contact == "wallt_y":
                area, area_grads = self.constraint_area_and_grads(positions, cid)
                energy += self.wallt * abs(area)
                sign = 1.0 if area >= 0.0 else -1.0
                for vid, g in area_grads.items():
                    accumulate(grads, vid, v_scale(g, self.wallt * sign))
        return energy, grads

    def contact_energy_and_grads_np(
        self, positions: Dict[int, Vector], pos: "numeric.np.ndarray"
    ) -> Tuple[float, "numeric.np.ndarray"]:
        grads = numeric.np.zeros_like(pos)
        if not self.boundary_edges or abs(self.wallt) < 1e-12:
            return 0.0, grads
        energy = 0.0
        for cid, info in self.constraints.items():
            exprs = info.get("energy")
            if exprs:
                e, g = self.constraint_energy_and_grads(positions, cid, exprs)
                energy += e
                for vid, gv in g.items():
                    grads[self.index[vid]] += numeric.np.array(gv)
                continue
            contact = self.constraint_contacts.get(cid)
            if contact == "wallt_y":
                area, area_grads = self.constraint_area_and_grads(positions, cid)
                energy += self.wallt * abs(area)
                sign = 1.0 if area >= 0.0 else -1.0
                for vid, g in area_grads.items():
                    idx = self.index[vid]
                    grads[idx] += numeric.np.array(g) * (self.wallt * sign)
        return energy, grads

    def constraint_energy_and_grads(
        self, positions: Dict[int, Vector], cid: int, exprs: Dict[str, str]
    ) -> Tuple[float, Dict[int, Vector]]:
        edges = [
            eid for eid in self.boundary_edges if self.edges[eid].constraint == cid
        ]
        if not edges:
            return 0.0, {}
        energy = self.constraint_energy(positions, cid, exprs)
        grads: Dict[int, Vector] = {}
        eps = 1e-6
        axis = self.constraint_axes.get(cid, 2)
        affected = set()
        for eid in edges:
            edge = self.edges[eid]
            affected.add(edge.tail)
            affected.add(edge.head)
        for vid in affected:
            if vid in self.fixed_ids:
                grads[vid] = (0.0, 0.0, 0.0)
                continue
            base = positions[vid]
            deriv = [0.0, 0.0, 0.0]
            for dim in range(3):
                if dim == axis:
                    continue
                plus = list(base)
                minus = list(base)
                plus[dim] += eps
                minus[dim] -= eps
                pos_plus = dict(positions)
                pos_minus = dict(positions)
                pos_plus[vid] = self.project_vertex(vid, (plus[0], plus[1], plus[2]))
                pos_minus[vid] = self.project_vertex(
                    vid, (minus[0], minus[1], minus[2])
                )
                e_plus = self.constraint_energy(pos_plus, cid, exprs)
                e_minus = self.constraint_energy(pos_minus, cid, exprs)
                deriv[dim] = (e_plus - e_minus) / (2.0 * eps)
            grads[vid] = (deriv[0], deriv[1], deriv[2])
        return energy, grads

    def constraint_energy(
        self, positions: Dict[int, Vector], cid: int, exprs: Dict[str, str]
    ) -> float:
        edges = [
            eid for eid in self.boundary_edges if self.edges[eid].constraint == cid
        ]
        if not edges:
            return 0.0
        loops = build_edge_loops(edges, self.edges)
        if not loops:
            return 0.0
        energy = 0.0
        for loop in loops:
            n = len(loop)
            for i in range(n):
                a = positions[loop[i]]
                b = positions[loop[(i + 1) % n]]
                mid = ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5, (a[2] + b[2]) * 0.5)
                names = {
                    "x": mid[0],
                    "y": mid[1],
                    "z": mid[2],
                    "x1": mid[0],
                    "x2": mid[1],
                    "x3": mid[2],
                }
                names.update(self.params)
                for key, expr in self.defines.items():
                    try:
                        names[key] = eval_expr(expr, names)
                    except Exception:
                        pass
                e1 = eval_expr(exprs.get("e1", "0"), names)
                e2 = eval_expr(exprs.get("e2", "0"), names)
                e3 = eval_expr(exprs.get("e3", "0"), names)
                dx = b[0] - a[0]
                dy = b[1] - a[1]
                dz = b[2] - a[2]
                energy += e1 * dx + e2 * dy + e3 * dz
        return energy

    def constraint_area_and_grads(
        self, positions: Dict[int, Vector], cid: int
    ) -> Tuple[float, Dict[int, Vector]]:
        edges = [
            eid for eid in self.boundary_edges if self.edges[eid].constraint == cid
        ]
        loops = build_edge_loops(edges, self.edges)
        if not loops:
            return 0.0, {}
        axis = self.constraint_axes.get(cid, 2)
        energy_area = 0.0
        grads: Dict[int, Vector] = {}
        for loop in loops:
            area, loop_grads = polygon_area_and_grads(loop, positions, axis)
            energy_area += area
            for vid, g in loop_grads.items():
                accumulate(grads, vid, g)
        return energy_area, grads

    def gravity_energy_and_grads(
        self, positions: Dict[int, Vector]
    ) -> Tuple[float, Dict[int, Vector]]:
        g = self.gravity_constant * self.body.density
        if abs(g) < 1e-12:
            return 0.0, {}
        energy = 0.0
        grads: Dict[int, Vector] = {}
        face_ids = self.body.faces or list(self.faces.keys())
        for fid in face_ids:
            face = self.faces.get(abs(fid))
            if not face:
                continue
            for a, b, c, vids in self.face_triangles(face, positions):
                vol = v_dot(a, v_cross(b, c)) / 6.0
                sum_z = (a[2] + b[2] + c[2]) / 4.0
                energy += g * vol * sum_z
                dV_da = v_scale(v_cross(b, c), 1.0 / 6.0)
                dV_db = v_scale(v_cross(c, a), 1.0 / 6.0)
                dV_dc = v_scale(v_cross(a, b), 1.0 / 6.0)
                grad_a = v_add(v_scale(dV_da, g * sum_z), (0.0, 0.0, g * vol / 4.0))
                grad_b = v_add(v_scale(dV_db, g * sum_z), (0.0, 0.0, g * vol / 4.0))
                grad_c = v_add(v_scale(dV_dc, g * sum_z), (0.0, 0.0, g * vol / 4.0))
                accumulate(grads, vids[0], grad_a)
                accumulate(grads, vids[1], grad_b)
                accumulate(grads, vids[2], grad_c)
        return energy, grads

    def gravity_energy_and_grads_np(
        self,
        positions: Dict[int, Vector],
        pos: "numeric.np.ndarray",
        a: "numeric.np.ndarray",
        b: "numeric.np.ndarray",
        c: "numeric.np.ndarray",
        vol_terms: "numeric.np.ndarray",
        cross_bc: "numeric.np.ndarray",
        cross_ca: "numeric.np.ndarray",
        cross_ab: "numeric.np.ndarray",
    ) -> Tuple[float, "numeric.np.ndarray"]:
        g = self.gravity_constant * self.body.density
        grads = numeric.np.zeros_like(pos)
        if abs(g) < 1e-12:
            return 0.0, grads
        sum_z = (a[:, 2] + b[:, 2] + c[:, 2]) / 4.0
        energy = float(numeric.np.sum(g * vol_terms * sum_z))
        tri = numeric.np.array(self.tri_idx, dtype=int)
        coeff = g * sum_z / 6.0
        numeric.np.add.at(grads, tri[:, 0], cross_bc * coeff[:, None])
        numeric.np.add.at(grads, tri[:, 1], cross_ca * coeff[:, None])
        numeric.np.add.at(grads, tri[:, 2], cross_ab * coeff[:, None])
        z_term = (g * vol_terms / 4.0)[:, None]
        add_z = numeric.np.zeros_like(a)
        add_z[:, 2] = z_term[:, 0]
        numeric.np.add.at(grads, tri[:, 0], add_z)
        numeric.np.add.at(grads, tri[:, 1], add_z)
        numeric.np.add.at(grads, tri[:, 2], add_z)
        return energy, grads


def triangulate_face(vertex_loop: Sequence[int]) -> List[Tuple[int, int, int]]:
    if len(vertex_loop) < 3:
        return []
    if len(vertex_loop) == 3:
        return [tuple(vertex_loop)]  # type: ignore[return-value]
    tris = []
    v0 = vertex_loop[0]
    for i in range(1, len(vertex_loop) - 1):
        tris.append((v0, vertex_loop[i], vertex_loop[i + 1]))
    return tris

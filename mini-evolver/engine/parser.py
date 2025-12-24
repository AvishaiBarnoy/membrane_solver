"""Input parsing for .fe/.json/.yaml geometry."""

from __future__ import annotations

import json
import re
from typing import Dict, List

from .mesh import Body, Edge, Face, Mesh, Quantity, Vertex
from .refine import build_edges_from_faces, triangulate_mesh


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    lines = []
    for line in text.splitlines():
        clean = line.split("//", 1)[0].strip()
        if clean:
            lines.append(clean)
    return "\n".join(lines)


def parse_face_vertices(edge_ids: List[int], edges: Dict[int, Edge]) -> List[int]:
    vertices: List[int] = []
    for i, eid in enumerate(edge_ids):
        edge = edges[abs(eid)]
        if eid > 0:
            tail, head = edge.tail, edge.head
        else:
            tail, head = edge.head, edge.tail
        if i == 0:
            vertices.extend([tail, head])
        else:
            if vertices[-1] != tail:
                raise ValueError("Edge loop is not continuous")
            vertices.append(head)
    if vertices[0] == vertices[-1]:
        vertices.pop()
    return vertices


def parse_fe(path: str) -> Mesh:
    with open(path, "r", encoding="ascii") as f:
        raw = f.read()
    text = strip_comments(raw)
    macros: Dict[str, str] = {}
    defines: Dict[str, str] = {}
    read_commands: List[str] = []
    for name, body_text in re.findall(r"(\w+)\s*:=\s*\{([^}]*)\}", text):
        macros[name.strip().lower()] = body_text.strip()
    vertices: Dict[int, Vertex] = {}
    edges: Dict[int, Edge] = {}
    faces: Dict[int, Face] = {}
    body: Body | None = None
    bodies: List[Body] = []
    params: Dict[str, float] = {}
    quantities: List[Quantity] = []
    gravity_constant = 0.0
    constraints: Dict[int, Dict[str, float]] = {}
    section = None
    data_done = False
    current_constraint: int | None = None
    in_energy_block = False
    after_read = False
    for line in text.splitlines():
        lower = line.lower()
        if lower.startswith("parameter"):
            parts = line.split()
            if len(parts) >= 4 and parts[2] == "=":
                try:
                    params[parts[1]] = float(parts[3])
                except ValueError:
                    pass
            continue
        if lower.startswith("quantity"):
            tokens = line.split()
            name = tokens[1] if len(tokens) > 1 else "quantity"
            qtype = tokens[2].lower() if len(tokens) > 2 else "energy"
            method = ""
            modulus = 1.0
            qty_params: Dict[str, float] = {}
            scope = "global"
            targets: List[int] = []
            if "modulus" in (t.lower() for t in tokens):
                idx = [i for i, t in enumerate(tokens) if t.lower() == "modulus"][0]
                try:
                    modulus = float(tokens[idx + 1])
                except (ValueError, IndexError):
                    try:
                        name = tokens[idx + 1]
                    except IndexError:
                        name = ""
                    if name and name in params:
                        modulus = float(params[name])
                        qty_params["modulus_param"] = name
                    else:
                        modulus = 1.0
            if "method" in (t.lower() for t in tokens):
                idx = [i for i, t in enumerate(tokens) if t.lower() == "method"][0]
                try:
                    method = tokens[idx + 1].lower()
                except IndexError:
                    method = ""
            if "body" in (t.lower() for t in tokens):
                idx = [i for i, t in enumerate(tokens) if t.lower() == "body"][0]
                try:
                    targets = [int(tokens[idx + 1])]
                    scope = "body"
                except (ValueError, IndexError):
                    scope = "body"
            if "facet" in (t.lower() for t in tokens):
                idx = [i for i, t in enumerate(tokens) if t.lower() == "facet"][0]
                try:
                    targets = [int(tokens[idx + 1])]
                    scope = "facet"
                except (ValueError, IndexError):
                    scope = "facet"
            if "global" in (t.lower() for t in tokens):
                scope = "global"
            if method:
                quantities.append(
                    Quantity(
                        name=name,
                        qtype=qtype,
                        method=method,
                        modulus=modulus,
                        scope=scope,
                        targets=targets or None,
                        params=qty_params or None,
                    )
                )
            continue
        if lower.startswith("gravity_constant"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    gravity_constant = float(parts[1])
                except ValueError:
                    pass
            continue
        if lower.startswith("constraint"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    current_constraint = int(parts[1])
                except ValueError:
                    current_constraint = None
            in_energy_block = False
            continue
        if current_constraint is not None and lower.startswith("formula"):
            continue
        if current_constraint is not None and lower.startswith("energy"):
            in_energy_block = True
            continue
        if (
            current_constraint is not None
            and in_energy_block
            and lower.startswith("e1")
        ):
            expr = line.split(":", 1)[1].strip()
            constraints.setdefault(current_constraint, {}).setdefault("energy", {})[
                "e1"
            ] = expr
            continue
        if (
            current_constraint is not None
            and in_energy_block
            and lower.startswith("e2")
        ):
            expr = line.split(":", 1)[1].strip()
            constraints.setdefault(current_constraint, {}).setdefault("energy", {})[
                "e2"
            ] = expr
            continue
        if (
            current_constraint is not None
            and in_energy_block
            and lower.startswith("e3")
        ):
            expr = line.split(":", 1)[1].strip()
            constraints.setdefault(current_constraint, {}).setdefault("energy", {})[
                "e3"
            ] = expr
            continue
        if lower.startswith("#define"):
            parts = line.split(None, 2)
            if len(parts) >= 3:
                defines[parts[1]] = parts[2].strip()
            continue
        if current_constraint is not None:
            match = re.search(r"x([123])\s*=\s*([-+]?\d+(?:\.\d+)?)", lower)
            if match:
                axis = int(match.group(1))
                value = float(match.group(2))
                constraints[current_constraint] = {"axis": axis, "value": value}
                current_constraint = None
                continue
        if lower == "vertices":
            section = "vertices"
            continue
        if lower == "edges":
            section = "edges"
            continue
        if lower == "faces":
            section = "faces"
            continue
        if lower == "bodies":
            section = "bodies"
            continue
        if lower == "read":
            data_done = True
            section = None
            after_read = True
            continue
        if data_done:
            if after_read and ":=" not in line:
                read_commands.append(line.strip())
            continue
        tokens = line.split()
        if not tokens:
            continue
        if section == "vertices":
            vid = int(tokens[0])
            x, y, z = map(float, tokens[1:4])
            tail_tokens = [t.lower() for t in tokens[4:]]
            fixed = "fixed" in tail_tokens
            constraint = None
            if "constraint" in tail_tokens:
                idx = tail_tokens.index("constraint")
                try:
                    constraint = int(tokens[4 + idx + 1])
                except (ValueError, IndexError):
                    constraint = None
            vertices[vid] = Vertex(vid, (x, y, z), fixed=fixed, constraint=constraint)
        elif section == "edges":
            eid = int(tokens[0])
            tail, head = map(int, tokens[1:3])
            constraint = None
            lower_tokens = [t.lower() for t in tokens[3:]]
            if "constraint" in lower_tokens:
                idx = lower_tokens.index("constraint")
                try:
                    constraint = int(tokens[3 + idx + 1])
                except (ValueError, IndexError):
                    constraint = None
            no_refine = "no_refine" in lower_tokens
            edges[eid] = Edge(
                eid, tail, head, constraint=constraint, no_refine=no_refine
            )
        elif section == "faces":
            fid = int(tokens[0])
            edge_loop: List[int] = []
            tail_tokens: List[str] = []
            for tok in tokens[1:]:
                try:
                    edge_loop.append(int(tok))
                except ValueError:
                    tail_tokens.append(tok.lower())
            no_refine = "no_refine" in tail_tokens
            faces[fid] = Face(
                fid, edge_loop=edge_loop, vertex_loop=[], no_refine=no_refine
            )
        elif section == "bodies":
            bid = int(tokens[0])
            face_list: List[int] = []
            target_volume = 0.0
            density = 1.0
            for tok in tokens[1:]:
                if tok.lower() == "volume":
                    break
                face_list.append(int(tok))
            if "volume" in (t.lower() for t in tokens):
                vol_index = [i for i, t in enumerate(tokens) if t.lower() == "volume"][
                    0
                ]
                target_volume = float(tokens[vol_index + 1])
            if "density" in (t.lower() for t in tokens):
                den_index = [i for i, t in enumerate(tokens) if t.lower() == "density"][
                    0
                ]
                try:
                    density = float(tokens[den_index + 1])
                except (ValueError, IndexError):
                    density = 1.0
            new_body = Body(
                bid, faces=face_list, target_volume=target_volume, density=density
            )
            bodies.append(new_body)
            if body is None:
                body = new_body
    for face in faces.values():
        face.vertex_loop = parse_face_vertices(face.edge_loop, edges)
    mesh = Mesh(
        vertices,
        edges,
        faces,
        body if body else Body(1, [], 0.0),
        bodies=bodies if bodies else None,
        quantities=quantities,
        macros=macros,
        params=params,
        gravity_constant=gravity_constant,
        surface_tension=float(params.get("surface_tension", 1.0)),
        square_curvature_modulus=float(params.get("square_curvature_modulus", 0.0)),
        constraints=constraints,
        defines=defines,
        read_commands=read_commands,
    )
    mesh.flip_orientation_if_needed(mesh.current_positions())
    return triangulate_mesh(mesh)


def parse_geometry_dict(data: Dict[str, object]) -> Mesh:
    vertices: Dict[int, Vertex] = {}
    for key, coords in (data.get("vertices") or {}).items():  # type: ignore[union-attr]
        vid = int(key)
        vertices[vid] = Vertex(
            vid, (float(coords[0]), float(coords[1]), float(coords[2]))
        )
    fixed_vertices = {int(v) for v in (data.get("fixed_vertices") or [])}
    for vid in fixed_vertices:
        if vid in vertices:
            vertices[vid].fixed = True
    vertex_constraints = data.get("vertex_constraints") or {}
    if isinstance(vertex_constraints, dict):
        for key, value in vertex_constraints.items():
            vid = int(key)
            if vid in vertices:
                vertices[vid].constraint = int(value)
    faces_vertices = [list(map(int, face)) for face in (data.get("faces") or [])]
    edges, faces = build_edges_from_faces(faces_vertices, vertices=vertices)
    no_refine_faces = {int(f) for f in (data.get("no_refine_faces") or [])}
    for fid in no_refine_faces:
        face = faces.get(fid)
        if face:
            face.no_refine = True
    no_refine_edges = data.get("no_refine_edges") or []
    if isinstance(no_refine_edges, list):
        edge_by_key = {
            (min(e.tail, e.head), max(e.tail, e.head)): e for e in edges.values()
        }
        for item in no_refine_edges:
            if isinstance(item, int):
                edge = edges.get(item)
                if edge:
                    edge.no_refine = True
                continue
            if (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and all(isinstance(v, (int, float, str)) for v in item)
            ):
                try:
                    a = int(item[0])
                    b = int(item[1])
                except (ValueError, TypeError):
                    continue
                key = (a, b) if a < b else (b, a)
                edge = edge_by_key.get(key)
                if edge:
                    edge.no_refine = True
    bodies_data = data.get("bodies")
    bodies: List[Body] = []
    if isinstance(bodies_data, list) and bodies_data:
        for item in bodies_data:
            faces_list = [int(f) for f in item.get("faces", [])]
            target_volume = float(item.get("target_volume", 0.0))
            density = float(item.get("density", 1.0))
            bid = int(item.get("id", len(bodies) + 1))
            bodies.append(Body(bid, faces_list, target_volume, density=density))
        body = bodies[0]
    else:
        target_volume = float(data.get("target_volume", 0.0))
        density = float(data.get("density", 1.0))
        body = Body(1, list(faces.keys()), target_volume, density=density)
    raw_macros = data.get("macros", {})
    macros = (
        {str(key).lower(): str(value) for key, value in raw_macros.items()}
        if isinstance(raw_macros, dict)
        else {}
    )
    params = {k: float(v) for k, v in (data.get("params") or {}).items()}
    gravity_constant = float(data.get("gravity_constant", 0.0))
    surface_tension = float(data.get("surface_tension", 1.0))
    square_curvature_modulus = float(data.get("square_curvature_modulus", 0.0))
    quantities: List[Quantity] = []
    raw_quantities = data.get("quantities") or []
    if isinstance(raw_quantities, list):
        for item in raw_quantities:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "quantity"))
            qtype = str(item.get("type", "energy")).lower()
            method = str(item.get("method", "")).lower()
            modulus_val = item.get("modulus", 1.0)
            modulus = 1.0
            qty_params: Dict[str, float] = {}
            if isinstance(modulus_val, (int, float)):
                modulus = float(modulus_val)
            elif isinstance(modulus_val, str):
                if modulus_val in params:
                    modulus = float(params[modulus_val])
                    qty_params["modulus_param"] = 1.0
                elif modulus_val in (data.get("params") or {}):
                    try:
                        modulus = float((data.get("params") or {})[modulus_val])
                        qty_params["modulus_param"] = 1.0
                    except (ValueError, TypeError):
                        modulus = 1.0
            scope = str(item.get("scope", "global")).lower()
            targets_raw = item.get("targets") or []
            targets: List[int] = []
            if isinstance(targets_raw, list):
                for val in targets_raw:
                    try:
                        targets.append(int(val))
                    except (ValueError, TypeError):
                        continue
            params_raw = item.get("params") or {}
            if isinstance(params_raw, dict):
                for key, value in params_raw.items():
                    try:
                        qty_params[str(key)] = float(value)
                    except (ValueError, TypeError):
                        continue
            if method:
                quantities.append(
                    Quantity(
                        name=name,
                        qtype=qtype,
                        method=method,
                        modulus=modulus,
                        scope=scope,
                        targets=targets or None,
                        params=qty_params or None,
                    )
                )
    constraints = data.get("constraints", {})
    defines = data.get("defines", {})
    read_commands = data.get("read_commands", [])
    mesh = Mesh(
        vertices,
        edges,
        faces,
        body,
        bodies=bodies if bodies else None,
        quantities=quantities,
        macros=macros,
        params=params,
        gravity_constant=gravity_constant,
        surface_tension=surface_tension,
        square_curvature_modulus=square_curvature_modulus,
        constraints=constraints,
        defines=defines,
        read_commands=read_commands,
    )
    mesh.flip_orientation_if_needed(mesh.current_positions())
    return triangulate_mesh(mesh)


def parse_json_geometry(path: str) -> Mesh:
    with open(path, "r", encoding="ascii") as f:
        data = json.load(f)
    return parse_geometry_dict(data)


def parse_yaml_geometry(path: str) -> Mesh:
    with open(path, "r", encoding="ascii") as f:
        raw = f.read()
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(raw)
    except Exception:
        data = json.loads(raw)
    return parse_geometry_dict(data)


def load_mesh(path: str) -> Mesh:
    lower = path.lower()
    if lower.endswith(".json"):
        return parse_json_geometry(path)
    if lower.endswith(".yaml") or lower.endswith(".yml"):
        return parse_yaml_geometry(path)
    return parse_fe(path)

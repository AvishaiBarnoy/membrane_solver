"""Command parsing and macro execution."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .geometry import Vector
from .mesh import Mesh
from .minimize import (
    enforce_volume_constraint,
    gradient_steps,
    hessian_bfgs,
    vertex_average,
)
from .plot import LivePlotter
from .refine import refine_edges_on_constraint, refine_mesh


def set_mesh_positions(mesh: Mesh, positions: Dict[int, Vector]) -> None:
    for vid, pos in positions.items():
        if vid in mesh.vertices:
            mesh.vertices[vid].pos = pos


def expand_macros(mesh: Mesh, text: str, depth: int = 0) -> List[str]:
    if depth > 5:
        return []
    expanded: List[str] = []
    for part in text.split(";"):
        item = part.strip()
        if not item:
            continue
        name = item.split()[0].lower()
        if name in mesh.macros:
            expanded.extend(expand_macros(mesh, mesh.macros[name], depth + 1))
        else:
            expanded.append(item)
    return expanded


def parse_command(line: str) -> Tuple[str, List[str]]:
    tokens = line.replace(":=", " :=").split()
    if not tokens:
        return "", []
    head = tokens[0]
    if head in {"G", "V"}:
        return head, tokens[1:]
    if head.lower().startswith("g") and head[1:].isdigit():
        return "g", [head[1:]]
    return head.lower(), tokens[1:]


def execute_commands(
    mesh: Mesh,
    positions: Dict[int, Vector],
    commands: List[str],
    penalty: float,
    enforce_volume: bool,
    hessian_steps: int,
    volume_mode: str,
    step_scale: float = 1.0,
    on_step: callable | None = None,
) -> Tuple[Mesh, Dict[int, Vector], int]:
    step_count = 0

    def maybe_update_plotter_mesh() -> None:
        if not on_step:
            return
        target = getattr(on_step, "__self__", None)
        if isinstance(target, LivePlotter):
            target.mesh = mesh

    for cmd in commands:
        if not cmd:
            continue
        macro_key = cmd if cmd in mesh.macros else cmd.lower()
        if macro_key in mesh.macros:
            expanded = expand_macros(mesh, mesh.macros[macro_key])
            mesh, positions, steps = execute_commands(
                mesh,
                positions,
                expanded,
                penalty,
                enforce_volume,
                hessian_steps,
                volume_mode,
                step_scale=step_scale,
                on_step=on_step,
            )
            step_count += steps
            continue
        name, args = parse_command(cmd)
        if name and len(args) >= 2 and args[0] == ":=":
            key = name
            try:
                value = float(args[1])
            except ValueError:
                continue
            mesh.params[key] = value
            mesh.wallt = mesh.compute_wallt()
            continue
        if name == "g" and args:
            try:
                count = int(args[0])
            except ValueError:
                continue
            positions = gradient_steps(
                mesh,
                positions,
                penalty=penalty,
                steps=count,
                enforce_volume=enforce_volume,
                step_scale=step_scale,
                volume_mode=volume_mode,
            )
            set_mesh_positions(mesh, positions)
            step_count += count
            if on_step:
                on_step(positions, step_count)
        elif name == "r":
            set_mesh_positions(mesh, positions)
            mesh = refine_mesh(mesh)
            positions = mesh.current_positions()
            maybe_update_plotter_mesh()
            if on_step:
                on_step(positions, step_count)
        elif name == "hessian":
            positions = hessian_bfgs(
                mesh,
                positions,
                penalty=penalty,
                steps=hessian_steps,
                volume_mode=volume_mode,
                enforce_volume=enforce_volume,
            )
            set_mesh_positions(mesh, positions)
            step_count += hessian_steps
            if on_step:
                on_step(positions, step_count)
        elif name == "refine":
            if args and args[0].lower() == "edges" and "on_constraint" in args:
                try:
                    idx = args.index("on_constraint")
                    cid = int(args[idx + 1])
                except (ValueError, IndexError):
                    cid = None
                if cid is not None:
                    set_mesh_positions(mesh, positions)
                    mesh = refine_edges_on_constraint(mesh, cid)
                    positions = mesh.current_positions()
                    maybe_update_plotter_mesh()
                    if on_step:
                        on_step(positions, step_count)
            else:
                set_mesh_positions(mesh, positions)
                mesh = refine_mesh(mesh)
                positions = mesh.current_positions()
                maybe_update_plotter_mesh()
                if on_step:
                    on_step(positions, step_count)
        elif name == "re":
            set_mesh_positions(mesh, positions)
            mesh = refine_edges_on_constraint(mesh, 1)
            positions = mesh.current_positions()
            maybe_update_plotter_mesh()
            if on_step:
                on_step(positions, step_count)
        elif name == "t" and args:
            try:
                step_scale = float(args[0])
            except ValueError:
                pass
        elif name == "V":
            positions = vertex_average(mesh, positions)
            positions, _ = enforce_volume_constraint(mesh, positions)
            set_mesh_positions(mesh, positions)
            if on_step:
                on_step(positions, step_count)
        elif name == "G" and args:
            try:
                mesh.gravity_constant = float(args[0])
            except ValueError:
                pass
        elif name == "unset" and len(args) >= 3:
            if args[0].lower() == "vertex" and args[1].lower() == "constraint":
                try:
                    cid = int(args[2])
                except ValueError:
                    continue
                for vid, v in mesh.vertices.items():
                    if v.constraint == cid:
                        v.constraint = None
                mesh.build_cache()
                positions = mesh.current_positions()
    return mesh, positions, step_count


def run_gogo(
    mesh: Mesh,
    penalty: float,
    enforce_volume: bool,
    hessian_steps: int = 5,
    macro_name: str = "gogo",
    volume_mode: str = "penalty",
    on_step: callable | None = None,
) -> Tuple[Mesh, Dict[int, Vector], int]:
    positions = mesh.current_positions()
    expanded = expand_macros(mesh, mesh.macros.get(macro_name.lower(), ""))
    return execute_commands(
        mesh,
        positions,
        expanded,
        penalty,
        enforce_volume,
        hessian_steps,
        volume_mode,
        on_step=on_step,
    )

import re

import numpy as np

from commands.base import Command


class QuitCommand(Command):
    def execute(self, context, args):
        context.should_exit = True
        print("Exiting interactive mode.")


class HelpCommand(Command):
    def execute(self, context, args):
        print("Interactive commands:")
        print("  gN            Run N minimization steps (e.g. g5, g10)")
        print("  gd / cg       Switch to Gradient Descent / Conjugate Gradient stepper")
        print("  tX            Fix step size to X (e.g. t1e-3)")
        print("  tf / t free   Re-enable adaptive step sizing")
        print("  r             Refine mesh (triangle refinement + polygonal)")
        print("  V / Vn        Vertex averaging once or n times (e.g. V5)")
        print("  vertex_average Same as V")
        print("  u             Equiangulate mesh")
        print(
            "  visualize / s [tilt|div|plain] [arrows|noarrows] Plot current geometry"
        )
        print("  properties    Print physical properties (area, volume, surface Rg)")
        print(
            "  print [entity] [id|filter] Query geometry (e.g. print edges len > 0.1)"
        )
        print(
            "  set [param] [value]       Set global parameter (e.g. set surface_tension 1.0)"
        )
        print(
            "                            or entity property (e.g. set vertex 0 fixed true)"
        )
        print(
            "  live_vis / lv [tilt|div|plain] [arrows|noarrows] Turn on/off live visualization during minimization"
        )
        print("  save          Save geometry to 'interactive.temp'")
        print("  energy        Shortcut for 'print energy breakdown'")
        print("  history       Show commands entered in this session")
        print("  tilt_stats    Print |tilt| and div(tilt) diagnostics")
        print("  refresh       Reload energy/constraint modules from mesh state")
        print("  quit / exit / q  Leave interactive mode")


class StepSizeCommand(Command):
    """Set minimizer step size (shortcut: tX)."""

    def execute(self, context, args):
        if not args:
            print("Usage: tX (e.g. t1e-3)")
            return

        token = str(args[0]).strip().lower()
        if token in {"f", "free"}:
            context.mesh.global_parameters.set("step_size_mode", "adaptive")
            print("Step size mode set to adaptive.")
            return

        try:
            step_size = float(token)
        except (TypeError, ValueError):
            print(f"Invalid step size: {args[0]!r} (use tX or tf)")
            return

        if step_size <= 0.0:
            print("Step size must be positive.")
            return

        mesh = context.mesh
        mesh.global_parameters.set("step_size", step_size)
        mesh.global_parameters.set("step_size_mode", "fixed")
        if getattr(context, "minimizer", None) is not None:
            context.minimizer.step_size = step_size

        print(f"Step size fixed to {step_size:g}")


class EnergyCommand(Command):
    """Print total energy and per-module breakdown."""

    def execute(self, context, args):
        mode = "breakdown"
        if args:
            mode = str(args[0]).lower().strip()

        if mode in {"breakdown", "details", "detail"}:
            breakdown = context.minimizer.compute_energy_breakdown()
            total = sum(breakdown.values())
            print(f"Current Total Energy: {total:.10f}")
            for name, value in breakdown.items():
                print(f"  {name}: {value:.10f}")
            return

        if mode in {"stats", "curvature"}:
            from geometry.curvature import compute_curvature_fields

            mesh = context.mesh
            mesh.build_position_cache()
            positions = mesh.positions_view()
            index_map = mesh.vertex_index_to_row
            fields = compute_curvature_fields(mesh, positions, index_map)

            H = fields.mean_curvature
            boundary_vids = getattr(mesh, "boundary_vertex_ids", None) or []
            boundary_rows = np.array(
                [index_map[vid] for vid in boundary_vids if vid in index_map],
                dtype=int,
            )

            mask_interior = np.ones(len(H), dtype=bool)
            if boundary_rows.size:
                mask_interior[boundary_rows] = False

            def _stats(name: str, values):
                if values.size == 0:
                    print(f"{name}: (no vertices)")
                    return
                vals = np.asarray(values, dtype=float)
                q = np.quantile(vals, [0.0, 0.5, 0.9, 0.99, 1.0])
                print(
                    f"{name}: min={q[0]:.4e} med={q[1]:.4e} "
                    f"p90={q[2]:.4e} p99={q[3]:.4e} max={q[4]:.4e}"
                )

            print("Curvature diagnostics (|H|):")
            print(f"  vertices: {len(H)} (boundary {int(boundary_rows.size)})")
            _stats("  all", H)
            if np.any(mask_interior):
                _stats("  interior", H[mask_interior])
            return

        if mode in {"total", "sum"}:
            E = context.minimizer.compute_energy()
            print(f"Current Total Energy: {E:.10f}")
            return

        print("Usage: energy [breakdown|total]")


class RefreshModulesCommand(Command):
    """Reload energy/constraint modules from the current mesh state."""

    def execute(self, context, args):
        minimizer = getattr(context, "minimizer", None)
        if minimizer is None:
            print("No minimizer available to refresh modules.")
            return
        minimizer.refresh_modules()


class TiltStatsCommand(Command):
    """Print summary statistics for tilt magnitude and divergence."""

    def execute(self, context, args):
        from geometry.tilt_operators import p1_vertex_divergence

        mesh = context.mesh
        mesh.build_position_cache()
        positions = mesh.positions_view()
        tri_rows, _ = mesh.triangle_row_cache()
        if tri_rows is None or len(tri_rows) == 0:
            print("Tilt diagnostics: no triangles available.")
            return

        boundary_vids = getattr(mesh, "boundary_vertex_ids", None) or []
        boundary_rows = np.array(
            [
                mesh.vertex_index_to_row[vid]
                for vid in boundary_vids
                if vid in mesh.vertex_index_to_row
            ],
            dtype=int,
        )
        mask_interior = np.ones(len(mesh.vertex_ids), dtype=bool)
        if boundary_rows.size:
            mask_interior[boundary_rows] = False

        def _stats(label: str, values: np.ndarray) -> None:
            values = np.asarray(values, dtype=float)
            if values.size == 0:
                print(f"{label}: (no vertices)")
                return
            q = np.quantile(values, [0.0, 0.5, 0.9, 0.99, 1.0])
            print(
                f"{label}: min={q[0]:.4e} med={q[1]:.4e} "
                f"p90={q[2]:.4e} p99={q[3]:.4e} max={q[4]:.4e}"
            )

        def _report(name: str, tilts: np.ndarray) -> None:
            tilts = np.asarray(tilts, dtype=float)
            if tilts.size == 0:
                print(f"{name}: (no tilt data)")
                return
            mags = np.linalg.norm(tilts, axis=1)
            div_v, _areas = p1_vertex_divergence(
                n_vertices=len(mesh.vertex_ids),
                positions=positions,
                tilts=tilts,
                tri_rows=tri_rows,
            )

            print(f"{name} (|t|):")
            _stats("  all", mags)
            if np.any(mask_interior):
                _stats("  interior", mags[mask_interior])

            print(f"{name} (div t):")
            _stats("  all", div_v)
            if np.any(mask_interior):
                _stats("  interior", div_v[mask_interior])

        mode = args[0].strip().lower() if args else "both"
        if mode in {"tilt", "legacy", "single"}:
            _report("tilt", mesh.tilts_view())
            return
        if mode in {"in", "inner", "tilt_in"}:
            _report("tilt_in", mesh.tilts_in_view())
            return
        if mode in {"out", "outer", "tilt_out"}:
            _report("tilt_out", mesh.tilts_out_view())
            return

        # Default: report all available fields.
        if hasattr(mesh, "tilts_in_view") and hasattr(mesh, "tilts_out_view"):
            _report("tilt_in", mesh.tilts_in_view())
            _report("tilt_out", mesh.tilts_out_view())
        else:
            _report("tilt", mesh.tilts_view())


class SetCommand(Command):
    """Refactored logic for 'set'."""

    def execute(self, context, args):
        # args expected to be tokens after 'set', e.g. ['vertex', '0', 'fixed', 'true']
        if len(args) < 2:
            print("Usage: set [param] [value] OR set [entity] [id] [prop] [value]")
            return

        mesh = context.mesh
        first = args[0].lower()

        def _parse_value(text: str):
            """Parse common CLI literals (bool/none/float) and fall back to string."""
            raw = str(text).strip()
            low = raw.lower()
            if low == "true":
                return True
            if low == "false":
                return False
            if low in {"none", "null"}:
                return None
            try:
                return float(raw)
            except ValueError:
                return raw

        def _parse_filter(tokens: list[str]) -> tuple[str, str, object]:
            """Parse a simple `where` filter like `key=value` or `key > 1.0`."""
            if not tokens:
                raise ValueError("Empty where clause.")
            if len(tokens) == 1:
                m = re.match(r"^([A-Za-z_][\w]*)(>=|<=|!=|==|=|>|<)(.+)$", tokens[0])
                if not m:
                    raise ValueError(f"Invalid where expression: {tokens[0]!r}")
                key, op, raw_val = m.groups()
                return str(key), str(op), _parse_value(raw_val)

            if len(tokens) >= 3:
                key = tokens[0]
                op = tokens[1]
                raw_val = " ".join(tokens[2:])
                return str(key), str(op), _parse_value(raw_val)

            raise ValueError("Invalid where clause; use `key=value` or `key op value`.")

        def _get_attr_or_option(obj, key: str):
            """Return `obj.key` if present, else `obj.options[key]` when available."""
            if hasattr(obj, key):
                return getattr(obj, key)
            opts = getattr(obj, "options", None) or {}
            if isinstance(opts, dict) and key in opts:
                return opts[key]
            return None

        def _to_float(value):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        def _matches_filter(obj, key: str, op: str, expected) -> bool:
            """Return True when `obj` matches the filter condition."""
            actual = _get_attr_or_option(obj, key)
            if actual is None:
                return False

            op = "==" if op == "=" else op
            if op in {"==", "!="}:
                a_num = _to_float(actual)
                e_num = _to_float(expected)
                if a_num is not None and e_num is not None:
                    hit = a_num == e_num
                else:
                    hit = str(actual) == str(expected)
                return hit if op == "==" else (not hit)

            a_num = _to_float(actual)
            e_num = _to_float(expected)
            if a_num is None or e_num is None:
                return False

            if op == ">":
                return a_num > e_num
            if op == "<":
                return a_num < e_num
            if op == ">=":
                return a_num >= e_num
            if op == "<=":
                return a_num <= e_num
            return False

        # Check if first token is an entity type
        if first in [
            "vertex",
            "edge",
            "facet",
            "body",
            "vertices",
            "edges",
            "facets",
            "bodies",
        ]:
            if len(args) < 4:
                print("Usage: set [entity] [id] [prop] [value]")
                return

            entity_type = first
            id_token = str(args[1]).strip().lower()
            prop = args[2]
            val_str = args[3]

            val = _parse_value(val_str)

            entities = None
            if entity_type.startswith("vert"):
                entities = mesh.vertices
            elif entity_type.startswith("edge"):
                entities = mesh.edges
            elif entity_type.startswith(("facet", "face")):
                entities = mesh.facets
            elif entity_type.startswith("body"):
                entities = mesh.bodies
            else:
                entities = {}

            targets = []
            idx = None
            if id_token in {"all", "*"}:
                targets = list(entities.values())
            else:
                try:
                    idx = int(id_token)
                except ValueError:
                    print("ID must be an integer or 'all'.")
                    return
                obj = entities.get(idx)
                if not obj:
                    print(f"Entity {idx} not found.")
                    return
                targets = [obj]

            # Optional: apply filter clause
            if len(args) > 4:
                if args[4].lower() != "where":
                    print("Usage: set [entity] [id|all] [prop] [value] [where ...]")
                    return
                try:
                    f_key, f_op, f_val = _parse_filter([str(t) for t in args[5:]])
                except ValueError as exc:
                    print(f"Invalid where clause: {exc}")
                    return
                targets = [
                    obj for obj in targets if _matches_filter(obj, f_key, f_op, f_val)
                ]
                if not targets:
                    print("No entities matched the filter.")
                    return

            for obj in targets:
                # Set property
                if prop == "fixed":
                    obj.fixed = bool(val)
                    if obj.fixed and entity_type.startswith("edge"):
                        mesh.vertices[obj.tail_index].fixed = True
                        mesh.vertices[obj.head_index].fixed = True
                elif entity_type.startswith("body") and prop == "target_volume":
                    obj.target_volume = None if val is None else float(val)
                    if not obj.options:
                        obj.options = {}
                    obj.options["target_volume"] = obj.target_volume
                elif entity_type.startswith("vert") and prop in {"x", "y", "z"}:
                    coord = _to_float(val)
                    if coord is None:
                        continue
                    axis = {"x": 0, "y": 1, "z": 2}[prop]
                    obj.position[axis] = coord
                else:
                    if not obj.options:
                        obj.options = {}
                    obj.options[prop] = val

            if id_token not in {"all", "*"} and len(targets) == 1:
                obj = targets[0]
                if prop == "fixed":
                    print(f"Set {entity_type} {idx} fixed={obj.fixed}")
                elif entity_type.startswith("body") and prop == "target_volume":
                    print(f"Set {entity_type} {idx} target_volume={obj.target_volume}")
                elif entity_type.startswith("vert") and prop in {"x", "y", "z"}:
                    axis = {"x": 0, "y": 1, "z": 2}[prop]
                    print(
                        f"Set {entity_type} {idx} position[{prop}] = {obj.position[axis]}"
                    )
                else:
                    print(f"Set {entity_type} {idx} options['{prop}'] = {val}")
            else:
                print(f"Updated {len(targets)} {entity_type}(s).")

        else:
            # Global parameter set
            param = args[0]
            val_str = args[1]
            try:
                val = float(val_str)
            except ValueError:
                val = val_str

            mesh.global_parameters.set(param, val)
            print(f"Global parameter '{param}' set to {val}")

        bump = getattr(mesh, "increment_version", None)
        if callable(bump):
            bump()


class PrintEntityCommand(Command):
    """Refactored logic for 'print'."""

    def execute(self, context, args):
        if len(args) < 1:
            print("Usage: print [vertices|edges|facets|bodies] [id|filter] ...")
            return

        mesh = context.mesh
        entity_type = args[0].lower()

        entities = None
        name = ""
        if entity_type in ["vertices", "vertex"]:
            entities = mesh.vertices
            name = "Vertex"
        elif entity_type in ["edges", "edge"]:
            entities = mesh.edges
            name = "Edge"
        elif entity_type in ["facets", "facet", "faces", "face"]:
            entities = mesh.facets
            name = "Facet"
        elif entity_type in ["bodies", "body"]:
            entities = mesh.bodies
            name = "Body"
        elif entity_type == "energy":
            if len(args) > 1 and args[1].lower() in {"breakdown", "details"}:
                breakdown = context.minimizer.compute_energy_breakdown()
                total = sum(breakdown.values())
                print(f"Current Total Energy: {total:.10f}")
                for name, value in breakdown.items():
                    print(f"  {name}: {value:.10f}")
            else:
                E = context.minimizer.compute_energy()
                print(f"Current Total Energy: {E:.10f}")
            return
        elif entity_type in ["macros", "macro"]:
            macros = getattr(mesh, "macros", {}) or {}
            if not macros:
                print("No macros defined.")
                return
            print("Macros:")
            for name, steps in macros.items():
                if isinstance(steps, list):
                    body = "; ".join(str(step) for step in steps)
                else:
                    body = str(steps)
                print(f"  {name}: {body}")
            return
        else:
            print(f"Unknown entity type: {entity_type}")
            return

        # Mode 1: print entity ID
        if len(args) == 2 and args[1].isdigit():
            idx = int(args[1])
            if idx in entities:
                print(f"{name} {idx}: {entities[idx]}")
                if hasattr(entities[idx], "options"):
                    print(f"  Options: {entities[idx].options}")
                if hasattr(entities[idx], "position"):
                    print(f"  Position: {entities[idx].position}")
            else:
                print(f"{name} {idx} not found.")
            return

        # Mode 2: print all or filter
        targets = entities.items()

        if len(args) >= 4:
            # prop operator value
            prop = args[1]
            op = args[2]
            val_str = args[3]

            try:
                val = float(val_str)
            except ValueError:
                val = val_str

            def get_val(obj, key):
                if hasattr(obj, key):
                    return getattr(obj, key)
                if hasattr(obj, "options") and obj.options and key in obj.options:
                    return obj.options[key]
                if key == "len" and hasattr(obj, "compute_length"):
                    return obj.compute_length(mesh)
                if key == "area" and hasattr(obj, "compute_area"):
                    return obj.compute_area(mesh)
                return None

            filtered = []
            for k, obj in targets:
                v = get_val(obj, prop)
                if v is None:
                    continue

                match = False
                if op == ">":
                    match = v > val
                elif op == "<":
                    match = v < val
                elif op == ">=":
                    match = v >= val
                elif op == "<=":
                    match = v <= val
                elif op == "==" or op == "=":
                    match = v == val
                elif op == "!=":
                    match = v != val

                if match:
                    filtered.append((k, obj))
            targets = filtered
            print(f"Found {len(targets)} {entity_type} matching filter.")

        print(f"List of {entity_type} ({len(targets)}):")
        for k, obj in list(targets)[:20]:
            info = ""
            if entity_type.startswith("edge"):
                info = f"len={obj.compute_length(mesh):.4f}"
            elif entity_type.startswith("facet"):
                info = f"area={obj.compute_area(mesh):.4f}"
            print(f"  [{k}]: {info} {obj.options if hasattr(obj, 'options') else ''}")
        if len(targets) > 20:
            print("  ... (showing first 20)")


class HistoryCommand(Command):
    """Prints command history."""

    def execute(self, context, args):
        history = getattr(context, "history", None)
        if not history:
            print("No command history recorded.")
            return
        print("Command history:")
        for idx, line in enumerate(history, start=1):
            print(f"  {idx}: {line}")

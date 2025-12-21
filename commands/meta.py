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
        print("  tX            Set step size to X (e.g. t1e-3)")
        print("  r             Refine mesh (triangle refinement + polygonal)")
        print("  V / Vn        Vertex averaging once or n times (e.g. V5)")
        print("  vertex_average Same as V")
        print("  u             Equiangulate mesh")
        print("  visualize / s Plot current geometry")
        print("  properties    Print physical properties (area, volume, etc.)")
        print(
            "  print [entity] [id|filter] Query geometry (e.g. print edges len > 0.1)"
        )
        print(
            "  set [param] [value]       Set global parameter (e.g. set surface_tension 1.0)"
        )
        print(
            "                            or entity property (e.g. set vertex 0 fixed true)"
        )
        print("  live_vis / lv Turn on/off live visualization during minimization")
        print("  save          Save geometry to 'interactive.temp'")
        print("  quit / exit / q  Leave interactive mode")


class SetCommand(Command):
    """Refactored logic for 'set'."""

    def execute(self, context, args):
        # args expected to be tokens after 'set', e.g. ['vertex', '0', 'fixed', 'true']
        if len(args) < 2:
            print("Usage: set [param] [value] OR set [entity] [id] [prop] [value]")
            return

        mesh = context.mesh
        first = args[0].lower()

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
            try:
                idx = int(args[1])
            except ValueError:
                print("ID must be an integer.")
                return

            prop = args[2]
            val_str = args[3]

            # Helper to interpret value
            val_lower = val_str.lower()
            if val_lower == "true":
                val = True
            elif val_lower == "false":
                val = False
            elif val_lower in {"none", "null"}:
                val = None
            else:
                try:
                    val = float(val_str)
                except ValueError:
                    val = val_str

            # Find entity
            obj = None
            if entity_type.startswith("vert"):
                obj = mesh.vertices.get(idx)
            elif entity_type.startswith("edge"):
                obj = mesh.edges.get(idx)
            elif entity_type.startswith("facet") or entity_type.startswith("face"):
                obj = mesh.facets.get(idx)
            elif entity_type.startswith("body"):
                obj = mesh.bodies.get(idx)

            if not obj:
                print(f"Entity {idx} not found.")
                return

            # Set property
            if prop == "fixed":
                obj.fixed = bool(val)
                print(f"Set {entity_type} {idx} fixed={obj.fixed}")
            elif entity_type.startswith("body") and prop == "target_volume":
                obj.target_volume = None if val is None else float(val)
                if not obj.options:
                    obj.options = {}
                obj.options["target_volume"] = obj.target_volume
                print(f"Set {entity_type} {idx} target_volume={obj.target_volume}")
            else:
                if not obj.options:
                    obj.options = {}
                obj.options[prop] = val
                print(f"Set {entity_type} {idx} options['{prop}'] = {val}")

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

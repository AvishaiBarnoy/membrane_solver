# geometry_io.py
import json
import logging

import numpy as np
import yaml

from core.expr_eval import eval_expr
from core.ordered_unique_list import OrderedUniqueList
from core.parameters.global_parameters import GlobalParameters
from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from runtime.refinement import refine_polygonal_facets

logger = logging.getLogger("membrane_solver")

_CONSTRAINT_NAME_ALIASES = {
    "pin_surface_group_to_shape": "pin_to_plane",
}

_PIN_TO_PLANE_KEY_ALIASES = {
    "pin_surface_group_to_shape_mode": "pin_to_plane_mode",
    "pin_surface_group_to_shape_group": "pin_to_plane_group",
    "pin_surface_group_to_shape_normal": "pin_to_plane_normal",
    "pin_surface_group_to_shape_point": "pin_to_plane_point",
}


def load_data(filename):
    """Load geometry from a JSON file.

    Expected JSON or YAML format:
    {
        "vertices": [[x, y, z], ...],
        "faces": [
            [i, j, k, ...] or [i, j, k, ..., {"refine": false, "surface_tension": 0.8}],
            ...
        ]
    }"""
    filename_str = str(filename)
    with open(filename_str, "r") as f:
        if filename_str.endswith((".yaml", ".yml")):
            data = yaml.safe_load(f)
        elif filename_str.endswith(".json"):
            data = json.load(f)
        else:
            logger.error(f"Unsupported file format for: {filename_str}")
            raise ValueError(f"Unsupported file format for: {filename_str}")

    return data


def parse_geometry(data: dict) -> Mesh:
    mesh = Mesh()

    # Initialize global parameters
    mesh.global_parameters = GlobalParameters()

    def _canonical_constraint_name(name: str) -> str:
        canonical = _CONSTRAINT_NAME_ALIASES.get(name, name)
        if canonical != name:
            logger.info(
                "Constraint alias '%s' mapped to '%s'.",
                name,
                canonical,
            )
        return canonical

    def _normalize_constraint_names(raw_constraints) -> list[str]:
        if raw_constraints is None:
            return []
        if isinstance(raw_constraints, str):
            values = [raw_constraints]
        elif isinstance(raw_constraints, list):
            values = list(raw_constraints)
        else:
            err_msg = "constraint modules should be in a list or a single string"
            logger.error(err_msg)
            raise TypeError(err_msg)
        return [_canonical_constraint_name(str(value)) for value in values]

    def _apply_pin_to_plane_aliases(options: dict) -> dict:
        if not isinstance(options, dict):
            return options
        for alias_key, canonical_key in _PIN_TO_PLANE_KEY_ALIASES.items():
            if alias_key not in options:
                continue
            if canonical_key not in options:
                options[canonical_key] = options[alias_key]
            options.pop(alias_key, None)
        return options

    # Override global parameters with values from the input file.
    input_global_params = dict(data.get("global_parameters", {}) or {})
    _apply_pin_to_plane_aliases(input_global_params)
    mesh.global_parameters.update(input_global_params)

    def _coerce_float_param(key: str) -> None:
        """Coerce numeric global parameters that may parse as strings in YAML."""
        val = mesh.global_parameters.get(key)
        if isinstance(val, str):
            try:
                mesh.global_parameters.set(key, float(val))
            except ValueError:
                logger.warning(
                    "global_parameters.%s should be numeric; got %r", key, val
                )

    for _key in (
        "surface_tension",
        "volume_stiffness",
        "volume_tolerance",
        "step_size",
        "step_size_floor",
        "intrinsic_curvature",
        "bending_modulus",
        "gaussian_modulus",
        "line_tension",
    ):
        _coerce_float_param(_key)

    def _numeric_value(value):
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value.strip())
            except ValueError:
                return None
        return None

    def _build_define_names(extra: dict | None = None) -> dict:
        names = {}
        for key, val in mesh.global_parameters.to_dict().items():
            num = _numeric_value(val)
            if num is not None:
                names[key] = num
        if extra:
            names.update(extra)
        return names

    def _evaluate_defines(defines: dict | None) -> None:
        if not defines:
            return
        if not isinstance(defines, dict):
            raise TypeError("defines must be a mapping of name -> expression")
        pending = dict(defines)
        resolved: dict[str, float] = {}
        for _ in range(len(pending) + 1):
            progress = False
            for key, expr in list(pending.items()):
                num = _numeric_value(expr)
                if num is not None:
                    resolved[key] = num
                    pending.pop(key)
                    progress = True
                    continue
                if not isinstance(expr, str):
                    raise TypeError(
                        f"define {key!r} must be a number or expression string"
                    )
                try:
                    val = eval_expr(expr, _build_define_names(resolved))
                except Exception as exc:  # pragma: no cover - defensive
                    if isinstance(exc, ValueError) and "Unknown name" in str(exc):
                        continue
                    raise ValueError(
                        f"Invalid define expression for {key!r}: {exc}"
                    ) from exc
                resolved[key] = float(val)
                pending.pop(key)
                progress = True
            if not progress:
                break
        if pending:
            missing = ", ".join(sorted(pending))
            raise ValueError(f"Could not resolve defines: {missing}")
        for key, val in resolved.items():
            mesh.global_parameters.set(key, val)

    _evaluate_defines(data.get("defines"))

    def _parse_id(value, *, label: str) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            text = value.strip()
            if text.lstrip("-").isdigit():
                return int(text)
        raise TypeError(
            f"{label} IDs must be integers (or integer strings); got {value!r}"
        )

    # Stabilise volume constraint defaults: enforce complementary pairs.
    has_mode = "volume_constraint_mode" in input_global_params
    has_proj = "volume_projection_during_minimization" in input_global_params
    if not has_mode and not has_proj:
        mesh.global_parameters.set("volume_constraint_mode", "lagrange")
        mesh.global_parameters.set("volume_projection_during_minimization", False)
    elif has_mode and not has_proj:
        mode = str(
            mesh.global_parameters.get("volume_constraint_mode", "lagrange")
        ).lower()
        proj = False if mode == "lagrange" else True
        mesh.global_parameters.set("volume_projection_during_minimization", proj)
    elif has_proj and not has_mode:
        proj = bool(
            mesh.global_parameters.get("volume_projection_during_minimization", True)
        )
        mode = "penalty" if proj else "lagrange"
        mesh.global_parameters.set("volume_constraint_mode", mode)

    # Warn about unstable combinations.
    mode = str(mesh.global_parameters.get("volume_constraint_mode", "lagrange")).lower()
    proj_flag = bool(
        mesh.global_parameters.get("volume_projection_during_minimization", False)
    )
    if mode == "lagrange" and proj_flag:
        logger.warning(
            "volume_constraint_mode='lagrange' with volume_projection_during_minimization=True "
            "is known to be unstable; consider disabling projection."
        )
    if mode == "penalty" and not proj_flag:
        logger.warning(
            "volume_constraint_mode='penalty' without geometric projection is not supported; "
            "consider enabling volume_projection_during_minimization."
        )

    # Initialize module-name list with deterministic order. Keep only the
    # first occurrence of each module to avoid duplicates without hash-order
    # non-determinism across processes.
    energy_module_names = OrderedUniqueList(data.get("energy_modules", []))
    # If the input doesn't specify any modules but has surface tension,
    # default to 'surface' for backward compatibility.
    if (
        not energy_module_names
        and mesh.global_parameters.get("surface_tension", 0.0) > 0
    ):
        energy_module_names.append("surface")

    # Allow explicit constraint modules at the top level (e.g. "global_area")
    # in addition to those inferred from per‑entity "constraints" options.
    constraint_module_names = OrderedUniqueList(
        _normalize_constraint_names(data.get("constraint_modules", []))
    )
    # If the input specifies a global target surface area, automatically load
    # the corresponding constraint so users do not have to list the module
    # manually.
    if mesh.global_parameters.get("target_surface_area") is not None:
        constraint_module_names.append("global_area")

    definitions = data.get("definitions", {})
    mesh.definitions = definitions.copy() if isinstance(definitions, dict) else {}

    def resolve_options(raw_options):
        if not raw_options:
            return {}
        preset_name = raw_options.get("preset")
        if preset_name:
            if preset_name not in definitions:
                raise ValueError(f"Preset '{preset_name}' not found in definitions.")
            # Merge: preset first, then raw_options overrides
            merged = definitions[preset_name].copy()
            merged.update(raw_options)
            # We keep the 'preset' key so we can filter entities by preset later.
            if "preset" not in merged:
                merged["preset"] = preset_name
            return _apply_pin_to_plane_aliases(merged)
        return _apply_pin_to_plane_aliases(raw_options)

    def normalize_constraints(options: dict, *, fixed_setter) -> list[str]:
        """Normalize an entity 'constraints' option to a list and handle 'fixed'.

        The string 'fixed' is treated as a structural flag (sets entity.fixed)
        and is not kept as a constraint module name.
        """
        raw = options.get("constraints")
        if raw is None:
            if options.get("fixed", False):
                fixed_setter(True)
            return []

        constraints = _normalize_constraint_names(raw)

        if "fixed" in constraints:
            fixed_setter(True)
            constraints = [c for c in constraints if c != "fixed"]

        if constraints:
            options["constraints"] = constraints
        else:
            options.pop("constraints", None)

        if options.get("fixed", False):
            fixed_setter(True)

        return constraints

    # Vertices
    vertices = data.get("vertices") or data.get("Vertices")
    if vertices is None:
        raise ValueError("Geometry file must contain 'vertices'")

    if isinstance(vertices, dict):
        vertex_items = []
        for raw_vid, entry in vertices.items():
            vid = _parse_id(raw_vid, label="vertex")
            vertex_items.append((vid, entry))
        vertex_items.sort(key=lambda item: item[0])
    else:
        vertex_items = list(enumerate(vertices))

    for vid, entry in vertex_items:
        *position, raw_opts = entry if isinstance(entry[-1], dict) else (*entry, {})
        options = resolve_options(raw_opts)

        pos_array = np.asarray(position, dtype=float)
        if np.any(np.isnan(pos_array)):
            raise ValueError(f"Vertex {vid} has NaN coordinates.")
        if np.any(np.isinf(pos_array)):
            raise ValueError(f"Vertex {vid} has infinite coordinates.")

        tilt_fixed_val = options.get("tilt_fixed", options.get("fixed_tilt", False))
        if isinstance(tilt_fixed_val, str):
            tilt_fixed_val = tilt_fixed_val.strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
                "on",
            )
        tilt_fixed_in_val = options.get("tilt_fixed_in", False)
        tilt_fixed_out_val = options.get("tilt_fixed_out", False)
        if isinstance(tilt_fixed_in_val, str):
            tilt_fixed_in_val = tilt_fixed_in_val.strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
                "on",
            )
        if isinstance(tilt_fixed_out_val, str):
            tilt_fixed_out_val = tilt_fixed_out_val.strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
                "on",
            )
        options.pop("tilt_fixed", None)
        options.pop("fixed_tilt", None)
        options.pop("tilt_fixed_in", None)
        options.pop("tilt_fixed_out", None)

        raw_tilt = options.get("tilt")
        if raw_tilt is not None:
            if (
                not isinstance(raw_tilt, (list, tuple))
                or len(raw_tilt) not in (2, 3)
                or not all(isinstance(val, (int, float)) for val in raw_tilt)
            ):
                raise TypeError(
                    f"Vertex {vid} tilt must be a 2- or 3-vector of numbers; got {raw_tilt!r}"
                )
        raw_tilt_in = options.get("tilt_in")
        if raw_tilt_in is not None:
            if (
                not isinstance(raw_tilt_in, (list, tuple))
                or len(raw_tilt_in) not in (2, 3)
                or not all(isinstance(val, (int, float)) for val in raw_tilt_in)
            ):
                raise TypeError(
                    f"Vertex {vid} tilt_in must be a 2- or 3-vector of numbers; got {raw_tilt_in!r}"
                )
        raw_tilt_out = options.get("tilt_out")
        if raw_tilt_out is not None:
            if (
                not isinstance(raw_tilt_out, (list, tuple))
                or len(raw_tilt_out) not in (2, 3)
                or not all(isinstance(val, (int, float)) for val in raw_tilt_out)
            ):
                raise TypeError(
                    f"Vertex {vid} tilt_out must be a 2- or 3-vector of numbers; got {raw_tilt_out!r}"
                )

        def _tilt_to_array(raw, name):
            if raw is None:
                return None
            arr = np.asarray(raw, dtype=float)
            if arr.shape == (2,):
                arr = np.array([arr[0], arr[1], 0.0], dtype=float)
            elif arr.shape != (3,):
                raise ValueError(
                    f"Vertex {vid} {name} must have length 2 or 3; got {arr!r}"
                )
            return arr

        tilt_arr = _tilt_to_array(raw_tilt, "tilt")
        tilt_in_arr = _tilt_to_array(raw_tilt_in, "tilt_in")
        tilt_out_arr = _tilt_to_array(raw_tilt_out, "tilt_out")

        vertex = Vertex(
            index=vid,
            position=pos_array,
            options=options,
            tilt_fixed=bool(tilt_fixed_val),
            tilt_fixed_in=bool(tilt_fixed_in_val),
            tilt_fixed_out=bool(tilt_fixed_out_val),
            tilt=tilt_arr if tilt_arr is not None else np.zeros(3, dtype=float),
            tilt_in=tilt_in_arr
            if tilt_in_arr is not None
            else np.zeros(3, dtype=float),
            tilt_out=tilt_out_arr
            if tilt_out_arr is not None
            else np.zeros(3, dtype=float),
        )
        mesh.vertices[vid] = vertex

        if "energy" in options:
            if isinstance(options["energy"], list):
                energy_module_names.update(options["energy"])
            elif isinstance(options["energy"], str):
                energy_module_names.add(options["energy"])
            else:
                err_msg = "energy modules should be in a list or a single string"
                logger.error(err_msg)
                raise err_msg
        # Expression energy auto-enable.
        if (
            options.get("expression")
            or options.get("energy_expression")
            or options.get("expr")
        ):
            if "energy" not in options:
                options["energy"] = ["expression"]
            elif isinstance(options["energy"], list):
                if "expression" not in options["energy"]:
                    options["energy"].append("expression")
            elif isinstance(options["energy"], str):
                if options["energy"] != "expression":
                    options["energy"] = [options["energy"], "expression"]
            energy_module_names.add("expression")

        # Uncomment to add a default energy moduel to Vertices
        # elif "energy" not in options:
        # mesh.vertices[i].options["energy"] = ["surface"]

        # Vertex constraint modules
        constraints = normalize_constraints(
            mesh.vertices[vid].options,
            fixed_setter=lambda flag, idx=vid: setattr(
                mesh.vertices[idx], "fixed", flag
            ),
        )
        constraint_module_names.extend(constraints)
        if (
            mesh.vertices[vid].options.get("constraint_expression") is not None
            or mesh.vertices[vid].options.get("expression_constraint") is not None
        ):
            if "expression" not in constraints:
                constraints.append("expression")
                mesh.vertices[vid].options["constraints"] = constraints
                constraint_module_names.append("expression")

    # Edges
    edges = data.get("edges") or data.get("Edges")
    if edges is None:
        err_msg = "Input geometry is missing required 'edges' section."
        logger.error(err_msg)
        raise KeyError(err_msg)

    edges_are_explicit = isinstance(edges, dict)
    if edges_are_explicit:
        edge_items = []
        for raw_eid, entry in edges.items():
            eid = _parse_id(raw_eid, label="edge")
            edge_items.append((eid, entry))
        edge_items.sort(key=lambda item: item[0])
    else:
        edge_items = [(i + 1, entry) for i, entry in enumerate(edges)]

    for eid, entry in edge_items:
        tail_index, head_index, *opts = entry
        tail_index = _parse_id(tail_index, label="vertex")
        head_index = _parse_id(head_index, label="vertex")

        if tail_index not in mesh.vertices:
            raise ValueError(f"Edge {eid} references missing tail vertex {tail_index}")
        if head_index not in mesh.vertices:
            raise ValueError(f"Edge {eid} references missing head vertex {head_index}")

        raw_opts = opts[0] if opts else {}
        options = resolve_options(raw_opts)
        mesh.edges[eid] = Edge(
            index=eid, tail_index=tail_index, head_index=head_index, options=options
        )

        if "energy" in options:
            if isinstance(options["energy"], list):
                energy_module_names.update(options["energy"])

            elif isinstance(options["energy"], str):
                energy_module_names.add(options["energy"])
            else:
                err_msg = "energy modules should be in a list or a single string"
                logger.error(err_msg)
                raise err_msg
        # Uncomment to add a default energy moduel for Edges
        # elif "energy" not in options:
        # mesh.edges[i+1].options["energy"] = ["surface"]

        if (
            options.get("expression")
            or options.get("energy_expression")
            or options.get("expr")
        ):
            if "energy" not in options:
                options["energy"] = ["expression"]
            elif isinstance(options["energy"], list):
                if "expression" not in options["energy"]:
                    options["energy"].append("expression")
            elif isinstance(options["energy"], str):
                if options["energy"] != "expression":
                    options["energy"] = [options["energy"], "expression"]
            energy_module_names.add("expression")

        # Edges constraint modules
        constraints = normalize_constraints(
            mesh.edges[eid].options,
            fixed_setter=lambda flag, idx=eid: setattr(mesh.edges[idx], "fixed", flag),
        )
        constraint_module_names.extend(constraints)
        if (
            mesh.edges[eid].options.get("constraint_expression") is not None
            or mesh.edges[eid].options.get("expression_constraint") is not None
        ):
            if "expression" not in constraints:
                constraints.append("expression")
                mesh.edges[eid].options["constraints"] = constraints
                constraint_module_names.append("expression")

        if mesh.edges[eid].fixed:
            mesh.vertices[tail_index].fixed = True
            mesh.vertices[head_index].fixed = True

    # Facets (optional for line‑only geometries)
    faces_section = data.get("faces") or data.get("Faces") or data.get("Facets") or []
    faces_are_explicit = isinstance(faces_section, dict)
    if faces_are_explicit:
        face_items = []
        for raw_fid, entry in faces_section.items():
            fid = _parse_id(raw_fid, label="face")
            face_items.append((fid, entry))
        face_items.sort(key=lambda item: item[0])
    else:
        face_items = list(enumerate(faces_section))

    def parse_edge_ref(e):
        if edges_are_explicit:
            if isinstance(e, str) and e.startswith("r"):
                return -_parse_id(e[1:], label="edge")
            return _parse_id(e, label="edge")
        if isinstance(e, str) and e.startswith("r"):
            return -(int(e[1:]) + 1)  # "r0" -> -1
        i = int(e)
        if i >= 0:
            return i + 1  # 0 -> 1, 1 -> 2, etc.
        return i - 1  # -11 -> -12

    for fid, entry in face_items:
        *raw_edges, raw_opts = entry if isinstance(entry[-1], dict) else (*entry, {})
        options = resolve_options(raw_opts)

        edge_indices = [parse_edge_ref(e) for e in raw_edges]
        mesh.facets[fid] = Facet(index=fid, edge_indices=edge_indices, options=options)

        if "energy" in options:
            if isinstance(options["energy"], list):
                energy_module_names.update(options["energy"])
            elif isinstance(options["energy"], str):
                energy_module_names.add(options["energy"])
                mesh.facets[fid].options["energy"] = [
                    mesh.facets[fid].options["energy"]
                ]
            else:
                err_msg = "energy modules should be in a list or a single string"
                logger.error(err_msg)
                raise err_msg
        elif "energy" not in options:
            surface_tension = options.get(
                "surface_tension", mesh.global_parameters.get("surface_tension", 0.0)
            )
            try:
                surface_tension_value = float(surface_tension)
            except (TypeError, ValueError):
                surface_tension_value = 0.0

            # Only enable surface energy by default when it would contribute.
            if surface_tension_value != 0.0:
                mesh.facets[fid].options["energy"] = ["surface"]
                energy_module_names.add("surface")
            else:
                mesh.facets[fid].options["energy"] = []

        if (
            options.get("expression")
            or options.get("energy_expression")
            or options.get("expr")
        ):
            if "energy" not in options:
                options["energy"] = ["expression"]
            elif isinstance(options["energy"], list):
                if "expression" not in options["energy"]:
                    options["energy"].append("expression")
            elif isinstance(options["energy"], str):
                if options["energy"] != "expression":
                    options["energy"] = [options["energy"], "expression"]
            energy_module_names.add("expression")

        # Ensure all facets have surface_tension set
        if "surface_tension" not in options:
            mesh.facets[fid].options["surface_tension"] = mesh.global_parameters.get(
                "surface_tension", 1.0
            )

        # Facets constraint modules
        facet_constraints = normalize_constraints(
            mesh.facets[fid].options,
            fixed_setter=lambda flag, idx=fid: setattr(mesh.facets[idx], "fixed", flag),
        )

        if options.get("target_area") is not None:
            if not facet_constraints:
                facet_constraints = []
            if "fix_facet_area" not in facet_constraints:
                facet_constraints.append("fix_facet_area")
                mesh.facets[fid].options["constraints"] = facet_constraints

        if facet_constraints:
            constraint_module_names.extend(facet_constraints)
        if (
            mesh.facets[fid].options.get("constraint_expression") is not None
            or mesh.facets[fid].options.get("expression_constraint") is not None
        ):
            if "expression" not in facet_constraints:
                facet_constraints.append("expression")
                mesh.facets[fid].options["constraints"] = facet_constraints
                constraint_module_names.append("expression")

    vol_mode = mesh.global_parameters.get("volume_constraint_mode", "lagrange")
    if vol_mode == "penalty":
        energy_module_names.add("volume")

    # Bodies
    bodies_section = data.get("bodies") or data.get("Bodies")
    if bodies_section:
        explicit_body_map = (
            isinstance(bodies_section, dict)
            and "faces" not in bodies_section
            and all(
                isinstance(spec, dict) and "faces" in spec
                for spec in bodies_section.values()
            )
        )
        if explicit_body_map:
            for raw_bid, spec in bodies_section.items():
                bid = _parse_id(raw_bid, label="body")
                if not isinstance(spec, dict):
                    raise TypeError("body specs must be mappings")
                facet_indices = spec.get("faces")
                if not isinstance(facet_indices, list):
                    raise TypeError("body.faces must be a list of face IDs")
                facet_indices = [_parse_id(fid, label="face") for fid in facet_indices]

                body_options = {k: v for k, v in spec.items() if k not in {"faces"}}
                target_volume = body_options.pop("target_volume", None)
                target_area = body_options.get("target_area")
                if target_area is not None:
                    body_options["target_area"] = float(target_area)

                body = Body(
                    index=bid,
                    facet_indices=facet_indices,
                    target_volume=target_volume,
                    options=body_options,
                )
                if target_volume is not None:
                    if (
                        isinstance(target_volume, str)
                        and target_volume.lower() == "initial"
                    ):
                        body.options["target_volume"] = body.compute_volume(mesh)
                    else:
                        body.options["target_volume"] = float(target_volume)

                mesh.bodies[bid] = body

                # Load energy modules defined on the body
                energy_spec = body.options.get("energy")
                if energy_spec:
                    if isinstance(energy_spec, list):
                        energy_module_names.update(energy_spec)
                    elif isinstance(energy_spec, str):
                        energy_module_names.add(energy_spec)

                constraint_spec = body.options.get("constraints", [])
                body_constraints = _normalize_constraint_names(constraint_spec)

                if (
                    target_volume is not None
                    and vol_mode == "lagrange"
                    and "volume" not in body_constraints
                ):
                    body_constraints.append("volume")

                if (
                    body.options.get("target_area") is not None
                    and "body_area" not in body_constraints
                ):
                    body_constraints.append("body_area")

                if body_constraints:
                    body.options["constraints"] = body_constraints
                    constraint_module_names.extend(body_constraints)
                if (
                    body.options.get("constraint_expression") is not None
                    or body.options.get("expression_constraint") is not None
                ):
                    if "expression" not in body_constraints:
                        body_constraints.append("expression")
                        body.options["constraints"] = body_constraints
                        constraint_module_names.append("expression")

                if (
                    body.options.get("expression")
                    or body.options.get("energy_expression")
                    or body.options.get("expr")
                ):
                    energy_module_names.add("expression")
            # Skip legacy block.
            bodies_section = None

    if bodies_section:
        face_groups = bodies_section["faces"]
        volumes = bodies_section.get("target_volume", [None] * len(face_groups))
        areas = bodies_section.get("target_area", [None] * len(face_groups))

        # ``energy`` may be:
        #   - a list parallel to ``faces`` (per‑body specs), or
        #   - a single string/dict applying to all bodies.
        energy_entries = bodies_section.get("energy", [None] * len(face_groups))
        if not isinstance(energy_entries, list) or len(energy_entries) != len(
            face_groups
        ):
            energy_entries = [energy_entries] * len(face_groups)

        constraint_entries = bodies_section.get(
            "constraints", [None] * len(face_groups)
        )
        if not isinstance(constraint_entries, list) or len(constraint_entries) != len(
            face_groups
        ):
            constraint_entries = [constraint_entries] * len(face_groups)

        for i, (facet_indices, volume, area, energy_spec, constraint_spec) in enumerate(
            zip(face_groups, volumes, areas, energy_entries, constraint_entries)
        ):
            # Start with an options dict derived from the energy specification.
            body_options = {}
            if isinstance(energy_spec, dict):
                body_options.update(energy_spec)
            elif energy_spec is not None:
                body_options["energy"] = energy_spec

            # Merge root-level constraints (if provided)
            merged_constraints = []
            if constraint_spec is not None:
                merged_constraints = _normalize_constraint_names(constraint_spec)
            if merged_constraints:
                existing = body_options.get("constraints")
                if existing is None:
                    body_options["constraints"] = merged_constraints
                else:
                    if isinstance(existing, str):
                        existing = [existing]
                    body_options["constraints"] = list(
                        dict.fromkeys(list(existing) + merged_constraints)
                    )

            # Merge root-level target_area if present
            if area is not None and "target_area" not in body_options:
                body_options["target_area"] = float(area)

            body = Body(
                index=i,
                facet_indices=facet_indices,
                target_volume=volume,
                options=body_options,
            )
            if volume is not None:
                if isinstance(volume, str) and volume.lower() == "initial":
                    body.options["target_volume"] = body.compute_volume(mesh)
                else:
                    body.options["target_volume"] = float(volume)

            mesh.bodies[i] = body

            # Energy modules (opt‑in). If omitted, bodies have no explicit
            # volume penalty term and are expected to be governed by hard
            # constraints instead.
            energy_spec = body.options.get("energy")
            if energy_spec is not None:
                if isinstance(energy_spec, list):
                    energy_module_names.update(energy_spec)
                elif isinstance(energy_spec, str):
                    energy_module_names.add(energy_spec)
                    body.options["energy"] = [energy_spec]
                else:
                    err_msg = "energy modules should be in a list or a single string"
                    logger.error(err_msg)
                    raise err_msg
            if (
                body.options.get("expression")
                or body.options.get("energy_expression")
                or body.options.get("expr")
            ):
                energy_module_names.add("expression")

            # Body constraint modules. If a target volume is specified,
            # automatically enable the volume constraint module so bodies
            # behave like FIXEDVOL in Evolver, on top of any explicit
            # constraints configured.
            constraint_spec = body.options.get("constraints", [])
            body_constraints = _normalize_constraint_names(constraint_spec)

            if (
                volume is not None
                and vol_mode == "lagrange"
                and "volume" not in body_constraints
            ):
                body_constraints.append("volume")

            if (
                body.options.get("target_area") is not None
                and "body_area" not in body_constraints
            ):
                body_constraints.append("body_area")

            if body_constraints:
                body.options["constraints"] = body_constraints
                constraint_module_names.extend(body_constraints)
            if (
                body.options.get("constraint_expression") is not None
                or body.options.get("expression_constraint") is not None
            ):
                if "expression" not in body_constraints:
                    body_constraints.append("expression")
                    body.options["constraints"] = body_constraints
                    constraint_module_names.append("expression")

    # Instructions
    mesh.instructions = data.get("instructions", [])

    def _split_command_list(text: str) -> list[str]:
        parts: list[str] = []
        for chunk in text.replace("\n", ";").split(";"):
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk)
        return parts

    def _normalize_macro_body(value) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return _split_command_list(value)
        if isinstance(value, list):
            lines: list[str] = []
            for item in value:
                if isinstance(item, str):
                    lines.extend(_split_command_list(item))
                else:
                    raise TypeError("macro entries must be strings")
            return lines
        raise TypeError("macros must be a string or a list of strings")

    raw_macros = data.get("macros", {}) or {}
    if not isinstance(raw_macros, dict):
        raise TypeError("macros must be a mapping of name -> command string/list")
    macros: dict[str, list[str]] = {}
    for name, body in raw_macros.items():
        if not isinstance(name, str) or not name.strip():
            raise TypeError("macro names must be non-empty strings")
        macros[name.strip()] = _normalize_macro_body(body)
    mesh.macros = macros

    # Energy modules
    mesh.energy_modules = OrderedUniqueList(energy_module_names)

    # Constraint modules
    mesh.constraint_modules = OrderedUniqueList(constraint_module_names)

    def _strip_tilt_options(target: Mesh) -> None:
        for vertex in target.vertices.values():
            opts = getattr(vertex, "options", None)
            if isinstance(opts, dict):
                opts.pop("tilt", None)
                opts.pop("tilt_fixed", None)
                opts.pop("fixed_tilt", None)
                opts.pop("tilt_in", None)
                opts.pop("tilt_out", None)
                opts.pop("tilt_fixed_in", None)
                opts.pop("tilt_fixed_out", None)

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    mesh.initialize_tilts_from_options()

    # Basic validation (connectivity)
    try:
        mesh.validate_edge_indices()
    except Exception as e:
        logger.error(f"Mesh connectivity validation failed: {e}")
        raise

    # Automatically triangulate polygonal facets if needed
    if any(len(f.edge_indices) > 3 for f in mesh.facets.values()):
        refined = refine_polygonal_facets(mesh)
        refined.definitions = getattr(mesh, "definitions", {}).copy()
        refined.initialize_tilts_from_options()
        _strip_tilt_options(refined)
        try:
            refined.full_mesh_validate()
        except Exception as e:
            logger.error(f"Refined mesh validation failed: {e}")
            raise
        return refined

    _strip_tilt_options(mesh)
    try:
        mesh.full_mesh_validate()
    except Exception as e:
        logger.error(f"Mesh validation failed: {e}")
        raise

    return mesh


def save_geometry(
    mesh: Mesh,
    path: str = "outputs/temp_output_file.json",
    *,
    compact: bool = False,
):
    def _to_builtin(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {k: _to_builtin(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_to_builtin(v) for v in value]
        return value

    def _sorted_keys(dct):
        return sorted(dct.keys())

    # IMPORTANT: The on-disk format encodes indices implicitly by list order:
    # - vertices are 0..N-1 by position in "vertices" list
    # - edges are 1..E by position in "edges" list (with 0-based references in faces)
    # - faces are 0..F-1 by position in "faces" list
    #
    # In-memory meshes can have sparse/non-contiguous IDs after refinement or
    # equiangulation (e.g. deleting edge 10 and creating edge 500). When saving,
    # reindex all entities to a compact, contiguous numbering so that a
    # save→load roundtrip produces a valid mesh.
    vertex_ids = _sorted_keys(mesh.vertices)
    vertex_id_map = {old: new for new, old in enumerate(vertex_ids)}

    edge_ids = _sorted_keys(mesh.edges)
    edge_id_map = {old: new + 1 for new, old in enumerate(edge_ids)}  # 1-based

    facet_ids = _sorted_keys(mesh.facets)
    facet_id_map = {old: new for new, old in enumerate(facet_ids)}  # 0-based
    pos_view = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    def export_edge_index(old_signed_edge_index: int):
        sign = -1 if old_signed_edge_index < 0 else 1
        old_abs = abs(int(old_signed_edge_index))
        if old_abs not in edge_id_map:
            raise KeyError(
                f"Cannot save geometry: facet references missing edge {old_signed_edge_index}."
            )
        new_abs = edge_id_map[old_abs]
        new_signed = sign * new_abs
        if new_signed < 0:
            return f"r{new_abs - 1}"  # -1 → "r0"
        return new_abs - 1  # 1 → 0

    def prepare_options(entity):
        opts = entity.options.copy() if entity.options else {}
        if entity.fixed:
            opts["fixed"] = True
        if isinstance(entity, Vertex):
            if hasattr(entity, "tilt") and np.any(entity.tilt):
                opts["tilt"] = entity.tilt.tolist()
            if hasattr(entity, "tilt_fixed") and entity.tilt_fixed:
                opts["tilt_fixed"] = True
            if hasattr(entity, "tilt_in") and np.any(entity.tilt_in):
                opts["tilt_in"] = entity.tilt_in.tolist()
            if hasattr(entity, "tilt_out") and np.any(entity.tilt_out):
                opts["tilt_out"] = entity.tilt_out.tolist()
            if hasattr(entity, "tilt_fixed_in") and entity.tilt_fixed_in:
                opts["tilt_fixed_in"] = True
            if hasattr(entity, "tilt_fixed_out") and entity.tilt_fixed_out:
                opts["tilt_fixed_out"] = True
        return opts if opts else None

    data = {
        "vertices": [
            (
                [
                    *(
                        pos_view[index_map[int(old_vid)]]
                        if int(old_vid) in index_map
                        else mesh.vertices[old_vid].position
                    ).tolist(),
                    prepare_options(mesh.vertices[old_vid]),
                ]
                if prepare_options(mesh.vertices[old_vid])
                else (
                    pos_view[index_map[int(old_vid)]]
                    if int(old_vid) in index_map
                    else mesh.vertices[old_vid].position
                ).tolist()
            )
            for old_vid in vertex_ids
        ],
        "edges": [
            (
                [
                    vertex_id_map[int(mesh.edges[old_eid].tail_index)],
                    vertex_id_map[int(mesh.edges[old_eid].head_index)],
                    prepare_options(mesh.edges[old_eid]),
                ]
                if prepare_options(mesh.edges[old_eid])
                else [
                    vertex_id_map[int(mesh.edges[old_eid].tail_index)],
                    vertex_id_map[int(mesh.edges[old_eid].head_index)],
                ]
            )
            for old_eid in edge_ids
        ],
        "faces": [
            (
                [
                    *map(
                        export_edge_index,
                        mesh.facets[old_fid].edge_indices,
                    ),
                    prepare_options(mesh.facets[old_fid]),
                ]
                if prepare_options(mesh.facets[old_fid])
                else list(map(export_edge_index, mesh.facets[old_fid].edge_indices))
            )
            for old_fid in facet_ids
        ],
        "bodies": {
            "faces": [
                [facet_id_map[int(fid)] for fid in mesh.bodies[b].facet_indices]
                for b in mesh.bodies.keys()
            ],
            "target_volume": [mesh.bodies[b].target_volume for b in mesh.bodies.keys()],
            "target_area": [
                mesh.bodies[b].options.get("target_area") for b in mesh.bodies.keys()
            ],
            "energy": [
                mesh.bodies[b].options.get("energy", {}) for b in mesh.bodies.keys()
            ],
            "constraints": [
                mesh.bodies[b].options.get("constraints", [])
                for b in mesh.bodies.keys()
            ],
        },
        "energy_modules": list(mesh.energy_modules),
        "constraint_modules": list(mesh.constraint_modules),
        "global_parameters": mesh.global_parameters.to_dict(),
        "instructions": mesh.instructions,
    }
    definitions = (
        mesh.definitions.copy()
        if isinstance(getattr(mesh, "definitions", None), dict)
        else {}
    )
    used_presets: set[str] = set()
    for vertex in mesh.vertices.values():
        preset = (vertex.options or {}).get("preset")
        if preset:
            used_presets.add(str(preset))
    for edge in mesh.edges.values():
        preset = (edge.options or {}).get("preset")
        if preset:
            used_presets.add(str(preset))
    for facet in mesh.facets.values():
        preset = (facet.options or {}).get("preset")
        if preset:
            used_presets.add(str(preset))
    if used_presets:
        for preset in used_presets:
            definitions.setdefault(preset, {})
        data["definitions"] = definitions
    elif definitions:
        data["definitions"] = definitions

    filename_str = str(path)
    with open(path, "w") as f:
        if filename_str.endswith((".yaml", ".yml")):
            yaml.safe_dump(
                _to_builtin(data),
                f,
                default_flow_style=bool(compact),
                sort_keys=False,
                allow_unicode=True,
            )
        else:
            if compact:
                json.dump(
                    _to_builtin(data), f, separators=(",", ":"), ensure_ascii=False
                )
            else:
                json.dump(_to_builtin(data), f, indent=4, ensure_ascii=False)

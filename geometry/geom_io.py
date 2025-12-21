# geometry_io.py
import json

import numpy as np
import yaml

from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from parameters.global_parameters import GlobalParameters
from runtime.logging_config import setup_logging
from runtime.refinement import refine_polygonal_facets

logger = setup_logging("membrane_solver.log")


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

    # Override global parameters with values from the input file
    input_global_params = data.get("global_parameters", {})
    mesh.global_parameters.update(input_global_params)

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

    # Initialize module_name list
    energy_module_names = set()
    # Allow explicit constraint modules at the top level (e.g. "global_area")
    # in addition to those inferred from per‑entity "constraints" options.
    constraint_module_names = list(data.get("constraint_modules", []))
    # If the input specifies a global target surface area, automatically load
    # the corresponding constraint so users do not have to list the module
    # manually.
    if mesh.global_parameters.get("target_surface_area") is not None:
        constraint_module_names.append("global_area")

    definitions = data.get("definitions", {})

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
            # Remove the 'preset' key from the final dict to avoid confusion
            merged.pop("preset", None)
            return merged
        return raw_options

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

        if isinstance(raw, str):
            constraints = [raw]
        elif isinstance(raw, list):
            constraints = list(raw)
        else:
            err_msg = "constraint modules should be in a list or a single string"
            logger.error(err_msg)
            raise TypeError(err_msg)

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

    for i, entry in enumerate(vertices):
        *position, raw_opts = entry if isinstance(entry[-1], dict) else (*entry, {})
        options = resolve_options(raw_opts)

        pos_array = np.asarray(position, dtype=float)
        if np.any(np.isnan(pos_array)):
            raise ValueError(f"Vertex {i} has NaN coordinates.")
        if np.any(np.isinf(pos_array)):
            raise ValueError(f"Vertex {i} has infinite coordinates.")

        mesh.vertices[i] = Vertex(index=i, position=pos_array, options=options)

        if "energy" in options:
            if isinstance(options["energy"], list):
                energy_module_names.update(options["energy"])
            elif isinstance(options["energy"], str):
                energy_module_names.add(options["energy"])
            else:
                err_msg = "energy modules should be in a list or a single string"
                logger.error(err_msg)
                raise err_msg
        # Uncomment to add a default energy moduel to Vertices
        # elif "energy" not in options:
        # mesh.vertices[i].options["energy"] = ["surface"]

        # Vertex constraint modules
        constraints = normalize_constraints(
            mesh.vertices[i].options,
            fixed_setter=lambda flag, idx=i: setattr(mesh.vertices[idx], "fixed", flag),
        )
        constraint_module_names.extend(constraints)

    # Edges
    edges = data.get("edges") or data.get("Edges")
    if edges is None:
        err_msg = "Input geometry is missing required 'edges' section."
        logger.error(err_msg)
        raise KeyError(err_msg)

    for i, entry in enumerate(edges):
        tail_index, head_index, *opts = entry

        if tail_index not in mesh.vertices:
            raise ValueError(
                f"Edge {i + 1} references missing tail vertex {tail_index}"
            )
        if head_index not in mesh.vertices:
            raise ValueError(
                f"Edge {i + 1} references missing head vertex {head_index}"
            )

        raw_opts = opts[0] if opts else {}
        options = resolve_options(raw_opts)
        mesh.edges[i + 1] = Edge(
            index=i + 1, tail_index=tail_index, head_index=head_index, options=options
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

        # Edges constraint modules
        constraints = normalize_constraints(
            mesh.edges[i + 1].options,
            fixed_setter=lambda flag, idx=i: setattr(
                mesh.edges[idx + 1], "fixed", flag
            ),
        )
        constraint_module_names.extend(constraints)

    # Facets (optional for line‑only geometries)
    faces_section = data.get("faces") or data.get("Faces") or data.get("Facets") or []
    for i, entry in enumerate(faces_section):
        *raw_edges, raw_opts = entry if isinstance(entry[-1], dict) else (*entry, {})
        options = resolve_options(raw_opts)

        def parse_edge(e):
            if isinstance(e, str) and e.startswith("r"):
                return -(int(e[1:]) + 1)  # "r0" -> -1
            i = int(e)
            if i >= 0:
                return i + 1  # 0 -> 1, 1 -> 2, etc.
            elif i < 0:
                return i - 1  # -11 -> -12

        edge_indices = [parse_edge(e) for e in raw_edges]
        mesh.facets[i] = Facet(index=i, edge_indices=edge_indices, options=options)

        if "energy" in options:
            if isinstance(options["energy"], list):
                energy_module_names.update(options["energy"])
            elif isinstance(options["energy"], str):
                energy_module_names.add(options["energy"])
                mesh.facets[i].options["energy"] = [mesh.facets[i].options["energy"]]
            else:
                err_msg = "energy modules should be in a list or a single string"
                logger.error(err_msg)
                raise err_msg
        elif "energy" not in options:
            mesh.facets[i].options["energy"] = ["surface"]
            energy_module_names.add("surface")

        # Ensure all facets have surface_tension set
        if "surface_tension" not in options:
            mesh.facets[i].options["surface_tension"] = mesh.global_parameters.get(
                "surface_tension", 1.0
            )

        # Facets constraint modules
        facet_constraints = normalize_constraints(
            mesh.facets[i].options,
            fixed_setter=lambda flag, idx=i: setattr(mesh.facets[idx], "fixed", flag),
        )

        if options.get("target_area") is not None:
            if not facet_constraints:
                facet_constraints = []
            if "fix_facet_area" not in facet_constraints:
                facet_constraints.append("fix_facet_area")
                mesh.facets[i].options["constraints"] = facet_constraints

        if facet_constraints:
            constraint_module_names.extend(facet_constraints)

    vol_mode = mesh.global_parameters.get("volume_constraint_mode", "lagrange")
    if vol_mode == "penalty":
        energy_module_names.add("volume")

    # Bodies
    bodies_section = data.get("bodies") or data.get("Bodies")
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
                if isinstance(constraint_spec, str):
                    merged_constraints = [constraint_spec]
                elif isinstance(constraint_spec, list):
                    merged_constraints = list(constraint_spec)
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

            # Body constraint modules. If a target volume is specified,
            # automatically enable the volume constraint module so bodies
            # behave like FIXEDVOL in Evolver, on top of any explicit
            # constraints configured.
            constraint_spec = body.options.get("constraints", [])
            if isinstance(constraint_spec, str):
                body_constraints = [constraint_spec]
            elif isinstance(constraint_spec, list):
                body_constraints = list(constraint_spec)
            else:
                body_constraints = []

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

    # Instructions
    mesh.instructions = data.get("instructions", [])

    # Energy modules
    mesh.energy_modules = list(energy_module_names)

    # Constraint modules
    mesh.constraint_modules = list(set(constraint_module_names))

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()

    # Basic validation (connectivity)
    try:
        mesh.validate_edge_indices()
    except Exception as e:
        logger.error(f"Mesh connectivity validation failed: {e}")
        raise

    # Automatically triangulate polygonal facets if needed
    if any(len(f.edge_indices) > 3 for f in mesh.facets.values()):
        refined = refine_polygonal_facets(mesh)
        try:
            refined.full_mesh_validate()
        except Exception as e:
            logger.error(f"Refined mesh validation failed: {e}")
            raise
        return refined

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
        return opts if opts else None

    data = {
        "vertices": [
            (
                [
                    *mesh.vertices[old_vid].position.tolist(),
                    prepare_options(mesh.vertices[old_vid]),
                ]
                if prepare_options(mesh.vertices[old_vid])
                else mesh.vertices[old_vid].position.tolist()
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
        "global_parameters": mesh.global_parameters.to_dict(),
        "instructions": mesh.instructions,
    }
    with open(path, "w") as f:
        if compact:
            json.dump(data, f, separators=(",", ":"), ensure_ascii=False)
        else:
            json.dump(data, f, indent=4, ensure_ascii=False)

# geometry_io.py
import json

import numpy as np
import yaml

from geometry.entities import Body, Edge, Facet, Mesh, Vertex
from parameters.global_parameters import GlobalParameters
from runtime.logging_config import setup_logging
from runtime.refinement import refine_polygonal_facets

logger = setup_logging('membrane_solver.log')

def load_data(filename):
    """Load geometry from a JSON file.

    Expected JSON format:
    {
        "vertices": [[x, y, z], ...],
        "faces": [
            [i, j, k, ...] or [i, j, k, ..., {"refine": false, "surface_tension": 0.8}],
            ...
        ]
    }"""
    filename_str = str(filename)
    with open(filename_str, 'r') as f:
        if filename_str.endswith((".yaml", ".yml")):
            data = yaml.safe_load(f)
            logger.error("Currently yaml format not supported.")
            raise ValueError("Currently yaml format not supported.")
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
        mode = str(mesh.global_parameters.get("volume_constraint_mode", "lagrange")).lower()
        proj = False if mode == "lagrange" else True
        mesh.global_parameters.set("volume_projection_during_minimization", proj)
    elif has_proj and not has_mode:
        proj = bool(mesh.global_parameters.get("volume_projection_during_minimization", True))
        mode = "penalty" if proj else "lagrange"
        mesh.global_parameters.set("volume_constraint_mode", mode)

    # Warn about unstable combinations.
    mode = str(mesh.global_parameters.get("volume_constraint_mode", "lagrange")).lower()
    proj_flag = bool(mesh.global_parameters.get("volume_projection_during_minimization", False))
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

    # TODO: add option to read both lowercase and uppercase title Vertices/vertices
    # Vertices
    for i, entry in enumerate(data["vertices"]):
        *position, options = entry if isinstance(entry[-1], dict) else (*entry, {})
        mesh.vertices[i] = Vertex(index=i, position=np.asarray(position,
                                                               dtype=float), options=options)

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
        #elif "energy" not in options:
            #mesh.vertices[i].options["energy"] = ["surface"]

        # Vertex constraint modules
        if "constraints" in options:
            if isinstance(options["constraints"], list):
                constraint_module_names.extend(options["constraints"])
            elif isinstance(options["constraints"], str):
                constraint_module_names.append(options["constraints"])
                mesh.vertices[i].options["constraints"] = [mesh.vertices[i].options["constraints"]]
            else:
                err_msg = "constraint modules should be in a list or a single string"
                logger.error(err_msg)
                raise err_msg
            if "fixed" in options["constraints"]:
                # TODO: move fixed out of out constraints and make
                #       a fixed_edges_map in meshes, when we do energy/grad
                #       calculations the map zeros or just skips these edges
                #       make sure the fixed flag is turned on
                # constraint_module_names.append("fixed") # Removed: 'fixed' is not a module
                mesh.vertices[i].fixed = True
        if options.get("fixed", False):
            # constraint_module_names.append("fixed") # Removed: 'fixed' is not a module
            mesh.vertices[i].fixed = True # Ensure fixed flag is set if from top-level option

    # Edges
    if "edges" not in data:
        err_msg = "Input geometry is missing required 'edges' section."
        logger.error(err_msg)
        raise KeyError(err_msg)

    for i, entry in enumerate(data["edges"]):
        tail_index, head_index, *opts = entry
        options = opts[0] if opts else {}
        mesh.edges[i+1] = Edge(index=i+1, tail_index=tail_index,
                               head_index=head_index, options=options)

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
        #elif "energy" not in options:
            #mesh.edges[i+1].options["energy"] = ["surface"]

        # Edges constraint modules
        if "constraints" in options:
            if isinstance(options["constraints"], list):
                constraint_module_names.extend(options["constraints"])
            elif isinstance(options["constraints"], str):
                constraint_module_names.append(options["constraints"])
                mesh.edges[i+1].options["constraints"] = [mesh.edges[i+1].options["constraints"]]
            else:
                err_msg = "constraint modules should be in a list or a single string"
                logger.error(err_msg)
                raise err_msg
            if "fixed" in options["constraints"]:
                # constraint_module_names.append("fixed") # Removed: 'fixed' is not a module
                mesh.edges[i+1].fixed = True
        if options.get("fixed", False):
            # constraint_module_names.append("fixed") # Removed: 'fixed' is not a module
            mesh.edges[i+1].fixed = True # Ensure fixed flag is set if from top-level option

    # Facets (optional for line‑only geometries)
    faces_section = data.get("faces", [])
    for i, entry in enumerate(faces_section):
        *raw_edges, options = entry if isinstance(entry[-1], dict) else (*entry, {})
        def parse_edge(e):
            if isinstance(e, str) and e.startswith("r"):
                return -(int(e[1:]) + 1)    # "r0" -> -1
            i = int(e)
            if i >= 0: return i + 1     # 0 -> 1, 1 -> 2, etc.
            elif i < 0: return i - 1    # -11 -> -12
        edge_indices = [parse_edge(e) for e in raw_edges]
        mesh.facets[i] = Facet(index=i, edge_indices=edge_indices,
                                 options=options)

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
            mesh.facets[i].options["surface_tension"] = mesh.global_parameters.get("surface_tension", 1.0)

        # Facets constraint modules
        facet_constraints = options.get("constraints")
        if facet_constraints is not None and not isinstance(facet_constraints, (list, str)):
            err_msg = "constraint modules should be in a list or a single string"
            logger.error(err_msg)
            raise err_msg

        if isinstance(facet_constraints, str):
            facet_constraints = [facet_constraints]
            mesh.facets[i].options["constraints"] = facet_constraints
        elif isinstance(facet_constraints, list):
            facet_constraints = list(facet_constraints)

        if options.get("target_area") is not None:
            if facet_constraints is None:
                facet_constraints = []
            if "fix_facet_area" not in facet_constraints:
                facet_constraints.append("fix_facet_area")
                mesh.facets[i].options["constraints"] = facet_constraints

        if facet_constraints:
            constraint_module_names.extend(facet_constraints)
            if "fixed" in facet_constraints:
                # constraint_module_names.append("fixed") # Removed: 'fixed' is not a module
                mesh.facets[i].fixed = True

        if options.get("fixed", False):
            # constraint_module_names.append("fixed") # Removed: 'fixed' is not a module
            mesh.facets[i].fixed = True # Ensure fixed flag is set if from top-level option

    vol_mode = mesh.global_parameters.get("volume_constraint_mode", "lagrange")
    if vol_mode == "penalty":
        energy_module_names.add("volume")

    # Bodies
    if "bodies" in data:
        bodies_section = data["bodies"]
        face_groups = bodies_section["faces"]
        volumes = bodies_section.get("target_volume", [None] * len(face_groups))
        areas = bodies_section.get("target_area", [None] * len(face_groups))

        # ``energy`` may be:
        #   - a list parallel to ``faces`` (per‑body specs), or
        #   - a single string/dict applying to all bodies.
        energy_entries = bodies_section.get("energy", [None] * len(face_groups))
        if not isinstance(energy_entries, list) or len(energy_entries) != len(face_groups):
            energy_entries = [energy_entries] * len(face_groups)

        constraint_entries = bodies_section.get("constraints", [None] * len(face_groups))
        if not isinstance(constraint_entries, list) or len(constraint_entries) != len(face_groups):
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

            if body.options.get("target_area") is not None and "body_area" not in body_constraints:
                body_constraints.append("body_area")

            if body_constraints:
                body.options["constraints"] = body_constraints
                constraint_module_names.extend(body_constraints)

    # Instructions
    mesh.instructions = data.get("instructions", [])

    # Energy modules
    mesh.energy_modules= list(energy_module_names)

    # Constraint modules
    mesh.constraint_modules = list(set(constraint_module_names))

    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()

    # Automatically triangulate polygonal facets if needed
    if any(len(f.edge_indices) > 3 for f in mesh.facets.values()):
        refined = refine_polygonal_facets(mesh)
        return refined

    return mesh

def save_geometry(mesh: Mesh, path: str = "outputs/temp_output_file.json"):
    def export_edge_index(i):
        if i < 0:
            return f"r{abs(i) - 1}"     # -1 → "r0"
        return i - 1                    # 1 → 0

    def prepare_options(entity):
        opts = entity.options.copy() if entity.options else {}
        if entity.fixed:
            opts["fixed"] = True
        return opts if opts else None

    data = {
        "vertices": [[*mesh.vertices[v].position.tolist(),
                        prepare_options(mesh.vertices[v])] if prepare_options(mesh.vertices[v]) else
                        mesh.vertices[v].position.tolist() for v in mesh.vertices.keys()],
        "edges": [[mesh.edges[e].tail_index, mesh.edges[e].head_index,
                   prepare_options(mesh.edges[e])] if prepare_options(mesh.edges[e]) else
                  [mesh.edges[e].tail_index, mesh.edges[e].head_index] for e in mesh.edges.keys()],
        "faces": [
            [*map(export_edge_index, mesh.facets[facet_idx].edge_indices),
             prepare_options(mesh.facets[facet_idx])] if prepare_options(mesh.facets[facet_idx]) else
            list(map(export_edge_index, mesh.facets[facet_idx].edge_indices))
            for facet_idx in mesh.facets.keys()
        ],
        "bodies": {
            "faces": [mesh.bodies[b].facet_indices for b in mesh.bodies.keys()],
            "target_volume": [mesh.bodies[b].target_volume for b in mesh.bodies.keys()],
            "target_area": [mesh.bodies[b].options.get("target_area") for b in mesh.bodies.keys()],
            "energy": [mesh.bodies[b].options.get("energy", {}) for b in mesh.bodies.keys()],
            "constraints": [mesh.bodies[b].options.get("constraints", []) for b in mesh.bodies.keys()]
        },
        "global_parameters": mesh.global_parameters.to_dict(),
        "instructions": mesh.instructions
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

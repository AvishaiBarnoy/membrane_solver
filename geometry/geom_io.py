# geometry_io.py 
import json #, yaml
from geometry.entities import Vertex, Edge, Facet, Body, Mesh
from parameters.global_parameters import GlobalParameters
from runtime.refinement import refine_polygonal_facets
import importlib
import numpy as np
import logging
from logging_config import setup_logging
import sys
logger = logging.getLogger('membrane_solver')

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
    with open(filename, 'r') as f:
        if filename.endswith((".yaml", ".yml")):
            data = yaml.safe_load(f)
            logger.error("Currently yaml format not supported.")
            raise ValueError("Currently yaml format not supported.")
        elif filename.endswith(".json"):
            data = json.load(f)
        else:
            logger.error(f"Unsupported file format for: {filename}")
            raise ValueError(f"Unsupported file format for: {filename}")

    return data

def parse_geometry(data: dict) -> Mesh:
    mesh = Mesh()

    # Initialize global parameters
    mesh.global_parameters = GlobalParameters()

    # Override global parameters with values from the input file
    input_global_params = data.get("global_parameters", {})
    mesh.global_parameters.update(input_global_params)

    # Initialize module_name list
    energy_module_names = []
    constraint_module_names = []

    # TODO: add option to read both lowercase and uppercase title Vertices/vertices
    # Vertices
    for i, entry in enumerate(data["vertices"]):
        *position, options = entry if isinstance(entry[-1], dict) else (*entry, {})
        mesh.vertices[i] = Vertex(index=i, position=np.asarray(position,
                                                               dtype=float), options=options)

        if "energy" in options:
            if isinstance(options["energy"], list):
                energy_module_names.extend(options["energy"])
            elif isinstance(options["energy"], str):
                energy_module_names.append(options["energy"])
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
                constraint_module_names.append("fixed")
        if options.get("fixed", False):
            constraint_module_names.append("fixed")

    # Edges
    for i, entry in enumerate(data["edges"]):
        tail_index, head_index, *opts = entry
        options = opts[0] if opts else {}
        mesh.edges[i+1] = Edge(index=i+1, tail_index=tail_index,
                               head_index=head_index, options=options)

        if "energy" in options:
            if isinstance(options["energy"], list):
                energy_module_names.extend(options["energy"])

            elif isinstance(options["energy"], str):
                energy_module_names.append(options["energy"])
            else:
                err_msg = "energy modules should be in a list or a single string"
                logger.error(err_msg)
                raise err_msg
        # Uncomment to add a default energy moduel to Edges
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
                constraint_module_names.append("fixed")
        if options.get("fixed", False):
            constraint_module_names.append("fixed")

    # Facets
    for i, entry in enumerate(data["faces"]):
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
                energy_module_names.extend(options["energy"])
            elif isinstance(options["energy"], str):
                energy_module_names.append(options["energy"])
                mesh.facets[i].options["energy"] = [mesh.facets[i].options["energy"]]
            else:
                err_msg = "energy modules should be in a list or a single string"
                logger.error(err_msg)
                raise err_msg
        elif "energy" not in options:
            mesh.facets[i].options["energy"] = ["surface"]
            energy_module_names.append("surface")
        # Facets constraint modules
        if "constraints" in options:
            if isinstance(options["constraints"], list):
                constraint_module_names.extend(options["constraints"])
            elif isinstance(options["constraints"], str):
                constraint_module_names.append(options["constraints"])
                mesh.facets[i].options["constraints"] = [mesh.facets[i].options["constraints"]]
            else:
                err_msg = "constraint modules should be in a list or a single string"
                logger.error(err_msg)
                raise err_msg

    # Instructions
    # Bodies
    if "bodies" in data:
        face_groups = data["bodies"]["faces"]
        volumes = data["bodies"].get("target_volume"), [None] * len(face_groups)
        options = data["bodies"].get("energy", [{}] * len(face_groups))
        for i, (facet_indices, volume, options) in enumerate(zip(face_groups, volumes, options)):
            body = Body(index=i, facet_indices=facet_indices,
                                  target_volume=volume[0], options={"energy": options})
            body.options["target_volume"] = float(volume[0])
            mesh.bodies[i] = body
            # Energy modules
            if "energy" in options:
                if isinstance(options["energy"], list):
                    energy_module_names.extend(options["energy"])
                elif isinstance(options["energy"], str):
                    energy_module_names.append(options["energy"])
                    mesh.bodies[i].options["energy"] = [mesh.bodies[i].options["energy"]]
                else:
                    err_msg = "energy modules should be in a list or a single string"
                    logger.error(err_msg)
                    raise err_msg
            elif "energy" not in options:
                if len(mesh.bodies) > 0:
                    mesh.bodies[i].options["energy"] = ["volume"]
                    energy_module_names.append("volume")
            # Body constraint modules
            if "constraints" in options:
                if isinstance(options["energy"], list):
                    energy_module_names.extend(options["energy"])
                elif isinstance(options["energy"], str):
                    energy_module_names.append(options["energy"])
                    mesh.bodies[i].options["energy"] = [mesh.bodies[i].options["energy"]]
                else:
                    err_msg = "constraint modules should be in a list or a single string"
                    logger.error(err_msg)
                    raise err_msg

    # Instructions
    mesh.instructions = data.get("instructions", [])
    # Energy modules
    mesh.energy_modules= list(set(energy_module_names))
    # Constraint modules
    mesh.constraint_modules = list(set(constraint_module_names))

    new_mesh = refine_polygonal_facets(mesh)

    return new_mesh

def save_geometry(mesh: Mesh, path: str = "temp_output_file.json"):
    def export_edge_index(i):
        if i < 0:
            return f"r{abs(i) - 1}"     # -1 → "r0"
        return i - 1                    # 1 → 0

    data = {
        "vertices": [[*mesh.vertices[v].position.tolist(),
                        mesh.vertices[v].options] if mesh.vertices[v].options else
                        mesh.vertices[v].position.tolist() for v in mesh.vertices.keys()],
        "edges": [[mesh.edges[e].tail_index, mesh.edges[e].head_index,
                   mesh.edges[e].options] if mesh.edges[e].options else
                  [mesh.edges[e].tail_index, mesh.edges[e].head_index] for e in mesh.edges.keys()],
        "faces": [
            [*map(export_edge_index, mesh.facets[facet_idx].edge_indices),
             mesh.facets[facet_idx].options] if mesh.facets[facet_idx].options else
            list(map(export_edge_index, mesh.facets[facet_idx].edge_indices))
            for facet_idx in mesh.facets.keys()
        ],
        "bodies": {
            "faces": [mesh.bodies[b].facet_indices for b in mesh.bodies.keys()],
            "target_volume": [mesh.bodies[b].target_volume for b in mesh.bodies.keys()],
            "energy": [mesh.bodies[b].options.get("energy", {}) for b in mesh.bodies.keys()]
        },
        "global_parameters": mesh.global_parameters.to_dict(),
        "instructions": mesh.instructions
    }
    dfaces = data["faces"]
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


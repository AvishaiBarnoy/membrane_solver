# geometry_io.py 
import json #, yaml
from geometry.entities import Vertex, Edge, Facet, Body, Mesh
from parameters.global_parameters import GlobalParameters
#from runtime.refinement import refine_polygonal_facets
import importlib
import numpy as np
import logging
from logging_config import setup_logging
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
    logger.info(f"######################################")
    logger.info(f"Loading file: {filename}")
    logger.info(f"######################################")
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

    # TODO: add option to read both lowercase and uppercase title Vertices/vertices
    # Vertices
    for i, entry in enumerate(data["vertices"]):
        *position, options = entry if isinstance(entry[-1], dict) else (*entry, {})
        mesh.vertices[i] = Vertex(index=i, position=np.array(position),
                                    options=options)

    # Edges
    for i, entry in enumerate(data["edges"]):
        tail_index, head_index, *opts = entry
        options = opts[0] if opts else {}
        mesh.edges[i+1] = Edge(index=i+1, tail_index=tail_index,
                               head_index=head_index, options=options)

    # Facets
    for i, entry in enumerate(data["faces"]):
        # TODO: should accept lsot data["facets"]
        *raw_edges, options = entry if isinstance(entry[-1], dict) else (*entry, {})
        def parse_edge(e):
            if isinstance(e, str) and e.startswith("r"):
                return -(int(e[1:]) + 1)    # "r0" -> -1
            i = int(e)
            if i >= 0: return i + 1     # 0 -> 1, 1 -> 2, etc.
            elif i < 0: return i - 1    # -11 -> -12
        edge_indices = [parse_edge(e) for e in raw_edges]
        # TODO: here do refine_polygonal_facets 
        mesh.facets[i] = Facet(index=i, edge_indices=edge_indices,
                                 options=options)

    # Bodies
    if "bodies" in data:
        face_groups = data["bodies"]["faces"]
        volumes = data["bodies"].get("target_volume"), [None] * len(face_groups)
        energies = data["bodies"].get("energy", [{}] * len(face_groups))
        for i, (facet_indices, volume, energy) in enumerate(zip(face_groups, volumes, energies)):
            mesh.bodies[i] = Body(index=i, facet_indices=facet_indices, target_volume=volume, options={"energy": energy})

    # Global parameters and instructions
    mesh.global_parameters = data.get("global_parameters", {})
    mesh.instructions = data.get("instructions", [])
    return mesh

def save_geometry(mesh: Mesh, path: str = "temp_output_file.json"):
    def export_edge_index(i):
        if i < 0:
            return f"r{abs(i) - 1}"     # -1 → "r0"
        return i - 1                    # 1 → 0

    data = {
        #"vertices": [[*v.position.tolist(), v.options] if v.options else v.position.tolist() for v in mesh.vertices],
        "vertices": [[*mesh.vertices[v].position.tolist(),
                        mesh.vertices[v].options] if mesh.vertices[v].options else
                        mesh.vertices[v].position.tolist() for v in mesh.vertices.keys()],
        #"edges": [[e.tail_index, e.head_index, e.options] if e.options else [e.tail_index, e.head_index] for e in mesh.edges],
        "edges": [[mesh.edges[e].tail_index, mesh.edges[e].head_index,
                   mesh.edges[e].options] if mesh.edges[e].options else
                  [mesh.edges[e].tail_index, mesh.edges[e].head_index] for e in mesh.edges.keys()],
        "faces": [
            [*map(export_edge_index, mesh.facets[facet_idx].edge_indices),
             mesh.facets[facet_idx].options] if mesh.facets[facet_idx].options else
            list(map(export_edge_index, mesh.facets[facet_idx].edge_indices))
            for facet_idx in mesh.facets.keys()
        ],
        #"faces": [
        #    [*map(export_edge_index, facet.edge_indices), facet.options] if facet.options else
        #    list(map(export_edge_index, facet.edge_indices))
        #    for facet in mesh.facets
        #],
        "bodies": {
            "faces": [mesh.bodies[b].facet_indices for b in mesh.bodies.keys()],
            "target_volume": [mesh.bodies[b].target_volume for b in mesh.bodies.keys()],
            "energy": [mesh.bodies[b].options.get("energy", {}) for b in mesh.bodies.keys()]
        },
        "global_parameters": mesh.global_parameters,
        "instructions": mesh.instructions
    }
    dfaces = data["faces"]
    print(f"data[faces] {dfaces}")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


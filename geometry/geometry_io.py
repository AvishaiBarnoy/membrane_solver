# geometry_io.py 
import json, yaml
from geometry.geometry_entities import Vertex, Edge, Facet, Body
from parameters.global_parameters import GlobalParameters
import numpy as np
import os
import sys
import importlib
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

def build_vertices(vertices_data):
    # TODO: add documentation
    # construct vertex objects
    vertices = []
    for idx, vertex in enumerate(vertices_data):
        coords = vertex[:3]   # Extract first 3 elements as coordinates
        options = vertex[3] if len(vertex) > 3 else {}
        if options and not isinstance(options, dict):
            raise ValueError(f"Vertex #{idx} options must be a dictionary: {options}")
        vertex = Vertex(coords, idx, options)    # Pass attributes as keyword args
        logger.info(f"Build vertex: {vertex}")
        vertices.append(vertex)
    return vertices

def build_edges(edges_data, vertices):
    # TODO: add documentation
    # construct edges objects
    edge_map = {}  # key = (i, j) → Edge instance, oriented
    edges = []     # flat list of all unique Edge objects

    for idx, entry in enumerate(edges_data):
        if not isinstance(entry, list) or len(entry) < 2:
            raise ValueError(f"Edge #{idx} must have at least two vertex indices: {entry}")

        i, j = entry[:2]
        if not isinstance(i, int) or not isinstance(j, int):
            raise ValueError(f"Edge #{idx} vertex indices must be integers: {entry}")

        options = entry[2] if len(entry) > 2 else {}
        if options and not isinstance(options, dict):
            raise ValueError(f"Edge #{idx} options must be a dictionary: {options}")

        key = (i, j)    # Preserve user-defined orientation 
        if key not in edge_map:
            tail, head = vertices[i], vertices[j]
            edge = Edge(tail, head, idx, options)
            logger.info(f"Build edge: {edge}")
            edge_map[key] = edge
            edges.append(edge)
            logger.debug(f"Created edge {key} with options: {options}")
        else:
            # Optional: merge or warn on duplicate with different options
            logger.debug(f"Duplicate edge {key} found. Skipping.")
    return edges

def parse_edge_spec(edge_spec):
    """
    Parses an edge specification.
    If it's an integer and negative, treat it as reversed and return (abs(index), True).
    If it's a string starting with 'r' (or 'R'), then the edge is reversed.
    If it's a string representing a number, convert it accordingly.
    If it's a list/tuple, expect the form [edge_index, reversed_flag].
    """
    if isinstance(edge_spec, int):
        if edge_spec < 0:
            return abs(edge_spec), True
        else:
            return edge_spec, False
    elif isinstance(edge_spec, str):
        # Remove any whitespace
        s = edge_spec.strip()
        if s.lower().startswith('r'):
            return int(s[1:]), True
        else:
            val = int(s)
            if val < 0:
                return abs(val), True
            return val, False
    elif isinstance(edge_spec, (list, tuple)):
        if len(edge_spec) != 2:
            raise ValueError(f"Invalid edge specification list: {edge_spec}")
        edge_index, flag = edge_spec
        # If the edge index is negative here, also adjust it
        if int(edge_index) < 0:
            return abs(int(edge_index)), True
        return int(edge_index), bool(flag)
    else:
        raise TypeError(f"Unknown type for edge specification: {edge_spec}")

def build_facet_edges(edge_specs, edges):
    """
    Build a list of oriented edges for a single facet from its edge specifications.
    Edge specifications can be provided as integers, strings (e.g., "r5"), or lists/tuples.
    The parse_edge_spec function is used to determine the edge index and whether it should be reversed.
    Orientation matters for defining a consistent facet normal.
    """
    if not edge_specs:
        logger.error("No edges specified for facet!")
        raise ValueError("No edges specified for facet!")

    if len(edge_specs) < 3:
        logger.critical(f"Facet with edges {edge_specs} has fewer than three edges! Please check input file.")
        raise ValueError(f"Facet with edges {edge_specs} has fewer than three edges!")

    oriented_edges = []

    # First pass: build oriented edges using parse_edge_spec
    for spec in edge_specs:
        try:
            edge_id, reverse = parse_edge_spec(spec)
            logger.debug(f"Parsed edge spec {spec} as edge_id {edge_id} with reverse={reverse}.")
        except Exception as e:
            logger.error(f"Failed to parse edge specification {spec}: {e}")
            raise

        if edge_id < 0 or edge_id >= len(edges):
            logger.error(f"Edge index {edge_id} out of range. Please check input file.")
            raise IndexError(f"Edge index {edge_id} out of range.")

        original_edge = edges[edge_id]
        if reverse:
            # Create a reversed edge: swap tail and head, and update the vector.
            new_edge = Edge(original_edge.head, original_edge.tail, original_edge.index, original_edge.options.copy())
            new_edge.vector = new_edge.head.position - new_edge.tail.position
            oriented_edges.append(new_edge)
            logger.debug(f"Edge {edge_id} reversed successfully.")
        else:
            oriented_edges.append(original_edge)
            logger.debug(f"Edge {edge_id} used in normal orientation.")

    # Second pass: verify that the sequence of oriented edges is continuous.
    n = len(oriented_edges)
    for i in range(n):
        current_edge = oriented_edges[i]
        next_edge = oriented_edges[(i + 1) % n]  # wrap-around for closure check
        if current_edge.head != next_edge.tail:
            # TODO: Replace with np.allclose for numerical reasons.
            logger.error("Facet edges don't create a closed loop.")
            raise ValueError(
                f"Facet edge loop is not continuous at edge {i}: "
                f"{current_edge.head.position} != {next_edge.tail.position}\n"
                f"Failed building facet {edge_specs}"
            )

    logger.info(f"Finished building edges for facet: {edge_specs}")
    return oriented_edges

def build_facets(facets_data, edges, global_params):
    # TODO: add documentation
    facets = []
    for idx, facet in enumerate(facets_data):
        if facet and isinstance(facet[-1], dict):
            facet_edges_idx = facet[:-1]
            options = facet[-1]
        else:
            facet_edges_idx = facet
            options = {}

        # Merge local options with global defaults
        full_options = {
            "surface_tension": global_params.surface_tension,
            "bending_modulus": global_params.bending_modulus,
            "gaussian_modulus": global_params.gaussian_modulus,
            "energy": options.get("energy", "surface"), # default energy module 
            **options  # Local overrides take precedence
        }

        facet_edges = build_facet_edges(facet_edges_idx, edges)
        facet = Facet(facet_edges, idx, options=full_options)
        logger.info(f"Finished building facet: {facet}")
        facets.append(facet)
    return facets

def build_bodies(bodies_data, facets, global_params):
    # TODO: add documentation
    bodies = []
    for idx, body in enumerate(bodies_data["faces"]):
        # TODO: add extra
        # TODO: fix for case where target_volume is not given, what is wanted behavior?

        options = {
            "target_volume": bodies_data.get("target_volume", None)[idx],
            "volume_stiffness": global_params.volume_stiffness,
            "energy": bodies_data["energy"][idx]
        }

        body = Body(facets, index=idx, options=options)
        body.calculate_volume()
        logger.info(f"Finished building body object: {body}")
        bodies.append(body)
    return bodies

    bodies = []
    for idx, face_indices in enumerate(bodies_data["faces"]):
        options = {
            "target_volume": bodies_data.get("target_volume", [None])[idx],
            "volume_stiffness": global_params.volume_stiffness
        }
        body_facets = [facets[i] for i in face_indices]
        body = Body(body_facets, index=idx, options=options)
        body.calculate_volume()
        bodies.append(body)
    logger.info(f"Finished building body object: {body}")
    return bodies

def parse_geometry(data):
    """
    Takes loaded data from geometry file

    Returns:
        vertices (list of Vertex): List of vertex objects.
        facets (list of Facet): List of Facet objects.
        volume: Calculated volume of object.
    """
    global_params_data = data.get("global_parameters", {})
    global_params = GlobalParameters(global_params_data)
    logger.info(f"Loaded global parameters:\n\t{global_params}")

    # build vertix objects
    vertices = build_vertices(data["vertices"])
    logger.info(f"--Successfuly built initial vertices--\n")

    # build edge object 
    edges = build_edges(data["edges"], vertices)
    logger.info(f"--Successfuly built initial edges--\n")

    # build facets objects 
    facets = build_facets(data["faces"], edges, global_params)
    logger.info(f"--Sucessfuly building initial facets--\n")

    # build body object
    # Currently working HERE
    bodies = build_bodies(data["bodies"], facets, global_params)
    logger.info(f"--Sucessfuly building initial bodies--\n")

    return vertices, facets, bodies, global_params

def initial_triangulation(vertices, facets):
    """
    Converts all facets with more than three vertices into triangles.
    Unlike subsequent refinement steps, the initial triangulation always
    subdivides a facet into triangles (even if its options include "refine": False)
    because energy computations are applied only to simplex triangles.

    For each n-gon (n > 3), a new vertex is added at the centroid and the facet
    is subdivided into n triangles by connecting each edge of the polygon to the centroid.

    Child facets inherit a copy of the parent facet’s options.

    Args:
        vertices (list of Vertex): The list of vertices.
        facets (list of Facet): The list of facets.

    Returns:
        (vertices, new_facets): The updated list of vertices and a new list of facets.
    """
    new_facets = []
    for facet in facets:
        n = len(facet.edges)

        # Always triangulate if the facet is non-triangle.
        if n < 3:
            logger.critical("Facet edges: " + str(facet.edges))
            logger.critical("""Critical error! Cannot proceed further.
                            Facet with fewer than three vertices encounterd!""")
            raise ValueError("Facet with fewer than three vertices encountered!")

        elif n == 3:
            new_facets.append(facet)
            continue

        # --- Step 1: Reconstruct ordered vertex loop from edges ---
        vertex_loop = [facet.edges[0].tail]
        for edge in facet.edges:
            if vertex_loop[-1] != edge.tail:
                raise ValueError(f"Edge loop is not continuous,{edge}")
            vertex_loop.append(edge.head)

        # Ensure loop is closed
        if vertex_loop[0] == vertex_loop[-1]:
            vertex_loop = vertex_loop[:-1]

        if len(vertex_loop) < 3:
            raise ValueError("Cannot triangulate facet with <3 vertices after loop reconstruction.")

        # --- Step 2: Compute centroid position ---
        centroid_pos = np.mean([v.position for v in vertex_loop], axis=0)
        centroid_vertex = Vertex(centroid_pos, len(vertices))
        vertices.append(centroid_vertex)

        # --- Step 3: Determine reference normal ---
        v0, v1, v2 = vertex_loop[0].position, vertex_loop[1].position, vertex_loop[2].position
        ref_normal = np.cross(v1 - v0, v2 - v0)

        # --- Step 4: Fan triangulation with orientation correction ---
        for i in range(len(vertex_loop)):
            a = vertex_loop[i]
            b = vertex_loop[(i + 1) % len(vertex_loop)]

            # Compute triangle normal (a → b → centroid)
            tri_normal = np.cross(b.position - a.position, centroid_vertex.position - a.position)

            # Flip to match orientation if needed
            if np.dot(tri_normal, ref_normal) < 0:
                a, b = b, a

            e1 = Edge(a, b)
            e2 = Edge(b, centroid_vertex)
            e3 = Edge(centroid_vertex, a)

            new_facets.append(Facet([e1, e2, e3], options=facet.options.copy()))
    return vertices, new_facets

def save_geometry(filename, vertices, facets, volume):
    """
    Saves the geometry (vertices, facets, and computed volume) to a JSON file.
    If the filename already exists, prints a warning message and adjusts the output name.

    The saved JSON has the following format:
    {
        "vertices": [[x, y, z], ...],
        "faces": [
            [i, j, k, ...] or [i, j, k, ..., {facet options}],
            ...
        ],
        "volume": <calculated volume>
    }
    """
    data = {
        "vertices": [],
        "faces": [],
        "volume": volume
    }
    for v in vertices:
        if v.options:
            data["vertices"].append(list(v.position) + [v.attributes])
        else:
            data["vertices"].append(list(v.position))

    for facet in facets:
        # If there are facet options, append them as the last element.
        if facet.options:
            data["faces"].append(list(facet.edges) + [facet.options])
        else:
            data["faces"].append(list(facet.edges))

    original_filename = filename
    counter = 1
    while os.path.exists(filename):
        logger.warning(f"Warning: File '{filename}' already exists. Adjusting output name.")
        base, ext = os.path.splitext(original_filename)
        filename = f"{base}_{counter}{ext}"
        counter += 1

    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Geometry saved to {filename}")

def parse_inputfile(data):
    # TODO: add documentation
    vertices, facets, bodies, global_params = parse_geometry(data=data)

    logger.info("########################")
    logger.info("# Input file containts #")
    logger.info("########################")
    logger.info(f"Number of vertices: {len(vertices)}")
    logger.info(f"Number of facets: {len(facets)}")
    logger.info(f"number of bodies: {len(bodies)}, with volume {[body.volume for body in bodies]}")

    # TODO: remove redundant loaded geometry information, change lots of info to debug
    """logger.info("Loaded vertices:")
    for v in vertices:
        logger.info(v.position)
    logger.info("Loaded facets:")
    for facet in facets:
        logger.info(f"{facet} {facet.options}")
    logger.info("Loaded global parameters:")
    logger.info(f"{global_params}")"""

    # Perform the initial triangulation (always subdividing non-simplex facets).
    vertices, tri_facets = initial_triangulation(vertices, facets)
    sys.exit(1)
    logger.info("\nAfter initial triangulation:")
    logger.info(f"Number of vertices: {len(vertices)}")
    for v in vertices:
        logger.info(v.position)
    logger.info(f"Number of facets: {len(tri_facets)}")
    logger.info("Triangulated facets:")
    for facet in tri_facets:
        logger.info(f"{facet} {facet.options}")

    logger.info("Loaded global parameters:")
    body = Body(tri_facets)  # or pass your facets list here
    body.facets = tri_facets
    volume = body.calculate_volume()
    logger.info(f"Initial volume of object: {volume}")

    try:
        body.target_volume = Body(["target_volume"][0])
    except:
        logger.warning('Potential issue detected: No target volume given.')

    return vertices, tri_facets, bodies, global_params

if __name__ == '__main__':
    logger = setup_logging('membrane_solver.log')

    inpfile = "meshes/sample_geometry.json"
    data = load_data(inpfile)
    vertices, facets, body, global_params = parse_inputfile(data=data)


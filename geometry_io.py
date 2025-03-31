# geometry_io.py 
import json, yaml
from geometry_entities import Vertex, Edge, Facet, Body
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
    with open(filename, 'r') as f:
        if filename.endswith((".yaml", ".yml")):
            data = yaml.safe_load(f)
            raise ValueError("Currently yaml format not supported.")
        elif filename.endswith(".json"):
            data = json.load(f)
        else:
            raise ValueError(f"Unsupported file format for: {filename}")

    return data

def build_vertices(vertices_data):
    # construct vertex objects
    vertices = []
    for idx, vertex in enumerate(data["vertices"]):
        coords = vertex[:3]   # Extract first 3 elements as coordinates
        options = vertex[3] if len(vertex) > 3 else {}
        if options and not isinstance(options, dict):
            raise ValueError(f"Vertex #{idx} options must be a dictionary: {options}")
        vertex = Vertex(coords, idx, options)    # Pass attributes as keyword args
        vertices.append(vertex)
    return vertices

def build_edges(edges_data, vertices):
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
            edge = Edge(tail, head, options)
            edge_map[key] = edge
            edges.append(edge)
            logger.debug(f"Created edge {key} with options: {options}")
        else:
            # Optional: merge or warn on duplicate with different options
            logger.debug(f"Duplicate edge {key} found. Skipping.")
    return edges

def build_facet_edges(edge_indices, edges):
    """
    takes facet data and oriented edges list
    builds the correct oriented edges for the facet

    Build a list of edges for a single facet from its edge indices.
    Negative indices mean the edge is reversed.
    Orientation matters for defining a consistent normal.
    """
    if not edges_indices:
        # TODO: add to logger
        raise ValueError("No edges specified for facet!")

    oriented_facets = []
    # First pass: build oriented edges (with reversed direction if needed)
    for idx in edge_indices:
        reverse = (idx<0)
        edge_id = abs(idx)

        if edge_id >= len(edes):
            # TODO: add to logger
            raise IndexError(f"Edge index {idx} out of range.")
        e = edges[edge_id]
        if reverse:
            # Create a reversed edge
            new_edge = Edge(e.head_vertex, e.tail_vertex, e.options.copy())
            # Recompute the vector explicitly (if needed)
            new_edge.vector = new_edge.head - new_edge.tail
        else:
            new_edge = e
        oriented_edges.append(new_edge)

    # Second pass: verify sequence of oriented edges is continuous
    n = len(oriented_edes)
    for i in range(n):
        current_edge = oriented_edge[i]
        next_edge = oriented_edge[(i + 1) % n]  # wrap-around for closure checkup
        if current_edge.head != next_edge.tail:
            # TODO: replace with np.allclose for numerical reasone 
            # TODO: add to logger
            raise ValueError(
                "Facet edge loop is not continuous at edge {i}: "
                f"{current_edge.head_vertex.position} != {next_edge.tail_vertex.position}"
            )
    return oriented_edges

def build_facets(facets_data, edges):
    facets = []
    for idx, facet in enumerate(facets_data):
        if facet and isinstance(facet[-1], dict):
            facet_edges_idx = facet[:-1]
            options = facet[-1]
        else:
            facet_edges_idx = facet
            options = {}

        facet_edges = build_facet_edges(facet_edges_idx, edges)
        # TODO: I am working on this section now
        """
        build_facet_edges now gets the indices of its edges, and the full edges
        list.
        should take the relevant edges, orient them correctly in space, and
        check that the shape is closed. Retuen the properly oriented edges.
        Should I refactor the flip vector so that I can use it later, maybe in
        the refining functions?
        """
        facet = Facet(facet_edges, options)
        facets.append(facet)
    sys.exit("Facet edges break point")

    # construct facets objects
    facets = []

    for face in data["faces"]:
        if face and isinstance(face[-1], dict):
            edge_indices = face[:-1]
            options = face[-1]
        else:
            edge_indices = face
            options = {}

        facet_edges = []

        # Test if facet is a closed loop
        first_vertex = edges[edge_indices[0]].tail.position
        last_vertex = edges[edge_indices[-1]].head.position
        # Use allclose to take care of numerical errors 
        if not np.allclose(first_vertex, last_vertex):
            # continue
            raise ValueError(f"Edge loop is not continuous on facet, {face}")

        for ei in edge_indices:
            edges[edge_indices[-1]]
            reverse = ei < 0
            edge_index = abs(ei)

            if edge_index >= len(edges):
                raise IndexError(f"Edge index {ei} out of range for edge list of length {len(edges)}")

            original_edge = edges[edge_index]

            if reverse:
                # Create a reversed Edge object (new head/tail)
                reversed_edge = Edge(original_edge.head,
                                     original_edge.tail, original_edge.options.copy())
                facet_edges.append(reversed_edge)
            else:
                facet_edges.append(original_edge)

        if len(facet_edges) < 3:
            raise ValueError(f"Facet with edges {edge_indices} has fewer than three edges!")

        facets.append(Facet(facet_edges, options))

    return facets

def build_body(body_data, facets):
    return bodies

def load_geometry(data):
    """
    Takes loaded data from geometry file

    Returns:
        vertices (list of Vertex): List of vertex objects.
        facets (list of Facet): List of Facet objects.
        volume: Calculated volume of object.
    """
    vertices = build_vertices(data["vertices"])
    logger.info(f"Successfuly building vertices.")

    edges = build_edges(data["edges"], vertices)
    logger.info(f"Successfuly building edges.")

    facets = build_facets(data["faces"], edges)
    logger.info(f"Message on facets building")
    sys.exit(1)

    # build body object
    body_data = data.get("body", {})
    global_params = data.get("global_parameters", {})

    body = Body(facets)
    body.target_volume = body_data.get("target_volume", 0)[0]
    body.calculate_volume()

    print("##################################")
    print(f"body: {body_data}")
    print(f"Initial volume: {body.volume}")
    print(f"Target volume: {body.target_volume}")
    print("##################################")

    # Default modules logic
    # TODO: other modules should be loaded, not just from body, should be
    #           appended when each object is loaded/built 
    energy_modules = set(body_data.get("energy_modules", []))

    if "surface" not in energy_modules:
        energy_modules.add("surface")

    if "target_volume" in body_data and "volume" not in energy_modules:
        energy_modules.add("volume")

    # Load all modules now
    loaded_modules = {}
    for module_name in energy_modules:
        try:
            module = module = importlib.import_module(f"modules.{module_name}")
            loaded_modules[module_name] = module
            logger.info(f"Loaded module: {module_name}")
        except ImportError as e:
            logger.error(f"Module '{module_name}' loading error: {e}")

    if not loaded_modules:
        raise ValueError("No energy modules loaded. Check input file.")

    return vertices, facets, global_params, body, loaded_modules

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
        centroid_vertex = Vertex(centroid_pos)
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

def main(data):
    vertices, facets, global_params, body, modules = load_geometry(data=data)

    logger.info(f"Number of vertices: {len(vertices)}")
    logger.info("Loaded vertices:")
    for v in vertices:
        logger.info(v.position)
    logger.info(f"Number of facets: {len(facets)}")
    logger.info("Loaded facets:")
    for facet in facets:
        logger.info(f"{facet} {facet.options}")
    logger.info("Loaded global parameters:")
    for param in global_params.keys():
        logger.info(f"{param} = {global_params[param]}")

    # Perform the initial triangulation (always subdividing non-simplex facets).
    vertices, tri_facets = initial_triangulation(vertices, facets)
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

    return vertices, tri_facets, global_params, body, modules

if __name__ == '__main__':
    logger = setup_logging('membrane_solver.log')

    inpfile = "meshes/sample_geometry.json"
    data = load_data(inpfile)
    vertices, facets, global_params, body, modules = main(data=data)


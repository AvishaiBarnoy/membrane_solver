# geometry_io.py 
import json
from geometry_entities import Vertex, Facet, Body
import os
import sys
from logging_config import setup_logging
import importlib

def load_geometry(filename):
    """
    Load geometry from a JSON file.

    Expected JSON format:
    {
        "vertices": [[x, y, z], ...],
        "faces": [
            [i, j, k, ...] or [i, j, k, ..., {"refine": false, "surface_tension": 0.8}],
            ...
        ]
    }

    Returns:
        vertices (list of Vertex): List of vertex objects.
        facets (list of Facet): List of Facet objects.
        volume: Calculated volume of object.
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    vertices = []
    for item in data["vertices"]:
        coords = item[:3]   # Extract first 3 elements as coordinates
        attributes = item[3] if len(item) > 3 else {}   # Extract optional attributes
        vertex = Vertex(coords, **attributes)    # Pass attributes as keyword args
        vertices.append(vertex)

    facets = []
    for face in data["faces"]:
        if face and isinstance(face[-1], dict):
            indices = face[:-1]
            options = face[-1]
        else:
            indices = face
            options = {}
        facets.append(Facet(indices, options))

    body_data = data.get("body", {})
    global_params = data.get("global_parameters", {})

    body = Body(facets)
    initial_volume = body.calculate_volume(vertices)

    # Default modules logic
    energy_modules = set(body_data.get("energy_modules", []))

    if "target_volume" in body_data and "volume" not in energy_modules:
        energy_modules.add("volume")

    if "surface_energy" not in energy_modules:
        energy_modules.add("surface_energy")

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
        # Always triangulate if the facet is non-triangle.
        if len(facet.indices) == 3:
            new_facets.append(facet)
        elif len(facet.indices) > 3:
            # Compute centroid as the average of the vertex positions.
            pts = [vertices[i].position for i in facet.indices]
            centroid = [sum(coords)/len(pts) for coords in zip(*pts)]
            centroid_index = len(vertices)
            vertices.append(Vertex(centroid))
            n = len(facet.indices)
            # Create n new triangles using each edge and the centroid.
            for i in range(n):
                tri_indices = (facet.indices[i], facet.indices[(i + 1) % n], centroid_index)
                # Use a copy of the parent's options for the child facets.
                new_facets.append(Facet(tri_indices, facet.options.copy()))
        else:
            raise ValueError("Facet with fewer than three vertices encountered!")
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
        "vertices": [list(v.position) for v in vertices],
        "faces": [],
        "volume": volume
    }

    for facet in facets:
        # If there are facet options, append them as the last element.
        if facet.options:
            data["faces"].append(list(facet.indices) + [facet.options])
        else:
            data["faces"].append(list(facet.indices))

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

def main():
    # TODO:
    # 1. run the read input file
    # 2. triangulate
    # 3. calculate volume
    # 4. return all relevant values
    try:
        inpfile = sys.argv[1]
    except IndexError:
        inpfile = "meshes/sample_geometry.json"

    vertices, facets, global_params, body, modules = load_geometry(inpfile)
    logger.info(f"Number of vertices: {len(vertices)}")
    logger.info("Loaded vertices:")
    for v in vertices:
        logger.info(v.position)
    logger.info(f"Number of facets: {len(facets)}")
    logger.info("Loaded facets:")
    for facet in facets:
        logger.info(f"{facet.indices} {facet.options}")
    logger.info("Loaded global parameters:")
    for param in global_params.keys():
        logger.info(f"{param} = {global_params[param]}")

    # Perform the initial triangulation (always subdividing non-simplex facets).
    vertices, tri_facets = initial_triangulation(vertices, facets)
    logger.info("\nAfter initial triangulation:")
    logger.info(f"Number of vertices: {len(vertices)}")
    for v in vertices:
        logger.info(v.position)
    logger.info(f"Number of facets: {len(facets)}")
    logger.info("Triangulated facets:")
    for facet in facets:
        logger.info(f"{facet.indices} {facet.options}")

    logger.info("Loaded global parameters:")
    body = Body(tri_facets)  # or pass your facets list here
    body.calculate_volume(vertices)   # calculates volume at loading
    try:
        body.target_volume = Body(["target_volume"][0])
    except:
        logger.warning('Potential issue detected: No target volume given.')

    initial_volume = body.calculate_volume(vertices)
    logger.info(f"Initial volume of object: {body.volume}")

    return vertices, tri_facets, global_params, body, modules

if __name__ == '__main__':
    logger = setup_logging()

    vertices, facets, global_params, body, modules = main()


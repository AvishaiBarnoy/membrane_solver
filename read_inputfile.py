# read_inputfile.py
import json
from geometry_entities import Vertex, Facet
import sys
from logging_config import setup_logging

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
        volume: Current volume of the object if volume is given
    """
    with open(filename, 'r') as f:
        data = json.load(f)

    # vertices = [Vertex(pos) for pos in data["vertices"]]  # depracted definition
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
    volumes = {'target volume': None, 'current volume': None} # initialize volume dicitionary
    # TODO: replace direct access to volume with dynamic access to multiple objects
    volumes['target volume'] = data['body']['target_volume'][0]
    volumes['current volume'] = calculate_volume()
    return vertices, facets, volumes

def calculate_volume():
    """
    Calculates volume of input geometry
    target volume is the volume the object wants to be with without any
    constraints
    """
    # TODO: fill this 
    return None

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
        # Always triangulate if the facet is non-simplex.
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

if __name__ == '__main__':
    try:
        inpfile = sys.argv[1]
    except IndexError:
        inpfile = "meshes/sample_geometry.json"

    logger = setup_logging()

    vertices, facets, volumes = load_geometry(inpfile)
    logger.info("Loaded vertices:")
    for v in vertices:
        logger.info(v.position)
    logger.info("Loaded facets:")
    for facet in facets:
        logger.info(f"{facet.indices} {facet.options}")

    # Perform the initial triangulation (always subdividing non-simplex facets).
    vertices, tri_facets = initial_triangulation(vertices, facets)
    logger.info("\nAfter initial triangulation:")
    logger.info(f"Number of vertices: {len(vertices)}")
    for facet in tri_facets:
        logger.info(f"{facet.indices} {facet.options}")
    logger.info(f"Initial volume of object: {volumes['current volume']}")
    logger.info(f"Target volume of object: {volumes['target volume']}")

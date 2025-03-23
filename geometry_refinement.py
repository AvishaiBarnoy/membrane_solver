# geometry_refinement.py
# TODO:
#   1. refining options:
#       1.1 center of mass of face - default
#       1.2  Delaunay algorithm
# geometry_refinement.py

from geometry_entities  import Vertex, Facet
from logging_config import setup_logging

def refine_mesh(vertices, facets):
    """
    Refines a triangular mesh by subdividing every edge for each facet whose
    options permit refinement. For each triangle that is refined, a new vertex
    is inserted at the midpoint of each edge, and the triangle is subdivided into
    four smaller triangles.

    Facets with the option {"refine": False} are left unmodified.

    Child facets inherit a copy of the parent facet's options.

    This function uses a cache (dictionary) so that the midpoint for a given edge is
    computed only once.

    Args:
        vertices (list of Vertex): The list of vertices.
        facets (list of Facet): The list of triangular facets.

    Returns:
        (vertices, new_facets): The updated list of vertices and the new list of facets.
    """
    midpoint_cache = {}
    new_facets = []

    def get_midpoint(i, j):
        # Use a sorted tuple as key so that edge (i,j) is the same as (j,i)
        key = tuple(sorted((i, j)))
        if key in midpoint_cache:
            return midpoint_cache[key]
        else:
            pos1 = vertices[i].position
            pos2 = vertices[j].position
            midpoint = [(a + b) / 2 for a, b in zip(pos1, pos2)]
            mid_index = len(vertices)
            vertices.append(Vertex(midpoint))
            midpoint_cache[key] = mid_index
            return mid_index

    # Process each facet in the mesh.
    for facet in facets:
        # Check the refine option for this facet.
        if facet.options.get("refine", True) is False:
            new_facets.append(facet)
        else:
            # Ensure the facet is triangular.
            if len(facet.indices) != 3:
                raise ValueError("Refinement expects triangular facets only!")
            a, b, c = facet.indices
            ab = get_midpoint(a, b)
            bc = get_midpoint(b, c)
            ca = get_midpoint(c, a)

            # Subdivide the triangle into 4 smaller triangles.
            new_facets.append(Facet((a, ab, ca), facet.options.copy()))
            new_facets.append(Facet((b, bc, ab), facet.options.copy()))
            new_facets.append(Facet((c, ca, bc), facet.options.copy()))
            new_facets.append(Facet((ab, bc, ca), facet.options.copy()))

    return vertices, new_facets

if __name__ == '__main__':
    import sys
    from geometry_io import load_geometry, initial_triangulation

    logger = setup_logging()

    try:
        inpfile = sys.argv[1]
    except IndexError:
        inpfile = "meshes/sample_geometry.json"

    vertices, facets, volume = load_geometry(inpfile)
    logger.info("Loaded vertices:")
    for v in vertices:
        logger.info(v.position)
    logger.info("Loaded facets:")
    for facet in facets:
        logger.info(f"{facet.indices} {facet.options}")

    # Perform the initial triangulation (always subdividing to triangles).
    vertices, tri_facets = initial_triangulation(vertices, facets)
    logger.info("\nAfter initial triangulation:")
    logger.info(f"Number of vertices: {len(vertices)}")
    for facet in tri_facets:
        logger.info(f"{facet.indices} {facet.options}")

    # Refine the mesh: facets with {"refine": False} remain unchanged.
    vertices, refined_facets = refine_mesh(vertices, tri_facets)
    logger.info("\nAfter refinement:")
    logger.info(f"Number of vertices: {len(vertices)}")
    for facet in refined_facets:
        logger.info(f"{facet.indices} {facet.options}")


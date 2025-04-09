# visualize_geometry.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# visualize_geometry.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

# Import functions from your modules.
from geometry.geometry_io import parse_geometry, load_data
from runtime.refinement import refine_triangle_mesh, refine_polygonal_facets

import logging
from logging_config import setup_logging
logger = logging.getLogger('membrane_solver')

# TODO: shading option when rotating
# TODO: opaque scatter when transparent=False


def plot_geometry(vertices, facets, show_indices=False, ax=None,
                  transparent=False):
    """
    Visualizes the triangulated geometry.

    Args:
        vertices (list of Vertex): geometry vertices.
        facets (list of Facet): list of triangle facets (must be triangulated).
        show_indices (bool): show vertex indices.
        ax (mpl_toolkits.mplot3d.Axes3D): optional matplotlib axis.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    vertex_positions = [v.position for v in vertices]
    X, Y, Z = zip(*vertex_positions)

    # Plot triangle facets
    triangles = []
    for facet in facets:
        # Reconstruct the triangle from edges
        if len(facet.edges) != 3:
            continue
        tri = [e.tail.position for e in facet.edges]
        triangles.append(tri)

    if transparent:
        alpha = 0.4
    else:
        alpha = 1
    tri_collection = Poly3DCollection(triangles, alpha=alpha, edgecolor='k')
    tri_collection.set_facecolor((0.6, 0.8, 1.0))
    ax.add_collection3d(tri_collection)

    # Optional: plot vertices
    ax.scatter(X, Y, Z, color='r', s=20)

    if show_indices:
        for v in vertices:
            ax.text(*v.position, f"{v.index}", color='k', fontsize=8)
    # TODO: add option to show index of facet

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Refined Geometry")
    ax.auto_scale_xyz(X, Y, Z)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    logger = setup_logging('membrane_solver.log')

    try:
        inpfile = sys.argv[1]
        if not inpfile.endwith('.json'):
            raise ValueError("Input file must be a JSON file (.json).")
        if not os.path.isfile(inpfile):
            raise FileNotFoundError(f"Input file '{inpfile}' not found!")
    except IndexError:
        inpfile = "meshes/sample_geometry.json"

    # Load geometry from the input file.
    # vertices, facets, volume = load_geometry(inpfile)
    data = load_data(inpfile)
    vertices, edges, facets, bodies, global_params = parse_geometry(data=data)
    vertices, edges, tri_facets, bodies = refine_polygonal_facets(vertices, edges, facets, bodies, global_params) # initial triangulation

    # Perform the initial triangulation on loaded facets.
    #vertices, tri_facets = initial_triangulation(vertices, facets)

    # Optionally, perform a refinement step.
    vertices, edges, tri_facets, bodies = refine_triangle_mesh(vertices, edges, tri_facets, bodies)
    #vertices, edges, tri_facets = refine_triangle_mesh(vertices, edges, tri_facets)

    # Visualize the resulting triangulated geometry.
    plot_geometry(vertices, tri_facets, show_indices=False)

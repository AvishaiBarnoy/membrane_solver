# visualize_geometry.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import sys

# Import functions from your modules.
from geometry_io import load_geometry, initial_triangulation
from geometry_refinement import refine_mesh

def plot_geometry(vertices, facets, title="Triangulated Geometry"):
    """
    Creates a 3D plot of the triangulated geometry.
    
    Args:
        vertices (list): List of Vertex objects with a .position attribute.
        facets (list): List of Facet objects (each facet has an .indices attribute).
        title (str): Title for the plot.
    """
    # Extract vertex positions as lists.
    points = [v.position for v in vertices]

    # Build the list of face polygons from vertex positions using each facet's indices.
    polygons = [[points[idx] for idx in facet.indices] for facet in facets]
    
    # Create a 3D plot.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    poly3d = Poly3DCollection(polygons, edgecolor='k', facecolor='cyan', alpha=0.6)
    ax.add_collection3d(poly3d)
    
    # Extract coordinates for scaling the axes.
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    zs = [p[2] for p in points]
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(min(ys), max(ys))
    ax.set_zlim(min(zs), max(zs))
    
    ax.set_title(title)
    plt.show()

if __name__ == '__main__':
    try:
        inpfile = sys.argv[1]
    except IndexError:
        inpfile = "meshes/sample_geometry.json"
    
    # Load geometry from the input file.
    vertices, facets, volume = load_geometry(inpfile)
    
    # Perform the initial triangulation on loaded facets.
    vertices, tri_facets = initial_triangulation(vertices, facets)
    
    # Optionally, perform a refinement step.
    vertices, tri_facets = refine_mesh(vertices, tri_facets)
    
    # Visualize the resulting triangulated geometry.
    plot_geometry(vertices, tri_facets, title="Initial Triangulation")


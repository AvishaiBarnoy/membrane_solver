# visualize_geometry.py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os, sys
import numpy as np

# Import functions from your modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from geometry.geom_io import parse_geometry, load_data
from runtime.refinement import refine_triangle_mesh, refine_polygonal_facets
from logging_config import setup_logging

logger = setup_logging('membrane_solver')
logger = logger.getChild('visualizer')

def plot_geometry(mesh, show_indices=False, scatter=False, ax=None,
                  transparent=False):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    vertex_positions = [mesh.vertices[v].position for v in mesh.vertices.keys()]
    X, Y, Z = zip(*vertex_positions)

    # Plot triangle facets
    triangles = []
    for facet in mesh.facets.values():
        if len(facet.edge_indices) < 3:
            logger.warning(f"Skipping facet {facet.index}: too few edges")
            continue
        tri = [mesh.vertices[mesh.get_edge(e).tail_index].position for e in facet.edge_indices]
        triangles.append(tri)

    alpha = 0.4 if transparent else 1.0
    tri_collection = Poly3DCollection(triangles, alpha=alpha, edgecolor='k')
    tri_collection.set_facecolor((0.6, 0.8, 1.0))  # sky blue
    ax.add_collection3d(tri_collection)

    if scatter:
        ax.scatter(X, Y, Z, color='r', s=20)

    if show_indices:
        for v in mesh.vertices.values():
            ax.text(*v.position, f"{v.index}", color='k', fontsize=8)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Refined Geometry")
    ax.auto_scale_xyz(X, Y, Z)
    plt.tight_layout()
    plt.show()

def compute_sphericity_from_mesh(mesh):
    vertex_positions = np.array([v.position for v in mesh.vertices.values()])
    center = np.mean(vertex_positions, axis=0)
    radii = np.linalg.norm(vertex_positions - center, axis=1)
    avg_radius = np.mean(radii)
    std_radius = np.std(radii)
    min_radius = np.min(radii)
    max_radius = np.max(radii)

    def triangle_area(v0, v1, v2):
        return 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))

    area = 0
    for facet in mesh.facets.values():
        if len(facet.edge_indices) != 3:
            continue
        v0 = mesh.vertices[mesh.get_edge(facet.edge_indices[0]).tail_index].position
        v1 = mesh.vertices[mesh.get_edge(facet.edge_indices[1]).tail_index].position
        v2 = mesh.vertices[mesh.get_edge(facet.edge_indices[2]).tail_index].position
        area += triangle_area(v0, v1, v2)

    expected_area = 4 * np.pi * avg_radius**2
    volume = (4/3) * np.pi * avg_radius**3
    sphericity = (np.pi**(1/3) * (6 * volume)**(2/3)) / area if area > 0 else np.nan

    return {
        "center": center,
        "average_radius": avg_radius,
        "radius_std_dev": std_radius,
        "min_radius": min_radius,
        "max_radius": max_radius,
        "surface_area": area,
        "expected_area": expected_area,
        "area_ratio": area / expected_area if expected_area > 0 else np.nan,
        "sphericity": sphericity
    }

if __name__ == '__main__':
    try:
        inpfile = sys.argv[1]
        if not inpfile.endswith('.json'):
            raise ValueError("Input file must be a JSON file (.json).")
        if not os.path.isfile(inpfile):
            raise FileNotFoundError(f"Input file '{inpfile}' not found!")
    except IndexError:
        inpfile = "meshes/sample_geometry.json"

    # Load geometry from the input file
    data = load_data(inpfile)
    mesh = parse_geometry(data=data)

    # Compute and print sphere-likeness metrics
    metrics = compute_sphericity_from_mesh(mesh)
    print("\nSphere-Likeness Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}")

    # Visualize the mesh
    plot_geometry(mesh, show_indices=False, scatter=True)

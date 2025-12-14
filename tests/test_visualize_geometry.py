import os
import sys

import matplotlib

# Use a non-interactive backend suitable for testing.
matplotlib.use("Agg")

import numpy as np

# Ensure project root is on sys.path for direct test execution.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Mesh, Vertex
from geometry.geom_io import parse_geometry
from tests.sample_meshes import SAMPLE_GEOMETRY
from visualize_geometry import plot_geometry


def _build_edge_only_mesh() -> Mesh:
    mesh = Mesh()
    mesh.vertices[0] = Vertex(index=0, position=np.array([0.0, 0.0, 0.0]))
    mesh.vertices[1] = Vertex(index=1, position=np.array([1.0, 0.0, 0.0]))
    mesh.edges[1] = Edge(index=1, tail_index=0, head_index=1)
    mesh.build_connectivity_maps()
    return mesh


def test_plot_geometry_edges_only_does_not_crash():
    """plot_geometry should handle meshes with only edges."""
    mesh = _build_edge_only_mesh()
    # Should run without raising; drawing happens on an Agg canvas.
    plot_geometry(mesh, draw_facets=False, draw_edges=True, scatter=False)


def test_plot_geometry_colors_from_options_and_kwargs():
    """plot_geometry should respect facet and edge color options."""
    data = SAMPLE_GEOMETRY.copy()
    mesh = parse_geometry(data)

    # Tag one facet and one edge with explicit colors via options.
    if mesh.facets:
        first_facet = next(iter(mesh.facets.values()))
        first_facet.options["color"] = "red"
    if mesh.edges:
        first_edge = next(iter(mesh.edges.values()))
        first_edge.options["color"] = "green"

    # Also supply default colors via kwargs; per-entity options should override.
    plot_geometry(
        mesh,
        draw_facets=True,
        draw_edges=True,
        facet_color="blue",
        edge_color="black",
    )

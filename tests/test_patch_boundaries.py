import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.entities import Edge, Facet, Mesh, Vertex
from runtime.diagnostics.patches import patch_boundary_edges, patch_boundary_lengths
from visualization.plotting import plot_geometry


def _square_two_triangles() -> Mesh:
    mesh = Mesh()
    mesh.vertices = {
        0: Vertex(0, np.array([0.0, 0.0, 0.0])),
        1: Vertex(1, np.array([1.0, 0.0, 0.0])),
        2: Vertex(2, np.array([1.0, 1.0, 0.0])),
        3: Vertex(3, np.array([0.0, 1.0, 0.0])),
    }
    mesh.edges = {
        1: Edge(1, 0, 1),
        2: Edge(2, 1, 2),
        3: Edge(3, 2, 3),
        4: Edge(4, 3, 0),
        5: Edge(5, 0, 2),
    }
    mesh.facets = {
        1: Facet(1, [1, 2, -5], options={"disk_patch": "top"}),
        2: Facet(2, [5, 3, 4], options={}),
    }
    mesh.build_connectivity_maps()
    mesh.build_facet_vertex_loops()
    return mesh


def test_patch_boundary_edges_excludes_mesh_boundary():
    mesh = _square_two_triangles()
    groups = patch_boundary_edges(mesh, patch_key="disk_patch")

    assert set(groups.keys()) == {"top"}
    assert {e.index for e in groups["top"]} == {5}


def test_patch_boundary_edges_between_two_patches_includes_both():
    mesh = _square_two_triangles()
    mesh.facets[2].options["disk_patch"] = "bottom"

    groups = patch_boundary_edges(mesh, patch_key="disk_patch")
    assert set(groups.keys()) == {"bottom", "top"}
    assert {e.index for e in groups["top"]} == {5}
    assert {e.index for e in groups["bottom"]} == {5}


def test_patch_boundary_lengths_match_diagonal():
    mesh = _square_two_triangles()
    lengths = patch_boundary_lengths(mesh, patch_key="disk_patch")

    assert set(lengths.keys()) == {"top"}
    assert np.isclose(lengths["top"], np.sqrt(2.0))


def test_plot_geometry_can_overlay_patch_and_boundary_loops():
    mesh = _square_two_triangles()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_geometry(
        mesh,
        ax=ax,
        show=False,
        draw_edges=False,
        draw_facets=False,
        show_patch_boundaries=True,
        show_boundary_loops=True,
        annotate_boundary_geodesic=True,
    )

    labels = {getattr(coll, "get_label", lambda: None)() for coll in ax.collections}
    assert "_patch_boundaries" in labels
    assert any(
        isinstance(coll, Line3DCollection) and coll.get_label() == "_boundary_loops"
        for coll in ax.collections
    )

    plt.close(fig)

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from geometry.entities import Edge, Facet, Mesh, Vertex
from visualization.plotting import (
    _colormap_norm_for_scalars,
    plot_geometry,
    update_live_vis,
)


def test_colormap_norm():
    # Test tilt_mag (Sequential)
    vals = np.array([0.0, 0.5, 1.0])
    cmap, norm = _colormap_norm_for_scalars("tilt_mag", vals)
    assert cmap.name == "viridis"
    assert norm.vmin == 0.0
    assert norm.vmax == 1.0

    # Test tilt_div (Divergent)
    vals = np.array([-1.0, 0.0, 2.0])
    cmap, norm = _colormap_norm_for_scalars("tilt_div", vals)
    assert cmap.name == "coolwarm"
    assert norm.vcenter == 0.0
    assert norm.vmin == -2.0
    assert norm.vmax == 2.0


def test_plot_geometry_tilt_features():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(
        0, np.array([0.0, 0.0, 0.0]), tilt=np.array([1.0, 0.0, 0.0])
    )
    mesh.vertices[1] = Vertex(
        1, np.array([1.0, 0.0, 0.0]), tilt=np.array([1.0, 0.0, 0.0])
    )
    mesh.vertices[2] = Vertex(
        2, np.array([0.0, 1.0, 0.0]), tilt=np.array([1.0, 0.0, 0.0])
    )
    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 2)
    mesh.edges[3] = Edge(3, 2, 0)
    mesh.facets[0] = Facet(0, [1, 2, 3])
    mesh.build_facet_vertex_loops()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Test with colorbar and arrows
    plot_geometry(
        mesh,
        ax=ax,
        color_by="tilt_mag",
        show_colorbar=True,
        show_tilt_arrows=True,
        show=False,
    )

    # Check if collections were added
    poly_cols = [c for c in ax.collections if "Poly3DCollection" in str(type(c))]
    line_cols = [c for c in ax.collections if "Line3DCollection" in str(type(c))]

    assert len(poly_cols) >= 1
    # Tilt arrows have label "_tilt_arrows"
    arrow_cols = [c for c in line_cols if c.get_label() == "_tilt_arrows"]
    assert len(arrow_cols) == 1

    # Check if colorbar was set on axis
    assert hasattr(ax, "_membrane_colorbar")
    assert ax._membrane_colorbar is not None

    plt.close(fig)


def test_update_live_vis_colorbar():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(
        0, np.array([0.0, 0.0, 0.0]), tilt=np.array([1.0, 0.0, 0.0])
    )
    mesh.vertices[1] = Vertex(
        1, np.array([1.0, 0.0, 0.0]), tilt=np.array([1.0, 0.0, 0.0])
    )
    mesh.vertices[2] = Vertex(
        2, np.array([0.0, 1.0, 0.0]), tilt=np.array([1.0, 0.0, 0.0])
    )
    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 2)
    mesh.edges[3] = Edge(3, 2, 0)
    mesh.facets[0] = Facet(0, [1, 2, 3])
    mesh.build_facet_vertex_loops()

    # Initial slow path call
    state = update_live_vis(mesh, color_by="tilt_mag", show_colorbar=True)
    assert state["colorbar"] is not None
    assert state["color_by"] == "tilt_mag"

    # Fast update path call (topology same)
    state2 = update_live_vis(mesh, state=state, color_by="tilt_mag", show_colorbar=True)
    assert state2["colorbar"] is state["colorbar"]

    plt.close(state["fig"])


def test_update_live_vis_bilayer():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(
        0,
        np.array([0.0, 0.0, 0.0]),
        tilt_in=np.array([0.0, 1.0, 0.0]),
        tilt_out=np.array([0.0, 0.0, 1.0]),
    )
    mesh.vertices[1] = Vertex(
        1,
        np.array([1.0, 0.0, 0.0]),
        tilt_in=np.array([0.0, 1.0, 0.0]),
        tilt_out=np.array([0.0, 0.0, 1.0]),
    )
    mesh.vertices[2] = Vertex(
        2,
        np.array([0.0, 1.0, 0.0]),
        tilt_in=np.array([0.0, 1.0, 0.0]),
        tilt_out=np.array([0.0, 0.0, 1.0]),
    )
    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 2)
    mesh.edges[3] = Edge(3, 2, 0)
    mesh.facets[0] = Facet(0, [1, 2, 3])
    mesh.build_facet_vertex_loops()

    state = update_live_vis(mesh, color_by="tilt_bilayer", show_colorbar=True)
    assert state["tri_collection_top"] is not None
    assert state["tri_collection_bottom"] is not None

    state2 = update_live_vis(
        mesh, state=state, color_by="tilt_bilayer", show_colorbar=True
    )
    assert state2["tri_collection_top"] is state["tri_collection_top"]
    assert state2["tri_collection_bottom"] is state["tri_collection_bottom"]

    plt.close(state["fig"])


def test_plot_geometry_tilt_leaflet_fields():
    mesh = Mesh()
    mesh.vertices[0] = Vertex(
        0,
        np.array([0.0, 0.0, 0.0]),
        tilt=np.array([1.0, 0.0, 0.0]),
        tilt_in=np.array([0.0, 1.0, 0.0]),
        tilt_out=np.array([0.0, 0.0, 1.0]),
    )
    mesh.vertices[1] = Vertex(
        1,
        np.array([1.0, 0.0, 0.0]),
        tilt=np.array([1.0, 0.0, 0.0]),
        tilt_in=np.array([0.0, 1.0, 0.0]),
        tilt_out=np.array([0.0, 0.0, 1.0]),
    )
    mesh.vertices[2] = Vertex(
        2,
        np.array([0.0, 1.0, 0.0]),
        tilt=np.array([1.0, 0.0, 0.0]),
        tilt_in=np.array([0.0, 1.0, 0.0]),
        tilt_out=np.array([0.0, 0.0, 1.0]),
    )
    mesh.edges[1] = Edge(1, 0, 1)
    mesh.edges[2] = Edge(2, 1, 2)
    mesh.edges[3] = Edge(3, 2, 0)
    mesh.facets[0] = Facet(0, [1, 2, 3])
    mesh.build_facet_vertex_loops()

    for color_by in (
        "tilt_in",
        "tilt_out",
        "tilt_div_in",
        "tilt_div_out",
        "tilt_bilayer",
    ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plot_geometry(mesh, ax=ax, color_by=color_by, show=False)
        fig.canvas.draw()
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__])

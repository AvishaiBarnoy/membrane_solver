import os
import sys

import matplotlib

# Use a non-interactive backend suitable for testing.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from visualization.plotting import plot_geometry


def _triangle_mesh_with_tilt() -> dict:
    return {
        "vertices": [
            [0.0, 0.0, 0.0, {"tilt": [1.0, 0.0]}],
            [1.0, 0.0, 0.0, {"tilt": [0.0, 0.0]}],
            [0.0, 1.0, 0.0, {"tilt": [0.0, 0.0]}],
        ],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
        "energy_modules": ["tilt"],
        "global_parameters": {"surface_tension": 0.0, "volume_constraint_mode": "none"},
        "instructions": [],
    }


def test_plot_geometry_adds_colorbar_when_coloring_by_tilt():
    mesh = parse_geometry(_triangle_mesh_with_tilt())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_geometry(mesh, ax=ax, show=False, draw_edges=False, color_by="tilt_mag")

    cbar = getattr(ax, "_membrane_colorbar", None)
    assert cbar is not None
    assert cbar.ax.get_ylabel() in {"|t|", "div(t)"}


def test_plot_geometry_removes_colorbar_when_returning_to_plain():
    mesh = parse_geometry(_triangle_mesh_with_tilt())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_geometry(mesh, ax=ax, show=False, draw_edges=False, color_by="tilt_mag")
    assert getattr(ax, "_membrane_colorbar", None) is not None

    plot_geometry(mesh, ax=ax, show=False, draw_edges=False, color_by=None)
    assert getattr(ax, "_membrane_colorbar", None) is None


def test_plot_geometry_can_overlay_tilt_arrows():
    mesh = parse_geometry(_triangle_mesh_with_tilt())

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_geometry(
        mesh,
        ax=ax,
        show=False,
        draw_edges=False,
        color_by=None,
        show_tilt_arrows=True,
    )

    assert any(
        isinstance(coll, Line3DCollection) and coll.get_label() == "_tilt_arrows"
        for coll in ax.collections
    )

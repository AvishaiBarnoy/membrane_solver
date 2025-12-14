import logging
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from geometry.entities import Mesh

logger = logging.getLogger("membrane_solver.log")


def plot_geometry(
    mesh: Mesh,
    show_indices: bool = False,
    scatter: bool = False,
    ax=None,
    transparent: bool = False,
    draw_facets: bool = True,
    draw_edges: bool = False,
    facet_color: Any = None,
    edge_color: str = "k",
    facet_colors: Optional[Dict[int, Any]] = None,
    edge_colors: Optional[Dict[int, Any]] = None,
    no_axes: bool = False,
    show: bool = True,
) -> None:
    """
    Visualize a mesh in 3D using Matplotlib.

    Parameters
    ----------
    mesh :
        The :class:`~geometry.entities.Mesh` instance to visualize.
    show_indices : bool, optional
        If ``True``, draw vertex indices next to each vertex.
    scatter : bool, optional
        If ``True``, draw vertices as red scatter points.
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Optional Matplotlib 3D axis. If omitted, a new figure and axis
        are created.
    transparent : bool, optional
        If ``True``, draw facets semi‑transparent.
    draw_facets : bool, optional
        If ``True`` (default), draw polygonal facets as filled surfaces.
    draw_edges : bool, optional
        If ``True`` (default), draw all edges as line segments. This
        includes edges that are not part of any facet, so wire‑frame
        or line‑only meshes can be visualized.
    facet_color :
        Default color to use for facets when no per‑facet color is
        provided. If ``None``, a light blue color is used.
    edge_color :
        Default color to use for edges when no per‑edge color is
        provided. Defaults to ``"k"`` (black).
    facet_colors : dict[int, Any], optional
        Optional mapping ``facet_index -> color``. When provided it
        overrides both ``facet_color`` and any ``"color"`` entry in
        ``facet.options``.
    edge_colors : dict[int, Any], optional
        Optional mapping ``edge_index -> color``. When provided it
        overrides both ``edge_color`` and any ``"color"`` entry in
        ``edge.options``.
    show : bool, optional
        If ``True`` (default), call :func:`matplotlib.pyplot.show` after
        drawing. Set to ``False`` when using non‑interactive backends
        or when the caller is responsible for displaying or saving the
        figure.
    """
    if not mesh.vertices:
        logger.warning("Mesh has no vertices to visualize.")
        return

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    vertex_positions = [mesh.vertices[v].position for v in mesh.vertices.keys()]
    X, Y, Z = zip(*vertex_positions)

    # Plot facets as filled polygons.
    if draw_facets and mesh.facets:
        triangles = []
        face_colors = []
        default_facet_color = (
            facet_color if facet_color is not None else (0.6, 0.8, 1.0)
        )

        for facet in mesh.facets.values():
            # Facets with fewer than 3 edges cannot form a polygonal area.
            # Silently skip them here so that line‑only meshes (or auxiliary
            # facets used for line energies) do not generate noisy warnings
            # during visualization.
            if len(facet.edge_indices) < 3:
                continue

            tri = [
                mesh.vertices[mesh.get_edge(e).tail_index].position
                for e in facet.edge_indices
            ]
            triangles.append(tri)

            if facet_colors is not None and facet.index in facet_colors:
                color = facet_colors[facet.index]
            else:
                color = facet.options.get("color", default_facet_color)
            face_colors.append(color)

        if triangles:
            alpha = 0.7 if transparent else 1.0
            tri_collection = Poly3DCollection(
                triangles,
                alpha=alpha,
                edgecolor=edge_color if not draw_edges else "k",
            )
            tri_collection.set_facecolor(face_colors)
            ax.add_collection3d(tri_collection)

    # Plot all edges as line segments, including standalone edges.
    if draw_edges and mesh.edges:
        segments = []
        line_colors = []
        for edge in mesh.edges.values():
            tail = mesh.vertices[edge.tail_index].position
            head = mesh.vertices[edge.head_index].position
            segments.append([tail, head])
            if edge_colors is not None and edge.index in edge_colors:
                color = edge_colors[edge.index]
            else:
                color = edge.options.get("color", edge_color)
            line_colors.append(color)

        if segments:
            line_collection = Line3DCollection(
                segments, colors=line_colors, linewidths=1.0
            )
            ax.add_collection3d(line_collection)

    # Optional: plot vertices
    if scatter:
        ax.scatter(X, Y, Z, color="r", s=20)

    if show_indices:
        for v in mesh.vertices.values():
            ax.text(*v.position, f"{v.index}", color="k", fontsize=8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Refined Geometry")
    ax.auto_scale_xyz(X, Y, Z)
    if no_axes:
        ax.set_axis_off()
    plt.tight_layout()

    if show:
        plt.show()


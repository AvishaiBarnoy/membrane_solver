import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from geometry.entities import Mesh

logger = logging.getLogger("membrane_solver")


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
        If ``True``, draw all edges as line segments. This includes
        edges that are not part of any facet, so wire‑frame or line‑only
        meshes can be visualized. When the mesh has no facets, edges are
        automatically drawn even if ``draw_edges`` is ``False``.
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

    # For line-only meshes, automatically draw edges so the plot isn't empty.
    draw_edges = bool(draw_edges or (not mesh.facets and mesh.edges))

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    vertex_positions = [mesh.vertices[v].position for v in mesh.vertices.keys()]
    X, Y, Z = zip(*vertex_positions)

    # Plot facets as filled polygons.
    if draw_facets:
        # Try to use vectorized triangle cache first
        tri_rows, tri_facets = mesh.triangle_row_cache()
        positions = mesh.positions_view()

        triangles = []
        face_colors = []
        default_facet_color = (
            facet_color if facet_color is not None else (0.6, 0.8, 1.0)
        )

        # 1. Vectorized triangles
        if tri_rows is not None and len(tri_rows) > 0:
            # (N_tri, 3, 3) array of positions
            tri_data = positions[tri_rows]
            triangles.extend(list(tri_data))

            # Map colors
            # We need to map tri_facets (list of fid) to colors
            for fid in tri_facets:
                if facet_colors is not None and fid in facet_colors:
                    face_colors.append(facet_colors[fid])
                else:
                    # Fallback to option or default
                    # Accessing mesh.facets[fid] might be slow if loop is huge,
                    # but usually options are sparse.
                    opts = mesh.facets[fid].options
                    face_colors.append(opts.get("color", default_facet_color))

        # 2. Non-triangle facets (polygons) - Fallback
        # We need to find facets that are NOT in tri_facets
        # A set lookup is fast
        tri_set = set(tri_facets) if tri_facets else set()

        for facet in mesh.facets.values():
            if facet.index in tri_set:
                continue

            if len(facet.edge_indices) < 3:
                continue

            # Slow path for polygons
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
            alpha = 0.4 if transparent else 1.0
            tri_collection = Poly3DCollection(
                triangles,
                alpha=alpha,
                edgecolor=edge_color if not draw_edges else "k",
                linewidths=0.5 if draw_edges else 0.0,
            )
            tri_collection.set_facecolor(face_colors)
            ax.add_collection3d(tri_collection)

    # Plot all edges as line segments, including standalone edges.
    if draw_edges and mesh.edges:
        # Try vectorized edge extraction
        positions = mesh.positions_view()
        # mesh.vertex_index_to_row tells us where each vertex is in positions

        segments = []
        line_colors = []

        # Build edge indices list
        # This is still a loop, but we avoid point lookups
        idx_map = mesh.vertex_index_to_row
        edge_indices = []

        # Helper to get color
        def get_edge_color(e):
            if edge_colors is not None and e.index in edge_colors:
                return edge_colors[e.index]
            return e.options.get("color", edge_color)

        # Pre-allocate if possible? No, edges is dict.
        for edge in mesh.edges.values():
            t = idx_map.get(edge.tail_index)
            h = idx_map.get(edge.head_index)
            if t is not None and h is not None:
                edge_indices.append((t, h))
                line_colors.append(get_edge_color(edge))

        if edge_indices:
            edge_indices = np.array(edge_indices)
            # vectorized lookup: (N_edges, 2, 3)
            segments_arr = positions[edge_indices]
            segments = list(segments_arr)

            line_collection = Line3DCollection(
                segments, colors=line_colors, linewidths=0.5
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

    # Set equal aspect ratio
    # X, Y, Z are tuples from zip(*vertex_positions)
    # Convert to array for min/max
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    max_range = np.array(
        [X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]
    ).max()
    mid_x = (X.max() + X.min()) * 0.5
    mid_y = (Y.max() + Y.min()) * 0.5
    mid_z = (Z.max() + Z.min()) * 0.5

    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    if no_axes:
        ax.set_axis_off()

    plt.tight_layout()

    if show:
        plt.show()


def update_live_vis(
    mesh: Mesh,
    *,
    state: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
) -> Dict[str, Any]:
    """Update or create a live visualization window for a mesh."""
    import matplotlib.pyplot as plt

    if state is None:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        state = {"fig": fig, "ax": ax, "mesh_version": -1}
    else:
        fig = state["fig"]
        ax = state["ax"]

    # Check if we can do a fast update (topology unchanged)
    # We rely on mesh._topology_version or similar.
    # But mesh._version increments on positions too.
    # Let's check mesh._topology_version

    # Actually, plot_geometry creates collections. We need to store them in state to update them.
    # If state doesn't have collections, or topology changed, we redraw.

    fast_update = False
    current_topo_version = getattr(mesh, "_topology_version", -1)
    last_topo_version = state.get("topology_version", -2)

    if (
        "tri_collection" in state
        and "line_collection" in state
        and current_topo_version == last_topo_version
    ):
        fast_update = True

    if fast_update:
        # Update positions only
        positions = mesh.positions_view()

        # Triangles
        tri_rows, _ = mesh.triangle_row_cache()
        if tri_rows is not None and state["tri_collection"]:
            tri_data = positions[tri_rows]
            state["tri_collection"].set_verts(list(tri_data))

        # Edges
        # We need cached edge indices to be fast.
        # If we didn't cache them, we have to loop again.
        # But looping edges is faster than creating new Collection.
        # Let's see if we stored edge_indices in state.
        if "edge_indices" in state and state["line_collection"]:
            edge_indices = state["edge_indices"]
            # Check bounds? Assumed safe if topo version matched
            segments_arr = positions[edge_indices]
            state["line_collection"].set_segments(list(segments_arr))

        if title:
            ax.set_title(title)
        fig.canvas.draw_idle()
        plt.pause(0.001)
        return state

    # Slow path: Full redraw
    ax.cla()

    # We call plot_geometry but we need to intercept the collections it creates.
    # Since plot_geometry doesn't return them, we might have to inline the logic or modify plot_geometry.
    # Or, we can just copy the logic here for live viz since it's specialized.
    # MODIFYING plot_geometry to return collections would be cleaner but changes API.
    # Let's stick to calling plot_geometry for now, but to support fast update next time,
    # we need to capture the collections.

    # Actually, let's just make plot_geometry return the collections if we want to reuse them.
    # Or we can inspect ax.collections after calling.

    plot_geometry(mesh, ax=ax, show=False)

    # Capture collections for next time
    # ax.collections usually has [Poly3DCollection, Line3DCollection] (or similar order)
    # We need to be sure which is which.
    tri_col = None
    line_col = None

    for col in ax.collections:
        if isinstance(col, Poly3DCollection):
            tri_col = col
        elif isinstance(col, Line3DCollection):
            line_col = col

    state["tri_collection"] = tri_col
    state["line_collection"] = line_col
    state["topology_version"] = current_topo_version

    # Cache edge indices for next time
    idx_map = mesh.vertex_index_to_row
    edge_indices = []
    for edge in mesh.edges.values():
        t = idx_map.get(edge.tail_index)
        h = idx_map.get(edge.head_index)
        if t is not None and h is not None:
            edge_indices.append((t, h))
    if edge_indices:
        state["edge_indices"] = np.array(edge_indices)
    else:
        state.pop("edge_indices", None)

    if title:
        ax.set_title(title)
    fig.canvas.draw_idle()
    plt.pause(0.001)
    return state

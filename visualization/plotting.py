import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from typing import Any, Dict, Optional

import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from geometry.entities import Mesh

logger = logging.getLogger("membrane_solver")

_TILT_COLOR_BY = {"tilt_mag", "tilt_div"}


def triangle_tilt_magnitudes(mesh: Mesh) -> tuple[np.ndarray, list[int]]:
    """Return per-triangle mean tilt magnitudes for facets in triangle_row_cache."""
    tri_rows, tri_facets = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return np.array([], dtype=float), []
    mags = np.linalg.norm(mesh.tilts_view(), axis=1)
    values = mags[tri_rows].mean(axis=1)
    return values, tri_facets


def _triangle_divergence_from_arrays(
    positions: np.ndarray, tilts: np.ndarray, tri_rows: np.ndarray
) -> np.ndarray:
    """Compute per-triangle divergence values for a nodal vector field.

    The nodal field is assumed piecewise-linear on each triangle. Divergence is
    constant per triangle and returned as a dense array with one value per row
    in ``tri_rows``.
    """
    v0 = positions[tri_rows[:, 0]]
    v1 = positions[tri_rows[:, 1]]
    v2 = positions[tri_rows[:, 2]]

    t0 = tilts[tri_rows[:, 0]]
    t1 = tilts[tri_rows[:, 1]]
    t2 = tilts[tri_rows[:, 2]]

    n = np.cross(v1 - v0, v2 - v0)
    denom = np.einsum("ij,ij->i", n, n)
    good = denom > 1e-24

    g0 = np.zeros_like(n)
    g1 = np.zeros_like(n)
    g2 = np.zeros_like(n)

    g0[good] = np.cross(n[good], v2[good] - v1[good]) / denom[good][:, None]
    g1[good] = np.cross(n[good], v0[good] - v2[good]) / denom[good][:, None]
    g2[good] = np.cross(n[good], v1[good] - v0[good]) / denom[good][:, None]

    div = (
        np.einsum("ij,ij->i", t0, g0)
        + np.einsum("ij,ij->i", t1, g1)
        + np.einsum("ij,ij->i", t2, g2)
    )
    div[~good] = 0.0
    return div


def triangle_tilt_divergence(mesh: Mesh) -> tuple[np.ndarray, list[int]]:
    """Return per-triangle divergence of the tilt field for facets in triangle_row_cache."""
    tri_rows, tri_facets = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return np.array([], dtype=float), []
    positions = mesh.positions_view()
    tilts = mesh.tilts_view()
    values = _triangle_divergence_from_arrays(positions, tilts, tri_rows)
    return values, tri_facets


def _colors_from_scalars(color_by: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        values = np.zeros_like(values)
        finite = np.ones_like(values, dtype=bool)

    if color_by == "tilt_mag":
        vmax = float(values[finite].max()) if finite.any() else 1.0
        vmax = vmax if vmax > 0 else 1.0
        norm = mpl_colors.Normalize(vmin=0.0, vmax=vmax)
        cmap = plt.get_cmap("viridis")
    elif color_by == "tilt_div":
        vlim = float(np.abs(values[finite]).max()) if finite.any() else 1.0
        vlim = vlim if vlim > 0 else 1.0
        norm = mpl_colors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
        cmap = plt.get_cmap("coolwarm")
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported color_by={color_by!r}")

    colors = cmap(norm(np.nan_to_num(values, nan=0.0)))
    colors = np.asarray(colors, dtype=float)
    if colors.ndim == 2 and colors.shape[1] == 4:
        colors[:, 3] = 1.0
    return colors


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
    color_by: Optional[str] = None,
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
    color_by : str, optional
        When set, override facet colors with a colormap based on tilt values.
        Supported values are ``"tilt_mag"`` (mean ``|t|`` per facet) and
        ``"tilt_div"`` (piecewise-linear per-triangle ``div(t)``).
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

    if color_by is not None and color_by not in _TILT_COLOR_BY:
        raise ValueError(
            f"Unsupported color_by={color_by!r}; expected one of {sorted(_TILT_COLOR_BY)}"
        )

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    vertex_positions = [mesh.vertices[v].position for v in mesh.vertices.keys()]
    X, Y, Z = zip(*vertex_positions)

    # Plot edges first so opaque facets can occlude back-facing edges in the
    # 3D painter's algorithm (reduces the "transparent facets" look).
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
                segments, colors=line_colors, linewidths=1.0
            )
            ax.add_collection3d(line_collection)

    # Plot facets as filled polygons.
    if draw_facets:
        # Try to use vectorized triangle cache first
        tri_rows, tri_facets = mesh.triangle_row_cache()
        positions = mesh.positions_view()

        triangles = []
        face_colors = []
        scalar_values: list[float] = []
        default_facet_color = (
            facet_color if facet_color is not None else (0.6, 0.8, 1.0)
        )

        # 1. Vectorized triangles
        if tri_rows is not None and len(tri_rows) > 0:
            # (N_tri, 3, 3) array of positions
            tri_data = positions[tri_rows]
            triangles.extend(list(tri_data))

            if color_by is None:
                # Map colors
                for fid in tri_facets:
                    if facet_colors is not None and fid in facet_colors:
                        face_colors.append(facet_colors[fid])
                    else:
                        opts = mesh.facets[fid].options
                        face_colors.append(opts.get("color", default_facet_color))
            elif color_by == "tilt_mag":
                tri_vals, _ = triangle_tilt_magnitudes(mesh)
                scalar_values.extend(list(map(float, tri_vals)))
            elif color_by == "tilt_div":
                tri_vals, _ = triangle_tilt_divergence(mesh)
                scalar_values.extend(list(map(float, tri_vals)))

        # 2. Non-triangle facets (polygons) - Fallback
        tri_set = set(tri_facets) if tri_facets else set()

        for facet in mesh.facets.values():
            if facet.index in tri_set:
                continue

            if len(facet.edge_indices) < 3:
                continue

            tri = [
                mesh.vertices[mesh.get_edge(e).tail_index].position
                for e in facet.edge_indices
            ]
            triangles.append(tri)

            if color_by is None:
                if facet_colors is not None and facet.index in facet_colors:
                    color = facet_colors[facet.index]
                else:
                    color = facet.options.get("color", default_facet_color)
                face_colors.append(color)
            elif color_by == "tilt_mag":
                mags = [
                    float(
                        np.linalg.norm(mesh.vertices[mesh.get_edge(e).tail_index].tilt)
                    )
                    for e in facet.edge_indices
                ]
                scalar_values.append(float(np.mean(mags)) if mags else 0.0)
            elif color_by == "tilt_div":
                vids = [mesh.get_edge(e).tail_index for e in facet.edge_indices]
                if len(vids) < 3:
                    scalar_values.append(0.0)
                else:
                    v0 = mesh.vertices[int(vids[0])]
                    total_area = 0.0
                    weighted_div = 0.0
                    for i in range(1, len(vids) - 1):
                        v1 = mesh.vertices[int(vids[i])]
                        v2 = mesh.vertices[int(vids[i + 1])]
                        tri_pos = np.stack(
                            [v0.position, v1.position, v2.position], axis=0
                        )
                        tri_tilt = np.stack([v0.tilt, v1.tilt, v2.tilt], axis=0)
                        n = np.cross(tri_pos[1] - tri_pos[0], tri_pos[2] - tri_pos[0])
                        area2 = float(np.linalg.norm(n))
                        if area2 <= 1e-12:
                            continue
                        area = 0.5 * area2
                        tri_rows_local = np.array([[0, 1, 2]], dtype=np.int32)
                        div_val = float(
                            _triangle_divergence_from_arrays(
                                tri_pos, tri_tilt, tri_rows_local
                            )[0]
                        )
                        weighted_div += area * div_val
                        total_area += area
                    scalar_values.append(
                        weighted_div / total_area if total_area > 0 else 0.0
                    )

        if triangles:
            if color_by is not None:
                face_colors = list(
                    _colors_from_scalars(color_by, np.asarray(scalar_values))
                )
            alpha = 0.4 if transparent else 1.0
            tri_collection = Poly3DCollection(
                triangles,
                alpha=alpha,
                edgecolor=edge_color if not draw_edges else (0.2, 0.2, 0.2),
                linewidths=1.0 if draw_edges else 0.0,
            )
            tri_collection.set_facecolor(face_colors)
            ax.add_collection3d(tri_collection)

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
    color_by: Optional[str] = None,
) -> Dict[str, Any]:
    """Update or create a live visualization window for a mesh."""
    import matplotlib.pyplot as plt

    if color_by is not None and color_by not in _TILT_COLOR_BY:
        raise ValueError(
            f"Unsupported color_by={color_by!r}; expected one of {sorted(_TILT_COLOR_BY)}"
        )

    if state is None:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        state = {"fig": fig, "ax": ax, "mesh_version": -1, "color_by": color_by}
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

    if current_topo_version == last_topo_version:
        tri_rows = state.get("tri_rows")
        if (
            "tri_collection" in state
            and "line_collection" in state
            and tri_rows is not None
            and state.get("triangle_mesh_only", False)
            and state.get("color_by") == color_by
        ):
            fast_update = True

    if fast_update:
        # Update positions only
        positions = mesh.positions_view()

        # Triangles
        tri_rows = state.get("tri_rows")
        if tri_rows is not None and state["tri_collection"] is not None:
            tri_data = positions[tri_rows]
            state["tri_collection"].set_verts(list(tri_data))
            if color_by is not None:
                tilts = mesh.tilts_view()
                if color_by == "tilt_mag":
                    mags = np.linalg.norm(tilts, axis=1)
                    values = mags[tri_rows].mean(axis=1)
                else:
                    values = _triangle_divergence_from_arrays(
                        positions, tilts, tri_rows
                    )
                colors = _colors_from_scalars(color_by, values)
                state["tri_collection"].set_facecolor(colors)

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

    # Live visualization should be an inspection tool: keep facets opaque and
    # draw edges so structure is readable while stepping.
    plot_geometry(
        mesh,
        ax=ax,
        show=False,
        draw_edges=True,
        transparent=False,
        color_by=color_by,
    )

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
    state["color_by"] = color_by

    tri_rows, tri_facets = mesh.triangle_row_cache()
    state["triangle_mesh_only"] = tri_rows is not None and len(tri_facets) == len(
        mesh.facets
    )
    state["tri_rows"] = tri_rows if state["triangle_mesh_only"] else None

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

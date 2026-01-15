import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
import warnings
from typing import Any, Dict, Optional

import matplotlib.colors as mpl_colors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from geometry.entities import Mesh

logger = logging.getLogger("membrane_solver")

_TILT_COLOR_BY = {
    "tilt_mag",
    "tilt_div",
    "tilt_in",
    "tilt_out",
    "tilt_div_in",
    "tilt_div_out",
    "tilt_bilayer",
}


def _tilt_field_for_color_by(mesh: Mesh, color_by: str | None) -> np.ndarray:
    if color_by in {"tilt_in", "tilt_div_in"}:
        return mesh.tilts_in_view()
    if color_by in {"tilt_out", "tilt_div_out"}:
        return mesh.tilts_out_view()
    return mesh.tilts_view()


def _bilayer_offset_scale(positions: np.ndarray) -> float:
    if positions.size == 0:
        return 0.0
    span = positions.max(axis=0) - positions.min(axis=0)
    max_range = float(np.max(span)) if span.size else 0.0
    return 0.01 * max_range if max_range > 0 else 0.0


def _triangle_unit_normals(tri_data: np.ndarray) -> np.ndarray:
    normals = np.cross(tri_data[:, 1] - tri_data[:, 0], tri_data[:, 2] - tri_data[:, 0])
    norms = np.linalg.norm(normals, axis=1)
    unit = np.zeros_like(normals)
    good = norms > 1e-12
    unit[good] = normals[good] / norms[good][:, None]
    return unit


def _loop_unit_normal(loop_positions: np.ndarray) -> np.ndarray:
    if loop_positions.shape[0] < 3:
        return np.zeros(3, dtype=float)
    normal = np.cross(
        loop_positions[1] - loop_positions[0],
        loop_positions[2] - loop_positions[0],
    )
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-12:
        return np.zeros(3, dtype=float)
    return normal / norm


def _safe_pause(interval: float) -> None:
    """Pause briefly to let interactive backends process events.

    Matplotlib emits a UserWarning on non-interactive backends (e.g. Agg); we
    silence that warning so headless/test runs remain clean even if warnings are
    treated as errors.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*non-interactive.*cannot be shown.*",
            category=UserWarning,
        )
        plt.pause(interval)


def triangle_tilt_magnitudes(
    mesh: Mesh, *, tilts: np.ndarray | None = None
) -> tuple[np.ndarray, list[int]]:
    """Return per-triangle mean tilt magnitudes for facets in triangle_row_cache."""
    tri_rows, tri_facets = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return np.array([], dtype=float), []
    if tilts is None:
        tilts = mesh.tilts_view()
    mags = np.linalg.norm(tilts, axis=1)
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


def triangle_tilt_divergence(
    mesh: Mesh, *, tilts: np.ndarray | None = None
) -> tuple[np.ndarray, list[int]]:
    """Return per-triangle divergence of the tilt field for facets in triangle_row_cache."""
    tri_rows, tri_facets = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return np.array([], dtype=float), []
    positions = mesh.positions_view()
    if tilts is None:
        tilts = mesh.tilts_view()
    values = _triangle_divergence_from_arrays(positions, tilts, tri_rows)
    return values, tri_facets


def _colormap_norm_for_scalars(
    color_by: str, values: np.ndarray
) -> tuple[mpl_colors.Colormap, mpl_colors.Normalize]:
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        values = np.zeros_like(values)
        finite = np.ones_like(values, dtype=bool)

    if color_by in {"tilt_mag", "tilt_in", "tilt_out"}:
        vmax = float(values[finite].max()) if finite.any() else 1.0
        vmax = vmax if vmax > 0 else 1.0
        norm = mpl_colors.Normalize(vmin=0.0, vmax=vmax)
        cmap = plt.get_cmap("viridis")
    elif color_by in {"tilt_div", "tilt_div_in", "tilt_div_out"}:
        vlim = float(np.abs(values[finite]).max()) if finite.any() else 1.0
        vlim = vlim if vlim > 0 else 1.0
        norm = mpl_colors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
        cmap = plt.get_cmap("coolwarm")
    else:  # pragma: no cover - defensive
        raise ValueError(f"Unsupported color_by={color_by!r}")
    return cmap, norm


def _colors_from_scalars(color_by: str, values: np.ndarray) -> np.ndarray:
    cmap, norm = _colormap_norm_for_scalars(color_by, values)
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
    show_colorbar: bool | None = None,
    show_tilt_arrows: bool = False,
    tilt_arrows_max: int | None = 2000,
    tilt_arrow_scale: float = 0.1,
    show_tilt_streamlines: bool = False,
    tilt_streamlines_max: int = 200,
    tilt_streamlines_steps: int = 80,
    tilt_streamlines_cos_min: float = 0.2,
    show_patch_boundaries: bool = False,
    patch_key: str = "disk_patch",
    show_boundary_loops: bool = False,
    annotate_boundary_geodesic: bool = False,
    no_axes: bool = False,
    show: bool = True,
    tight_layout: bool = True,
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
        Use ``tilt_bilayer`` to render outer vs inner leaflet magnitudes on
        dual offset surfaces.
        Supported values are ``"tilt_mag"`` (mean ``|t|`` per facet) and
        ``"tilt_div"`` (piecewise-linear per-triangle ``div(t)``).
    show_colorbar : bool, optional
        When ``True``, draw a colorbar matching ``color_by``. Defaults to
        ``True`` when ``color_by`` is provided.
    show_tilt_arrows : bool, optional
        When ``True``, draw short arrows at vertices showing the tilt
        direction (in 3D). This is an overlay to complement magnitude/divergence
        coloring.
    tilt_arrows_max : int, optional
        Maximum number of arrows to draw when ``show_tilt_arrows`` is enabled.
        Large meshes can produce unreadable plots if every vertex is drawn.
        Set to ``None`` to draw all arrows.
    tilt_arrow_scale : float, optional
        Arrow length as a fraction of the plot bounding-box span.
    show_tilt_streamlines : bool, optional
        When ``True``, draw simple mesh-graph streamlines that follow the tilt
        direction. These are qualitative diagnostics and are not a substitute
        for a proper covariant transport scheme.
    show_patch_boundaries : bool, optional
        When ``True``, draw edges separating facets with different values for
        ``facet.options[patch_key]`` (e.g. disk patch rims).
    patch_key : str, optional
        Facet option key storing patch labels (default: ``"disk_patch"``).
    show_boundary_loops : bool, optional
        When ``True``, overlay mesh boundary loops ("holes") as polylines.
    annotate_boundary_geodesic : bool, optional
        When ``True``, annotate boundary loops with their discrete geodesic
        curvature sums (Gauss–Bonnet turning angles).
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
    fig = ax.get_figure()

    show_colorbar_flag = (
        color_by is not None if show_colorbar is None else bool(show_colorbar)
    )
    if not draw_facets:
        show_colorbar_flag = False
    if not show_colorbar_flag or color_by is None:
        prev_cbar = getattr(ax, "_membrane_colorbar", None)
        if prev_cbar is not None:
            try:
                prev_cbar.remove()
            except Exception:
                try:
                    prev_cbar.ax.remove()
                except Exception:
                    pass
        setattr(ax, "_membrane_colorbar", None)
        setattr(ax, "_membrane_mappable", None)

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
            line_collection.set_label("_mesh_edges")
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

        if color_by == "tilt_bilayer":
            triangles_top = []
            triangles_bottom = []
            scalar_values_in: list[float] = []
            scalar_values_out: list[float] = []
            offset = _bilayer_offset_scale(positions)

            # 1. Vectorized triangles
            if tri_rows is not None and len(tri_rows) > 0:
                tri_data = positions[tri_rows]
                unit = _triangle_unit_normals(tri_data)
                offsets = unit * offset
                triangles_top.extend(list(tri_data + offsets[:, None, :]))
                triangles_bottom.extend(list(tri_data - offsets[:, None, :]))

                tilts_in = _tilt_field_for_color_by(mesh, "tilt_in")
                tilts_out = _tilt_field_for_color_by(mesh, "tilt_out")
                vals_in, _ = triangle_tilt_magnitudes(mesh, tilts=tilts_in)
                vals_out, _ = triangle_tilt_magnitudes(mesh, tilts=tilts_out)
                scalar_values_in.extend(list(map(float, vals_in)))
                scalar_values_out.extend(list(map(float, vals_out)))

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
                loop_positions = np.asarray(tri, dtype=float)
                unit = _loop_unit_normal(loop_positions)
                triangles_top.append(loop_positions + offset * unit)
                triangles_bottom.append(loop_positions - offset * unit)

                mags_in = [
                    float(
                        np.linalg.norm(
                            mesh.vertices[mesh.get_edge(e).tail_index].tilt_in
                        )
                    )
                    for e in facet.edge_indices
                ]
                mags_out = [
                    float(
                        np.linalg.norm(
                            mesh.vertices[mesh.get_edge(e).tail_index].tilt_out
                        )
                    )
                    for e in facet.edge_indices
                ]
                scalar_values_in.append(float(np.mean(mags_in)) if mags_in else 0.0)
                scalar_values_out.append(float(np.mean(mags_out)) if mags_out else 0.0)

            if triangles_top:
                values_in = np.asarray(scalar_values_in, dtype=float)
                values_out = np.asarray(scalar_values_out, dtype=float)
                values_all = np.concatenate([values_in, values_out])
                cmap, norm = _colormap_norm_for_scalars("tilt_mag", values_all)
                colors_in = cmap(norm(values_in))
                colors_out = cmap(norm(values_out))
                if colors_in.ndim == 2 and colors_in.shape[1] == 4:
                    colors_in[:, 3] = 1.0
                if colors_out.ndim == 2 and colors_out.shape[1] == 4:
                    colors_out[:, 3] = 1.0

                alpha = 0.4 if transparent else 1.0
                top_collection = Poly3DCollection(
                    triangles_top,
                    alpha=alpha,
                    edgecolor=edge_color if not draw_edges else (0.2, 0.2, 0.2),
                    linewidths=1.0 if draw_edges else 0.0,
                )
                top_collection.set_facecolor(colors_out)
                top_collection.set_label("_mesh_facets_out")
                ax.add_collection3d(top_collection)

                bottom_collection = Poly3DCollection(
                    triangles_bottom,
                    alpha=alpha,
                    edgecolor=edge_color if not draw_edges else (0.2, 0.2, 0.2),
                    linewidths=1.0 if draw_edges else 0.0,
                )
                bottom_collection.set_facecolor(colors_in)
                bottom_collection.set_label("_mesh_facets_in")
                ax.add_collection3d(bottom_collection)

                if show_colorbar_flag and values_all.size and fig is not None:
                    prev_cbar = getattr(ax, "_membrane_colorbar", None)
                    if prev_cbar is not None:
                        try:
                            prev_cbar.remove()
                        except Exception:
                            try:
                                prev_cbar.ax.remove()
                            except Exception:
                                pass
                    mappable = ScalarMappable(norm=norm, cmap=cmap)
                    mappable.set_array(values_all)
                    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05)
                    cbar.set_label("|t|")
                    setattr(ax, "_membrane_colorbar", cbar)
                    setattr(ax, "_membrane_mappable", mappable)
                else:
                    setattr(ax, "_membrane_colorbar", None)
                    setattr(ax, "_membrane_mappable", None)
        else:
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
                elif color_by in {"tilt_mag", "tilt_in", "tilt_out"}:
                    tilts = _tilt_field_for_color_by(mesh, color_by)
                    tri_vals, _ = triangle_tilt_magnitudes(mesh, tilts=tilts)
                    scalar_values.extend(list(map(float, tri_vals)))
                elif color_by in {"tilt_div", "tilt_div_in", "tilt_div_out"}:
                    tilts = _tilt_field_for_color_by(mesh, color_by)
                    tri_vals, _ = triangle_tilt_divergence(mesh, tilts=tilts)
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
                elif color_by in {"tilt_mag", "tilt_in", "tilt_out"}:
                    tilt_attr = "tilt"
                    if color_by == "tilt_in":
                        tilt_attr = "tilt_in"
                    elif color_by == "tilt_out":
                        tilt_attr = "tilt_out"
                    mags = [
                        float(
                            np.linalg.norm(
                                getattr(
                                    mesh.vertices[mesh.get_edge(e).tail_index],
                                    tilt_attr,
                                )
                            )
                        )
                        for e in facet.edge_indices
                    ]
                    scalar_values.append(float(np.mean(mags)) if mags else 0.0)
                elif color_by in {"tilt_div", "tilt_div_in", "tilt_div_out"}:
                    tilt_attr = "tilt"
                    if color_by == "tilt_div_in":
                        tilt_attr = "tilt_in"
                    elif color_by == "tilt_div_out":
                        tilt_attr = "tilt_out"
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
                            tri_tilt = np.stack(
                                [
                                    getattr(v0, tilt_attr),
                                    getattr(v1, tilt_attr),
                                    getattr(v2, tilt_attr),
                                ],
                                axis=0,
                            )
                            n = np.cross(
                                tri_pos[1] - tri_pos[0], tri_pos[2] - tri_pos[0]
                            )
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
                    values_arr = np.asarray(scalar_values, dtype=float)
                    face_colors = list(_colors_from_scalars(color_by, values_arr))
                alpha = 0.4 if transparent else 1.0
                tri_collection = Poly3DCollection(
                    triangles,
                    alpha=alpha,
                    edgecolor=edge_color if not draw_edges else (0.2, 0.2, 0.2),
                    linewidths=1.0 if draw_edges else 0.0,
                )
                tri_collection.set_facecolor(face_colors)
                tri_collection.set_label("_mesh_facets")
                ax.add_collection3d(tri_collection)

                if (
                    color_by is not None
                    and show_colorbar_flag
                    and values_arr.size
                    and fig is not None
                ):
                    prev_cbar = getattr(ax, "_membrane_colorbar", None)
                    if prev_cbar is not None:
                        try:
                            prev_cbar.remove()
                        except Exception:
                            try:
                                prev_cbar.ax.remove()
                            except Exception:
                                pass
                    cmap, norm = _colormap_norm_for_scalars(color_by, values_arr)
                    mappable = ScalarMappable(norm=norm, cmap=cmap)
                    mappable.set_array(values_arr)
                    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, pad=0.05)
                    cbar.set_label(
                        "|t|"
                        if color_by
                        in {"tilt_mag", "tilt_in", "tilt_out", "tilt_bilayer"}
                        else "div(t)"
                    )
                    setattr(ax, "_membrane_colorbar", cbar)
                    setattr(ax, "_membrane_mappable", mappable)
                else:
                    setattr(ax, "_membrane_colorbar", None)
                    setattr(ax, "_membrane_mappable", None)

    if show_tilt_arrows:
        positions = mesh.positions_view()
        tilts = _tilt_field_for_color_by(mesh, color_by)
        mags = np.linalg.norm(tilts, axis=1)
        good_idx = np.where(mags > 1e-12)[0]
        if tilt_arrows_max is not None and good_idx.size > tilt_arrows_max:
            sample = np.linspace(0, good_idx.size - 1, int(tilt_arrows_max), dtype=int)
            good_idx = good_idx[sample]

        dirs = tilts[good_idx] / mags[good_idx][:, None]

        span = positions.max(axis=0) - positions.min(axis=0)
        max_range = float(np.max(span)) if span.size else 0.0
        arrow_len = float(tilt_arrow_scale) * max_range if max_range > 0 else 1.0

        starts = positions[good_idx]
        ends = starts + arrow_len * dirs
        segments_arr = np.stack([starts, ends], axis=1)
        arrow_collection = Line3DCollection(
            list(segments_arr),
            colors="k",
            linewidths=1.0,
            alpha=0.8,
        )
        arrow_collection.set_label("_tilt_arrows")
        ax.add_collection3d(arrow_collection)

    if show_tilt_streamlines:
        positions = mesh.positions_view()
        tilts = mesh.tilts_view()
        mags = np.linalg.norm(tilts, axis=1)
        good_rows = np.where(mags > 1e-12)[0]
        if good_rows.size:
            if good_rows.size > int(tilt_streamlines_max):
                sample = np.linspace(
                    0, good_rows.size - 1, int(tilt_streamlines_max), dtype=int
                )
                good_rows = good_rows[sample]

            neighbors: dict[int, list[int]] = {vid: [] for vid in mesh.vertex_ids}
            for edge in mesh.edges.values():
                neighbors[int(edge.tail_index)].append(int(edge.head_index))
                neighbors[int(edge.head_index)].append(int(edge.tail_index))
            for vids in neighbors.values():
                vids.sort()

            idx_map = mesh.vertex_index_to_row
            row_to_vid = [int(v) for v in mesh.vertex_ids]

            def _step(vid: int, direction: np.ndarray, visited: set[int]) -> int | None:
                row = idx_map.get(vid)
                if row is None:
                    return None
                origin = positions[row]
                best_vid = None
                best_cos = float(tilt_streamlines_cos_min)
                for nb in neighbors.get(vid, []):
                    if nb in visited:
                        continue
                    nb_row = idx_map.get(nb)
                    if nb_row is None:
                        continue
                    dpos = positions[nb_row] - origin
                    nd = float(np.linalg.norm(dpos))
                    if nd <= 1e-15:
                        continue
                    cosv = float(np.dot(dpos / nd, direction))
                    if cosv > best_cos:
                        best_cos = cosv
                        best_vid = nb
                return best_vid

            stream_segments: list[np.ndarray] = []

            for row in good_rows:
                seed_vid = row_to_vid[int(row)]
                seed_dir = tilts[int(row)]
                nrm = float(np.linalg.norm(seed_dir))
                if nrm <= 1e-12:
                    continue
                d0 = seed_dir / nrm

                def _trace(sign: float) -> list[int]:
                    path = [seed_vid]
                    visited = {seed_vid}
                    vid = seed_vid
                    direction = sign * d0
                    for _ in range(int(tilt_streamlines_steps)):
                        nxt = _step(vid, direction, visited)
                        if nxt is None:
                            break
                        path.append(nxt)
                        visited.add(nxt)
                        vid = nxt
                        row_n = idx_map.get(vid)
                        if row_n is None:
                            break
                        t = tilts[row_n]
                        nt = float(np.linalg.norm(t))
                        if nt <= 1e-12:
                            break
                        direction = sign * (t / nt)
                    return path

                backward = _trace(-1.0)
                forward = _trace(1.0)
                full = list(reversed(backward[:-1])) + forward
                if len(full) < 2:
                    continue

                pts = np.stack([mesh.vertices[vid].position for vid in full], axis=0)
                segs = np.stack([pts[:-1], pts[1:]], axis=1)
                stream_segments.extend(list(segs))

            if stream_segments:
                stream_collection = Line3DCollection(
                    stream_segments,
                    colors=(0.1, 0.1, 0.1, 0.6),
                    linewidths=1.0,
                )
                stream_collection.set_label("_tilt_streamlines")
                ax.add_collection3d(stream_collection)

    if show_patch_boundaries:
        from runtime.diagnostics.patches import patch_boundary_edges

        groups = patch_boundary_edges(mesh, patch_key=patch_key)
        if groups:
            positions = mesh.positions_view()
            idx_map = mesh.vertex_index_to_row
            cmap = plt.get_cmap("tab10")
            for idx, (label, edges) in enumerate(sorted(groups.items())):
                if not edges:
                    continue
                edge_rows = np.array(
                    [(idx_map[e.tail_index], idx_map[e.head_index]) for e in edges],
                    dtype=int,
                )
                segments_arr = positions[edge_rows]
                color = cmap(idx % 10)
                coll = Line3DCollection(
                    list(segments_arr),
                    colors=[color],
                    linewidths=2.0,
                    alpha=0.9,
                )
                coll.set_label("_patch_boundaries")
                ax.add_collection3d(coll)

    if show_boundary_loops:
        from runtime.diagnostics.gauss_bonnet import (
            boundary_geodesic_sum,
            extract_boundary_loops,
            find_boundary_edges,
        )

        boundary_edges = find_boundary_edges(mesh)
        loops = extract_boundary_loops(mesh, boundary_edges)
        per_loop = (
            boundary_geodesic_sum(mesh, loops) if annotate_boundary_geodesic else {}
        )

        cmap = plt.get_cmap("tab10")
        for idx, loop in enumerate(loops):
            if len(loop) < 2:
                continue
            pts = np.stack([v.position for v in loop] + [loop[0].position], axis=0)
            segs = np.stack([pts[:-1], pts[1:]], axis=1)
            color = cmap(idx % 10)
            coll = Line3DCollection(
                list(segs),
                colors=[color],
                linewidths=2.5,
                alpha=0.9,
            )
            coll.set_label("_boundary_loops")
            ax.add_collection3d(coll)

            if annotate_boundary_geodesic and idx in per_loop:
                c = pts[:-1].mean(axis=0)
                ax.text(
                    float(c[0]),
                    float(c[1]),
                    float(c[2]),
                    f"B{idx}={per_loop[idx]:.3g}",
                    fontsize=8,
                    color="k",
                )

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

    if tight_layout:
        plt.tight_layout()

    if show:
        plt.show()


def update_live_vis(
    mesh: Mesh,
    *,
    state: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    color_by: Optional[str] = None,
    show_colorbar: bool | None = None,
    show_tilt_arrows: bool = False,
) -> Dict[str, Any]:
    """Update or create a live visualization window for a mesh."""
    import matplotlib.pyplot as plt

    if color_by is not None and color_by not in _TILT_COLOR_BY:
        raise ValueError(
            f"Unsupported color_by={color_by!r}; expected one of {sorted(_TILT_COLOR_BY)}"
        )

    show_colorbar_flag = (
        color_by is not None if show_colorbar is None else bool(show_colorbar)
    )

    if state is None:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        state = {
            "fig": fig,
            "ax": ax,
            "mesh_version": -1,
            "color_by": color_by,
            "show_colorbar": show_colorbar_flag,
            "show_tilt_arrows": show_tilt_arrows,
        }
    else:
        fig = state["fig"]
        ax = state["ax"]

    current_topo_version = getattr(mesh, "_topology_version", -1)
    last_topo_version = state.get("topology_version", -2)

    def _tilt_divergence_for_loops(
        positions: np.ndarray, tilts: np.ndarray, loops: list[np.ndarray]
    ) -> np.ndarray:
        values = np.zeros(len(loops), dtype=float)
        tri_indices = [i for i, rows in enumerate(loops) if len(rows) == 3]
        if tri_indices:
            tri_rows = np.stack([loops[i] for i in tri_indices], axis=0).astype(
                np.int32, copy=False
            )
            values[np.array(tri_indices, dtype=int)] = _triangle_divergence_from_arrays(
                positions, tilts, tri_rows
            )

        for idx, rows in enumerate(loops):
            if len(rows) <= 3:
                continue

            r0 = int(rows[0])
            r1 = np.asarray(rows[1:-1], dtype=int)
            r2 = np.asarray(rows[2:], dtype=int)
            if r1.size == 0 or r2.size == 0:
                values[idx] = 0.0
                continue

            v0 = positions[r0]
            v1 = positions[r1]
            v2 = positions[r2]
            t0 = tilts[r0]
            t1 = tilts[r1]
            t2 = tilts[r2]

            v0s = np.broadcast_to(v0, v1.shape)
            t0s = np.broadcast_to(t0, t1.shape)

            n = np.cross(v1 - v0s, v2 - v0s)
            denom = np.einsum("ij,ij->i", n, n)
            good = denom > 1e-24
            if not np.any(good):
                values[idx] = 0.0
                continue

            g0 = np.zeros_like(n)
            g1 = np.zeros_like(n)
            g2 = np.zeros_like(n)

            g0[good] = np.cross(n[good], v2[good] - v1[good]) / denom[good][:, None]
            g1[good] = np.cross(n[good], v0s[good] - v2[good]) / denom[good][:, None]
            g2[good] = np.cross(n[good], v1[good] - v0s[good]) / denom[good][:, None]

            div = (
                np.einsum("ij,ij->i", t0s, g0)
                + np.einsum("ij,ij->i", t1, g1)
                + np.einsum("ij,ij->i", t2, g2)
            )
            div[~good] = 0.0

            area = 0.5 * np.sqrt(denom)
            area[~good] = 0.0
            total_area = float(area.sum())
            if total_area > 0:
                values[idx] = float((area * div).sum() / total_area)
            else:
                values[idx] = 0.0
        return values

    is_bilayer = color_by == "tilt_bilayer"
    mode_matches = (
        state.get("color_by") == color_by
        and state.get("show_colorbar") == show_colorbar_flag
        and state.get("show_tilt_arrows") == show_tilt_arrows
    )
    has_tri_collection = (
        state.get("tri_collection") is not None
        if not is_bilayer
        else state.get("tri_collection_top") is not None
        and state.get("tri_collection_bottom") is not None
    )
    fast_update = (
        current_topo_version == last_topo_version
        and has_tri_collection
        and mode_matches
        and (
            (
                state.get("tri_rows") is not None
                and state.get("triangle_mesh_only", False)
            )
            or state.get("facet_row_loops") is not None
        )
    )

    if fast_update:
        positions = mesh.positions_view()
        facet_row_loops = state.get("facet_row_loops")
        tri_rows = state.get("tri_rows")

        if is_bilayer:
            tri_collection_top = state.get("tri_collection_top")
            tri_collection_bottom = state.get("tri_collection_bottom")
            offset = _bilayer_offset_scale(positions)

            if facet_row_loops is not None:
                verts_top = []
                verts_bottom = []
                for rows in facet_row_loops:
                    loop_positions = positions[rows]
                    unit = _loop_unit_normal(loop_positions)
                    verts_top.append(loop_positions + offset * unit)
                    verts_bottom.append(loop_positions - offset * unit)
                tri_collection_top.set_verts(verts_top)
                tri_collection_bottom.set_verts(verts_bottom)
            elif tri_rows is not None:
                tri_data = positions[tri_rows]
                unit = _triangle_unit_normals(tri_data)
                offsets = unit * offset
                tri_collection_top.set_verts(list(tri_data + offsets[:, None, :]))
                tri_collection_bottom.set_verts(list(tri_data - offsets[:, None, :]))

            tilts_in = _tilt_field_for_color_by(mesh, "tilt_in")
            tilts_out = _tilt_field_for_color_by(mesh, "tilt_out")
            if facet_row_loops is not None:
                mags_in = np.linalg.norm(tilts_in, axis=1)
                mags_out = np.linalg.norm(tilts_out, axis=1)
                values_in = np.array(
                    [
                        float(mags_in[rows].mean()) if len(rows) else 0.0
                        for rows in facet_row_loops
                    ],
                    dtype=float,
                )
                values_out = np.array(
                    [
                        float(mags_out[rows].mean()) if len(rows) else 0.0
                        for rows in facet_row_loops
                    ],
                    dtype=float,
                )
            else:
                if tri_rows is None:  # pragma: no cover - defensive
                    values_in = np.array([], dtype=float)
                    values_out = np.array([], dtype=float)
                else:
                    mags_in = np.linalg.norm(tilts_in, axis=1)
                    mags_out = np.linalg.norm(tilts_out, axis=1)
                    values_in = mags_in[tri_rows].mean(axis=1)
                    values_out = mags_out[tri_rows].mean(axis=1)

            values_all = np.concatenate([values_in, values_out])
            cmap, norm = _colormap_norm_for_scalars("tilt_mag", values_all)
            colors_in = cmap(norm(values_in))
            colors_out = cmap(norm(values_out))
            if colors_in.ndim == 2 and colors_in.shape[1] == 4:
                colors_in[:, 3] = 1.0
            if colors_out.ndim == 2 and colors_out.shape[1] == 4:
                colors_out[:, 3] = 1.0
            tri_collection_top.set_facecolor(colors_out)
            tri_collection_bottom.set_facecolor(colors_in)

            if show_colorbar_flag:
                mappable = state.get("mappable")
                cbar = state.get("colorbar")
                if mappable is not None and cbar is not None:
                    mappable.set_array(values_all)
                    mappable.cmap = cmap
                    mappable.norm = norm
                    try:
                        cbar.update_normal(mappable)
                    except Exception:
                        pass
        else:
            tri_collection = state["tri_collection"]

            if facet_row_loops is not None:
                verts = [positions[rows] for rows in facet_row_loops]
                tri_collection.set_verts(verts)
            elif tri_rows is not None:
                tri_data = positions[tri_rows]
                tri_collection.set_verts(list(tri_data))

            if color_by is not None:
                tilts = _tilt_field_for_color_by(mesh, color_by)
                if facet_row_loops is not None:
                    if color_by in {"tilt_mag", "tilt_in", "tilt_out"}:
                        mags = np.linalg.norm(tilts, axis=1)
                        values = np.array(
                            [
                                float(mags[rows].mean()) if len(rows) else 0.0
                                for rows in facet_row_loops
                            ],
                            dtype=float,
                        )
                    else:
                        values = _tilt_divergence_for_loops(
                            positions, tilts, facet_row_loops
                        )
                else:
                    if tri_rows is None:  # pragma: no cover - defensive
                        values = np.array([], dtype=float)
                    elif color_by in {"tilt_mag", "tilt_in", "tilt_out"}:
                        mags = np.linalg.norm(tilts, axis=1)
                        values = mags[tri_rows].mean(axis=1)
                    else:
                        values = _triangle_divergence_from_arrays(
                            positions, tilts, tri_rows
                        )

                colors = _colors_from_scalars(color_by, values)
                tri_collection.set_facecolor(colors)

                if show_colorbar_flag:
                    mappable = state.get("mappable")
                    cbar = state.get("colorbar")
                    if mappable is not None and cbar is not None:
                        cmap, norm = _colormap_norm_for_scalars(color_by, values)
                        mappable.set_array(values)
                        mappable.cmap = cmap
                        mappable.norm = norm
                        try:
                            cbar.update_normal(mappable)
                        except Exception:
                            pass

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

        if show_tilt_arrows and state.get("tilt_arrows") is not None:
            tilts = _tilt_field_for_color_by(mesh, color_by)
            mags = np.linalg.norm(tilts, axis=1)
            good = mags > 1e-12
            dirs = np.zeros_like(tilts)
            dirs[good] = tilts[good] / mags[good][:, None]

            span = positions.max(axis=0) - positions.min(axis=0)
            max_range = float(np.max(span)) if span.size else 0.0
            arrow_len = 0.1 * max_range if max_range > 0 else 1.0

            ends = positions + arrow_len * dirs
            segments_arr = np.stack([positions, ends], axis=1)
            state["tilt_arrows"].set_segments(list(segments_arr))

        if title:
            ax.set_title(title)
        fig.canvas.draw_idle()
        _safe_pause(0.001)
        return state

    # Slow path: Full redraw
    prev_cbar = state.get("colorbar")
    if prev_cbar is not None:
        try:
            prev_cbar.remove()
        except Exception:
            try:
                prev_cbar.ax.remove()
            except Exception:
                pass
    state.pop("colorbar", None)
    state.pop("mappable", None)

    prev_view = None
    prev_limits = None
    prev_dist = None
    prev_axes_pos = None
    if "topology_version" in state:
        try:
            prev_view = (float(ax.elev), float(ax.azim))
        except Exception:
            prev_view = None
        try:
            prev_limits = (
                tuple(map(float, ax.get_xlim3d())),
                tuple(map(float, ax.get_ylim3d())),
                tuple(map(float, ax.get_zlim3d())),
            )
        except Exception:
            prev_limits = None
        try:
            prev_dist = float(getattr(ax, "dist", None))
        except Exception:
            prev_dist = None
        try:
            prev_axes_pos = ax.get_position().frozen()
        except Exception:
            prev_axes_pos = None

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
        show_colorbar=show_colorbar_flag,
        show_tilt_arrows=False,
        tight_layout=False,
    )

    # Capture collections for next time
    # ax.collections usually has [Poly3DCollection, Line3DCollection] (or similar order)
    # We need to be sure which is which.
    tri_col = None
    line_col = None
    tri_col_top = None
    tri_col_bottom = None

    for col in ax.collections:
        if isinstance(col, Poly3DCollection):
            label = col.get_label()
            if label == "_mesh_facets_out":
                tri_col_top = col
            elif label == "_mesh_facets_in":
                tri_col_bottom = col
            else:
                tri_col = col
        elif isinstance(col, Line3DCollection):
            if col.get_label() == "_mesh_edges":
                line_col = col

    if color_by == "tilt_bilayer":
        state["tri_collection"] = None
        state["tri_collection_top"] = tri_col_top
        state["tri_collection_bottom"] = tri_col_bottom
    else:
        state["tri_collection"] = tri_col
        state["tri_collection_top"] = None
        state["tri_collection_bottom"] = None
    state["line_collection"] = line_col
    state["topology_version"] = current_topo_version
    state["color_by"] = color_by
    state["show_colorbar"] = show_colorbar_flag
    state["show_tilt_arrows"] = show_tilt_arrows
    state["colorbar"] = getattr(ax, "_membrane_colorbar", None)
    state["mappable"] = getattr(ax, "_membrane_mappable", None)

    tri_rows, tri_facets = mesh.triangle_row_cache()
    state["triangle_mesh_only"] = tri_rows is not None and len(tri_facets) == len(
        mesh.facets
    )
    state["tri_rows"] = tri_rows if state["triangle_mesh_only"] else None
    state["facet_row_loops"] = None
    if not state["triangle_mesh_only"] and tri_col is not None:
        mesh.build_position_cache()
        idx_map = mesh.vertex_index_to_row
        tri_set = set(tri_facets) if tri_facets else set()

        loops: list[np.ndarray] = []
        if tri_rows is not None and len(tri_rows) > 0:
            loops.extend([np.asarray(row, dtype=np.int32) for row in tri_rows])

        for facet in mesh.facets.values():
            if facet.index in tri_set:
                continue
            if len(facet.edge_indices) < 3:
                continue
            rows: list[int] = []
            for signed_ei in facet.edge_indices:
                tail = mesh.get_edge(int(signed_ei)).tail_index
                row = idx_map.get(int(tail))
                if row is None:
                    rows = []
                    break
                rows.append(int(row))
            if rows:
                loops.append(np.array(rows, dtype=np.int32))
        state["facet_row_loops"] = loops

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

    if show_tilt_arrows:
        positions = mesh.positions_view()
        tilts = _tilt_field_for_color_by(mesh, color_by)
        mags = np.linalg.norm(tilts, axis=1)
        good = mags > 1e-12
        dirs = np.zeros_like(tilts)
        dirs[good] = tilts[good] / mags[good][:, None]

        span = positions.max(axis=0) - positions.min(axis=0)
        max_range = float(np.max(span)) if span.size else 0.0
        arrow_len = 0.1 * max_range if max_range > 0 else 1.0

        ends = positions + arrow_len * dirs
        segments_arr = np.stack([positions, ends], axis=1)
        arrow_collection = Line3DCollection(
            list(segments_arr),
            colors="k",
            linewidths=1.0,
            alpha=0.8,
        )
        arrow_collection.set_label("_tilt_arrows")
        ax.add_collection3d(arrow_collection)
        state["tilt_arrows"] = arrow_collection
    else:
        state["tilt_arrows"] = None

    if title:
        ax.set_title(title)
    if prev_view is not None:
        try:
            ax.view_init(elev=prev_view[0], azim=prev_view[1])
        except Exception:
            pass
    if prev_dist is not None:
        try:
            ax.dist = prev_dist
        except Exception:
            pass
    if prev_axes_pos is not None:
        try:
            ax.set_position(prev_axes_pos)
        except Exception:
            pass
    if prev_limits is not None:
        try:
            ax.set_xlim(prev_limits[0])
            ax.set_ylim(prev_limits[1])
            ax.set_zlim(prev_limits[2])
            ax.set_autoscale_on(False)
        except Exception:
            pass

    # Colorbar placement can change the axes rectangle (Matplotlib shrinks the
    # parent axes when attaching a colorbar). Live visualization should keep a
    # stable view when topology changes, so we pin the axes position and move
    # the colorbar into a fixed side gutter when possible.
    if prev_axes_pos is not None and show_colorbar_flag:
        cbar = getattr(ax, "_membrane_colorbar", None)
        if cbar is not None and getattr(cbar, "ax", None) is not None:
            try:
                gutter_pad = 0.02
                gutter_w = 0.03
                x0 = min(prev_axes_pos.x1 + gutter_pad, 0.95)
                max_w = max(0.0, 0.98 - x0)
                w = min(gutter_w, max_w)
                if w > 1e-6:
                    cbar.ax.set_position(
                        [x0, prev_axes_pos.y0, w, prev_axes_pos.height]
                    )
            except Exception:
                pass

    if state.get("axes_position") is None:
        try:
            state["axes_position"] = ax.get_position().frozen()
        except Exception:
            state["axes_position"] = None
    fig.canvas.draw_idle()
    _safe_pause(0.001)
    return state

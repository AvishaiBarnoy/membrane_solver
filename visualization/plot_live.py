import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import logging
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

from geometry.entities import Mesh
from visualization.plot_core import plot_geometry
from visualization.plot_data import (
    _TILT_COLOR_BY,
    _bilayer_offset_scale,
    _colormap_norm_for_scalars,
    _colors_from_scalars,
    _loop_unit_normal,
    _safe_pause,
    _tilt_field_for_color_by,
    _triangle_divergence_from_arrays,
    _triangle_unit_normals,
)

logger = logging.getLogger("membrane_solver")


def update_live_vis(
    mesh: Mesh,
    *,
    state: Optional[Dict[str, Any]] = None,
    title: Optional[str] = None,
    color_by: Optional[str] = None,
    show_colorbar: bool | None = None,
    show_tilt_arrows: bool = False,
    show_edges: bool = True,
) -> Dict[str, Any]:
    """Update or create a live visualization window for a mesh."""

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
            "show_edges": show_edges,
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
        draw_edges=show_edges,
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
    state["show_edges"] = show_edges
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

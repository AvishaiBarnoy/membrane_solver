"""Utilities for displaying meshes with PyVista."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from geometry.entities import Mesh


def _mesh_to_polydata(mesh: Mesh) -> pv.PolyData:
    """Convert ``Mesh`` to a PyVista ``PolyData`` object."""
    vertex_order = sorted(mesh.vertices.keys())
    points = np.array([mesh.vertices[idx].position for idx in vertex_order])
    index_map = {idx: i for i, idx in enumerate(vertex_order)}
    faces = []
    for facet in mesh.facets.values():
        v_ids = []
        for signed_ei in facet.edge_indices:
            edge = mesh.get_edge(signed_ei)
            vid = edge.tail_index if signed_ei > 0 else edge.head_index
            if not v_ids or v_ids[-1] != vid:
                v_ids.append(vid)
        if len(v_ids) >= 3:
            faces.append([len(v_ids)] + [index_map[v] for v in v_ids])
    face_array = np.hstack(faces).astype(int) if faces else np.array([], dtype=int)
    return pv.PolyData(points, face_array)


def plot_geometry(mesh: Mesh, show_indices: bool = False, transparent: bool = False) -> None:
    """Display the mesh in a PyVista window."""
    pv_mesh = _mesh_to_polydata(mesh)
    plotter = pv.Plotter()
    opacity = 0.4 if transparent else 1.0
    plotter.add_mesh(pv_mesh, color="lightblue", show_edges=True, opacity=opacity)
    if show_indices:
        for idx, point in enumerate(pv_mesh.points):
            label = str(sorted(mesh.vertices.keys())[idx])
            plotter.add_point_labels([point], [label], show_points=False)
    plotter.show()


if __name__ == "__main__":
    import argparse
    from geometry.geom_io import load_data, parse_geometry

    parser = argparse.ArgumentParser(description="Display a mesh with PyVista")
    parser.add_argument("input", help="Input mesh JSON file")
    args = parser.parse_args()

    data = load_data(args.input)
    mesh = parse_geometry(data)
    plot_geometry(mesh, show_indices=False)

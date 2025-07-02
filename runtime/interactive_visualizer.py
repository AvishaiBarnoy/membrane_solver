# runtime/interactive_visualizer.py
"""PyVista-based interactive visualization utilities."""

from __future__ import annotations

import numpy as np
import pyvista as pv

from geometry.entities import Mesh

class PyVistaVisualizer:
    """Visualize a mesh interactively using PyVista."""

    def __init__(self, mesh: Mesh) -> None:
        self.vertex_order = sorted(mesh.vertices.keys())
        self._points_array = self._gather_points(mesh)
        self.plotter = pv.Plotter()
        faces = self._build_faces(mesh)
        self.pv_mesh = pv.PolyData(self._points_array.copy(), faces)
        self.plotter.add_mesh(self.pv_mesh, color="lightblue", show_edges=True)
        self.plotter.show(auto_close=False, interactive_update=True)

    def _gather_points(self, mesh: Mesh) -> np.ndarray:
        return np.array([mesh.vertices[idx].position for idx in self.vertex_order])

    def _build_faces(self, mesh: Mesh) -> np.ndarray:
        faces_list = []
        index_map = {idx: i for i, idx in enumerate(self.vertex_order)}
        for facet in mesh.facets.values():
            v_ids = []
            for signed_ei in facet.edge_indices:
                edge = mesh.get_edge(signed_ei)
                vid = edge.tail_index if signed_ei > 0 else edge.head_index
                if not v_ids or v_ids[-1] != vid:
                    v_ids.append(vid)
            if len(v_ids) >= 3:
                faces_list.append([len(v_ids)] + [index_map[v] for v in v_ids])
        return np.hstack(faces_list).astype(int)

    def update(self, mesh: Mesh) -> None:
        self._points_array[:] = self._gather_points(mesh)
        self.pv_mesh.points = self._points_array
        self.plotter.render()

    def close(self) -> None:
        self.plotter.close()


def visualize_minimization(mesh: Mesh, minimizer, n_steps: int = 1) -> None:
    """Run minimization while updating a PyVista window."""
    visualizer = PyVistaVisualizer(mesh)

    def _cb(updated_mesh: Mesh) -> None:
        visualizer.update(updated_mesh)

    minimizer.minimize(n_steps=n_steps, callback=_cb)
    visualizer.close()

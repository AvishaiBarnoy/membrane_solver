"""Specialized shape-gradient projections for curved/free-disk simulations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from core.parameters.global_parameters import GlobalParameters
    from geometry.entities import Mesh


def project_curved_free_disk_shape_dofs(
    mesh: Mesh, global_params: GlobalParameters, grad_arr: np.ndarray
) -> None:
    """Restrict the shared-rim curved free-disk shape solve to height DOFs."""
    mode = str(global_params.get("rim_slope_match_mode") or "").strip().lower()
    if mode != "shared_rim_staggered_v1":
        return
    if not (
        global_params.get("rim_slope_match_group") is not None
        and global_params.get("rim_slope_match_outer_group") is not None
        and global_params.get("rim_slope_match_disk_group") is not None
    ):
        return
    grad_arr[:, :2] = 0.0
    _project_curved_free_disk_transition_shape_metric(mesh, global_params, grad_arr)


def _project_curved_free_disk_transition_shape_metric(
    mesh: Mesh, global_params: GlobalParameters, grad_arr: np.ndarray
) -> None:
    """Remove artificial shared-rim support-transition rows from shape descent."""
    support_group = str(global_params.get("rim_slope_match_outer_group") or "").strip()
    if not support_group:
        return

    support_rows = np.zeros(len(mesh.vertex_ids), dtype=bool)
    for row, vertex_id in enumerate(mesh.vertex_ids):
        vertex = mesh.vertices[int(vertex_id)]
        if vertex.options.get("rim_slope_match_group") == support_group:
            support_rows[row] = True
    if not np.any(support_rows):
        return

    tri_rows, _ = mesh.triangle_row_cache()
    if tri_rows is None or len(tri_rows) == 0:
        return

    transition_rows = np.zeros(len(mesh.vertex_ids), dtype=bool)
    transition_tris = np.any(support_rows[np.asarray(tri_rows, dtype=int)], axis=1)
    transition_rows[np.unique(np.asarray(tri_rows, dtype=int)[transition_tris])] = True
    grad_arr[transition_rows, 2] = 0.0

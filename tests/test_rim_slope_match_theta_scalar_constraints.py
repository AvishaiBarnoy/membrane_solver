from __future__ import annotations

from pathlib import Path

import numpy as np

from geometry.geom_io import load_data, parse_geometry
from modules.constraints import rim_slope_match_out


def test_rim_slope_match_theta_scalar_builds_inner_constraints():
    mesh_path = Path(
        "meshes/caveolin/kozlov_1disk_3d_tensionless_single_leaflet_profile_hard_rim_R12.yaml"
    )
    mesh = parse_geometry(load_data(str(mesh_path)))

    positions = mesh.positions_view()
    global_params = mesh.global_parameters
    global_params.set("rim_slope_match_thetaB_param", "tilt_thetaB_value")
    global_params.set("tilt_thetaB_value", 0.1)

    g_list = rim_slope_match_out.constraint_gradients_tilt_array(
        mesh,
        global_params,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
    )

    assert g_list, "Expected rim slope constraints with theta_scalar enabled"

    # Ensure that at least one constraint includes a non-null inner-tilt gradient.
    has_inner = any(g_in is not None and np.any(g_in) for g_in, _ in g_list)
    assert has_inner, "Expected inner tilt constraints when theta_scalar is set"

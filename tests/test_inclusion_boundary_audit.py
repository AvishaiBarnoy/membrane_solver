import logging

import numpy as np

from core.parameters.resolver import ParameterResolver
from geometry.geom_io import load_data, parse_geometry
from modules.constraints.rim_slope_match_payload import _build_matching_data
from modules.energy.tilt_thetaB_contact_in import compute_energy_array
from tools.diagnostics.inclusion_boundary_audit import run_inclusion_boundary_audit


def test_two_hole_audit_reports_local_geometry_and_legacy_frame_mismatch() -> None:
    audit = run_inclusion_boundary_audit(
        mesh_path="meshes/kozlov_two_holes.yaml",
        group="rim",
    )

    assert audit["component_count"] == 2
    assert audit["legacy_single_frame_unsafe"]
    assert [row["vertex_ids"] for row in audit["components"]] == [
        [7, 8, 9, 10],
        [11, 12, 13, 14],
    ]
    assert np.allclose(audit["components"][0]["center"], [3.0, 3.0, 0.0])
    assert np.allclose(audit["components"][1]["center"], [9.0, 3.0, 0.0])
    assert audit["components"][0]["configured_center_offset"] > 4.0
    assert audit["components"][1]["configured_center_offset"] > 9.0


def test_one_disk_audit_is_single_frame_safe() -> None:
    audit = run_inclusion_boundary_audit(
        mesh_path=(
            "tests/fixtures/"
            "kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml"
        ),
        group="disk",
    )

    assert audit["component_count"] == 1
    assert not audit["legacy_single_frame_unsafe"]
    assert audit["components"][0]["configured_center_offset"] < 1.0e-12


def test_legacy_operators_warn_once_for_disconnected_rims(caplog) -> None:
    mesh = parse_geometry(load_data("meshes/kozlov_two_holes.yaml"))
    mesh.build_position_cache()
    params = mesh.global_parameters
    resolver = ParameterResolver(params)

    with caplog.at_level(logging.WARNING, logger="membrane_solver"):
        for _ in range(2):
            compute_energy_array(
                mesh,
                params,
                resolver,
                positions=mesh.positions_view(),
                index_map=mesh.vertex_index_to_row,
            )
            _build_matching_data(mesh, params, mesh.positions_view())

    assert caplog.text.count("tilt_thetaB_contact_in") == 1
    assert caplog.text.count("rim_slope_match_out") == 1
    assert "2 disconnected components" in caplog.text

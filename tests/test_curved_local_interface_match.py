import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from geometry.geom_io import load_data, parse_geometry
from modules.constraints import curved_local_interface_match
from modules.constraints.local_interface_shells import (
    build_local_interface_shell_data,
    local_interface_constraint_diagnostics,
)
from tools.reproduce_flat_disk_one_leaflet import _boundary_at_R_parity_metrics


def _load_mesh():
    return parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )


def test_curved_local_interface_data_uses_shared_shell_builder() -> None:
    mesh = _load_mesh()
    positions = mesh.positions_view()

    data = curved_local_interface_match._build_local_interface_data(
        mesh,
        mesh.global_parameters,
        positions,
    )
    shell_data = build_local_interface_shell_data(mesh, positions=positions)

    assert data is not None
    assert data["shell_source"] == "disk_boundary_local_shells"
    assert data["matching_strategy"] == "nearest_azimuth"
    assert data["projection_mode"] == "vector_average"
    assert data["disk_radius"] == pytest.approx(shell_data.disk_radius, abs=1e-12)
    assert data["rim_radius"] == pytest.approx(shell_data.rim_radius, abs=1e-12)
    assert data["outer_radius"] == pytest.approx(shell_data.outer_radius, abs=1e-12)
    assert data["disk_rows"].shape == data["rim_rows"].shape
    assert data["rim_rows"].shape == data["outer_rows"].shape


def test_shared_shell_builder_matches_boundary_at_r_shell_family() -> None:
    mesh = _load_mesh()
    positions = mesh.positions_view()
    shell_data = build_local_interface_shell_data(mesh, positions=positions)
    boundary = _boundary_at_R_parity_metrics(mesh, theory_theta_value=0.4)

    assert int(boundary["disk_count"]) == int(shell_data.disk_rows.size)
    assert int(boundary["rim_count"]) == int(shell_data.rim_rows.size)
    assert int(boundary["outer_count"]) == int(shell_data.outer_rows.size)
    assert float(boundary["disk_radius"]) == pytest.approx(
        shell_data.disk_radius, abs=1.0e-12
    )
    assert float(boundary["rim_radius"]) == pytest.approx(
        shell_data.rim_radius, abs=1.0e-12
    )
    assert float(boundary["outer_radius"]) == pytest.approx(
        shell_data.outer_radius, abs=1.0e-12
    )


def test_local_interface_constraint_diagnostics_report_shared_shell_builder() -> None:
    mesh = _load_mesh()
    diagnostics = local_interface_constraint_diagnostics(
        mesh,
        positions=mesh.positions_view(),
        mode="local_vector_match_v1",
        active=False,
    )

    assert diagnostics["available"] is True
    assert diagnostics["reason"] == "ok"
    assert diagnostics["mode"] == "local_vector_match_v1"
    assert diagnostics["active"] is False
    assert diagnostics["uses_shared_shell_builder"] is True
    assert diagnostics["matching_strategy"] == "nearest_azimuth"
    assert diagnostics["shell_source"] == "disk_boundary_local_shells"
    assert int(diagnostics["disk_count"]) > 0
    assert int(diagnostics["rim_count"]) > 0
    assert int(diagnostics["outer_count"]) > 0
    assert float(diagnostics["rim_radius"]) > float(diagnostics["disk_radius"])
    assert float(diagnostics["outer_radius"]) > float(diagnostics["rim_radius"])


@pytest.mark.parametrize(
    ("raw_mode", "expected_mode"),
    [
        ("average", "vector_average"),
        ("vector_average", "vector_average"),
        ("mixed", "local_mixed_match_v1"),
        ("local_mixed_match_v1", "local_mixed_match_v1"),
        ("rim2disk", "rim_to_disk"),
        ("disk2rim", "disk_to_rim"),
    ],
)
def test_curved_local_interface_mode_aliases_normalize(
    raw_mode: str, expected_mode: str
) -> None:
    mesh = _load_mesh()
    mesh.global_parameters.set("curved_local_interface_match_mode", raw_mode)

    assert (
        curved_local_interface_match._resolve_projection_mode(mesh.global_parameters)
        == expected_mode
    )


def test_curved_local_interface_enforce_tilt_constraint_matches_in_plane_average() -> (
    None
):
    mesh = _load_mesh()
    positions = mesh.positions_view()
    data = curved_local_interface_match._build_local_interface_data(
        mesh,
        mesh.global_parameters,
        positions,
    )
    assert data is not None

    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")
    tilts_in[:] = 0.0
    tilts_out[:] = 0.0

    for idx, rim_row in enumerate(data["rim_rows"]):
        disk_row = int(data["disk_rows"][idx])
        rim_row = int(rim_row)
        u_vec = data["basis_u"][idx]
        v_vec = data["basis_v"][idx]
        tilts_in[disk_row] = 0.8 * u_vec - 0.2 * v_vec
        tilts_in[rim_row] = -0.4 * u_vec + 0.6 * v_vec
        tilts_out[disk_row] = 0.3 * u_vec + 0.9 * v_vec
        tilts_out[rim_row] = -0.1 * u_vec - 0.5 * v_vec

    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)

    curved_local_interface_match.enforce_tilt_constraint(
        mesh,
        global_params=mesh.global_parameters,
    )

    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    for idx, rim_row in enumerate(data["rim_rows"]):
        disk_row = int(data["disk_rows"][idx])
        rim_row = int(rim_row)
        u_vec = data["basis_u"][idx]
        v_vec = data["basis_v"][idx]
        in_disk = np.array(
            [np.dot(tilts_in[disk_row], u_vec), np.dot(tilts_in[disk_row], v_vec)]
        )
        in_rim = np.array(
            [np.dot(tilts_in[rim_row], u_vec), np.dot(tilts_in[rim_row], v_vec)]
        )
        out_disk = np.array(
            [np.dot(tilts_out[disk_row], u_vec), np.dot(tilts_out[disk_row], v_vec)]
        )
        out_rim = np.array(
            [np.dot(tilts_out[rim_row], u_vec), np.dot(tilts_out[rim_row], v_vec)]
        )
        assert in_disk == pytest.approx(in_rim, abs=1.0e-9)
        assert out_disk == pytest.approx(out_rim, abs=1.0e-9)


def test_curved_local_interface_dense_tilt_gradients_match_sparse_rows() -> None:
    mesh = _load_mesh()
    positions = mesh.positions_view()

    row_constraints = curved_local_interface_match.constraint_gradients_tilt_rows_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map={},
    )
    dense_constraints = curved_local_interface_match.constraint_gradients_tilt_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map={},
    )

    assert row_constraints is not None
    assert dense_constraints is not None
    assert len(dense_constraints) == len(row_constraints)

    for (row_in, row_out), (dense_in, dense_out) in zip(
        row_constraints, dense_constraints
    ):
        if row_in is None:
            assert dense_in is None
        else:
            rows, vecs = row_in
            expected_in = np.zeros_like(positions)
            np.add.at(expected_in, rows, vecs)
            assert dense_in is not None
            assert dense_in == pytest.approx(expected_in, abs=1.0e-12)
        if row_out is None:
            assert dense_out is None
        else:
            rows, vecs = row_out
            expected_out = np.zeros_like(positions)
            np.add.at(expected_out, rows, vecs)
            assert dense_out is not None
            assert dense_out == pytest.approx(expected_out, abs=1.0e-12)


def test_curved_local_interface_mixed_mode_constrains_only_tangential_component() -> (
    None
):
    mesh = _load_mesh()
    positions = mesh.positions_view()
    mesh.global_parameters.set(
        "curved_local_interface_match_mode", "local_mixed_match_v1"
    )
    data = curved_local_interface_match._build_local_interface_data(
        mesh,
        mesh.global_parameters,
        positions,
    )
    assert data is not None

    row_constraints = curved_local_interface_match.constraint_gradients_tilt_rows_array(
        mesh,
        mesh.global_parameters,
        positions=positions,
        index_map={},
    )

    assert row_constraints is not None
    assert len(row_constraints) == 2
    rows_in, _ = row_constraints[0]
    assert rows_in is not None
    _, vecs = rows_in
    assert vecs[0] == pytest.approx(data["basis_v"][0], abs=1.0e-12)


def test_curved_local_interface_mixed_mode_projection_uses_outer_shell_kink_proxy() -> (
    None
):
    mesh = _load_mesh()
    positions = mesh.positions_view()
    mesh.global_parameters.set(
        "curved_local_interface_match_mode", "local_mixed_match_v1"
    )
    data = curved_local_interface_match._build_local_interface_data(
        mesh,
        mesh.global_parameters,
        positions,
    )
    assert data is not None

    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")
    tilts_in[:] = 0.0
    tilts_out[:] = 0.0
    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)

    curved_local_interface_match.enforce_tilt_constraint(
        mesh,
        global_params=mesh.global_parameters,
    )

    tilts_in = mesh.tilts_in_view()
    tilts_out = mesh.tilts_out_view()
    idx = 0
    disk_row = int(data["disk_rows"][idx])
    rim_row = int(data["rim_rows"][idx])
    basis_u = data["basis_u"][idx]
    basis_v = data["basis_v"][idx]
    phi_target = float(data["phi"][idx])

    in_disk = np.array(
        [np.dot(tilts_in[disk_row], basis_u), np.dot(tilts_in[disk_row], basis_v)]
    )
    in_rim = np.array(
        [np.dot(tilts_in[rim_row], basis_u), np.dot(tilts_in[rim_row], basis_v)]
    )
    out_disk = np.array(
        [np.dot(tilts_out[disk_row], basis_u), np.dot(tilts_out[disk_row], basis_v)]
    )
    out_rim = np.array(
        [np.dot(tilts_out[rim_row], basis_u), np.dot(tilts_out[rim_row], basis_v)]
    )

    assert in_disk[0] == pytest.approx(-phi_target, abs=1.0e-9)
    assert in_rim[0] == pytest.approx(-phi_target, abs=1.0e-9)
    assert out_disk[0] == pytest.approx(phi_target, abs=1.0e-9)
    assert out_rim[0] == pytest.approx(phi_target, abs=1.0e-9)
    assert in_disk[1] == pytest.approx(in_rim[1], abs=1.0e-9)
    assert out_disk[1] == pytest.approx(out_rim[1], abs=1.0e-9)


def test_curved_local_interface_rim_to_disk_projection_uses_rim_coefficients() -> None:
    mesh = _load_mesh()
    positions = mesh.positions_view()
    mesh.global_parameters.set("curved_local_interface_match_mode", "rim_to_disk")
    data = curved_local_interface_match._build_local_interface_data(
        mesh, mesh.global_parameters, positions
    )
    assert data is not None

    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")
    idx = 0
    disk_row = int(data["disk_rows"][idx])
    rim_row = int(data["rim_rows"][idx])
    u_vec = data["basis_u"][idx]
    v_vec = data["basis_v"][idx]
    tilts_in[disk_row] = 0.8 * u_vec - 0.2 * v_vec
    tilts_in[rim_row] = -0.4 * u_vec + 0.6 * v_vec
    tilts_out[disk_row] = 0.3 * u_vec + 0.9 * v_vec
    tilts_out[rim_row] = -0.1 * u_vec - 0.5 * v_vec
    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)

    curved_local_interface_match.enforce_tilt_constraint(
        mesh, global_params=mesh.global_parameters
    )

    new_in = mesh.tilts_in_view()
    new_out = mesh.tilts_out_view()
    rim_in = np.array(
        [np.dot(tilts_in[rim_row], u_vec), np.dot(tilts_in[rim_row], v_vec)]
    )
    rim_out = np.array(
        [np.dot(tilts_out[rim_row], u_vec), np.dot(tilts_out[rim_row], v_vec)]
    )
    assert np.array(
        [np.dot(new_in[disk_row], u_vec), np.dot(new_in[disk_row], v_vec)]
    ) == pytest.approx(rim_in, abs=1.0e-9)
    assert np.array(
        [np.dot(new_out[disk_row], u_vec), np.dot(new_out[disk_row], v_vec)]
    ) == pytest.approx(rim_out, abs=1.0e-9)


def test_curved_local_interface_disk_to_rim_projection_uses_disk_coefficients() -> None:
    mesh = _load_mesh()
    positions = mesh.positions_view()
    mesh.global_parameters.set("curved_local_interface_match_mode", "disk_to_rim")
    data = curved_local_interface_match._build_local_interface_data(
        mesh, mesh.global_parameters, positions
    )
    assert data is not None

    tilts_in = mesh.tilts_in_view().copy(order="F")
    tilts_out = mesh.tilts_out_view().copy(order="F")
    idx = 0
    disk_row = int(data["disk_rows"][idx])
    rim_row = int(data["rim_rows"][idx])
    u_vec = data["basis_u"][idx]
    v_vec = data["basis_v"][idx]
    tilts_in[disk_row] = 0.8 * u_vec - 0.2 * v_vec
    tilts_in[rim_row] = -0.4 * u_vec + 0.6 * v_vec
    tilts_out[disk_row] = 0.3 * u_vec + 0.9 * v_vec
    tilts_out[rim_row] = -0.1 * u_vec - 0.5 * v_vec
    mesh.set_tilts_in_from_array(tilts_in)
    mesh.set_tilts_out_from_array(tilts_out)

    curved_local_interface_match.enforce_tilt_constraint(
        mesh, global_params=mesh.global_parameters
    )

    new_in = mesh.tilts_in_view()
    new_out = mesh.tilts_out_view()
    disk_in = np.array(
        [np.dot(tilts_in[disk_row], u_vec), np.dot(tilts_in[disk_row], v_vec)]
    )
    disk_out = np.array(
        [np.dot(tilts_out[disk_row], u_vec), np.dot(tilts_out[disk_row], v_vec)]
    )
    assert np.array(
        [np.dot(new_in[rim_row], u_vec), np.dot(new_in[rim_row], v_vec)]
    ) == pytest.approx(disk_in, abs=1.0e-9)
    assert np.array(
        [np.dot(new_out[rim_row], u_vec), np.dot(new_out[rim_row], v_vec)]
    ) == pytest.approx(disk_out, abs=1.0e-9)

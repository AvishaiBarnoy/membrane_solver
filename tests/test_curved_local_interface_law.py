import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.parameters.resolver import ParameterResolver
from geometry.geom_io import load_data, parse_geometry
from modules.constraints.local_interface_shells import build_local_interface_shell_data
from modules.energy import curved_local_interface_law


def _build_mesh():
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    mesh.global_parameters.set("curved_local_interface_law_strength", 2.5)
    return mesh


def _controlled_positions_and_tilts(
    mesh, *, phi_target: float
) -> tuple[np.ndarray, np.ndarray]:
    positions = mesh.positions_view().copy()
    shell_data = build_local_interface_shell_data(mesh, positions=positions)
    radii = np.linalg.norm(positions[:, :2], axis=1)
    rim_rows = np.asarray(shell_data.rim_rows_matched, dtype=int)
    outer_rows = np.asarray(shell_data.outer_rows, dtype=int)
    dr = radii[outer_rows] - radii[rim_rows]

    positions[outer_rows, 2] = positions[rim_rows, 2] + phi_target * dr

    tilts_out = np.zeros_like(mesh.tilts_out_view())
    tilts_out[rim_rows] = phi_target * shell_data.rim_r_hat
    return positions, tilts_out


def test_curved_local_interface_law_zero_on_matched_shape_aware_state() -> None:
    mesh = _build_mesh()
    resolver = ParameterResolver(mesh.global_parameters)
    positions, tilts_out = _controlled_positions_and_tilts(mesh, phi_target=0.2)
    grad_arr = np.zeros_like(positions)
    tilt_out_grad_arr = np.zeros_like(positions)

    energy = curved_local_interface_law.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=grad_arr,
        tilts_out=tilts_out,
        tilt_out_grad_arr=tilt_out_grad_arr,
    )

    assert float(energy) == pytest.approx(0.0, abs=1.0e-12)
    assert np.allclose(grad_arr, 0.0, atol=1.0e-12)
    assert np.allclose(tilt_out_grad_arr, 0.0, atol=1.0e-12)


def test_curved_local_interface_law_returns_shape_and_tilt_gradients_when_perturbed() -> (
    None
):
    mesh = _build_mesh()
    resolver = ParameterResolver(mesh.global_parameters)
    positions, tilts_out = _controlled_positions_and_tilts(mesh, phi_target=0.15)
    shell_data = build_local_interface_shell_data(mesh, positions=positions)
    rim_rows = np.asarray(shell_data.rim_rows_matched, dtype=int)
    outer_rows = np.asarray(shell_data.outer_rows, dtype=int)

    tilts_out[rim_rows[0]] += 0.1 * shell_data.rim_r_hat[0]

    grad_arr = np.zeros_like(positions)
    tilt_out_grad_arr = np.zeros_like(positions)
    energy_grad = curved_local_interface_law.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=grad_arr,
        tilts_out=tilts_out,
        tilt_out_grad_arr=tilt_out_grad_arr,
    )
    energy_only = curved_local_interface_law.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        tilts_out=tilts_out,
    )

    assert float(energy_only) == pytest.approx(float(energy_grad), abs=1.0e-12)
    assert float(energy_only) > 0.0
    assert np.linalg.norm(tilt_out_grad_arr[rim_rows[0]]) > 0.0
    assert np.linalg.norm(grad_arr[rim_rows[0]]) > 0.0
    assert np.linalg.norm(grad_arr[outer_rows[0]]) > 0.0
    assert grad_arr[rim_rows[0], 2] == pytest.approx(
        -grad_arr[outer_rows[0], 2], rel=1e-12, abs=1.0e-12
    )
    assert np.allclose(grad_arr[:, :2], 0.0, atol=1.0e-12)

import importlib

import numpy as np
import pytest

from core.parameters.resolver import ParameterResolver
from geometry.geom_io import load_data, parse_geometry
from modules.constraints.local_interface_shells import build_local_interface_shell_data

INTERFACE_ENERGIES = (("penalty", False), ("law", True))


def _build_mesh(variant: str):
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    mesh.global_parameters.set(f"curved_local_interface_{variant}_strength", 2.5)
    return mesh


def _controlled_positions_and_tilts(
    mesh, *, phi_target: float
) -> tuple[np.ndarray, np.ndarray]:
    positions = mesh.positions_view().copy()
    shell_data = build_local_interface_shell_data(mesh, positions=positions)
    radii = np.linalg.norm(positions[:, :2], axis=1)
    rim_rows = np.asarray(shell_data.rim_rows_matched, dtype=int)
    outer_rows = np.asarray(shell_data.outer_rows, dtype=int)

    positions[outer_rows, 2] = positions[rim_rows, 2] + phi_target * (
        radii[outer_rows] - radii[rim_rows]
    )
    tilts_out = np.zeros_like(mesh.tilts_out_view())
    tilts_out[rim_rows] = phi_target * shell_data.rim_r_hat
    return positions, tilts_out


def _module(variant: str):
    return importlib.import_module(f"modules.energy.curved_local_interface_{variant}")


@pytest.mark.parametrize("variant,shape_aware", INTERFACE_ENERGIES)
def test_curved_local_interface_energy_zero_on_matched_state(
    variant: str, shape_aware: bool
) -> None:
    module = _module(variant)
    mesh = _build_mesh(variant)
    resolver = ParameterResolver(mesh.global_parameters)
    positions, tilts_out = _controlled_positions_and_tilts(mesh, phi_target=0.2)
    grad_arr = np.zeros_like(positions)
    tilt_out_grad_arr = np.zeros_like(positions)

    energy = module.compute_energy_and_gradient_array(
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
    assert np.allclose(tilt_out_grad_arr, 0.0, atol=1.0e-12)
    if shape_aware:
        assert np.allclose(grad_arr, 0.0, atol=1.0e-12)


@pytest.mark.parametrize("variant,shape_aware", INTERFACE_ENERGIES)
def test_curved_local_interface_energy_is_positive_when_perturbed(
    variant: str, shape_aware: bool
) -> None:
    module = _module(variant)
    mesh = _build_mesh(variant)
    resolver = ParameterResolver(mesh.global_parameters)
    positions, tilts_out = _controlled_positions_and_tilts(mesh, phi_target=0.15)
    shell_data = build_local_interface_shell_data(mesh, positions=positions)
    rim_rows = np.asarray(shell_data.rim_rows_matched, dtype=int)
    outer_rows = np.asarray(shell_data.outer_rows, dtype=int)

    tilts_out[rim_rows[0]] += 0.1 * shell_data.rim_r_hat[0]

    grad_arr = np.zeros_like(positions)
    tilt_out_grad_arr = np.zeros_like(positions)
    energy_grad = module.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=mesh.vertex_index_to_row,
        grad_arr=grad_arr,
        tilts_out=tilts_out,
        tilt_out_grad_arr=tilt_out_grad_arr,
    )
    energy_only = module.compute_energy_array(
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

    if shape_aware:
        assert np.linalg.norm(grad_arr[rim_rows[0]]) > 0.0
        assert np.linalg.norm(grad_arr[outer_rows[0]]) > 0.0
        assert grad_arr[rim_rows[0], 2] == pytest.approx(
            -grad_arr[outer_rows[0], 2], rel=1e-12, abs=1.0e-12
        )
        assert np.allclose(grad_arr[:, :2], 0.0, atol=1.0e-12)
    else:
        tilt_direction = tilt_out_grad_arr[rim_rows[0]] / np.linalg.norm(
            tilt_out_grad_arr[rim_rows[0]]
        )
        radial_direction = shell_data.rim_r_hat[0] / np.linalg.norm(
            shell_data.rim_r_hat[0]
        )
        assert np.allclose(tilt_direction, radial_direction, atol=1.0e-12)

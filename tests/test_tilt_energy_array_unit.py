import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sample_meshes import cube_soft_volume_input  # noqa: E402

from core.parameters.resolver import ParameterResolver  # noqa: E402
from geometry.geom_io import parse_geometry  # noqa: E402
from modules.energy import (
    tilt,  # noqa: E402
    tilt_coupling,  # noqa: E402
    tilt_in,  # noqa: E402
    tilt_out,  # noqa: E402
    tilt_smoothness,  # noqa: E402
    tilt_smoothness_in,  # noqa: E402
    tilt_smoothness_out,  # noqa: E402
    tilt_splay_twist_in,  # noqa: E402
)


def _build_mesh():
    mesh = parse_geometry(cube_soft_volume_input("lagrange"))
    mesh.build_facet_vertex_loops()
    return mesh


def _rng_tilts(shape, seed):
    rng = np.random.default_rng(seed)
    return 1.0e-2 * rng.standard_normal(size=shape)


def test_tilt_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("tilt_rigidity", 1.3)
    param_resolver = ParameterResolver(mesh.global_parameters)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    tilts = _rng_tilts(mesh.tilts_view().shape, 0)
    grad_dummy = np.zeros_like(positions)
    tilt_grad = np.zeros_like(positions)

    e_grad = tilt.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts=tilts,
        tilt_grad_arr=tilt_grad,
    )
    e_only = tilt.compute_energy_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        tilts=tilts,
    )
    assert e_only == pytest.approx(e_grad, rel=1e-12, abs=1e-12)


def test_tilt_leaflet_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("tilt_modulus_in", 0.7)
    mesh.global_parameters.set("tilt_modulus_out", 0.9)
    param_resolver = ParameterResolver(mesh.global_parameters)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    tilts_in = _rng_tilts(mesh.tilts_in_view().shape, 1)
    tilts_out = _rng_tilts(mesh.tilts_out_view().shape, 2)
    grad_dummy = np.zeros_like(positions)
    tilt_grad_in = np.zeros_like(positions)
    tilt_grad_out = np.zeros_like(positions)

    e_in_grad = tilt_in.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=tilts_in,
        tilt_in_grad_arr=tilt_grad_in,
    )
    e_in_only = tilt_in.compute_energy_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts_in,
    )
    assert e_in_only == pytest.approx(e_in_grad, rel=1e-12, abs=1e-12)

    e_out_grad = tilt_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_out=tilts_out,
        tilt_out_grad_arr=tilt_grad_out,
    )
    e_out_only = tilt_out.compute_energy_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        tilts_out=tilts_out,
    )
    assert e_out_only == pytest.approx(e_out_grad, rel=1e-12, abs=1e-12)


def test_tilt_coupling_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("tilt_coupling_modulus", 0.4)
    mesh.global_parameters.set("tilt_coupling_mode", "difference")
    param_resolver = ParameterResolver(mesh.global_parameters)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    tilts_in = _rng_tilts(mesh.tilts_in_view().shape, 3)
    tilts_out = _rng_tilts(mesh.tilts_out_view().shape, 4)
    grad_dummy = np.zeros_like(positions)
    tilt_grad_in = np.zeros_like(positions)
    tilt_grad_out = np.zeros_like(positions)

    e_grad = tilt_coupling.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=tilts_in,
        tilts_out=tilts_out,
        tilt_in_grad_arr=tilt_grad_in,
        tilt_out_grad_arr=tilt_grad_out,
    )
    e_only = tilt_coupling.compute_energy_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts_in,
        tilts_out=tilts_out,
    )
    assert e_only == pytest.approx(e_grad, rel=1e-12, abs=1e-12)


def test_tilt_smoothness_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("tilt_smoothness_rigidity", 0.8)
    param_resolver = ParameterResolver(mesh.global_parameters)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    tilts = _rng_tilts(mesh.tilts_view().shape, 5)
    grad_dummy = np.zeros_like(positions)
    tilt_grad = np.zeros_like(positions)

    e_grad = tilt_smoothness.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts=tilts,
        tilt_grad_arr=tilt_grad,
    )
    e_only = tilt_smoothness.compute_energy_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        tilts=tilts,
    )
    assert e_only == pytest.approx(e_grad, rel=1e-12, abs=1e-12)


def test_tilt_smoothness_leaflet_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("bending_modulus_in", 0.6)
    mesh.global_parameters.set("bending_modulus_out", 0.5)
    param_resolver = ParameterResolver(mesh.global_parameters)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    tilts_in = _rng_tilts(mesh.tilts_in_view().shape, 6)
    tilts_out = _rng_tilts(mesh.tilts_out_view().shape, 7)
    grad_dummy = np.zeros_like(positions)
    tilt_grad_in = np.zeros_like(positions)
    tilt_grad_out = np.zeros_like(positions)

    e_in_grad = tilt_smoothness_in.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=tilts_in,
        tilt_in_grad_arr=tilt_grad_in,
    )
    e_in_only = tilt_smoothness_in.compute_energy_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts_in,
    )
    assert e_in_only == pytest.approx(e_in_grad, rel=1e-12, abs=1e-12)

    e_out_grad = tilt_smoothness_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_out=tilts_out,
        tilt_out_grad_arr=tilt_grad_out,
    )
    e_out_only = tilt_smoothness_out.compute_energy_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        tilts_out=tilts_out,
    )
    assert e_out_only == pytest.approx(e_out_grad, rel=1e-12, abs=1e-12)


def test_tilt_splay_twist_in_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("tilt_splay_modulus_in", 0.6)
    mesh.global_parameters.set("tilt_twist_modulus_in", 0.2)
    param_resolver = ParameterResolver(mesh.global_parameters)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    tilts_in = _rng_tilts(mesh.tilts_in_view().shape, 8)
    grad_dummy = np.zeros_like(positions)
    tilt_grad_in = np.zeros_like(positions)

    e_in_grad = tilt_splay_twist_in.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=tilts_in,
        tilt_in_grad_arr=tilt_grad_in,
    )
    e_in_only = tilt_splay_twist_in.compute_energy_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts_in,
    )
    assert e_in_only == pytest.approx(e_in_grad, rel=1e-12, abs=1e-12)

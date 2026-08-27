import numpy as np

from core.parameters.resolver import ParameterResolver
from geometry.geom_io import parse_geometry
from modules.energy import (
    tilt_rim_source_bilayer,
    tilt_rim_source_in,
    tilt_rim_source_out,
)
from tests.sample_meshes import annulus_source_mesh_data


def _annulus_source_mesh(*, n: int = 10) -> dict:
    return annulus_source_mesh_data(
        n=n,
        global_parameters={
            "tilt_rim_source_center": [0.0, 0.0, 0.0],
            "tilt_rim_source_group": "inner",
            "tilt_rim_source_strength": 1.0,
            "tilt_rim_source_group_in": "inner",
            "tilt_rim_source_strength_in": 1.0,
            "tilt_rim_source_group_out": "inner",
            "tilt_rim_source_strength_out": 1.0,
        },
        energy_modules=[],
    )


def _set_radial_tilts(mesh) -> None:
    positions = mesh.positions_view()
    r = positions.copy()
    r[:, 2] = 0.0
    rn = np.linalg.norm(r, axis=1)
    radial = np.zeros_like(positions)
    good = rn > 1e-12
    radial[good] = r[good] / rn[good][:, None]
    mesh.set_tilts_in_from_array(radial)
    mesh.set_tilts_out_from_array(2.0 * radial)


def test_tilt_rim_source_out_selection_cache_reuses_and_invalidates():
    mesh = parse_geometry(_annulus_source_mesh())

    payload = tilt_rim_source_out._rim_selection_payload(
        mesh, group="inner", mode="boundary"
    )
    assert payload is not None
    original_edge_count = int(payload["edge_ids"].size)
    assert original_edge_count > 0

    cache = getattr(mesh, "_tilt_rim_source_out_selection_cache")
    sentinel = object()
    cache["value"] = sentinel

    cached = tilt_rim_source_out._rim_selection_payload(
        mesh, group="inner", mode="boundary"
    )
    assert cached is sentinel

    mesh.vertices[0].options["pin_to_circle_group"] = "other"
    mesh._vertex_ids_version += 1

    refreshed = tilt_rim_source_out._rim_selection_payload(
        mesh, group="inner", mode="boundary"
    )
    assert refreshed is not sentinel
    assert int(refreshed["edge_ids"].size) < original_edge_count


def test_tilt_rim_source_bilayer_selection_cache_reuses_and_invalidates():
    mesh = parse_geometry(_annulus_source_mesh())

    payload = tilt_rim_source_bilayer._rim_selection_payload(
        mesh, group="inner", mode="boundary"
    )
    assert payload is not None
    original_edge_count = int(payload["edge_ids"].size)
    assert original_edge_count > 0

    cache = getattr(mesh, "_tilt_rim_source_bilayer_selection_cache")
    sentinel = object()
    cache["value"] = sentinel

    cached = tilt_rim_source_bilayer._rim_selection_payload(
        mesh, group="inner", mode="boundary"
    )
    assert cached is sentinel

    mesh.vertices[0].options["pin_to_circle_group"] = "other"
    mesh._vertex_ids_version += 1

    refreshed = tilt_rim_source_bilayer._rim_selection_payload(
        mesh, group="inner", mode="boundary"
    )
    assert refreshed is not sentinel
    assert int(refreshed["edge_ids"].size) < original_edge_count


def test_tilt_rim_source_bilayer_matches_in_plus_out():
    mesh = parse_geometry(_annulus_source_mesh())
    _set_radial_tilts(mesh)
    resolver = ParameterResolver(mesh.global_parameters)

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_dummy = np.zeros_like(positions)

    grad_in_b = np.zeros_like(positions)
    grad_out_b = np.zeros_like(positions)
    e_b = tilt_rim_source_bilayer.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
        tilt_in_grad_arr=grad_in_b,
        tilt_out_grad_arr=grad_out_b,
    )

    grad_in = np.zeros_like(positions)
    grad_out = np.zeros_like(positions)
    e_in = tilt_rim_source_in.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=mesh.tilts_in_view(),
        tilt_in_grad_arr=grad_in,
    )
    e_out = tilt_rim_source_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=grad_out,
    )

    assert np.isclose(float(e_b), float(e_in + e_out), rtol=1e-12, atol=1e-12)


def test_tilt_rim_source_energy_array_matches_gradient_path():
    mesh = parse_geometry(_annulus_source_mesh())
    _set_radial_tilts(mesh)
    resolver = ParameterResolver(mesh.global_parameters)

    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    grad_dummy = np.zeros_like(positions)

    grad_in_b = np.zeros_like(positions)
    grad_out_b = np.zeros_like(positions)
    e_b_grad = tilt_rim_source_bilayer.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
        tilt_in_grad_arr=grad_in_b,
        tilt_out_grad_arr=grad_out_b,
    )
    e_b_only = tilt_rim_source_bilayer.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=mesh.tilts_in_view(),
        tilts_out=mesh.tilts_out_view(),
    )
    assert np.isclose(float(e_b_only), float(e_b_grad), rtol=1e-12, atol=1e-12)

    grad_in = np.zeros_like(positions)
    e_in_grad = tilt_rim_source_in.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=mesh.tilts_in_view(),
        tilt_in_grad_arr=grad_in,
    )
    e_in_only = tilt_rim_source_in.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=mesh.tilts_in_view(),
    )
    assert np.isclose(float(e_in_only), float(e_in_grad), rtol=1e-12, atol=1e-12)

    grad_out = np.zeros_like(positions)
    e_out_grad = tilt_rim_source_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_out=mesh.tilts_out_view(),
        tilt_out_grad_arr=grad_out,
    )
    e_out_only = tilt_rim_source_out.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_out=mesh.tilts_out_view(),
    )
    assert np.isclose(float(e_out_only), float(e_out_grad), rtol=1e-12, atol=1e-12)
    assert np.allclose(grad_in_b, grad_in)
    assert np.allclose(grad_out_b, grad_out)

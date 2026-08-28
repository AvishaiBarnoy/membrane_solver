import numpy as np
import pytest
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
from runtime.energy_context import EnergyContext  # noqa: E402


def _build_planar_patch_mesh():
    mesh = parse_geometry(
        {
            "vertices": {
                0: [0.0, 0.0, 0.0, {"fixed": True}],
                1: [1.0, 0.0, 0.0, {"fixed": True}],
                2: [1.0, 1.0, 0.0, {"fixed": True}],
                3: [0.0, 1.0, 0.0, {"fixed": True}],
                4: [0.5, 0.5, 0.0, {"fixed": True}],
            },
            "edges": {
                1: [0, 1],
                2: [1, 2],
                3: [2, 3],
                4: [3, 0],
                5: [0, 4],
                6: [1, 4],
                7: [2, 4],
                8: [3, 4],
            },
            "faces": {
                0: [1, 6, "r5"],
                1: [2, 7, "r6"],
                2: [3, 8, "r7"],
                3: [4, 5, "r8"],
            },
            "energy_modules": [],
            "global_parameters": {"surface_tension": 0.0},
            "instructions": [],
        }
    )
    mesh.build_facet_vertex_loops()
    return mesh


def _build_mesh():
    mesh = parse_geometry(cube_soft_volume_input("lagrange"))
    mesh.build_facet_vertex_loops()
    return mesh


def _rng_tilts(shape, seed):
    rng = np.random.default_rng(seed)
    return 1.0e-2 * rng.standard_normal(size=shape)


def _assert_energy_only_matches_gradient_path(
    module,
    mesh,
    param_resolver: ParameterResolver,
    *,
    tilt_fields: dict[str, np.ndarray],
    gradient_keys: tuple[str, ...],
) -> None:
    positions = mesh.positions_view()
    common = {
        "positions": positions,
        "index_map": mesh.vertex_index_to_row,
        **tilt_fields,
    }
    energy_with_gradient = module.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        grad_arr=np.zeros_like(positions),
        **common,
        **{key: np.zeros_like(positions) for key in gradient_keys},
    )
    energy_only = module.compute_energy_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        **common,
    )
    assert energy_only == pytest.approx(
        energy_with_gradient,
        rel=1e-12,
        abs=1e-12,
    )


def test_tilt_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("tilt_rigidity", 1.3)
    _assert_energy_only_matches_gradient_path(
        tilt,
        mesh,
        ParameterResolver(mesh.global_parameters),
        tilt_fields={"tilts": _rng_tilts(mesh.tilts_view().shape, 0)},
        gradient_keys=("tilt_grad_arr",),
    )


def test_tilt_leaflet_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("tilt_modulus_in", 0.7)
    mesh.global_parameters.set("tilt_modulus_out", 0.9)
    resolver = ParameterResolver(mesh.global_parameters)

    _assert_energy_only_matches_gradient_path(
        tilt_in,
        mesh,
        resolver,
        tilt_fields={"tilts_in": _rng_tilts(mesh.tilts_in_view().shape, 1)},
        gradient_keys=("tilt_in_grad_arr",),
    )
    _assert_energy_only_matches_gradient_path(
        tilt_out,
        mesh,
        resolver,
        tilt_fields={"tilts_out": _rng_tilts(mesh.tilts_out_view().shape, 2)},
        gradient_keys=("tilt_out_grad_arr",),
    )


def test_tilt_out_energy_array_matches_gradient_path_with_shared_rim_outer_exclusion():
    mesh = _build_planar_patch_mesh()
    for vid in (0, 1, 2, 3):
        mesh.vertices[vid].options["rim_slope_match_group"] = "outer"

    mesh.global_parameters.set("tilt_modulus_out", 0.9)
    mesh.global_parameters.set("rim_slope_match_mode", "shared_rim_staggered_v1")
    mesh.global_parameters.set("tilt_out_exclude_shared_rim_outer_rows", True)
    tilts_out = np.zeros_like(mesh.tilts_out_view())
    tilts_out[:4, 0] = 1.0e-2

    _assert_energy_only_matches_gradient_path(
        tilt_out,
        mesh,
        ParameterResolver(mesh.global_parameters),
        tilt_fields={"tilts_out": tilts_out},
        gradient_keys=("tilt_out_grad_arr",),
    )


def test_tilt_coupling_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("tilt_coupling_modulus", 0.4)
    mesh.global_parameters.set("tilt_coupling_mode", "difference")
    _assert_energy_only_matches_gradient_path(
        tilt_coupling,
        mesh,
        ParameterResolver(mesh.global_parameters),
        tilt_fields={
            "tilts_in": _rng_tilts(mesh.tilts_in_view().shape, 3),
            "tilts_out": _rng_tilts(mesh.tilts_out_view().shape, 4),
        },
        gradient_keys=("tilt_in_grad_arr", "tilt_out_grad_arr"),
    )


def test_tilt_smoothness_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("tilt_smoothness_rigidity", 0.8)
    _assert_energy_only_matches_gradient_path(
        tilt_smoothness,
        mesh,
        ParameterResolver(mesh.global_parameters),
        tilt_fields={"tilts": _rng_tilts(mesh.tilts_view().shape, 5)},
        gradient_keys=("tilt_grad_arr",),
    )


def test_tilt_smoothness_connection_v1_matches_ambient_on_planar_mesh():
    mesh = _build_planar_patch_mesh()
    mesh.global_parameters.set("tilt_smoothness_rigidity", 0.8)
    mesh.global_parameters.set("bending_modulus_in", 0.6)
    mesh.global_parameters.set("bending_modulus_out", 0.5)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    tilts = _rng_tilts(mesh.tilts_view().shape, 9)
    tilts_in = _rng_tilts(mesh.tilts_in_view().shape, 10)
    tilts_out = _rng_tilts(mesh.tilts_out_view().shape, 11)

    mesh.global_parameters.set("tilt_transport_model", "ambient_v1")
    resolver = ParameterResolver(mesh.global_parameters)
    ambient_base = tilt_smoothness.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts=tilts,
    )
    ambient_in = tilt_smoothness_in.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts_in,
    )
    ambient_out = tilt_smoothness_out.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_out=tilts_out,
    )

    mesh.global_parameters.set("tilt_transport_model", "connection_v1")
    resolver = ParameterResolver(mesh.global_parameters)
    connection_base = tilt_smoothness.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts=tilts,
    )
    connection_in = tilt_smoothness_in.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts_in,
    )
    connection_out = tilt_smoothness_out.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_out=tilts_out,
    )

    assert connection_base == pytest.approx(ambient_base, rel=1e-12, abs=1e-12)
    assert connection_in == pytest.approx(ambient_in, rel=1e-12, abs=1e-12)
    assert connection_out == pytest.approx(ambient_out, rel=1e-12, abs=1e-12)


def test_tilt_smoothness_connection_v1_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("tilt_smoothness_rigidity", 0.8)
    mesh.global_parameters.set("tilt_transport_model", "connection_v1")
    _assert_energy_only_matches_gradient_path(
        tilt_smoothness,
        mesh,
        ParameterResolver(mesh.global_parameters),
        tilt_fields={"tilts": _rng_tilts(mesh.tilts_view().shape, 10)},
        gradient_keys=("tilt_grad_arr",),
    )


def test_tilt_smoothness_leaflet_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("bending_modulus_in", 0.6)
    mesh.global_parameters.set("bending_modulus_out", 0.5)
    resolver = ParameterResolver(mesh.global_parameters)

    _assert_energy_only_matches_gradient_path(
        tilt_smoothness_in,
        mesh,
        resolver,
        tilt_fields={"tilts_in": _rng_tilts(mesh.tilts_in_view().shape, 6)},
        gradient_keys=("tilt_in_grad_arr",),
    )
    _assert_energy_only_matches_gradient_path(
        tilt_smoothness_out,
        mesh,
        resolver,
        tilt_fields={"tilts_out": _rng_tilts(mesh.tilts_out_view().shape, 7)},
        gradient_keys=("tilt_out_grad_arr",),
    )


def test_tilt_smoothness_leaflet_ctx_matches_plain_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("bending_modulus_in", 0.7)
    mesh.global_parameters.set("bending_modulus_out", 0.9)
    param_resolver = ParameterResolver(mesh.global_parameters)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    ctx = EnergyContext()
    ctx.ensure_for_mesh(mesh)

    tilts_in = _rng_tilts(mesh.tilts_in_view().shape, 10)
    tilts_out = _rng_tilts(mesh.tilts_out_view().shape, 11)
    grad_dummy = np.zeros_like(positions)
    tilt_grad_in_plain = np.zeros_like(positions)
    tilt_grad_out_plain = np.zeros_like(positions)
    tilt_grad_in_ctx = np.zeros_like(positions)
    tilt_grad_out_ctx = np.zeros_like(positions)

    e_in_plain = tilt_smoothness_in.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=tilts_in,
        tilt_in_grad_arr=tilt_grad_in_plain,
    )
    e_in_ctx = tilt_smoothness_in.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_in=tilts_in,
        tilt_in_grad_arr=tilt_grad_in_ctx,
        ctx=ctx,
    )
    assert e_in_ctx == pytest.approx(e_in_plain, rel=1e-12, abs=1e-12)
    assert np.allclose(tilt_grad_in_ctx, tilt_grad_in_plain, rtol=0.0, atol=1e-12)

    e_out_plain = tilt_smoothness_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_out=tilts_out,
        tilt_out_grad_arr=tilt_grad_out_plain,
    )
    e_out_ctx = tilt_smoothness_out.compute_energy_and_gradient_array(
        mesh,
        mesh.global_parameters,
        param_resolver,
        positions=positions,
        index_map=index_map,
        grad_arr=grad_dummy,
        tilts_out=tilts_out,
        tilt_out_grad_arr=tilt_grad_out_ctx,
        ctx=ctx,
    )
    assert e_out_ctx == pytest.approx(e_out_plain, rel=1e-12, abs=1e-12)
    assert np.allclose(tilt_grad_out_ctx, tilt_grad_out_plain, rtol=0.0, atol=1e-12)


def test_tilt_splay_twist_in_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("tilt_splay_modulus_in", 0.6)
    mesh.global_parameters.set("tilt_twist_modulus_in", 0.2)
    _assert_energy_only_matches_gradient_path(
        tilt_splay_twist_in,
        mesh,
        ParameterResolver(mesh.global_parameters),
        tilt_fields={"tilts_in": _rng_tilts(mesh.tilts_in_view().shape, 8)},
        gradient_keys=("tilt_in_grad_arr",),
    )


def test_tilt_splay_twist_in_connection_v1_matches_ambient_on_planar_mesh():
    mesh = _build_planar_patch_mesh()
    mesh.global_parameters.set("tilt_splay_modulus_in", 0.6)
    mesh.global_parameters.set("tilt_twist_modulus_in", 0.2)
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row
    tilts_in = _rng_tilts(mesh.tilts_in_view().shape, 11)

    mesh.global_parameters.set("tilt_transport_model", "ambient_v1")
    resolver = ParameterResolver(mesh.global_parameters)
    ambient_energy = tilt_splay_twist_in.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts_in,
    )

    mesh.global_parameters.set("tilt_transport_model", "connection_v1")
    resolver = ParameterResolver(mesh.global_parameters)
    connection_energy = tilt_splay_twist_in.compute_energy_array(
        mesh,
        mesh.global_parameters,
        resolver,
        positions=positions,
        index_map=index_map,
        tilts_in=tilts_in,
    )

    assert connection_energy == pytest.approx(ambient_energy, rel=1e-12, abs=1e-12)


def test_tilt_splay_twist_in_connection_v1_energy_array_matches_gradient_path():
    mesh = _build_mesh()
    mesh.global_parameters.set("tilt_splay_modulus_in", 0.6)
    mesh.global_parameters.set("tilt_twist_modulus_in", 0.2)
    mesh.global_parameters.set("tilt_transport_model", "connection_v1")
    _assert_energy_only_matches_gradient_path(
        tilt_splay_twist_in,
        mesh,
        ParameterResolver(mesh.global_parameters),
        tilt_fields={"tilts_in": _rng_tilts(mesh.tilts_in_view().shape, 12)},
        gradient_keys=("tilt_in_grad_arr",),
    )

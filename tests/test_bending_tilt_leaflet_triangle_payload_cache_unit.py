import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from modules.energy import bending_tilt_leaflet as bt_leaflet  # noqa: E402


def test_leaflet_triangle_payload_cache_reuses_and_invalidates_on_absence_change() -> (
    None
):
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    mesh.build_position_cache()
    gp = mesh.global_parameters
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    payload = bt_leaflet._leaflet_triangle_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        cache_tag="out",
    )
    assert payload["tri_rows"].shape[0] > 0
    assert payload["tri_rows"].shape[0] < payload["tri_rows_full"].shape[0]

    cache = getattr(mesh, "_bending_tilt_leaflet_triangle_payload_cache_out")
    sentinel = object()
    cache["value"] = sentinel

    cached = bt_leaflet._leaflet_triangle_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        cache_tag="out",
    )
    assert cached is sentinel

    gp.set("leaflet_out_absent_presets", ["__no_matching_preset__"])
    refreshed = bt_leaflet._leaflet_triangle_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        cache_tag="out",
    )
    assert refreshed is not sentinel
    assert refreshed["tri_rows"].shape[0] > payload["tri_rows"].shape[0]


def test_leaflet_static_tilt_payload_cache_reuses_and_invalidates_on_c0_change() -> (
    None
):
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    mesh.build_position_cache()
    gp = mesh.global_parameters
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    tri_payload = bt_leaflet._leaflet_triangle_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        cache_tag="out",
    )
    static_payload = bt_leaflet._leaflet_static_tilt_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        k_vecs=tri_payload["k_vecs"],
        vertex_areas_vor=tri_payload["vertex_areas_vor"],
        tri_rows=tri_payload["tri_rows"],
        kappa_key="bending_modulus_out",
        cache_tag="out",
    )
    assert static_payload["base_tri"].shape[0] == tri_payload["tri_rows"].shape[0]

    cache = getattr(mesh, "_bending_tilt_leaflet_static_cache_out")
    sentinel = object()
    cache["value"] = sentinel

    original = bt_leaflet._per_vertex_params_leaflet

    def _unexpected_params(*args, **kwargs):
        raise AssertionError(
            "hot static payload cache hit should bypass param assembly"
        )

    bt_leaflet._per_vertex_params_leaflet = _unexpected_params

    try:
        cached = bt_leaflet._leaflet_static_tilt_payload(
            mesh,
            gp,
            positions=positions,
            index_map=index_map,
            k_vecs=tri_payload["k_vecs"],
            vertex_areas_vor=tri_payload["vertex_areas_vor"],
            tri_rows=tri_payload["tri_rows"],
            kappa_key="bending_modulus_out",
            cache_tag="out",
        )
        assert cached is sentinel
    finally:
        bt_leaflet._per_vertex_params_leaflet = original

    gp.set("spontaneous_curvature_out", 1.0)
    refreshed = bt_leaflet._leaflet_static_tilt_payload(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
        k_vecs=tri_payload["k_vecs"],
        vertex_areas_vor=tri_payload["vertex_areas_vor"],
        tri_rows=tri_payload["tri_rows"],
        kappa_key="bending_modulus_out",
        cache_tag="out",
    )
    assert refreshed is not sentinel
    assert not (refreshed["base_tri"] == static_payload["base_tri"]).all()

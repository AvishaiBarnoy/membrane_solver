import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import load_data, parse_geometry  # noqa: E402
from modules.energy import tilt_smoothness as base_smoothness  # noqa: E402
from modules.energy import tilt_smoothness_out  # noqa: E402


def test_tilt_smoothness_out_mask_cache_reuses_and_invalidates() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    mesh.build_position_cache()
    gp = mesh.global_parameters
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    weights_full, tri_rows_full = base_smoothness._get_weights_and_tris(
        mesh, positions=positions, index_map=index_map
    )
    weights_a, tri_rows_a = tilt_smoothness_out._masked_weights_and_tris(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
    )

    assert tri_rows_full is not None
    assert weights_full is not None
    assert tri_rows_a is not None
    assert weights_a is not None
    assert tri_rows_a.shape[0] > 0
    assert tri_rows_a.shape[0] < tri_rows_full.shape[0]

    weights_b, tri_rows_b = tilt_smoothness_out._masked_weights_and_tris(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
    )

    assert weights_b is weights_a
    assert tri_rows_b is tri_rows_a

    gp.set("leaflet_out_absent_presets", ["__no_matching_preset__"])
    weights_c, tri_rows_c = tilt_smoothness_out._masked_weights_and_tris(
        mesh,
        gp,
        positions=positions,
        index_map=index_map,
    )

    assert weights_c is not weights_a
    assert tri_rows_c is not tri_rows_a
    assert tri_rows_c.shape[0] > tri_rows_a.shape[0]

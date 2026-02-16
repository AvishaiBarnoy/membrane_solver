import numpy as np

from geometry.curvature import compute_curvature_data
from geometry.geom_io import load_data, parse_geometry


def test_geometry_freeze_invalidates_stale_curvature_cache_for_new_positions() -> None:
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    mesh.build_position_cache()
    positions = mesh.positions_view()
    index_map = mesh.vertex_index_to_row

    # Seed curvature cache at the current positions.
    compute_curvature_data(mesh, positions, index_map)

    # Evaluate at a different (frozen) positions array without changing mesh version.
    frozen_positions = positions.copy(order="F")
    frozen_positions[0, 0] += 1e-3

    with mesh.geometry_freeze(frozen_positions):
        k_freeze, a_freeze, w_freeze, tri_freeze = compute_curvature_data(
            mesh, frozen_positions, index_map
        )

    # No-cache reference for the same positions must match the freeze-path result.
    k_ref, a_ref, w_ref, tri_ref = compute_curvature_data(
        mesh, frozen_positions.copy(order="F"), index_map
    )

    np.testing.assert_allclose(k_freeze, k_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a_freeze, a_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(w_freeze, w_ref, rtol=0, atol=1e-12)
    assert np.array_equal(tri_freeze, tri_ref)


def test_geometry_freeze_invalidates_reused_positions_buffer_after_version_bump() -> (
    None
):
    mesh = parse_geometry(
        load_data("tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml")
    )
    mesh.build_position_cache()
    index_map = mesh.vertex_index_to_row

    positions_v0 = mesh.positions_view()
    compute_curvature_data(mesh, positions_v0, index_map)

    # Mutate geometry and bump the mesh version. positions_view() typically
    # reuses the same array object, so id(positions) is not a safe cache key.
    vid0 = int(mesh.vertex_ids[0])
    mesh.vertices[vid0].position[0] += 1e-3
    mesh.increment_version()
    positions_v1 = mesh.positions_view()

    with mesh.geometry_freeze(positions_v1):
        k_freeze, a_freeze, w_freeze, tri_freeze = compute_curvature_data(
            mesh, positions_v1, index_map
        )

    # Fresh no-cache reference at the same geometry.
    mesh._curvature_version = -1
    k_ref, a_ref, w_ref, tri_ref = compute_curvature_data(
        mesh, positions_v1.copy(order="F"), index_map
    )

    np.testing.assert_allclose(k_freeze, k_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(a_freeze, a_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(w_freeze, w_ref, rtol=0, atol=1e-12)
    assert np.array_equal(tri_freeze, tri_ref)

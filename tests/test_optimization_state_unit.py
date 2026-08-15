from __future__ import annotations

import numpy as np

from geometry.geom_io import parse_geometry
from runtime.optimization_state import (
    capture_position_state,
    capture_tilt_state,
    commit_position_rows,
    restore_position_state,
    restore_tilt_state,
)


def test_vertex_tilt_snapshot_is_copy_and_restores_through_mesh_setter() -> None:
    mesh = parse_geometry(
        {
            "vertices": [
                [0.0, 0.0, 0.0, {"tilt": [1.0, 0.0]}],
                [1.0, 0.0, 0.0, {"tilt": [0.0, 1.0]}],
            ],
            "edges": [[0, 1]],
        }
    )
    snapshot = capture_tilt_state(mesh, uses_leaflet_tilts=False)
    expected = snapshot.tilts.copy()

    mesh.set_tilts_from_array(np.full_like(mesh.tilts_view(), 4.0))
    assert np.array_equal(snapshot.tilts, expected)

    restore_tilt_state(
        mesh,
        snapshot,
        set_leaflet_tilts=lambda _tilts_in, _tilts_out: None,
    )
    assert np.array_equal(mesh.tilts_view(), expected)


def test_leaflet_snapshot_uses_supplied_version_aware_restore_callback() -> None:
    mesh = parse_geometry(
        {
            "energy_modules": ["tilt_in", "tilt_out"],
            "vertices": [
                [
                    0.0,
                    0.0,
                    0.0,
                    {"tilt_in": [1.0, 0.0], "tilt_out": [0.0, 1.0]},
                ],
                [1.0, 0.0, 0.0],
            ],
            "edges": [[0, 1]],
        }
    )
    snapshot = capture_tilt_state(mesh, uses_leaflet_tilts=True)
    expected_in = snapshot.tilts_in.copy()
    expected_out = snapshot.tilts_out.copy()
    calls: list[tuple[np.ndarray, np.ndarray]] = []

    mesh.set_tilts_in_from_array(np.full_like(mesh.tilts_in_view(), 3.0))
    mesh.set_tilts_out_from_array(np.full_like(mesh.tilts_out_view(), -3.0))

    def restore_leaflets(tilts_in: np.ndarray, tilts_out: np.ndarray) -> None:
        calls.append((tilts_in, tilts_out))
        mesh.set_tilts_in_from_array(tilts_in)
        mesh.set_tilts_out_from_array(tilts_out)

    restore_tilt_state(mesh, snapshot, set_leaflet_tilts=restore_leaflets)

    assert len(calls) == 1
    assert calls[0][0] is snapshot.tilts_in
    assert calls[0][1] is snapshot.tilts_out
    assert np.array_equal(mesh.tilts_in_view(), expected_in)
    assert np.array_equal(mesh.tilts_out_view(), expected_out)


def test_position_snapshot_restores_and_commits_rows_without_touching_version() -> None:
    mesh = parse_geometry(
        {
            "vertices": [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            "edges": [[0, 1], [1, 2], [2, 0]],
        }
    )
    snapshot = capture_position_state(mesh)
    version = mesh._version
    trial = snapshot.positions.copy()
    trial[1] = np.array([2.0, 3.0, 4.0])

    commit_position_rows(
        mesh,
        vertex_ids=snapshot.vertex_ids,
        rows=np.asarray([1]),
        positions=trial,
    )
    assert np.array_equal(mesh.vertices[int(snapshot.vertex_ids[1])].position, trial[1])
    assert mesh._version == version

    restore_position_state(mesh, snapshot)
    assert np.array_equal(
        mesh.vertices[int(snapshot.vertex_ids[1])].position, snapshot.positions[1]
    )
    assert mesh._version == version

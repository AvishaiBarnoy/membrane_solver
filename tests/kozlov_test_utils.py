"""Shared construction helpers for Kozlov regression and E2E tests."""

from pathlib import Path

import numpy as np

from tests.minimizer_test_utils import build_minimizer as build_minimizer

FIXTURE_DIR = Path(__file__).with_name("fixtures")


def fixture_path(name: str) -> str:
    """Return an absolute path to a named test fixture."""
    return str(FIXTURE_DIR / name)


def collect_group_rows(mesh, key: str, value: str) -> np.ndarray:
    """Return mesh rows whose vertex option matches a group value."""
    rows = [
        mesh.vertex_index_to_row[int(vid)]
        for vid in mesh.vertex_ids
        if (getattr(mesh.vertices[int(vid)], "options", None) or {}).get(key) == value
    ]
    return np.asarray(rows, dtype=int)


def order_by_angle(positions: np.ndarray) -> np.ndarray:
    """Return XY angular order for a position array."""
    return np.argsort(np.arctan2(positions[:, 1], positions[:, 0]))


def radial_unit_vectors(positions: np.ndarray) -> np.ndarray:
    """Return XY radial unit vectors with zeros at the origin."""
    radii = np.linalg.norm(positions[:, :2], axis=1)
    radial = np.zeros_like(positions)
    good = radii > 1e-12
    radial[good, 0] = positions[good, 0] / radii[good]
    radial[good, 1] = positions[good, 1] / radii[good]
    return radial


def outer_free_ring_rows(mesh, positions: np.ndarray) -> np.ndarray:
    """Return outermost rows not pinned to the outer circle group."""
    rows: list[int] = []
    radii: list[float] = []
    for vid in mesh.vertex_ids:
        vertex = mesh.vertices[int(vid)]
        opts = getattr(vertex, "options", None) or {}
        if opts.get("pin_to_circle_group") == "outer":
            continue
        row = mesh.vertex_index_to_row[int(vid)]
        rows.append(row)
        radii.append(float(np.linalg.norm(positions[row, :2])))
    if not rows:
        return np.zeros(0, dtype=int)
    radii_arr = np.asarray(radii, dtype=float)
    rows_arr = np.asarray(rows, dtype=int)
    return rows_arr[np.abs(radii_arr - float(np.max(radii_arr))) <= 1e-6]

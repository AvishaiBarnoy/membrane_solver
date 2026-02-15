import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.energy.bending_tilt_leaflet import _scatter_add_vec_bincount  # noqa: E402


def test_scatter_add_vec_bincount_matches_add_at_with_duplicate_rows() -> None:
    rng = np.random.default_rng(123)
    n_rows = 32
    n_terms = 400
    rows = rng.integers(0, n_rows, size=n_terms, dtype=int)
    values = rng.normal(size=(n_terms, 3))

    ref = np.zeros((n_rows, 3), dtype=float)
    np.add.at(ref, rows, values)

    got = np.zeros((n_rows, 3), dtype=float)
    _scatter_add_vec_bincount(got, rows, values)

    np.testing.assert_allclose(got, ref, rtol=0.0, atol=0.0)


def test_scatter_add_vec_bincount_handles_empty_rows() -> None:
    dest = np.zeros((5, 3), dtype=float)
    _scatter_add_vec_bincount(dest, np.zeros(0, dtype=int), np.zeros((0, 3)))
    np.testing.assert_allclose(dest, np.zeros((5, 3), dtype=float), rtol=0.0, atol=0.0)

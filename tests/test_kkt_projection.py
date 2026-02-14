import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from runtime.constraint_manager import ConstraintModuleManager


class DummyConstraint:
    def __init__(self, gC):
        self._gC = gC

    def constraint_gradients(self, mesh, global_params):
        if isinstance(self._gC, list):
            return self._gC
        return None

    def constraint_gradient(self, mesh, global_params):
        if isinstance(self._gC, list):
            return None
        return self._gC


class DummyArrayConstraint:
    def __init__(self, constraints):
        self._constraints = constraints

    def constraint_gradients_array(self, mesh, global_params, **kwargs):
        _ = mesh, global_params, kwargs
        return self._constraints


class DummyRowArrayConstraint:
    def __init__(self, row_constraints):
        self._row_constraints = row_constraints

    def constraint_gradients_rows_array(self, mesh, global_params, **kwargs):
        _ = mesh, global_params, kwargs
        return self._row_constraints


class DummyTiltConstraint:
    def __init__(self, constraints):
        self._constraints = constraints

    def constraint_gradients_tilt_array(self, mesh, global_params, **kwargs):
        _ = mesh, global_params, kwargs
        return self._constraints


class DummyTiltRowConstraint:
    def __init__(self, row_constraints):
        self._row_constraints = row_constraints

    def constraint_gradients_tilt_rows_array(self, mesh, global_params, **kwargs):
        _ = mesh, global_params, kwargs
        return self._row_constraints


class DummyMesh:
    def __init__(self, n_rows: int):
        self._positions = np.zeros((n_rows, 3), dtype=float)
        self.vertex_index_to_row = {i: i for i in range(n_rows)}

    def build_position_cache(self):
        return None

    def positions_view(self):
        return self._positions

    def tilts_in_view(self):
        return np.zeros_like(self._positions)

    def tilts_out_view(self):
        return np.zeros_like(self._positions)


def test_single_constraint_kkt_projection_zeroes_parallel_component():
    cm = ConstraintModuleManager([])
    cm.modules = {
        "dummy": DummyConstraint(
            {
                0: np.array([1.0, 0.0, 0.0]),
                1: np.array([0.0, 1.0, 0.0]),
            }
        )
    }

    grad = {
        0: np.array([1.0, 0.0, 0.0]),
        1: np.array([0.0, 1.0, 0.0]),
    }

    cm.apply_gradient_modifications(grad, mesh=None, global_params=None)

    assert np.allclose(grad[0], np.zeros(3))
    assert np.allclose(grad[1], np.zeros(3))


def test_multi_constraint_kkt_projection_removes_all_components():
    cm = ConstraintModuleManager([])
    cm.modules = {
        "dummy": DummyConstraint(
            [
                {0: np.array([1.0, 0.0, 0.0])},
                {1: np.array([0.0, 1.0, 0.0])},
            ]
        )
    }

    grad = {
        0: np.array([2.0, 0.0, 0.0]),
        1: np.array([0.0, -3.0, 0.0]),
    }

    cm.apply_gradient_modifications(grad, mesh=None, global_params=None)

    assert np.allclose(grad[0], np.zeros(3))
    assert np.allclose(grad[1], np.zeros(3))


def test_array_constraint_kkt_projection_removes_components():
    cm = ConstraintModuleManager([])
    g0 = np.zeros((2, 3), dtype=float)
    g1 = np.zeros((2, 3), dtype=float)
    g0[0, 0] = 1.0
    g1[1, 1] = 1.0
    cm.modules = {"dummy": DummyArrayConstraint([g0, g1])}

    grad_arr = np.zeros((2, 3), dtype=float)
    grad_arr[0, 0] = 2.0
    grad_arr[1, 1] = -3.0
    mesh = DummyMesh(2)

    cm.apply_gradient_modifications_array(grad_arr, mesh=mesh, global_params=None)
    assert np.allclose(grad_arr, np.zeros_like(grad_arr))


def test_tilt_array_constraint_kkt_projection_removes_components():
    cm = ConstraintModuleManager([])
    g_in = np.zeros((2, 3), dtype=float)
    g_out = np.zeros((2, 3), dtype=float)
    g_in[0, 0] = 1.0
    g_out[1, 1] = 1.0
    cm.modules = {"dummy": DummyTiltConstraint([(g_in, None), (None, g_out)])}

    tilt_in_grad = np.zeros((2, 3), dtype=float)
    tilt_out_grad = np.zeros((2, 3), dtype=float)
    tilt_in_grad[0, 0] = 4.0
    tilt_out_grad[1, 1] = -5.0
    mesh = DummyMesh(2)

    cm.apply_tilt_gradient_modifications_array(
        tilt_in_grad, tilt_out_grad, mesh=mesh, global_params=None
    )
    assert np.allclose(tilt_in_grad, np.zeros_like(tilt_in_grad))
    assert np.allclose(tilt_out_grad, np.zeros_like(tilt_out_grad))


def test_row_constraint_kkt_projection_matches_dense():
    cm_dense = ConstraintModuleManager([])
    cm_rows = ConstraintModuleManager([])

    g0 = np.zeros((3, 3), dtype=float)
    g1 = np.zeros((3, 3), dtype=float)
    g0[0, :] = np.array([1.0, 2.0, 0.0])
    g0[2, :] = np.array([0.0, -1.0, 1.0])
    g1[1, :] = np.array([0.0, 1.0, 1.0])

    cm_dense.modules = {"dummy": DummyArrayConstraint([g0, g1])}
    cm_rows.modules = {
        "dummy": DummyRowArrayConstraint(
            [
                (
                    np.asarray([0, 2], dtype=int),
                    np.asarray([[1.0, 2.0, 0.0], [0.0, -1.0, 1.0]], dtype=float),
                ),
                (
                    np.asarray([1], dtype=int),
                    np.asarray([[0.0, 1.0, 1.0]], dtype=float),
                ),
            ]
        )
    }

    grad_dense = np.asarray(
        [[1.0, -2.0, 0.5], [3.0, 4.0, -1.0], [0.1, 0.2, 0.3]], dtype=float
    )
    grad_rows = grad_dense.copy()
    mesh = DummyMesh(3)

    cm_dense.apply_gradient_modifications_array(
        grad_dense, mesh=mesh, global_params=None
    )
    cm_rows.apply_gradient_modifications_array(grad_rows, mesh=mesh, global_params=None)

    assert np.allclose(grad_rows, grad_dense, atol=1e-12, rtol=0.0)


def test_tilt_row_constraint_kkt_projection_matches_dense():
    cm_dense = ConstraintModuleManager([])
    cm_rows = ConstraintModuleManager([])

    g_in = np.zeros((2, 3), dtype=float)
    g_out = np.zeros((2, 3), dtype=float)
    g_in[0] = np.array([1.0, 0.0, 0.0])
    g_out[1] = np.array([0.0, 2.0, 0.0])

    cm_dense.modules = {"dummy": DummyTiltConstraint([(g_in, None), (None, g_out)])}
    cm_rows.modules = {
        "dummy": DummyTiltRowConstraint(
            [
                ((np.asarray([0], dtype=int), np.asarray([[1.0, 0.0, 0.0]])), None),
                (None, (np.asarray([1], dtype=int), np.asarray([[0.0, 2.0, 0.0]]))),
            ]
        )
    }

    tilt_in_dense = np.asarray([[5.0, 1.0, -2.0], [0.0, 0.0, 0.0]], dtype=float)
    tilt_out_dense = np.asarray([[0.0, 0.0, 0.0], [-4.0, 3.0, 1.0]], dtype=float)
    tilt_in_rows = tilt_in_dense.copy()
    tilt_out_rows = tilt_out_dense.copy()
    mesh = DummyMesh(2)

    cm_dense.apply_tilt_gradient_modifications_array(
        tilt_in_dense, tilt_out_dense, mesh=mesh, global_params=None
    )
    cm_rows.apply_tilt_gradient_modifications_array(
        tilt_in_rows, tilt_out_rows, mesh=mesh, global_params=None
    )

    assert np.allclose(tilt_in_rows, tilt_in_dense, atol=1e-12, rtol=0.0)
    assert np.allclose(tilt_out_rows, tilt_out_dense, atol=1e-12, rtol=0.0)


def test_tilt_mixed_constraints_match_stacked_reference():
    cm = ConstraintModuleManager([])

    g_in_dense = np.zeros((3, 3), dtype=float)
    g_out_dense = np.zeros((3, 3), dtype=float)
    g_in_dense[0] = np.array([1.0, -2.0, 0.5])
    g_out_dense[2] = np.array([0.25, 1.0, -1.5])

    rows_in = np.asarray([1], dtype=int)
    vecs_in = np.asarray([[0.5, 0.0, 1.0]], dtype=float)
    rows_out = np.asarray([0, 1], dtype=int)
    vecs_out = np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=float)

    cm.modules = {
        "dense": DummyTiltConstraint([(g_in_dense, g_out_dense)]),
        "rows": DummyTiltRowConstraint([((rows_in, vecs_in), (rows_out, vecs_out))]),
    }

    tilt_in = np.asarray(
        [[2.0, -1.0, 0.0], [0.5, 0.25, -0.5], [0.0, 0.0, 1.0]], dtype=float
    )
    tilt_out = np.asarray(
        [[-1.0, 0.5, 0.0], [1.5, -0.25, 0.75], [0.1, -0.1, 0.3]], dtype=float
    )
    mesh = DummyMesh(3)

    got_in = tilt_in.copy()
    got_out = tilt_out.copy()
    cm.apply_tilt_gradient_modifications_array(
        got_in, got_out, mesh=mesh, global_params=None
    )

    # Stacked reference formulation (the pre-optimization algebra).
    n_single = tilt_in.size
    n_total = 2 * n_single
    C = np.zeros((2, n_total), dtype=float)
    C[0, :n_single] = g_in_dense.reshape(-1)
    C[0, n_single:] = g_out_dense.reshape(-1)
    # Sparse row payload in/out packed into a single stacked row.
    for row, vec in zip(rows_in, vecs_in):
        base = 3 * int(row)
        C[1, base : base + 3] += vec
    for row, vec in zip(rows_out, vecs_out):
        base = n_single + 3 * int(row)
        C[1, base : base + 3] += vec

    stacked = np.concatenate([tilt_in.reshape(-1), tilt_out.reshape(-1)])
    b = C @ stacked
    A = C @ C.T
    A[np.diag_indices_from(A)] += 1e-18
    lam = np.linalg.solve(A, b)
    stacked -= C.T @ lam
    ref_in = stacked[:n_single].reshape(tilt_in.shape)
    ref_out = stacked[n_single:].reshape(tilt_out.shape)

    assert np.allclose(got_in, ref_in, atol=1e-12, rtol=0.0)
    assert np.allclose(got_out, ref_out, atol=1e-12, rtol=0.0)

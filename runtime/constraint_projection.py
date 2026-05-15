import hashlib
import logging

import numpy as np

logger = logging.getLogger("membrane_solver")


def _as_row_gradient_payload(payload):
    """Normalize a sparse row-gradient payload to ``(rows, vecs)`` arrays.

    Expected shapes:
      - rows: (m,), integer row ids in ``[0, n_rows)``
      - vecs: (m, 3), float vectors at each row
    """
    if not isinstance(payload, tuple) or len(payload) != 2:
        raise ValueError("row gradient payload must be a (rows, vectors) tuple")
    rows = np.asarray(payload[0], dtype=int).reshape(-1)
    vecs = np.asarray(payload[1], dtype=float)
    if vecs.ndim != 2 or vecs.shape[1] != 3:
        raise ValueError("row gradient vectors must have shape (m, 3)")
    if rows.shape[0] != vecs.shape[0]:
        raise ValueError("row gradient rows/vectors length mismatch")
    return rows, vecs


def _accumulate_sparse_row_into_dense_flat(
    dense_row: np.ndarray, rows: np.ndarray, vecs: np.ndarray
) -> None:
    if rows.size == 0:
        return
    # Dense row is flattened (N,3); accumulate directly in 2D to avoid
    # allocating index-expansion buffers on every constraint row.
    dense_row_2d = dense_row.reshape(-1, 3)
    np.add.at(dense_row_2d, rows, vecs)


def _coalesce_sparse_row_payload(
    rows: np.ndarray, vecs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic sparse payload with sorted, coalesced rows."""
    if rows.size == 0:
        return rows, vecs
    if rows.size == 1:
        return rows.reshape(-1), vecs.reshape(1, 3)
    order = np.argsort(rows, kind="stable")
    rows = rows[order]
    vecs = vecs[order]
    uniq_rows, inv = np.unique(rows, return_inverse=True)
    if uniq_rows.size == rows.size:
        return rows, vecs
    uniq_vecs = np.zeros((uniq_rows.size, 3), dtype=float)
    np.add.at(uniq_vecs, inv, vecs)
    return uniq_rows, uniq_vecs


def _solve_kkt_system(A: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    """Solve regularized KKT normal equations with SPD fast path."""
    try:
        L = np.linalg.cholesky(A)
        y = np.linalg.solve(L, b)
        return np.linalg.solve(L.T, y)
    except np.linalg.LinAlgError:
        try:
            return np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None


def _project_dense_gradient_against_constraints(
    grad_arr: np.ndarray, constraints: list[np.ndarray]
) -> None:
    """Project ``grad_arr`` onto the complement of dense constraint rows.

    Uses a vectorized KKT solve:
      A = C C^T, b = C g, lam = solve(A, b), g <- g - C^T lam
    where each row of ``C`` is one flattened dense constraint gradient.
    """
    if not constraints:
        return
    if len(constraints) == 1:
        gC = constraints[0]
        norm_sq = float(np.sum(gC * gC))
        if norm_sq > 1e-18:
            lam = float(np.sum(grad_arr * gC)) / norm_sq
            grad_arr -= lam * gC
        return

    grad_flat = grad_arr.reshape(-1)
    C = np.stack([np.asarray(gC, dtype=float).reshape(-1) for gC in constraints])
    b = C @ grad_flat
    A = C @ C.T
    A[np.diag_indices_from(A)] += 1e-18
    try:
        lam = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return
    grad_flat -= C.T @ lam


def _project_mixed_gradient_against_constraints(
    grad_arr: np.ndarray,
    *,
    dense_constraints: list[np.ndarray],
    sparse_constraints: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Project using dense and sparse-row constraints in one KKT solve."""
    n_dof = int(grad_arr.size)
    k = len(dense_constraints) + len(sparse_constraints)
    if k == 0:
        return

    C = np.zeros((k, n_dof), dtype=float)
    row_idx = 0
    for gC in dense_constraints:
        C[row_idx, :] = np.asarray(gC, dtype=float).reshape(-1)
        row_idx += 1
    for rows, vecs in sparse_constraints:
        _accumulate_sparse_row_into_dense_flat(C[row_idx, :], rows, vecs)
        row_idx += 1

    grad_flat = grad_arr.reshape(-1)
    b = C @ grad_flat
    A = C @ C.T
    A[np.diag_indices_from(A)] += 1e-18
    lam = _solve_kkt_system(A, b)
    if lam is None:
        return
    grad_flat -= C.T @ lam


def _row_constraints_payload_token(
    row_constraints: list[
        tuple[
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
        ]
    ],
) -> int:
    """Return a content hash token for sparse row constraints."""
    h = hashlib.blake2b(digest_size=16)
    for in_part, out_part in row_constraints:
        for tag, part in ((b"i", in_part), (b"o", out_part)):
            h.update(tag)
            if part is None:
                h.update(b"\x00")
                continue
            rows, vecs = part
            h.update(b"\x01")
            rows64 = np.ascontiguousarray(rows, dtype=np.int64)
            vecs64 = np.ascontiguousarray(vecs, dtype=np.float64)
            h.update(rows64.tobytes())
            h.update(vecs64.tobytes())
    return int.from_bytes(h.digest(), "little")


def _build_leaflet_sparse_projection_operator(
    *,
    row_constraints: list[
        tuple[
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
        ]
    ],
    n_single: int,
) -> dict[str, np.ndarray] | None:
    """Build a reusable sparse leaflet projection operator payload."""
    k = len(row_constraints)
    if k == 0:
        return None
    if k == 1:
        return None

    n_total = 2 * n_single
    axes = np.asarray([0, 1, 2], dtype=int)
    cols_chunks: list[np.ndarray] = []
    for in_part, out_part in row_constraints:
        if in_part is not None:
            in_rows, _ = in_part
            cols_chunks.append((3 * in_rows[:, None] + axes[None, :]).reshape(-1))
        if out_part is not None:
            out_rows, _ = out_part
            cols_chunks.append(
                (n_single + 3 * out_rows[:, None] + axes[None, :]).reshape(-1)
            )
    if not cols_chunks:
        return None

    active_cols = np.unique(np.concatenate(cols_chunks))
    col_to_comp = np.full(n_total, -1, dtype=int)
    col_to_comp[active_cols] = np.arange(active_cols.size, dtype=int)
    in_mask = active_cols < n_single
    in_pos = np.nonzero(in_mask)[0]
    out_pos = np.nonzero(~in_mask)[0]
    active_in_cols = active_cols[in_pos]
    active_out_cols = active_cols[out_pos] - n_single

    C = np.zeros((k, active_cols.size), dtype=float)
    for i, (in_part, out_part) in enumerate(row_constraints):
        if in_part is not None:
            in_rows, in_vecs = in_part
            in_cols = (3 * in_rows[:, None] + axes[None, :]).reshape(-1)
            C[i, col_to_comp[in_cols]] = in_vecs.reshape(-1)
        if out_part is not None:
            out_rows, out_vecs = out_part
            out_cols = (n_single + 3 * out_rows[:, None] + axes[None, :]).reshape(-1)
            C[i, col_to_comp[out_cols]] = out_vecs.reshape(-1)

    A = C @ C.T
    A[np.diag_indices_from(A)] += 1e-18
    chol_L = None
    solve_mat = None
    try:
        chol_L = np.linalg.cholesky(A)
        ident = np.eye(A.shape[0], dtype=float)
        y = np.linalg.solve(chol_L, ident)
        solve_mat = np.linalg.solve(chol_L.T, y)
    except np.linalg.LinAlgError:
        pass
    if solve_mat is None:
        try:
            solve_mat = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            solve_mat = None
    return {
        "active_cols": active_cols,
        "active_in_pos": in_pos,
        "active_in_rows": active_in_cols // 3,
        "active_in_comp": active_in_cols % 3,
        "active_out_pos": out_pos,
        "active_out_rows": active_out_cols // 3,
        "active_out_comp": active_out_cols % 3,
        "C": C,
        "A": A,
        "chol_L": chol_L,
        "solve_mat": solve_mat,
        "proj_active": (
            None if solve_mat is None else np.asarray(C.T @ solve_mat @ C, dtype=float)
        ),
    }


def _solve_leaflet_sparse_kkt(
    operator_cache: dict[str, np.ndarray], b: np.ndarray
) -> np.ndarray | None:
    """Solve ``A * lam = b`` from cached operator payload."""
    solve_mat = operator_cache.get("solve_mat")
    if solve_mat is not None:
        return solve_mat @ b
    chol_L = operator_cache.get("chol_L")
    if chol_L is not None:
        try:
            y = np.linalg.solve(chol_L, b)
            return np.linalg.solve(chol_L.T, y)
        except np.linalg.LinAlgError:
            # Fall back to direct solve below if triangular solve fails.
            pass
    A = operator_cache.get("A")
    if A is None:
        return None
    return _solve_kkt_system(A, b)


def _project_leaflet_sparse_rows_against_constraints(
    tilt_in_grad_arr: np.ndarray,
    tilt_out_grad_arr: np.ndarray,
    row_constraints: list[
        tuple[
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
        ]
    ],
    *,
    operator_cache: dict[str, np.ndarray] | None = None,
) -> None:
    """Project leaflet gradients using sparse row constraints only.

    This avoids constructing a dense stacked constraint matrix over all
    leaflet DOFs when all active constraints are row-sparse payloads.
    """
    k = len(row_constraints)
    if k == 0:
        return

    if k == 1:
        in_part, out_part = row_constraints[0]
        norm_sq = 0.0
        dot = 0.0
        if in_part is not None:
            in_rows, in_vecs = in_part
            norm_sq += float(np.sum(in_vecs * in_vecs))
            dot += float(np.sum(tilt_in_grad_arr[in_rows] * in_vecs))
        if out_part is not None:
            out_rows, out_vecs = out_part
            norm_sq += float(np.sum(out_vecs * out_vecs))
            dot += float(np.sum(tilt_out_grad_arr[out_rows] * out_vecs))
        if norm_sq <= 1e-18:
            return
        scale = -(dot / norm_sq)
        if in_part is not None:
            in_rows, in_vecs = in_part
            np.add.at(tilt_in_grad_arr, in_rows, scale * in_vecs)
        if out_part is not None:
            out_rows, out_vecs = out_part
            np.add.at(tilt_out_grad_arr, out_rows, scale * out_vecs)
        return

    n_single = int(tilt_in_grad_arr.size)
    if operator_cache is None:
        operator_cache = _build_leaflet_sparse_projection_operator(
            row_constraints=row_constraints,
            n_single=n_single,
        )
    if operator_cache is None:
        return
    active_cols = operator_cache["active_cols"]
    active_in_pos = operator_cache["active_in_pos"]
    active_in_rows = operator_cache["active_in_rows"]
    active_in_comp = operator_cache["active_in_comp"]
    active_out_pos = operator_cache["active_out_pos"]
    active_out_rows = operator_cache["active_out_rows"]
    active_out_comp = operator_cache["active_out_comp"]
    C = operator_cache["C"]
    proj_active = operator_cache.get("proj_active")

    grad_active = np.empty(active_cols.shape[0], dtype=float)
    if active_in_pos.shape[0]:
        grad_active[active_in_pos] = tilt_in_grad_arr[active_in_rows, active_in_comp]
    if active_out_rows.shape[0]:
        grad_active[active_out_pos] = tilt_out_grad_arr[
            active_out_rows, active_out_comp
        ]
    if proj_active is not None:
        delta = proj_active @ grad_active
    else:
        b = C @ grad_active
        lam = _solve_leaflet_sparse_kkt(operator_cache, b)
        if lam is None:
            return
        delta = C.T @ lam

    if active_in_pos.shape[0]:
        tilt_in_grad_arr[active_in_rows, active_in_comp] -= delta[active_in_pos]
    if active_out_rows.shape[0]:
        tilt_out_grad_arr[active_out_rows, active_out_comp] -= delta[active_out_pos]


def _joint_row_constraints_payload_token(
    row_constraints: list[
        tuple[
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
        ]
    ],
) -> int:
    """Return a content hash token for sparse joint row constraints."""
    h = hashlib.blake2b(digest_size=16)
    for shape_part, in_part, out_part in row_constraints:
        for tag, part in (
            (b"s", shape_part),
            (b"i", in_part),
            (b"o", out_part),
        ):
            h.update(tag)
            if part is None:
                h.update(b"\x00")
                continue
            rows, vecs = part
            h.update(b"\x01")
            rows64 = np.ascontiguousarray(rows, dtype=np.int64)
            vecs64 = np.ascontiguousarray(vecs, dtype=np.float64)
            h.update(rows64.tobytes())
            h.update(vecs64.tobytes())
    return int.from_bytes(h.digest(), "little")


def _build_joint_sparse_projection_operator(
    *,
    row_constraints: list[
        tuple[
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
        ]
    ],
    n_single: int,
) -> dict[str, np.ndarray] | None:
    """Build a reduced-column operator for sparse joint constraints."""
    k = len(row_constraints)
    if k <= 1:
        return None

    n_total = 3 * n_single
    axes = np.asarray([0, 1, 2], dtype=int)
    cols_chunks: list[np.ndarray] = []
    for shape_part, in_part, out_part in row_constraints:
        if shape_part is not None:
            shape_rows, _ = shape_part
            cols_chunks.append((3 * shape_rows[:, None] + axes[None, :]).reshape(-1))
        if in_part is not None:
            in_rows, _ = in_part
            cols_chunks.append(
                (n_single + 3 * in_rows[:, None] + axes[None, :]).reshape(-1)
            )
        if out_part is not None:
            out_rows, _ = out_part
            cols_chunks.append(
                (2 * n_single + 3 * out_rows[:, None] + axes[None, :]).reshape(-1)
            )
    if not cols_chunks:
        return None

    active_cols = np.unique(np.concatenate(cols_chunks))
    col_to_comp = np.full(n_total, -1, dtype=int)
    col_to_comp[active_cols] = np.arange(active_cols.size, dtype=int)

    shape_mask = active_cols < n_single
    in_mask = (active_cols >= n_single) & (active_cols < 2 * n_single)
    out_mask = active_cols >= 2 * n_single

    shape_pos = np.nonzero(shape_mask)[0]
    in_pos = np.nonzero(in_mask)[0]
    out_pos = np.nonzero(out_mask)[0]

    active_shape_cols = active_cols[shape_pos]
    active_in_cols = active_cols[in_pos] - n_single
    active_out_cols = active_cols[out_pos] - 2 * n_single

    C = np.zeros((k, active_cols.size), dtype=float)
    for i, (shape_part, in_part, out_part) in enumerate(row_constraints):
        if shape_part is not None:
            shape_rows, shape_vecs = shape_part
            shape_cols = (3 * shape_rows[:, None] + axes[None, :]).reshape(-1)
            C[i, col_to_comp[shape_cols]] = shape_vecs.reshape(-1)
        if in_part is not None:
            in_rows, in_vecs = in_part
            in_cols = (n_single + 3 * in_rows[:, None] + axes[None, :]).reshape(-1)
            C[i, col_to_comp[in_cols]] = in_vecs.reshape(-1)
        if out_part is not None:
            out_rows, out_vecs = out_part
            out_cols = (2 * n_single + 3 * out_rows[:, None] + axes[None, :]).reshape(
                -1
            )
            C[i, col_to_comp[out_cols]] = out_vecs.reshape(-1)

    A = C @ C.T
    A[np.diag_indices_from(A)] += 1e-18
    chol_L = None
    solve_mat = None
    try:
        chol_L = np.linalg.cholesky(A)
        ident = np.eye(A.shape[0], dtype=float)
        y = np.linalg.solve(chol_L, ident)
        solve_mat = np.linalg.solve(chol_L.T, y)
    except np.linalg.LinAlgError:
        pass
    if solve_mat is None:
        try:
            solve_mat = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            solve_mat = None

    return {
        "active_cols": active_cols,
        "shape_pos": shape_pos,
        "shape_rows": active_shape_cols // 3,
        "shape_comp": active_shape_cols % 3,
        "in_pos": in_pos,
        "in_rows": active_in_cols // 3,
        "in_comp": active_in_cols % 3,
        "out_pos": out_pos,
        "out_rows": active_out_cols // 3,
        "out_comp": active_out_cols % 3,
        "C": C,
        "A": A,
        "chol_L": chol_L,
        "solve_mat": solve_mat,
        "proj_active": (
            None if solve_mat is None else np.asarray(C.T @ solve_mat @ C, dtype=float)
        ),
    }


def _project_joint_sparse_rows_against_constraints(
    grad_arr: np.ndarray,
    tilt_in_grad_arr: np.ndarray,
    tilt_out_grad_arr: np.ndarray,
    row_constraints: list[
        tuple[
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
            tuple[np.ndarray, np.ndarray] | None,
        ]
    ],
    *,
    operator_cache: dict[str, np.ndarray] | None = None,
) -> None:
    """Project shape and tilt gradients using sparse joint constraints only."""
    k = len(row_constraints)
    if k == 0:
        return

    if k == 1:
        shape_part, in_part, out_part = row_constraints[0]
        norm_sq = 0.0
        dot = 0.0
        if shape_part is not None:
            shape_rows, shape_vecs = shape_part
            norm_sq += float(np.sum(shape_vecs * shape_vecs))
            dot += float(np.sum(grad_arr[shape_rows] * shape_vecs))
        if in_part is not None:
            in_rows, in_vecs = in_part
            norm_sq += float(np.sum(in_vecs * in_vecs))
            dot += float(np.sum(tilt_in_grad_arr[in_rows] * in_vecs))
        if out_part is not None:
            out_rows, out_vecs = out_part
            norm_sq += float(np.sum(out_vecs * out_vecs))
            dot += float(np.sum(tilt_out_grad_arr[out_rows] * out_vecs))
        if norm_sq <= 1e-18:
            return
        scale = -(dot / norm_sq)
        if shape_part is not None:
            shape_rows, shape_vecs = shape_part
            np.add.at(grad_arr, shape_rows, scale * shape_vecs)
        if in_part is not None:
            in_rows, in_vecs = in_part
            np.add.at(tilt_in_grad_arr, in_rows, scale * in_vecs)
        if out_part is not None:
            out_rows, out_vecs = out_part
            np.add.at(tilt_out_grad_arr, out_rows, scale * out_vecs)
        return

    n_single = int(grad_arr.size)
    if operator_cache is None:
        operator_cache = _build_joint_sparse_projection_operator(
            row_constraints=row_constraints,
            n_single=n_single,
        )
    if operator_cache is None:
        return

    active_cols = operator_cache["active_cols"]
    shape_pos = operator_cache["shape_pos"]
    shape_rows = operator_cache["shape_rows"]
    shape_comp = operator_cache["shape_comp"]
    in_pos = operator_cache["in_pos"]
    in_rows = operator_cache["in_rows"]
    in_comp = operator_cache["in_comp"]
    out_pos = operator_cache["out_pos"]
    out_rows = operator_cache["out_rows"]
    out_comp = operator_cache["out_comp"]
    C = operator_cache["C"]
    proj_active = operator_cache.get("proj_active")

    grad_active = np.empty(active_cols.shape[0], dtype=float)
    if shape_pos.shape[0]:
        grad_active[shape_pos] = grad_arr[shape_rows, shape_comp]
    if in_pos.shape[0]:
        grad_active[in_pos] = tilt_in_grad_arr[in_rows, in_comp]
    if out_pos.shape[0]:
        grad_active[out_pos] = tilt_out_grad_arr[out_rows, out_comp]

    if proj_active is not None:
        delta = proj_active @ grad_active
    else:
        b = C @ grad_active
        lam = _solve_leaflet_sparse_kkt(operator_cache, b)
        if lam is None:
            return
        delta = C.T @ lam

    if shape_pos.shape[0]:
        grad_arr[shape_rows, shape_comp] -= delta[shape_pos]
    if in_pos.shape[0]:
        tilt_in_grad_arr[in_rows, in_comp] -= delta[in_pos]
    if out_pos.shape[0]:
        tilt_out_grad_arr[out_rows, out_comp] -= delta[out_pos]

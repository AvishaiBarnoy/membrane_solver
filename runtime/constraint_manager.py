# runtime/constraint_manager.py

import hashlib
import importlib
import logging

import numpy as np

logger = logging.getLogger("ConstraintManager")


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
    """Combine duplicate row ids in a sparse row-gradient payload."""
    if rows.size <= 1:
        return rows, vecs
    uniq_rows, inv = np.unique(rows, return_inverse=True)
    if uniq_rows.size == rows.size:
        return rows, vecs
    uniq_vecs = np.zeros((uniq_rows.size, 3), dtype=float)
    np.add.at(uniq_vecs, inv, vecs)
    return uniq_rows, uniq_vecs


class ConstraintModuleManager:
    def __init__(self, module_names):
        self.modules = {}
        self._warned_no_grad = set()
        self._leaflet_sparse_projection_cache: dict[str, object] | None = None
        for name in module_names:
            try:
                self.modules[name] = importlib.import_module(
                    f"modules.constraints.{name}"
                )
                logger.info(f"Loaded constraint module: {name}")
            except ImportError as e:
                logger.error(f"Could not load constraint module '{name}': {e}")
                raise
        # self.modules = self._load_modules(module_names)

    def get_module(self, mod):
        """
        Retrieve a loaded constraint module by name.
        """
        if mod in self.modules:
            return self.modules[mod]
        try:
            module = importlib.import_module(f"modules.constraints.{mod}")
        except ImportError as exc:
            raise KeyError(f"Constraint module '{mod}' not found.") from exc
        self.modules[mod] = module
        logger.info("Loaded constraint module (lazy): %s", mod)
        return module

    def get_constraint(self, mod):
        """Backward-compatible alias for ``get_module``."""
        return self.get_module(mod)

    def _load_modules(self, names):
        loaded = {}
        for name in names:
            try:
                module = importlib.import_module(f"modules.constraints.{name}")
                if hasattr(module, "enforce_constraint"):
                    loaded[name] = module
                else:
                    logger.warning(
                        f"Constraint module '{name}' lacks 'enforce_constraint' function."
                    )
            except ImportError as e:
                logger.warning(f"Could not load constraint module '{name}': {e}")
        return loaded

    @staticmethod
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

    def apply_gradient_modifications(self, grad, mesh, global_params):
        """Apply KKT-style gradient projection from constraint modules.

        Constraint modules may supply ``constraint_gradients`` (or the legacy
        singular form) to participate in the KKT projection path. Constraints
        that do not supply gradients will only be enforced geometrically via
        ``enforce_constraint``.
        """
        kkt_candidates: list[tuple[str, list[dict[int, np.ndarray]]]] = []
        for name, module in self.modules.items():
            g_list = None
            if hasattr(module, "constraint_gradients"):
                try:
                    g_list = module.constraint_gradients(mesh, global_params)
                except TypeError:
                    g_list = module.constraint_gradients(mesh)
                if g_list:
                    kkt_candidates.append((name, g_list))
                    continue
            if hasattr(module, "constraint_gradient"):
                try:
                    gC = module.constraint_gradient(mesh, global_params)
                except TypeError:
                    gC = module.constraint_gradient(mesh)
                if gC:
                    kkt_candidates.append((name, [gC]))
            if (
                name not in self._warned_no_grad
                and not hasattr(module, "constraint_gradients")
                and not hasattr(module, "constraint_gradient")
            ):
                logger.warning(
                    "Constraint module '%s' lacks constraint gradients; "
                    "it will not participate in KKT projection and will only be "
                    "enforced geometrically.",
                    name,
                )
                self._warned_no_grad.add(name)

        if not kkt_candidates:
            return

        all_constraints: list[dict[int, np.ndarray]] = []
        for _, grads in kkt_candidates:
            all_constraints.extend(grads)

        k = len(all_constraints)
        if k == 1:
            gC = all_constraints[0]
            norm_sq = 0.0
            dot = 0.0
            for vidx, gvec in gC.items():
                if vidx in grad:
                    dot += float(np.dot(grad[vidx], gvec))
                norm_sq += float(np.dot(gvec, gvec))
            if norm_sq > 1e-18:
                lam = dot / norm_sq
                for vidx, gvec in gC.items():
                    if vidx in grad:
                        grad[vidx] -= lam * gvec
            return

        A = np.zeros((k, k), dtype=float)
        b = np.zeros(k, dtype=float)
        for i, gCi in enumerate(all_constraints):
            for vidx, gvec in gCi.items():
                if vidx in grad:
                    b[i] += float(np.dot(grad[vidx], gvec))
                for j in range(i, k):
                    gCj = all_constraints[j].get(vidx)
                    if gCj is None:
                        continue
                    val = float(np.dot(gvec, gCj))
                    A[i, j] += val
                    if j != i:
                        A[j, i] += val

        A[np.diag_indices_from(A)] += 1e-18
        lam = self._solve_kkt_system(A, b)
        if lam is None:
            return

        for gCi, lam_i in zip(all_constraints, lam):
            if lam_i == 0.0:
                continue
            for vidx, gvec in gCi.items():
                if vidx in grad:
                    grad[vidx] -= lam_i * gvec

    def apply_gradient_modifications_array(self, grad_arr, mesh, global_params):
        """Array-based variant of ``apply_gradient_modifications``.

        Preferred interface for constraint modules is ``constraint_gradients_array``,
        returning a list of dense arrays with the same shape as ``grad_arr``.
        For high-cardinality constraints, modules may provide
        ``constraint_gradients_rows_array`` returning sparse row payloads
        ``(rows, vectors)`` with shapes ``(m,)`` and ``(m, 3)``.
        The legacy dict-based ``constraint_gradients`` is still supported.
        """
        mesh.build_position_cache()
        positions = mesh.positions_view()
        index_map = mesh.vertex_index_to_row

        sparse_row_constraints: list[tuple[np.ndarray, np.ndarray]] = []
        all_constraints: list[np.ndarray] = []
        for name, module in self.modules.items():
            g_list_rows = None
            if hasattr(module, "constraint_gradients_rows_array"):
                try:
                    g_list_rows = module.constraint_gradients_rows_array(
                        mesh,
                        global_params,
                        positions=positions,
                        index_map=index_map,
                    )
                except TypeError:
                    g_list_rows = module.constraint_gradients_rows_array(
                        mesh, global_params
                    )
                if g_list_rows:
                    for payload in g_list_rows:
                        rows, vecs = _as_row_gradient_payload(payload)
                        if rows.size:
                            sparse_row_constraints.append((rows, vecs))
                    continue

            g_list_arr = None
            if hasattr(module, "constraint_gradients_array"):
                try:
                    g_list_arr = module.constraint_gradients_array(
                        mesh,
                        global_params,
                        positions=positions,
                        index_map=index_map,
                    )
                except TypeError:
                    g_list_arr = module.constraint_gradients_array(mesh, global_params)
                if g_list_arr:
                    all_constraints.extend(g_list_arr)
                    continue

            g_list = None
            if hasattr(module, "constraint_gradients"):
                try:
                    g_list = module.constraint_gradients(mesh, global_params)
                except TypeError:
                    g_list = module.constraint_gradients(mesh)
                if g_list:
                    for gC in g_list:
                        gC_arr = np.zeros_like(grad_arr)
                        for vidx, gvec in gC.items():
                            row = index_map.get(vidx)
                            if row is None:
                                continue
                            gC_arr[row] += gvec
                        all_constraints.append(gC_arr)
                    continue

            if hasattr(module, "constraint_gradient"):
                try:
                    gC = module.constraint_gradient(mesh, global_params)
                except TypeError:
                    gC = module.constraint_gradient(mesh)
                if gC:
                    gC_arr = np.zeros_like(grad_arr)
                    for vidx, gvec in gC.items():
                        row = index_map.get(vidx)
                        if row is None:
                            continue
                        gC_arr[row] += gvec
                    all_constraints.append(gC_arr)
                    continue

            if (
                name not in self._warned_no_grad
                and not hasattr(module, "constraint_gradients_rows_array")
                and not hasattr(module, "constraint_gradients_array")
                and not hasattr(module, "constraint_gradients")
                and not hasattr(module, "constraint_gradient")
            ):
                logger.warning(
                    "Constraint module '%s' lacks constraint gradients; "
                    "it will not participate in KKT projection.",
                    name,
                )
                self._warned_no_grad.add(name)

        if not all_constraints:
            if not sparse_row_constraints:
                return
            if len(sparse_row_constraints) == 1:
                rows, vecs = sparse_row_constraints[0]
                norm_sq = float(np.sum(vecs * vecs))
                if norm_sq <= 1e-18:
                    return
                dot = float(np.sum(grad_arr[rows] * vecs))
                grad_updates = np.zeros_like(grad_arr)
                np.add.at(grad_updates, rows, vecs)
                grad_arr -= (dot / norm_sq) * grad_updates
                return
            self._project_mixed_gradient_against_constraints(
                grad_arr,
                dense_constraints=[],
                sparse_constraints=sparse_row_constraints,
            )
            return

        k = len(all_constraints) + len(sparse_row_constraints)
        if k == 1:
            if all_constraints:
                gC_arr = all_constraints[0]
                norm_sq = float(np.sum(gC_arr * gC_arr))
                if norm_sq > 1e-18:
                    lam = float(np.sum(grad_arr * gC_arr)) / norm_sq
                    grad_arr -= lam * gC_arr
                return
            rows, vecs = sparse_row_constraints[0]
            norm_sq = float(np.sum(vecs * vecs))
            if norm_sq > 1e-18:
                dot = float(np.sum(grad_arr[rows] * vecs))
                grad_updates = np.zeros_like(grad_arr)
                np.add.at(grad_updates, rows, vecs)
                grad_arr -= (dot / norm_sq) * grad_updates
            return

        self._project_mixed_gradient_against_constraints(
            grad_arr,
            dense_constraints=all_constraints,
            sparse_constraints=sparse_row_constraints,
        )

    @staticmethod
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
        lam = ConstraintModuleManager._solve_kkt_system(A, b)
        if lam is None:
            return
        grad_flat -= C.T @ lam

    @staticmethod
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
        lam = ConstraintModuleManager._solve_kkt_system(A, b)
        if lam is None:
            return
        grad_flat -= C.T @ lam

    @staticmethod
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

    @staticmethod
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
                out_cols = (n_single + 3 * out_rows[:, None] + axes[None, :]).reshape(
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
        }

    @staticmethod
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
        return ConstraintModuleManager._solve_kkt_system(A, b)

    @staticmethod
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
            operator_cache = (
                ConstraintModuleManager._build_leaflet_sparse_projection_operator(
                    row_constraints=row_constraints,
                    n_single=n_single,
                )
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

        grad_active = np.empty(active_cols.shape[0], dtype=float)
        if active_in_pos.shape[0]:
            grad_active[active_in_pos] = tilt_in_grad_arr[
                active_in_rows, active_in_comp
            ]
        if active_out_rows.shape[0]:
            grad_active[active_out_pos] = tilt_out_grad_arr[
                active_out_rows, active_out_comp
            ]
        b = C @ grad_active
        lam = ConstraintModuleManager._solve_leaflet_sparse_kkt(operator_cache, b)
        if lam is None:
            return
        delta = C.T @ lam

        if active_in_pos.shape[0]:
            tilt_in_grad_arr[active_in_rows, active_in_comp] -= delta[active_in_pos]
        if active_out_rows.shape[0]:
            tilt_out_grad_arr[active_out_rows, active_out_comp] -= delta[active_out_pos]

    def apply_tilt_gradient_modifications_array(
        self,
        tilt_in_grad_arr: np.ndarray,
        tilt_out_grad_arr: np.ndarray,
        mesh,
        global_params,
        *,
        positions: np.ndarray | None = None,
        tilts_in: np.ndarray | None = None,
        tilts_out: np.ndarray | None = None,
    ) -> None:
        """Project leaflet-tilt gradients onto the constraint manifold.

        Constraint modules may supply ``constraint_gradients_tilt_array``, which
        returns a list of (gC_in, gC_out) arrays. Each gC_* must match the shape
        of the corresponding tilt gradient array.
        """
        if tilt_in_grad_arr.shape != tilt_out_grad_arr.shape:
            raise ValueError("tilt gradient arrays must have matching shapes")

        if positions is None:
            mesh.build_position_cache()
            positions = mesh.positions_view()
        if tilts_in is None:
            tilts_in = mesh.tilts_in_view()
        if tilts_out is None:
            tilts_out = mesh.tilts_out_view()

        index_map = mesh.vertex_index_to_row
        all_constraints: list[tuple[np.ndarray | None, np.ndarray | None]] = []
        row_constraints: list[
            tuple[
                tuple[np.ndarray, np.ndarray] | None,
                tuple[np.ndarray, np.ndarray] | None,
            ]
        ] = []

        for module in self.modules.values():
            if hasattr(module, "constraint_gradients_tilt_rows_array"):
                try:
                    g_list_rows = module.constraint_gradients_tilt_rows_array(
                        mesh,
                        global_params,
                        positions=positions,
                        index_map=index_map,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                    )
                except TypeError:
                    g_list_rows = module.constraint_gradients_tilt_rows_array(
                        mesh, global_params
                    )
                if g_list_rows:
                    for payload in g_list_rows:
                        if not isinstance(payload, tuple) or len(payload) != 2:
                            raise ValueError(
                                "tilt row payload must be a ((rows, vecs)|None, (rows, vecs)|None) tuple"
                            )
                        in_part = payload[0]
                        out_part = payload[1]
                        in_norm = None
                        out_norm = None
                        if in_part is not None:
                            in_rows, in_vecs = _as_row_gradient_payload(in_part)
                            in_rows, in_vecs = _coalesce_sparse_row_payload(
                                in_rows, in_vecs
                            )
                            if in_rows.size:
                                in_norm = (in_rows, in_vecs)
                        if out_part is not None:
                            out_rows, out_vecs = _as_row_gradient_payload(out_part)
                            out_rows, out_vecs = _coalesce_sparse_row_payload(
                                out_rows, out_vecs
                            )
                            if out_rows.size:
                                out_norm = (out_rows, out_vecs)
                        if in_norm is not None or out_norm is not None:
                            row_constraints.append((in_norm, out_norm))
                    continue

            if not hasattr(module, "constraint_gradients_tilt_array"):
                continue
            try:
                g_list = module.constraint_gradients_tilt_array(
                    mesh,
                    global_params,
                    positions=positions,
                    index_map=index_map,
                    tilts_in=tilts_in,
                    tilts_out=tilts_out,
                )
            except TypeError:
                g_list = module.constraint_gradients_tilt_array(mesh, global_params)
            if g_list:
                all_constraints.extend(g_list)

        if not all_constraints and not row_constraints:
            return

        if not all_constraints:
            operator_cache = None
            if mesh._geometry_cache_active(positions):
                cache_key = (
                    int(mesh._version),
                    int(mesh._vertex_ids_version),
                    int(getattr(mesh, "_facet_loops_version", -1)),
                    int(getattr(mesh, "_tilt_fixed_flags_version", -1)),
                    id(positions),
                    tilt_in_grad_arr.size,
                    self._row_constraints_payload_token(row_constraints),
                )
                cached = self._leaflet_sparse_projection_cache
                if cached is not None and cached.get("key") == cache_key:
                    operator_cache = cached.get("operator")
                else:
                    operator_cache = self._build_leaflet_sparse_projection_operator(
                        row_constraints=row_constraints,
                        n_single=int(tilt_in_grad_arr.size),
                    )
                    self._leaflet_sparse_projection_cache = {
                        "key": cache_key,
                        "operator": operator_cache,
                    }
            self._project_leaflet_sparse_rows_against_constraints(
                tilt_in_grad_arr,
                tilt_out_grad_arr,
                row_constraints,
                operator_cache=operator_cache,
            )
            return

        constraints_flat: list[np.ndarray] = []
        zeros_in = np.zeros_like(tilt_in_grad_arr)
        zeros_out = np.zeros_like(tilt_out_grad_arr)
        for gC_in, gC_out in all_constraints:
            g_in = zeros_in if gC_in is None else np.asarray(gC_in, dtype=float)
            g_out = zeros_out if gC_out is None else np.asarray(gC_out, dtype=float)
            constraints_flat.append(
                np.concatenate([g_in.reshape(-1), g_out.reshape(-1)])
            )

        stacked_grad = np.concatenate(
            [tilt_in_grad_arr.reshape(-1), tilt_out_grad_arr.reshape(-1)]
        )
        n_single = tilt_in_grad_arr.size
        n_total = 2 * n_single
        k_total = len(constraints_flat) + len(row_constraints)
        C = np.zeros((k_total, n_total), dtype=float)
        row_idx = 0
        for c_row in constraints_flat:
            C[row_idx, :] = c_row
            row_idx += 1
        for in_part, out_part in row_constraints:
            if in_part is not None:
                in_rows, in_vecs = in_part
                _accumulate_sparse_row_into_dense_flat(
                    C[row_idx, :n_single], in_rows, in_vecs
                )
            if out_part is not None:
                out_rows, out_vecs = out_part
                _accumulate_sparse_row_into_dense_flat(
                    C[row_idx, n_single:], out_rows, out_vecs
                )
            row_idx += 1

        b = C @ stacked_grad
        A = C @ C.T
        A[np.diag_indices_from(A)] += 1e-18
        lam = ConstraintModuleManager._solve_kkt_system(A, b)
        if lam is None:
            return
        stacked_grad -= C.T @ lam
        n_in = tilt_in_grad_arr.size
        tilt_in_grad_arr[:] = stacked_grad[:n_in].reshape(tilt_in_grad_arr.shape)
        tilt_out_grad_arr[:] = stacked_grad[n_in:].reshape(tilt_out_grad_arr.shape)

    def enforce_tilt_constraints(self, mesh, **kwargs) -> None:
        """Invoke tilt-only constraint projections on all loaded modules."""
        for name, module in self.modules.items():
            if not hasattr(module, "enforce_tilt_constraint"):
                continue
            logger.debug("Enforcing tilt constraint: %s", name)
            try:
                module.enforce_tilt_constraint(mesh, **kwargs)
            except TypeError as exc:
                logger.debug(
                    "Tilt constraint module '%s' rejected kwargs (%s); retrying.",
                    name,
                    exc,
                )
                module.enforce_tilt_constraint(mesh)

    def enforce_all(self, mesh, **kwargs):
        """Invoke ``enforce_constraint`` on all loaded constraint modules.

        Modules are called with ``mesh`` and any keyword arguments supplied.
        If a module does not accept the expanded signature, we gracefully
        fall back to calling it with just ``mesh`` to preserve backward
        compatibility.
        """
        context = kwargs.get("context", "minimize")
        global_params = kwargs.get("global_params")
        project_in_minimize = True
        if global_params is not None:
            project_in_minimize = global_params.get(
                "volume_projection_during_minimization", True
            )

        for name, module in self.modules.items():
            if not hasattr(module, "enforce_constraint"):
                logger.debug(
                    "Constraint module '%s' has no enforce_constraint; skipping.",
                    name,
                )
                continue

            logger.debug("Enforcing constraint: %s", name)
            try:
                # For the volume constraint we distinguish between two use cases:
                #   - During minimization steps (``context == 'minimize'``) we
                #     may skip geometric volume projection if global parameters
                #     request that the optimizer handle volume purely via
                #     Lagrange‑style gradient projection.
                #   - After discrete mesh operations such as refinement,
                #     equiangulation or vertex averaging (other contexts), we
                #     always apply a hard projection back to the target volume.
                if (
                    name == "volume"
                    and context == "minimize"
                    and not project_in_minimize
                ):
                    logger.debug(
                        "Skipping geometric volume projection during minimization; "
                        "hard volume is handled via gradient projection."
                    )
                    continue

                # Filter force_projection from kwargs if it's there to avoid duplication
                call_kwargs = kwargs.copy()
                if "force_projection" in call_kwargs:
                    del call_kwargs["force_projection"]

                if name == "volume":
                    module.enforce_constraint(
                        mesh, force_projection=True, **call_kwargs
                    )
                else:
                    module.enforce_constraint(mesh, **kwargs)
            except TypeError as e:
                logger.debug(
                    "Constraint module '%s' rejected kwargs (%s); retrying without kwargs.",
                    name,
                    e,
                )
                module.enforce_constraint(mesh)

    def __contains__(self, name):
        return name in self.modules

    def __getitem__(self, name):
        return self.modules[name]

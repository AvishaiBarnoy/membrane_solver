# runtime/constraint_manager.py

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
    flat_cols = np.empty(rows.size * 3, dtype=int)
    base = 3 * rows
    flat_cols[0::3] = base
    flat_cols[1::3] = base + 1
    flat_cols[2::3] = base + 2
    np.add.at(dense_row, flat_cols, vecs.reshape(-1))


class ConstraintModuleManager:
    def __init__(self, module_names):
        self.modules = {}
        self._warned_no_grad = set()
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
        try:
            lam = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
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
        try:
            lam = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
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
        try:
            lam = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return
        grad_flat -= C.T @ lam

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
                            if in_rows.size:
                                in_norm = (in_rows, in_vecs)
                        if out_part is not None:
                            out_rows, out_vecs = _as_row_gradient_payload(out_part)
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
        try:
            lam = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
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

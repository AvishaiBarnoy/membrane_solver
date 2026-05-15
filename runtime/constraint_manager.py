# runtime/constraint_manager.py

import importlib
import logging

import numpy as np

from .constraint_projection import (
    _accumulate_sparse_row_into_dense_flat,
    _as_row_gradient_payload,
    _build_joint_sparse_projection_operator,
    _build_leaflet_sparse_projection_operator,
    _coalesce_sparse_row_payload,
    _joint_row_constraints_payload_token,
    _project_joint_sparse_rows_against_constraints,
    _project_leaflet_sparse_rows_against_constraints,
    _project_mixed_gradient_against_constraints,
    _row_constraints_payload_token,
    _solve_kkt_system,
)

logger = logging.getLogger("ConstraintManager")


class ConstraintModuleManager:
    def __init__(self, module_names):
        self.modules = {}
        self._warned_no_grad = set()
        self._leaflet_sparse_projection_cache: dict[str, object] | None = None
        self._joint_sparse_projection_cache: dict[str, object] | None = None
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
        lam = _solve_kkt_system(A, b)
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
                        rows, vecs = _coalesce_sparse_row_payload(rows, vecs)
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
            _project_mixed_gradient_against_constraints(
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

        _project_mixed_gradient_against_constraints(
            grad_arr,
            dense_constraints=all_constraints,
            sparse_constraints=sparse_row_constraints,
        )

    def apply_joint_gradient_modifications_array(
        self,
        grad_arr: np.ndarray,
        tilt_in_grad_arr: np.ndarray,
        tilt_out_grad_arr: np.ndarray,
        mesh,
        global_params,
        *,
        positions: np.ndarray | None = None,
        tilts_in: np.ndarray | None = None,
        tilts_out: np.ndarray | None = None,
    ) -> None:
        """Project shape and leaflet gradients against a joint constraint manifold."""
        if (
            grad_arr.shape != tilt_in_grad_arr.shape
            or grad_arr.shape != tilt_out_grad_arr.shape
        ):
            raise ValueError("joint gradient arrays must have matching shapes")

        if positions is None:
            mesh.build_position_cache()
            positions = mesh.positions_view()
        if tilts_in is None:
            tilts_in = mesh.tilts_in_view()
        if tilts_out is None:
            tilts_out = mesh.tilts_out_view()

        index_map = mesh.vertex_index_to_row
        dense_constraints: list[
            tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]
        ] = []
        row_constraints: list[
            tuple[
                tuple[np.ndarray, np.ndarray] | None,
                tuple[np.ndarray, np.ndarray] | None,
                tuple[np.ndarray, np.ndarray] | None,
            ]
        ] = []

        for name, module in self.modules.items():
            if hasattr(module, "constraint_gradients_joint_array"):
                try:
                    g_list_joint = module.constraint_gradients_joint_array(
                        mesh,
                        global_params,
                        positions=positions,
                        index_map=index_map,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                    )
                except TypeError:
                    g_list_joint = module.constraint_gradients_joint_array(
                        mesh, global_params
                    )
                if g_list_joint:
                    for payload in g_list_joint:
                        if not isinstance(payload, tuple) or len(payload) != 3:
                            raise ValueError(
                                "joint row payload must be a (shape, tilt_in, tilt_out) tuple"
                            )
                        shape_norm = None
                        in_norm = None
                        out_norm = None
                        if payload[0] is not None:
                            shape_rows, shape_vecs = _as_row_gradient_payload(
                                payload[0]
                            )
                            shape_rows, shape_vecs = _coalesce_sparse_row_payload(
                                shape_rows, shape_vecs
                            )
                            if shape_rows.size:
                                shape_norm = (shape_rows, shape_vecs)
                        if payload[1] is not None:
                            in_rows, in_vecs = _as_row_gradient_payload(payload[1])
                            in_rows, in_vecs = _coalesce_sparse_row_payload(
                                in_rows, in_vecs
                            )
                            if in_rows.size:
                                in_norm = (in_rows, in_vecs)
                        if payload[2] is not None:
                            out_rows, out_vecs = _as_row_gradient_payload(payload[2])
                            out_rows, out_vecs = _coalesce_sparse_row_payload(
                                out_rows, out_vecs
                            )
                            if out_rows.size:
                                out_norm = (out_rows, out_vecs)
                        if (
                            shape_norm is not None
                            or in_norm is not None
                            or out_norm is not None
                        ):
                            row_constraints.append((shape_norm, in_norm, out_norm))
                    continue

            shape_added = False
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
                    shape_added = True
                    for payload in g_list_rows:
                        shape_rows, shape_vecs = _as_row_gradient_payload(payload)
                        shape_rows, shape_vecs = _coalesce_sparse_row_payload(
                            shape_rows, shape_vecs
                        )
                        if shape_rows.size:
                            row_constraints.append(
                                ((shape_rows, shape_vecs), None, None)
                            )
            elif hasattr(module, "constraint_gradients_array"):
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
                    shape_added = True
                    for g_shape in g_list_arr:
                        dense_constraints.append(
                            (np.asarray(g_shape, dtype=float), None, None)
                        )
            elif hasattr(module, "constraint_gradients"):
                try:
                    g_list = module.constraint_gradients(mesh, global_params)
                except TypeError:
                    g_list = module.constraint_gradients(mesh)
                if g_list:
                    shape_added = True
                    for gC in g_list:
                        g_shape = np.zeros_like(grad_arr)
                        for vidx, gvec in gC.items():
                            row = index_map.get(vidx)
                            if row is None:
                                continue
                            g_shape[row] += gvec
                        dense_constraints.append((g_shape, None, None))
            elif hasattr(module, "constraint_gradient"):
                try:
                    gC = module.constraint_gradient(mesh, global_params)
                except TypeError:
                    gC = module.constraint_gradient(mesh)
                if gC:
                    shape_added = True
                    g_shape = np.zeros_like(grad_arr)
                    for vidx, gvec in gC.items():
                        row = index_map.get(vidx)
                        if row is None:
                            continue
                        g_shape[row] += gvec
                    dense_constraints.append((g_shape, None, None))

            tilt_added = False
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
                    tilt_added = True
                    for payload in g_list_rows:
                        if not isinstance(payload, tuple) or len(payload) != 2:
                            raise ValueError(
                                "tilt row payload must be a ((rows, vecs)|None, (rows, vecs)|None) tuple"
                            )
                        in_norm = None
                        out_norm = None
                        if payload[0] is not None:
                            in_rows, in_vecs = _as_row_gradient_payload(payload[0])
                            in_rows, in_vecs = _coalesce_sparse_row_payload(
                                in_rows, in_vecs
                            )
                            if in_rows.size:
                                in_norm = (in_rows, in_vecs)
                        if payload[1] is not None:
                            out_rows, out_vecs = _as_row_gradient_payload(payload[1])
                            out_rows, out_vecs = _coalesce_sparse_row_payload(
                                out_rows, out_vecs
                            )
                            if out_rows.size:
                                out_norm = (out_rows, out_vecs)
                        if in_norm is not None or out_norm is not None:
                            row_constraints.append((None, in_norm, out_norm))
            elif hasattr(module, "constraint_gradients_tilt_array"):
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
                    tilt_added = True
                    for g_in, g_out in g_list:
                        dense_constraints.append(
                            (
                                None,
                                None if g_in is None else np.asarray(g_in, dtype=float),
                                None
                                if g_out is None
                                else np.asarray(g_out, dtype=float),
                            )
                        )

            if (
                not shape_added
                and not tilt_added
                and name not in self._warned_no_grad
                and not hasattr(module, "constraint_gradients_joint_array")
            ):
                logger.warning(
                    "Constraint module '%s' lacks joint-compatible gradients; "
                    "it will not participate in joint KKT projection.",
                    name,
                )
                self._warned_no_grad.add(name)

        if not dense_constraints and not row_constraints:
            return

        if not dense_constraints:
            operator_cache = None
            if mesh._geometry_cache_active(positions):
                cache_key = (
                    int(mesh._version),
                    int(mesh._vertex_ids_version),
                    int(getattr(mesh, "_facet_loops_version", -1)),
                    int(getattr(mesh, "_tilt_fixed_flags_version", -1)),
                    id(positions),
                    grad_arr.size,
                    _joint_row_constraints_payload_token(row_constraints),
                )
                cached = self._joint_sparse_projection_cache
                if cached is not None and cached.get("key") == cache_key:
                    operator_cache = cached.get("operator")
                else:
                    operator_cache = _build_joint_sparse_projection_operator(
                        row_constraints=row_constraints,
                        n_single=int(grad_arr.size),
                    )
                    self._joint_sparse_projection_cache = {
                        "key": cache_key,
                        "operator": operator_cache,
                    }
            _project_joint_sparse_rows_against_constraints(
                grad_arr,
                tilt_in_grad_arr,
                tilt_out_grad_arr,
                row_constraints,
                operator_cache=operator_cache,
            )
            return

        n_single = int(grad_arr.size)
        n_total = 3 * n_single
        stacked_grad = np.concatenate(
            [
                grad_arr.reshape(-1),
                tilt_in_grad_arr.reshape(-1),
                tilt_out_grad_arr.reshape(-1),
            ]
        )
        k_total = len(dense_constraints) + len(row_constraints)
        C = np.zeros((k_total, n_total), dtype=float)
        row_idx = 0
        for g_shape, g_in, g_out in dense_constraints:
            if g_shape is not None:
                C[row_idx, :n_single] = np.asarray(g_shape, dtype=float).reshape(-1)
            if g_in is not None:
                C[row_idx, n_single : 2 * n_single] = np.asarray(
                    g_in, dtype=float
                ).reshape(-1)
            if g_out is not None:
                C[row_idx, 2 * n_single :] = np.asarray(g_out, dtype=float).reshape(-1)
            row_idx += 1
        for shape_part, in_part, out_part in row_constraints:
            if shape_part is not None:
                shape_rows, shape_vecs = shape_part
                _accumulate_sparse_row_into_dense_flat(
                    C[row_idx, :n_single], shape_rows, shape_vecs
                )
            if in_part is not None:
                in_rows, in_vecs = in_part
                _accumulate_sparse_row_into_dense_flat(
                    C[row_idx, n_single : 2 * n_single], in_rows, in_vecs
                )
            if out_part is not None:
                out_rows, out_vecs = out_part
                _accumulate_sparse_row_into_dense_flat(
                    C[row_idx, 2 * n_single :], out_rows, out_vecs
                )
            row_idx += 1

        b = C @ stacked_grad
        A = C @ C.T
        A[np.diag_indices_from(A)] += 1e-18
        lam = ConstraintModuleManager._solve_kkt_system(A, b)
        if lam is None:
            return
        stacked_grad -= C.T @ lam

        grad_arr[:] = stacked_grad[:n_single].reshape(grad_arr.shape)
        tilt_in_grad_arr[:] = stacked_grad[n_single : 2 * n_single].reshape(
            tilt_in_grad_arr.shape
        )
        tilt_out_grad_arr[:] = stacked_grad[2 * n_single :].reshape(
            tilt_out_grad_arr.shape
        )

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
                    _row_constraints_payload_token(row_constraints),
                )
                cached = self._leaflet_sparse_projection_cache
                if cached is not None and cached.get("key") == cache_key:
                    operator_cache = cached.get("operator")
                else:
                    operator_cache = _build_leaflet_sparse_projection_operator(
                        row_constraints=row_constraints,
                        n_single=int(tilt_in_grad_arr.size),
                    )
                    self._leaflet_sparse_projection_cache = {
                        "key": cache_key,
                        "operator": operator_cache,
                    }
            _project_leaflet_sparse_rows_against_constraints(
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

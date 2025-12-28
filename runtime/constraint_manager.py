# runtime/constraint_manager.py

import importlib
import logging

import numpy as np

logger = logging.getLogger("ConstraintManager")


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
        """Array-based variant of ``apply_gradient_modifications``."""
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
                    "it will not participate in KKT projection.",
                    name,
                )
                self._warned_no_grad.add(name)

        if not kkt_candidates:
            return

        mesh.build_position_cache()
        index_map = mesh.vertex_index_to_row

        all_constraints: list[np.ndarray] = []
        for _, grads in kkt_candidates:
            for gC in grads:
                gC_arr = np.zeros_like(grad_arr)
                for vidx, gvec in gC.items():
                    row = index_map.get(vidx)
                    if row is None:
                        continue
                    gC_arr[row] += gvec
                all_constraints.append(gC_arr)

        k = len(all_constraints)
        if k == 1:
            gC_arr = all_constraints[0]
            norm_sq = float(np.sum(gC_arr * gC_arr))
            if norm_sq > 1e-18:
                lam = float(np.sum(grad_arr * gC_arr)) / norm_sq
                grad_arr -= lam * gC_arr
            return

        A = np.zeros((k, k), dtype=float)
        b = np.zeros(k, dtype=float)
        for i, gCi in enumerate(all_constraints):
            b[i] = float(np.sum(grad_arr * gCi))
            for j in range(i, k):
                gCj = all_constraints[j]
                A[i, j] = float(np.sum(gCi * gCj))
                if j != i:
                    A[j, i] = A[i, j]

        A[np.diag_indices_from(A)] += 1e-18
        try:
            lam = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return

        for gCi, lam_i in zip(all_constraints, lam):
            if lam_i == 0.0:
                continue
            grad_arr -= lam_i * gCi

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

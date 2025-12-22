# runtime/constraint_manager.py

import importlib
import logging

import numpy as np

logger = logging.getLogger("ConstraintManager")


class ConstraintModuleManager:
    def __init__(self, module_names):
        self.modules = {}
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
        if mod in self.modules.keys():
            return self.modules[mod]
        raise KeyError(f"Constraint module '{mod}' not found.")

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
        """Invoke ``apply_constraint_gradient`` on all loaded constraint modules.

        This allows constraints to modify the energy gradient directly, for example
        by applying Lagrange multipliers (soft constraints) or projection forces.
        """
        kkt_candidates = []
        for name, module in self.modules.items():
            if hasattr(module, "constraint_gradient"):
                try:
                    gC = module.constraint_gradient(mesh, global_params)
                except TypeError:
                    gC = module.constraint_gradient(mesh)
                if gC:
                    kkt_candidates.append((name, gC))

        if len(kkt_candidates) == 1:
            _, gC = kkt_candidates[0]
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

        for name, module in self.modules.items():
            if hasattr(module, "apply_constraint_gradient"):
                # Skip module-specific projection if we already did KKT for it.
                if len(kkt_candidates) == 1 and name == kkt_candidates[0][0]:
                    continue
                # Pass the global_params to the module so it can check configuration
                module.apply_constraint_gradient(grad, mesh, global_params)

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
                print(f"DEBUG: Caught TypeError: {e}")
                # Older modules may not accept extra keyword arguments.
                module.enforce_constraint(mesh)

    def __contains__(self, name):
        return name in self.modules

    def __getitem__(self, name):
        return self.modules[name]

# runtime/minimizer.py

import logging
from typing import Callable, Dict, List, Optional

import numpy as np

from geometry.entities import Mesh
from parameters.global_parameters import GlobalParameters
from parameters.resolver import ParameterResolver
from runtime.constraint_manager import ConstraintModuleManager
from runtime.diagnostics.gauss_bonnet import GaussBonnetMonitor
from runtime.energy_manager import EnergyModuleManager
from runtime.steppers.base import BaseStepper

logger = logging.getLogger("membrane_solver")


class Minimizer:
    """Coordinate the optimization loop for a mesh."""

    def __init__(
        self,
        mesh: Mesh,
        global_params: GlobalParameters,
        stepper: BaseStepper,
        energy_manager: EnergyModuleManager,
        constraint_manager: ConstraintModuleManager,
        energy_modules: Optional[List[str]] = None,
        constraint_modules: Optional[List[str]] = None,
        step_size: float = 1e-3,
        tol: float = 1e-6,
        quiet: bool = False,
    ) -> None:
        self.mesh = mesh
        self.global_params = global_params
        self.energy_manager = energy_manager
        self.constraint_manager = constraint_manager
        self.stepper = stepper
        self.step_size = step_size
        self.tol = tol
        self.quiet = quiet
        self.max_zero_steps = int(global_params.get("max_zero_steps", 10))
        self.step_size_floor = float(global_params.get("step_size_floor", 1e-8))

        # Use provided module lists or fall back to those defined on the mesh
        module_list = (
            energy_modules if energy_modules is not None else mesh.energy_modules
        )
        self.energy_module_names = list(module_list)
        self.energy_modules = [
            self.energy_manager.get_module(mod) for mod in module_list
        ]

        constraint_list = (
            constraint_modules
            if constraint_modules is not None
            else mesh.constraint_modules
        )
        self.constraint_modules = [
            self.constraint_manager.get_constraint(constraint)
            for constraint in constraint_list
        ]
        self._has_enforceable_constraints = any(
            hasattr(mod, "enforce_constraint") for mod in self.constraint_modules
        )

        logger.debug(f"Loaded energy modules: {self.energy_manager.modules.keys()}")
        logger.debug(f"Mesh energy_modules: {self.mesh.energy_modules}")

        self.param_resolver = ParameterResolver(global_params)
        self._gauss_bonnet_monitor: GaussBonnetMonitor | None = None

    def _check_gauss_bonnet(self) -> None:
        """Emit Gauss-Bonnet diagnostics if enabled."""
        if not bool(self.global_params.get("gauss_bonnet_monitor", False)):
            return

        if self._gauss_bonnet_monitor is None:
            eps_angle = float(self.global_params.get("gauss_bonnet_eps_angle", 1e-4))
            c1 = float(self.global_params.get("gauss_bonnet_c1", 1.0))
            c2 = float(self.global_params.get("gauss_bonnet_c2", 1.0))
            self._gauss_bonnet_monitor = GaussBonnetMonitor.from_mesh(
                self.mesh, eps_angle=eps_angle, c1=c1, c2=c2
            )

        report = self._gauss_bonnet_monitor.evaluate(self.mesh)
        if not report["ok"]:
            logger.warning(
                "Gauss-Bonnet drift exceeded tolerance: |ΔG|=%.3e (tol %.3e).",
                report["drift_G"],
                report["tol_G"],
            )

    def refresh_modules(self):
        """Re-load energy and constraint modules from the current mesh state."""
        # Refresh energy modules
        self.energy_modules = [
            self.energy_manager.get_module(mod) for mod in self.mesh.energy_modules
        ]
        # Refresh constraint modules
        self.constraint_modules = [
            self.constraint_manager.get_constraint(constraint)
            for constraint in self.mesh.constraint_modules
        ]
        self._has_enforceable_constraints = any(
            hasattr(mod, "enforce_constraint") for mod in self.constraint_modules
        )
        logger.info(
            f"Minimizer modules refreshed: {len(self.energy_modules)} energy, {len(self.constraint_modules)} constraint."
        )

    def __repr__(self):
        msg = f"""### MINIMIZER ###
MESH:\t {self.mesh}
GLOBAL PARAMETERS:\t {self.global_params}
STEPPER:\t {self.stepper}
STEP SIZE:\t {self.step_size}
############"""
        return msg

    def compute_energy_and_gradient_array(self):
        """Return total energy and dense gradient array for the current mesh."""
        positions = self.mesh.positions_view()
        vertex_ids = self.mesh.vertex_ids
        index_map = self.mesh.vertex_index_to_row
        grad_arr = np.zeros_like(positions)

        total_energy = 0.0
        for module in self.energy_modules:
            if hasattr(module, "compute_energy_and_gradient_array"):
                # Use fast array path
                E_mod = module.compute_energy_and_gradient_array(
                    self.mesh,
                    self.global_params,
                    self.param_resolver,
                    positions=positions,
                    index_map=index_map,
                    grad_arr=grad_arr,
                )
                total_energy += E_mod
                continue

            # Legacy path: compute dict and scatter
            E_mod, g_mod = module.compute_energy_and_gradient(
                self.mesh, self.global_params, self.param_resolver
            )
            total_energy += E_mod
            for vidx, gvec in g_mod.items():
                row = index_map.get(vidx)
                if row is not None:
                    grad_arr[row] += gvec

        # Apply constraint modifications to the gradient (e.g., Lagrange multipliers)
        if hasattr(self.constraint_manager, "apply_gradient_modifications_array"):
            self.constraint_manager.apply_gradient_modifications_array(
                grad_arr, self.mesh, self.global_params
            )

        # Always zero gradients for fixed vertices in the array pipeline.
        for row, vidx in enumerate(vertex_ids):
            if getattr(self.mesh.vertices[int(vidx)], "fixed", False):
                grad_arr[row] = 0.0

        return total_energy, grad_arr

    def compute_energy_and_gradient(self):
        """Return total energy and gradient for the current mesh."""

        # Use array backend for performance
        total_energy, grad_arr = self.compute_energy_and_gradient_array()

        # Convert back to dictionary for legacy components if needed
        # (Though we prefer staying in array space as long as possible)
        grad = self._grad_arr_to_dict(grad_arr)

        # Apply constraint modifications to the gradient (dictionary-based path)
        # Note: compute_energy_and_gradient_array already handled array modifications
        # but if we are here, we might need the dict modifications if array one was missing.
        if not hasattr(self.constraint_manager, "apply_gradient_modifications_array"):
            self.constraint_manager.apply_gradient_modifications(
                grad, self.mesh, self.global_params
            )
            self._zero_fixed_gradients(grad)

        # Optional DEBUG‑level diagnostic: in Lagrange mode the projected
        # gradient should be (numerically) tangent to each fixed‑volume
        # manifold, i.e. ⟨∇E, ∇V_body⟩ ≈ 0.
        if logger.isEnabledFor(logging.DEBUG):
            mode = self.global_params.get("volume_constraint_mode", "lagrange")
            if mode == "lagrange" and self.mesh.bodies:
                max_abs_dot = 0.0
                for body in self.mesh.bodies.values():
                    V_target = body.target_volume
                    if V_target is None:
                        V_target = body.options.get("target_volume")

                    if V_target is None:
                        continue

                    _, vol_grad = body.compute_volume_and_gradient(self.mesh)

                    dot = 0.0
                    for vidx, gVi in vol_grad.items():
                        gE = grad.get(vidx)
                        if gE is not None:
                            dot += float(np.dot(gE, gVi))
                    max_abs_dot = max(max_abs_dot, abs(dot))
                logger.debug(
                    "Lagrange tangency check: max |<∇E, ∇V>| = %.3e",
                    max_abs_dot,
                )

        return total_energy, grad

    def compute_energy(self):
        """Compute the total energy using the loaded modules."""
        positions = self.mesh.positions_view()
        index_map = self.mesh.vertex_index_to_row
        grad_dummy = np.zeros_like(positions)

        total_energy = 0.0
        for module in self.energy_modules:
            if hasattr(module, "compute_energy_and_gradient_array"):
                E_mod = module.compute_energy_and_gradient_array(
                    self.mesh,
                    self.global_params,
                    self.param_resolver,
                    positions=positions,
                    index_map=index_map,
                    grad_arr=grad_dummy,
                )
                total_energy += float(E_mod)
                continue

            E_mod, _ = module.compute_energy_and_gradient(
                self.mesh,
                self.global_params,
                self.param_resolver,
                compute_gradient=False,
            )
            total_energy += float(E_mod)
        return float(total_energy)

    def compute_energy_breakdown(self) -> Dict[str, float]:
        """Return per-module energy contributions for the current mesh."""
        positions = self.mesh.positions_view()
        index_map = self.mesh.vertex_index_to_row
        breakdown: Dict[str, float] = {}

        for name, module in zip(self.energy_module_names, self.energy_modules):
            if hasattr(module, "compute_energy_and_gradient_array"):
                grad_dummy = np.zeros_like(positions)
                E_mod = module.compute_energy_and_gradient_array(
                    self.mesh,
                    self.global_params,
                    self.param_resolver,
                    positions=positions,
                    index_map=index_map,
                    grad_arr=grad_dummy,
                )
            else:
                E_mod, _ = module.compute_energy_and_gradient(
                    self.mesh,
                    self.global_params,
                    self.param_resolver,
                    compute_gradient=False,
                )
            breakdown[name] = float(E_mod)
        return breakdown

    def _grad_arr_to_dict(self, grad_arr: np.ndarray) -> Dict[int, np.ndarray]:
        """Convert a dense gradient array into a sparse dict keyed by vertex id."""
        return {
            vid: grad_arr[row].copy()
            for row, vid in enumerate(self.mesh.vertex_ids)
            if np.any(grad_arr[row])
        }

    def _zero_fixed_gradients(self, grad: Dict[int, np.ndarray]) -> None:
        """Helper to zero out gradients for fixed vertices in dict format."""
        for vidx, vertex in self.mesh.vertices.items():
            if getattr(vertex, "fixed", False):
                grad[vidx][:] = 0.0

    def project_constraints(self, grad: Dict[int, np.ndarray] | np.ndarray) -> None:
        """Project gradients onto the feasible set defined by constraints."""

        if isinstance(grad, np.ndarray):
            self.project_constraints_array(grad)
            return

        # Legacy dict path
        self._zero_fixed_gradients(grad)
        # Add projection for non-fixed constraints if needed here

    def project_constraints_array(self, grad_arr: np.ndarray) -> None:
        """Array-based variant of ``project_constraints``.

        The array pipeline uses KKT projection for hard constraints. This hook
        only enforces fixed vertices for compatibility with legacy call sites.
        """
        vertex_ids = self.mesh.vertex_ids
        for row, vidx in enumerate(vertex_ids):
            if getattr(self.mesh.vertices[int(vidx)], "fixed", False):
                grad_arr[row] = 0.0

    def _enforce_constraints(self, mesh: Mesh | None = None):
        """Invoke all constraint modules on the current mesh."""
        if not self._has_enforceable_constraints:
            return

        target_mesh = mesh if mesh is not None else self.mesh
        self.constraint_manager.enforce_all(
            target_mesh,
            global_params=self.global_params,
            context="minimize",
        )

    def enforce_constraints_after_mesh_ops(self, mesh: Mesh | None = None):
        """Enforce constraints after discrete mesh operations."""
        if not self._has_enforceable_constraints:
            return

        target_mesh = mesh if mesh is not None else self.mesh
        self.constraint_manager.enforce_all(
            target_mesh,
            global_params=self.global_params,
            context="mesh_operation",
        )

    def minimize(
        self, n_steps: int = 1, callback: Optional[Callable[["Mesh", int], None]] = None
    ):
        """Run the optimization loop for ``n_steps`` iterations."""
        zero_step_counter = 0
        step_success = True

        if n_steps <= 0:
            E, grad = self.compute_energy_and_gradient()
            self._enforce_constraints()
            return {
                "energy": E,
                "gradient": grad,
                "mesh": self.mesh,
                "step_success": True,
                "iterations": 0,
                "terminated_early": True,
            }

        if self._has_enforceable_constraints:
            self.enforce_constraints_after_mesh_ops(self.mesh)

        self._check_gauss_bonnet()
        last_grad_arr = None
        for i in range(n_steps):
            if callback:
                callback(self.mesh, i)

            # Use array-based path for main loop
            E, grad_arr = self.compute_energy_and_gradient_array()
            self.project_constraints_array(grad_arr)
            last_grad_arr = grad_arr

            # check convergence by gradient norm
            grad_norm = float(np.linalg.norm(grad_arr))
            if grad_norm < self.tol:
                logger.debug("Converged: gradient norm below tolerance.")
                logger.info(f"Converged in {i} iterations; |∇E|={grad_norm:.3e}")
                return {
                    "energy": E,
                    "gradient": self._grad_arr_to_dict(grad_arr),
                    "mesh": self.mesh,
                    "step_success": True,
                    "iterations": i + 1,
                    "terminated_early": True,
                }
            logger.debug("Iteration %d: |∇E|=%.3e", i, grad_norm)

            if not self.quiet:
                # Compute total area only when needed for diagnostics
                total_area = sum(
                    facet.compute_area(self.mesh) for facet in self.mesh.facets.values()
                )
                print(
                    f"Step {i:4d}: Area = {total_area:.5f}, Energy = {E:.5f}, Step Size  = {self.step_size:.2e}"
                )

            step_success, self.step_size = self.stepper.step(
                self.mesh,
                grad_arr,
                self.step_size,
                self.compute_energy,
                constraint_enforcer=self._enforce_constraints
                if self._has_enforceable_constraints
                else None,
            )

            self._check_gauss_bonnet()
            if not step_success:
                if self.step_size <= self.step_size_floor:
                    zero_step_counter += 1
                    if zero_step_counter >= self.max_zero_steps:
                        logger.info(
                            "Terminating early after %d consecutive zero-steps "
                            "with step size <= %.2e.",
                            zero_step_counter,
                            self.step_size_floor,
                        )
                        return {
                            "energy": E,
                            "gradient": self._grad_arr_to_dict(last_grad_arr)
                            if last_grad_arr is not None
                            else {},
                            "mesh": self.mesh,
                            "step_success": False,
                            "iterations": i + 1,
                            "terminated_early": True,
                        }
                else:
                    zero_step_counter = 0
                reset = getattr(self.stepper, "reset", None)
                if callable(reset):
                    reset()
            else:
                zero_step_counter = 0

                if logger.isEnabledFor(logging.DEBUG):
                    E_after = self.compute_energy()
                    max_rel_violation_dbg = 0.0
                    vol_msgs: list[str] = []
                    if self.mesh.bodies:
                        for body in self.mesh.bodies.values():
                            target = body.target_volume
                            if target is None:
                                target = body.options.get("target_volume")
                            if target is None:
                                continue
                            current = body.compute_volume(self.mesh)
                            denom = max(abs(target), 1.0)
                            rel = (current - target) / denom
                            max_rel_violation_dbg = max(max_rel_violation_dbg, abs(rel))
                            vol_msgs.append(
                                "body %d: V=%.6f, V0=%.6f, relΔV=%.3e"
                                % (body.index, current, target, rel)
                            )
                    logger.debug(
                        "Accepted step %d: E_before=%.6f, E_after=%.6f, "
                        "step_size=%.3e, max_relΔV=%.3e",
                        i,
                        E,
                        E_after,
                        self.step_size,
                        max_rel_violation_dbg,
                    )
                    if vol_msgs:
                        logger.debug("Volume diagnostics: %s", "; ".join(vol_msgs))

                mode = self.global_params.get("volume_constraint_mode", "lagrange")
                proj_flag = self.global_params.get(
                    "volume_projection_during_minimization", True
                )
                vol_tol = float(self.global_params.get("volume_tolerance", 1e-3))
                if mode == "lagrange" and not proj_flag and self.mesh.bodies:
                    max_rel_violation = 0.0
                    for body in self.mesh.bodies.values():
                        target = body.target_volume
                        if target is None:
                            target = body.options.get("target_volume")
                        if target is None:
                            continue
                        current = body.compute_volume(self.mesh)
                        denom = max(abs(target), 1.0)
                        rel = abs(current - target) / denom
                        if rel > max_rel_violation:
                            max_rel_violation = rel

                    if max_rel_violation > vol_tol:
                        logger.debug(
                            "Volume drift %.3e exceeds tolerance %.3e; "
                            "applying geometric volume projection.",
                            max_rel_violation,
                            vol_tol,
                        )
                        self.enforce_constraints_after_mesh_ops(self.mesh)
                        reset = getattr(self.stepper, "reset", None)
                        if callable(reset):
                            reset()

        return {
            "energy": E,
            "gradient": self._grad_arr_to_dict(last_grad_arr)
            if last_grad_arr is not None
            else {},
            "mesh": self.mesh,
            "step_success": step_success,
            "iterations": n_steps,
            "terminated_early": False,
        }

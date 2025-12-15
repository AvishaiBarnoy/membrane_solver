# runtime/minimizer.py

import logging
from typing import Dict, List, Optional

import numpy as np

from geometry.entities import Body, Mesh
from parameters.global_parameters import GlobalParameters
from parameters.resolver import ParameterResolver
from runtime.constraint_manager import ConstraintModuleManager
from runtime.energy_manager import EnergyModuleManager
from runtime.steppers.base import BaseStepper

logger = logging.getLogger('membrane_solver')

class Minimizer:
    """Coordinate the optimization loop for a mesh."""

    def __init__(self,
                 mesh: Mesh,
                 global_params: GlobalParameters,
                 stepper: BaseStepper,
                 energy_manager: EnergyModuleManager,
                 constraint_manager: ConstraintModuleManager,
                 energy_modules: Optional[List[str]] = None,
                 constraint_modules: Optional[List[str]] = None,
                 step_size: float = 1e-3,
                 tol: float = 1e-6,
                 quiet: bool = False) -> None:
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
        module_list = energy_modules if energy_modules is not None else mesh.energy_modules
        self.energy_modules = [
            self.energy_manager.get_module(mod) for mod in module_list
        ]

        constraint_list = (
            constraint_modules if constraint_modules is not None else mesh.constraint_modules
        )
        self.constraint_modules = [
            self.constraint_manager.get_constraint(constraint) for constraint in constraint_list
        ]
        self._has_enforceable_constraints = any(
            hasattr(mod, "enforce_constraint") for mod in self.constraint_modules
        )

        logger.debug(f"Loaded energy modules: {self.energy_manager.modules.keys()}")
        logger.debug(f"Mesh energy_modules: {self.mesh.energy_modules}")

        self.param_resolver = ParameterResolver(global_params)

    def __repr__(self):
        msg = f"""### MINIMIZER ###
MESH:\t {self.mesh}
GLOBAL PARAMETERS:\t {self.global_params}
STEPPER:\t {self.stepper}
STEP SIZE:\t {self.step_size}
############"""
        return msg

    def compute_energy_and_gradient(self):
        """Return total energy and gradient for the current mesh."""

        # Invalidate any cached body volume/gradient before computing new values
        # since vertex positions may have changed after the last step.
        for body in self.mesh.bodies.values():
            if hasattr(body, "_last_volume"):
                body._last_volume = None
            if hasattr(body, "_last_volume_grad"):
                body._last_volume_grad = None

        # First, compute volume gradients for constrained bodies when using
        # Lagrange‑style hard volume constraints. This mirrors Evolver's
        # ``calc_volgrads`` phase and lets us reuse the same gradients both
        # for constraint projection and diagnostics.
        self._update_volume_gradients()

        total_energy = 0.0
        # initialize per-vertex gradient dict
        grad: Dict[int, np.ndarray] = {
            idx: np.zeros(3) for idx in self.mesh.vertices
        }

        for module in self.energy_modules:
            # Each energy module must implement compute_energy_and_gradient
            E_mod, g_mod = module.compute_energy_and_gradient(
                self.mesh, self.global_params, self.param_resolver)

            total_energy += E_mod
            for vidx, gvec in g_mod.items():
                grad[vidx] += gvec

        # Integrate hard volume constraints via a Lagrange‑style projection
        # of the gradient onto the tangent space of fixed‑volume manifolds,
        # analogous to Evolver's ``calc_lagrange`` + ``lagrange_adjust``.
        self._apply_volume_constraints_lagrange(grad)

        # Apply area constraint forces to drive bodies toward their target areas.
        self._apply_area_constraint_forces(grad)

        # Optional DEBUG‑level diagnostic: in Lagrange mode the projected
        # gradient should be (numerically) tangent to each fixed‑volume
        # manifold, i.e. ⟨∇E, ∇V_body⟩ ≈ 0.  This reuses the cached volume
        # gradients from ``_update_volume_gradients`` and is only executed
        # when verbose debugging is enabled.
        if logger.isEnabledFor(logging.DEBUG):
            mode = self.global_params.get("volume_constraint_mode", "lagrange")
            if mode == "lagrange" and self.mesh.bodies:
                max_abs_dot = 0.0
                for body in self.mesh.bodies.values():
                    vol_grad = getattr(body, "_last_volume_grad", None)
                    if not vol_grad:
                        continue
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

    def _update_volume_gradients(self) -> None:
        """Compute and cache body volume gradients when needed.

        In ``\"lagrange\"`` mode this plays the role of Evolver's
        ``calc_volgrads``: for each body with a target volume we compute
        ``(V, ∇V)`` once per iteration and stash them on the body for later
        use by the projection routine.
        """

        mode = self.global_params.get("volume_constraint_mode", "lagrange")
        if mode != "lagrange":
            return

        if not self.mesh.bodies:
            return

        for body in self.mesh.bodies.values():
            V_target = body.target_volume
            if V_target is None:
                V_target = body.options.get("target_volume")
            if V_target is None:
                continue

            vol, vol_grad = body.compute_volume_and_gradient(self.mesh)
            body._last_volume = vol
            body._last_volume_grad = vol_grad

    def _apply_volume_constraints_lagrange(self, grad: Dict[int, np.ndarray]) -> None:
        """Project the gradient to respect volume constraints.

        When ``global_parameters.volume_constraint_mode == "lagrange"``,
        project the gradient onto the subspace orthogonal to all active body
        volume gradients. This solves a small linear system in the space of
        volume constraints, analogous to a multi‑constraint Lagrange
        multiplier update, and generalises cleanly to multiple bodies.
        """

        mode = self.global_params.get("volume_constraint_mode", "lagrange")
        if mode != "lagrange":
            return

        # Gather all bodies that actually have a target volume and cached
        # gradients from ``_update_volume_gradients``.
        constrained_bodies: list[tuple["Body", Dict[int, np.ndarray]]] = []
        for body in self.mesh.bodies.values():
            V_target = body.target_volume
            if V_target is None:
                V_target = body.options.get("target_volume")
            if V_target is None:
                continue

            vol_grad = getattr(body, "_last_volume_grad", None)
            if not vol_grad:
                continue
            constrained_bodies.append((body, vol_grad))

        if not constrained_bodies:
            return

        k = len(constrained_bodies)
        # A_ij = <∇V_i, ∇V_j>; b_i = <∇E, ∇V_i>
        A = np.zeros((k, k), dtype=float)
        b_vec = np.zeros(k, dtype=float)

        # Build A and b by looping over vertices once per body pair.
        for i, (_, gradVi) in enumerate(constrained_bodies):
            for vidx, gVi in gradVi.items():
                if vidx in grad:
                    b_vec[i] += float(np.dot(grad[vidx], gVi))
                for j in range(i, k):
                    gVj = constrained_bodies[j][1].get(vidx)
                    if gVj is None:
                        continue
                    val = float(np.dot(gVi, gVj))
                    A[i, j] += val
                    if j != i:
                        A[j, i] += val

        # Regularise and solve A λ = b. If the system is ill‑conditioned or
        # singular, fall back to no projection rather than risking instability.
        eps = 1e-18
        A[np.diag_indices_from(A)] += eps
        try:
            lam = np.linalg.solve(A, b_vec)
        except np.linalg.LinAlgError:
            return

        # Apply the combined projection to the gradient.
        for (body, gradVi), lam_i in zip(constrained_bodies, lam):
            if lam_i == 0.0:
                continue
            for vidx, gVi in gradVi.items():
                if vidx in grad:
                    grad[vidx] -= lam_i * gVi

    def _apply_area_constraint_forces(self, grad: Dict[int, np.ndarray]) -> None:
        """Add a constraint force to drive each body toward its target area.

        This mimics a Lagrange multiplier update by solving a 1‑body scalar
        lambda = delta/||gradA||^2 and subtracting lambda * gradA from the
        energy gradient, so the direction both reduces energy (when feasible)
        and restores area when off‑target.
        """
        if not self.mesh.bodies:
            return

        for body in self.mesh.bodies.values():
            A_target = body.options.get("target_area")
            if A_target is None:
                continue

            area = 0.0
            gA = {}
            for facet_idx in body.facet_indices:
                facet = self.mesh.facets[facet_idx]
                a_f, g_f = facet.compute_area_and_gradient(self.mesh)
                area += a_f
                for vidx, vec in g_f.items():
                    if vidx not in gA:
                        gA[vidx] = vec.copy()
                    else:
                        gA[vidx] += vec

            delta = area - A_target
            norm_sq = sum(np.dot(v, v) for v in gA.values())
            if abs(delta) < 1e-12 or norm_sq < 1e-18:
                continue

            lam = delta / (norm_sq + 1e-18)
            for vidx, vec in gA.items():
                if vidx in grad:
                    grad[vidx] -= lam * vec

    def compute_energy(self):
        """Compute the total energy using the loaded modules."""
        total_energy = 0.0
        for module in self.energy_modules:
            E_mod, _ = module.compute_energy_and_gradient(
                self.mesh, self.global_params, self.param_resolver, compute_gradient=False
            )
            total_energy += E_mod
        return total_energy

    def project_constraints(self, grad: Dict[int, np.ndarray]) -> None:
        """Project gradients onto the feasible set defined by constraints."""

        # zero out fixed vertices and project others
        for vidx, vertex in self.mesh.vertices.items():
            if getattr(vertex, 'fixed', False):
                grad[vidx][:] = 0.0
            elif hasattr(vertex, 'constraint'):
                # project the gradient into tangent space of constraint
                grad[vidx] = vertex.constraint.project_gradient(grad[vidx])

        for eidx, edge in self.mesh.edges.items():
            # If has fixed attribute uncomment and change hasattr to elif
            # if geattr(edge, 'fixed', False): grad[eidx][:] = 0.0
            if hasattr(edge, 'constraint'):
                # project the gradient into tangent space of constraint
                grad[eidx] = edge.constraint.project_gradient(grad[eidx])

        for fidx, facet in self.mesh.facets.items():
            # If has fixed attribute uncomment and change hasattr to elif
            # if geattr(facet, 'fixed', False): grad[fidx][:] = 0.0
            if hasattr(facet, 'constraint'):
                # project the gradient into tangent space of constraint
                grad[fidx] = facet.constraint.project_gradient(grad[fidx])

        for bidx, body in self.mesh.bodies.items():
            # If has fixed attribute uncomment and change hasattr to elif
            # if geattr(body, 'fixed', False): grad[bidx][:] = 0.0
            if hasattr(body, 'constraint'):
                # project the gradient into tangent space of constraint
                grad[bidx] = body.constraint.project_gradient(grad[bidx])


    def _enforce_constraints(self, mesh: Mesh | None = None):
        """Invoke all constraint modules on the current mesh.

        Accepts an optional ``mesh`` argument so it can be used directly as the
        ``constraint_enforcer`` callback in line‑search routines that call it
        as ``constraint_enforcer(mesh)``. When ``mesh`` is omitted, the
        minimizer's own mesh is used.

        This method is intended for use *during minimization*. Depending on
        the global parameter ``volume_projection_during_minimization``, the
        volume constraint may be enforced either purely through gradient
        projection (Evolver‑like) or by an additional geometric projection
        step inside the line search (legacy behaviour).
        """
        if not self._has_enforceable_constraints:
            return

        target_mesh = mesh if mesh is not None else self.mesh
        self.constraint_manager.enforce_all(
            target_mesh,
            global_params=self.global_params,
            context="minimize",
        )

    def enforce_constraints_after_mesh_ops(self, mesh: Mesh | None = None) -> None:
        """Enforce constraints after discrete mesh operations.

        This is used after refinement, equiangulation, vertex averaging, etc.,
        where we do want to snap the geometry back to satisfy hard constraints
        such as fixed volume.
        """
        if not self._has_enforceable_constraints:
            return

        target_mesh = mesh if mesh is not None else self.mesh
        self.constraint_manager.enforce_all(
            target_mesh,
            global_params=self.global_params,
            context="mesh_operation",
        )

    def minimize(self, n_steps: int = 1):
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

        for i in range(n_steps):
            E, grad = self.compute_energy_and_gradient()
            self.project_constraints(grad)

            # check convergence by gradient norm
            grad_norm = np.sqrt(sum(np.dot(g, g) for g in grad.values()))
            if grad_norm < self.tol:
                logger.debug("Converged: gradient norm below tolerance.")
                logger.info(f"Converged in {i} iterations; |∇E|={grad_norm:.3e}")
                return {"energy": E, "gradient": grad, "mesh": self.mesh,
                        "step_success": True, "iterations": i + 1,
                        "terminated_early": True}
            logger.debug("Iteration %d: |∇E|=%.3e", i, grad_norm)

            if not self.quiet:
                # Compute total area only when needed for diagnostics
                total_area = sum(
                    facet.compute_area(self.mesh) for facet in self.mesh.facets.values()
                )
                print(f"Step {i:4d}: Area = {total_area:.5f}, Energy = {E:.5f}, Step Size  = {self.step_size:.2e}")

            step_success, self.step_size = self.stepper.step(
                self.mesh,
                grad,
                self.step_size,
                self.compute_energy,
                constraint_enforcer=self._enforce_constraints if self._has_enforceable_constraints else None,
            )

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
                            "gradient": grad,
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

                # DEBUG-level diagnostics for accepted steps: report the new
                # energy and volume error for constrained bodies. These checks
                # are intentionally guarded so that normal runs pay no extra
                # cost; they are meant for interactive debugging only.
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
                            max_rel_violation_dbg = max(
                                max_rel_violation_dbg, abs(rel)
                            )
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

                # In Lagrange mode, when geometric volume projection is
                # disabled during the line search, occasionally pull the
                # configuration back exactly onto the target volume manifolds
                # if we have drifted too far. This keeps hard volume
                # constraints honest without fighting the line search at every
                # trial step.
                mode = self.global_params.get("volume_constraint_mode", "lagrange")
                proj_flag = self.global_params.get(
                    "volume_projection_during_minimization", True
                )
                vol_tol = float(self.global_params.get("volume_tolerance", 1e-3))
                if (
                    mode == "lagrange"
                    and not proj_flag
                    and self.mesh.bodies
                ):
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
                        # After a hard projection, any CG history is stale.
                        reset = getattr(self.stepper, "reset", None)
                        if callable(reset):
                            reset()

        return {
            "energy": E,
            "gradient": grad,
            "mesh": self.mesh,
            "step_success": step_success,
            "iterations": n_steps,
            "terminated_early": False,
        }

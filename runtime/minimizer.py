# modules/minimizer.py

import sys
import os
import importlib
import numpy as np
from typing import Dict, List, Optional
from importlib import import_module
from parameters.resolver import ParameterResolver
from geometry.entities import Mesh
from parameters.global_parameters import GlobalParameters
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager
from modules.steppers.base import BaseStepper


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
                 tol: float = 1e-6) -> None:
        self.mesh = mesh
        self.global_params = global_params
        self.energy_manager = energy_manager
        self.constraint_manager = constraint_manager
        self.stepper = stepper
        self.step_size = step_size
        self.tol = tol

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

        print(f"[DEBUG] Loaded energy modules: {self.energy_manager.modules.keys()}")
        print(f"[DEBUG] Mesh energy_modules: {self.mesh.energy_modules}")

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

        total_energy = 0.0
        # initialize per-vertex gradient dict
        grad: Dict[int, np.ndarray] = {
            idx: np.zeros(3) for idx in self.mesh.vertices
        }

        #self.energy_modules = set(self.energy_modules)
        for module in self.energy_modules:
            # Each energy module must implement compute_energy_and_gradient
            E_mod, g_mod = module.compute_energy_and_gradient(
                self.mesh, self.global_params, self.param_resolver)

            total_energy += E_mod
            for vidx, gvec in g_mod.items():
                grad[vidx] += gvec

        V  = self.mesh.compute_total_volume()
        return total_energy, grad

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

        # TODO: HOW IS instance.contraint.project_gradient(...) TAKING CARE OF
        # ALL RELEVANT CONSTRAINTS?

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

    def take_step_with_backtracking(self, grad: Dict[int, np.ndarray],
                                    max_iter=10, alpha_init=None, beta=0.5,
                                    c=1e-4, gamma=1.2, alpha_max_factor=10):
        """
        Performs a single descent with Armijo backtracking line search.

        Parameters:
            - grad: dict of vertex_id -> gradient vector
            - max_iter: max backtracking iterations
            - alpha_init: starting step size (defaults to self.step_size)
            - beta: step size reduction facotr (e.g., 0.5)
            - c: Armijo condition parameter (e.g., 1e-4)
            - gamma: step size growth factor (e.g., 1.2)
        """
        # Store original positions
        original_positions = {vidx: v.position.copy() for vidx, v in
                              self.mesh.vertices.items() if not getattr(v,
                                                                       'fixed',
                                                                       False)}

        # Compute original energy without gradient for efficiency
        E0 = self.compute_energy()

        # Compute squared norm of full gradient
        grad_norm_squared = sum(np.dot(g, g) for g in grad.values())
        if grad_norm_squared < 1e-20:
            print(f"[DEBUG] Gradient norm too small; skipping step.")
            return

        alpha = alpha_init if alpha_init is not None else self.step_size
        alpha_max = alpha_max_factor * self.step_size

        for i in range(max_iter):
            # Trial step
            for vidx, vertex in self.mesh.vertices.items():
                if getattr(vertex, 'fixed', False):
                    continue
                vertex.position[:] = original_positions[vidx] - alpha * grad[vidx]
                if hasattr(vertex, 'constraint'):
                    vertex.position[:] = vertex.constraint.project_position(vertex.position)

            # Compute energy after trial step without recomputing gradient
            E_trial = self.compute_energy()

            if E_trial <= E0 - c * alpha * grad_norm_squared:
                print(f"[DEBUG] Line search accepted step size {alpha:.3e}")
                self.step_size = min(alpha * gamma, alpha_max)
                return True    # Accept step

            alpha *= beta

        print(f"[DEBUG] Line search failed to satisfy Armijo after {max_iter} iterations. Reverting.")
        # Revert to original positions if no acceptable step found
        for vidx, vertex in self.mesh.vertices.items():
            if getattr(vertex, 'fixed', False):
                continue
            vertex.position[:] = original_positions[vidx]

        # Report diagnostic
        print("[DIAGNOSTIC] Zero-step detected: no trial step reduced energy.")
        print(f"[DIAGNOSTIC] Current step_size = {self.step_size:.2e}")
        if self.step_size < 1e-8:
            print("[DIAGNOSTIC] Step size is very small - consider increasing tolerance or refining geometry.")

        return False  # Signal to minimize() that step was not accepted

    def minimize(self, n_steps: int = 1):
        """Run the optimization loop for ``n_steps`` iterations."""
        zero_step_counter = 0

        max_zero_steps = 5  # You can tune this

        for i in range(0, n_steps + 1):
            E, grad = self.compute_energy_and_gradient()
            self.project_constraints(grad)

            # check convergence by gradient norm
            grad_norm = np.sqrt(sum(np.dot(g, g) for g in grad.values()))
            if grad_norm < self.tol:
                print("[DEBUG] Converged: gradient norm below tolerance.")
                print(f"Converged in {i} iterations; |∇E|={grad_norm:.3e}")
                return {"energy": E, "gradient": grad, "mesh": self.mesh,
                        "step_success": True, "iterations": i + 1,
                        "terminated_early": True}

            # Compute total area
            total_area = sum(facet.compute_area(self.mesh) for facet in self.mesh.facets.values())
            # Print step details
            print(f"Step {i:4d}: Area = {total_area:.5f}, Energy = {E:.5f}, Step Size  = {self.step_size:.2e}")

            step_success = self.take_step_with_backtracking(grad)

            if not step_success:
                zero_step_counter += 1
                if zero_step_counter >= max_zero_steps:
                    print(f"[INFO] Terminating early after {zero_step_counter} consecutive zero-steps.")
                    return {"energy": E, "gradient": grad, "mesh": self.mesh,
                            "step_success": False, "iterations": i + 1,
                            "terminated_early": True}
            else:
                zero_step_counter = 0

        return {"energy": E, "gradient": grad, "mesh": self.mesh,
                "step_success": step_success, "iterations": n_steps,
                "terminated_early": False}


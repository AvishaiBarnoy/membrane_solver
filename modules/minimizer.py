# modules/minimizer.py

import sys
import os
import importlib
import numpy as np
from typing import Dict
from importlib import import_module
from .steppers.base import BaseStepper
#from runtime.energy_manager import EnergyModuleManager
#from runtime.constraint_manager import ConstraintModuleManager

class ParameterResolver:
    def __init__(self, global_params):
        self.global_params = global_params

    def get(self, obj, name: str):
        # look for facet-specific override, else global
        return obj.options.get(name, self.global_params.get(name))

class Minimizer:
    def __init__(self,
                 mesh,
                 global_params,
                 stepper: BaseStepper,
                 energy_manager,
                 constraint_manager,
                 energy_modules: list = [],
                 constraint_modules: list = [],
                 step_size: float = 1e-3,
                 tol: float = 1e-6,
                 n_steps: int = 100):
        self.mesh = mesh
        self.global_params = global_params
        self.energy_manager = energy_manager
        self.constraint_manager = constraint_manager
        self.stepper = stepper
        self.step_size = step_size
        self.tol = tol
        self.n_steps = n_steps

        # Use module_names from the mesh to initialize the energy manager
        self.energy_modules = [
            self.energy_manager.get_module(mod) for mod in mesh.energy_modules
            ]

        self.constraint_modules = [
            self.constraint_manager.get_constraint(constraint) for constraint
            in mesh.constraint_modules
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

        #if i == 0:
        V  = self.mesh.compute_total_volume()
        #Es = surface_energy_list[-1] if surface_energy_list else 0.0
        #Ev = volume_energy_list[-1]  if volume_energy_list  else 0.0
        #print(f"step i:2d : V = {V:6.4f}  energy_surf = {Es:7.4f}  energy_vol = {Ev:7.4f}")
        return total_energy, grad

    def compute_energy(self):
        total_energy = 0.0
        for module in self.energy_modules:
            E_mod, _ = module.compute_energy_and_gradient(
                self.mesh, self.global_params, self.param_resolver, compute_gradient=False
            )
            total_energy += E_mod
        return total_energy

    def project_constraints(self, grad: Dict[int, np.ndarray]):

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
                return # Accept step

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

    def minimize(self):
        for i in range(0, self.n_steps + 1):
            E, grad = self.compute_energy_and_gradient()

            self.project_constraints(grad)

            # check convergence by gradient norm
            grad_norm = np.sqrt(sum(np.dot(g, g) for g in grad.values()))

            if grad_norm < self.tol:
                print("[DEBUG] Converged: gradient norm below tolerance.")
                print(f"Converged in {i} iterations; |∇E|={grad_norm:.3e}")
                break

            # Compute total area
            total_area = sum(facet.compute_area(self.mesh) for facet in self.mesh.facets.values())

            # Print step details
            print(f"Step {i:4d}: Area = {total_area:.6f}, Energy = {E:.6f}, Step Size  = {self.step_size:.2e}")

            self.take_step_with_backtracking(grad)

        return {"energy": E, "mesh": self.mesh}


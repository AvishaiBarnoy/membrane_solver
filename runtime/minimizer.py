# runtime/minimizer.py

import sys
import os
import importlib
import logging
import numpy as np
from typing import Dict, List, Optional
from importlib import import_module
from parameters.resolver import ParameterResolver
from geometry.entities import Mesh
from parameters.global_parameters import GlobalParameters
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager
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


    def minimize(self, n_steps: int = 1, callback: Optional[callable] = None):
        """Run the optimization loop for ``n_steps`` iterations.

        Parameters
        ----------
        n_steps : int
            Number of optimization iterations to perform.
        callback : callable, optional
            If given, called after each iteration with the updated ``Mesh``.
        """
        if callback:
            callback(self.mesh)

        zero_step_counter = 0

        max_zero_steps = 5  # You can tune this

        for i in range(0, n_steps + 1):
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

            # Compute total area
            total_area = sum(facet.compute_area(self.mesh) for facet in self.mesh.facets.values())
            # Print step details
            if not self.quiet:
                print(f"Step {i:4d}: Area = {total_area:.5f}, Energy = {E:.5f}, Step Size  = {self.step_size:.2e}")

            step_success, self.step_size = self.stepper.step(
                self.mesh, grad, self.step_size, self.compute_energy
            )

            if callback:
                callback(self.mesh)

            if not step_success:
                zero_step_counter += 1
                if zero_step_counter >= max_zero_steps:
                    logger.info(
                        f"Terminating early after {zero_step_counter} consecutive zero-steps."
                    )
                    return {"energy": E, "gradient": grad, "mesh": self.mesh,
                            "step_success": False, "iterations": i + 1,
                            "terminated_early": True}
            else:
                zero_step_counter = 0

        return {"energy": E, "gradient": grad, "mesh": self.mesh,
                "step_success": step_success, "iterations": n_steps,
                "terminated_early": False}


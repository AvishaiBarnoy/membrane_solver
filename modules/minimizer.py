# modules/minimizer.py

import sys
import os
import importlib
import numpy as np
from typing import Dict
from importlib import import_module
from .steppers.base import BaseStepper
from runtime.energy_manager import EnergyModuleManager
from runtime.constraint_manager import ConstraintModuleManager

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
        self.constraint_modules = constraint_manager
        self.stepper = BaseStepper
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

        for fname in self.energy_manager.modules.values():
            self.energy_modules.append(fname)

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

        for mod in self.energy_modules:
            # E_mod - energy module, g_mod - gradient module
            # TODO: documentation:
            # write that in new energy modules compute_energy_and_gradient is
            #   a mandatory name
            #E_mod, g_mod = mod.compute_energy_and_gradient(
            #    self.mesh, self.global_params, self.param_resolver
            #)
            #volume_energy_list = []
            #surface_energy_list = []

            for module in self.energy_modules:
                E_mod, g_mod = module.compute_energy_and_gradient(
                    self.mesh, self.global_params, self.param_resolver)

                #if   module.__name__ ==   "modules.energy.volume": volume_energy_list.append(E_mod)
                #elif module.__name__ == "modules.energy.surface": surface_energy_list.append(E_mod)
        #print(f"[DEBUG] Surface energy: {surface_energy_list}, Volume energy: {volume_energy_list}")
                total_energy += E_mod
                for vidx, gvec in g_mod.items():
                    grad[vidx] += gvec

        #if i == 0:
        V  = self.mesh.compute_total_volume()
        #Es = surface_energy_list[-1] if surface_energy_list else 0.0
        #Ev = volume_energy_list[-1]  if volume_energy_list  else 0.0
        #print(f"step i:2d : V = {V:6.4f}  energy_surf = {Es:7.4f}  energy_vol = {Ev:7.4f}")
        return total_energy, grad

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

    def take_step(self, grad: Dict[int, np.ndarray]):
        for vidx, vertex in self.mesh.vertices.items():
            if getattr(vertex, 'fixed', False):
                continue
            old_pos = vertex.position.copy()
            new_pos = vertex.position - self.step_size * grad[vidx]
            # if there's a constraint object, project position back onto it
            if hasattr(vertex, 'constraint'):
                new_pos = vertex.constraint.project_position(new_pos)
            vertex.position[:] = new_pos
            # Debug: Print vertex movement
            #print(f"[DEBUG] Vertex {vidx}: moved {np.linalg.norm(new_pos - old_pos):.6e}")

    def minimize(self):
        for i in range(0, self.n_steps + 1):
            E, grad = self.compute_energy_and_gradient()
            #print(f"[DEBUG]")
            #print(f"step i:2d : V = {V:6.4f}  energy_surf = {Es:7.4f}  energy_vol = {Ev:7.4f}")
            self.project_constraints(grad)

            # check convergence by gradient norm
            grad_norm = np.sqrt(sum(np.dot(g, g) for g in grad.values()))
            #print(f"[DEBUG] Iter {i}: Energy={E:.6f}, grad norm={grad_norm:.6e}")
            # Print a few gradient values
            #for idx, g in list(grad.items())[:3]:
                #print(f"[DEBUG] grad[{idx}] = {g}")

            if grad_norm < self.tol:
                print("[DEBUG] Converged: gradient norm below tolerance.")
                print(f"Converged in {i} iterations; |∇E|={grad_norm:.3e}")
                break

            # Compute total area
            total_area = sum(facet.compute_area(self.mesh) for facet in self.mesh.facets.values())

            # Print step details
            print(f"Step {i:4d}: Area = {total_area:.6f}, Energy = {E:.6f}, Step Size  = {self.step_size:.2e}")

            self.take_step(grad)
            #print(f"[DEBUG] Current volume: {self.mesh.compute_total_volume()}")

        return {"energy": E, "mesh": self.mesh}


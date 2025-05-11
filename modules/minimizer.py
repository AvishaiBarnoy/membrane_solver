# modules/minimizer.py

import os
import importlib
import numpy as np
from typing import Dict
from importlib import import_module
from .steppers.base import BaseStepper

class ParameterResolver:
    def __init__(self, global_params):
        self.global_params = global_params

    def get(self, facet, name: str):
        # look for facet-specific override, else global
        return facet.options.get(name, getattr(self.global_params, name))

class Minimizer:
    def __init__(self,
                 mesh,
                 global_params,
                 stepper: BaseStepper,
                 step_size: float = 1e-3,
                 tol: float = 1e-6,
                 max_iter: int = 1000):
        self.mesh = mesh
        self.global_params = global_params
        self.stepper = BaseStepper
        self.step_size = step_size
        self.tol = tol
        self.max_iter = max_iter

        # dynamically discover all energy modules in modules/
        self.energy_modules = self._discover_modules('modules')


        # TODO: is this section needed? 
        modules_dir = os.path.join(os.path.dirname(__file__), 'modules')

        for fname in os.listdir(modules_dir):
            if fname.endswith('.py') and not fname.startswith('__'):
                name = fname[:-3]
                mod = importlib.import_module(f"modules.{name}")
                self.energy_modules.append(mod)

        self.param_resolver = ParameterResolver(global_params)

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
            E_mod, g_mod = mod.compute_energy_and_gradient(
                self.mesh, self.global_params, self.param_resolver
            )
            total_energy += E_mod
            for vidx, gvec in g_mod.items():
                grad[vidx] += gvec

        return total_energy, grad

    ###### I am going over this line now
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

        for fidx, edge in self.mesh.facets.items():
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
            new_pos = vertex.position - self.step_size * grad[vidx]
            # if there's a constraint object, project position back onto it
            if hasattr(vertex, 'constraint'):
                new_pos = vertex.constraint.project_position(new_pos)
            vertex.position[:] = new_pos

    def minimize(self):
        for i in range(1, self.max_iter + 1):
            E, grad = self.compute_energy_and_gradient()
            self.project_constraints(grad)

            # check convergence by gradient norm
            norm = np.sqrt(sum(np.dot(g, g) for g in grad.values()))
            if norm < self.tol:
                print(f"Converged in {i} iterations; |∇E|={norm:.3e}")
                break

            self.take_step(grad)

        return self.mesh

# a convenience function
def minimize(mesh, global_params, **kwargs):
    """
    Standalone function if you prefer functional style.
    """
    engine = Minimizer(mesh, global_params, **kwargs)
    return engine.minimize()


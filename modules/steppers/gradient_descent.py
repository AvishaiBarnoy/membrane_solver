# modules/steppers/gradient_descent.py

import numpy as np
from .base import BaseStepper

class GradientDescent(BaseStepper):
    def __init__(self, c1=1e-4, c2=0.9, max_iter=20):
        self.c1 = c1
        self.c2 = c2
        self.max_iter = max_iter

    def step(self, mesh, grad, step_size, energy_fn):
        # Flatten positions and gradients for line search
        x0 = self._flatten_positions(mesh)
        d = -self._flatten_gradient(grad, mesh)

        energy0 = energy_fn()
        grad0 = self._flatten_gradient(grad, mesh)
        g0_dot_d = np.dot(grad0, d)

        alpha = step_size
        for i in range(self.max_iter):
            self._set_positions(mesh, x0 + alpha * d)

            energy = energy_fn()
            grad_new = self._collect_gradient(mesh)
            g_new_dot_d = np.dot(self._flatten_gradient(grad_new, mesh), d)

            if energy > energy0 + self.c1 * alpha * g0_dot_d:
                alpha *= 0.5  # insufficient decrease
            elif g_new_dot_d < self.c2 * g0_dot_d:
                alpha *= 1.1  # insufficient curvature
            else:
                break  # Wolfe conditions met

        # Final update
        for vidx, vertex in mesh.vertices.items():
            if getattr(vertex, 'fixed', False):
                continue
            vertex.position = self._original_position[vidx] + alpha * d[self._vmap[vidx]]

            if hasattr(vertex, 'constraint'):
                vertex.position = vertex.constraint.project_position(vertex.position)

    def _flatten_positions(self, mesh):
        self._vmap = {}  # maps vertex id to index
        self._original_position = {}
        pos = []
        i = 0
        for vidx, v in mesh.vertices.items():
            if getattr(v, 'fixed', False):
                continue
            self._original_position[vidx] = v.position.copy()
            self._vmap[vidx] = slice(i, i+3)
            pos.extend(v.position)
            i += 3
        return np.array(pos)

    def _flatten_gradient(self, grad, mesh):
        flat = []
        for vidx in self._vmap:
            flat.extend(grad[vidx])
        return np.array(flat)

    def _set_positions(self, mesh, flat):
        for vidx in self._vmap:
            mesh.vertices[vidx].position = flat[self._vmap[vidx]].copy()

    def _collect_gradient(self, mesh):
        raise NotImplementedError("This should be passed externally. Wrap mesh.grad_fn(mesh) externally.")

    def __repr__(self): ...

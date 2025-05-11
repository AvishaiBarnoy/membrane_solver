# modules/steppers/conjugate_gradient.py
import numpy as np
from .base import BaseStepper

class ConjugateGradient(BaseStepper):
    def __init__(self):
        # store previous gradient and direction
        self.prev_grad = {}
        self.prev_dir  = {}

    def step(self, mesh, grad, step_size):
        # On first call, just use GD direction
        for vidx, vertex in mesh.vertices.items():
            if getattr(vertex, 'fixed', False):
                continue

            g = grad[vidx]
            if vidx not in self.prev_grad:
                d = -g
            else:
                # β = (g⋅g) / (g_prev⋅g_prev)
                b = (g @ g) / (self.prev_grad[vidx] @ self.prev_grad[vidx] + 1e-20)
                d = -g + b * self.prev_dir[vidx]

            # update position
            new_pos = vertex.position + step_size * d
            if hasattr(vertex, 'constraint'):
                new_pos = vertex.constraint.project_position(new_pos)

            vertex.position = new_pos
            # save for next iteration
            self.prev_grad[vidx] = g
            self.prev_dir [vidx] = d

    def __repr__(self): ...


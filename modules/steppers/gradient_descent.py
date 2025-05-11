# modules/steppers/gradient_descent.py

import numpy as np
from .base import BaseStepper

class GradientDescent(BaseStepper):
    def step(self, mesh, grad, step_size):
        for vidx, vertex in mesh.vertices.items():
            if getattr(vertex, 'fixed', False):
                continue
            # plain descent
            vertex.position -= step_size * grad[vidx]
            # re-project if constrained
            if hasattr(vertex, 'constraint'):
                vertex.position = vertex.constraint.project_position(vertex.position)

    def __repr__(self): ...


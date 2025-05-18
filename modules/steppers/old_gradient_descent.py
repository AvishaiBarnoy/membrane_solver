# modules/steppers/gradient_descent.py

import numpy as np
from .base import BaseStepper

class GradientDescent(BaseStepper):
    def step(self, mesh, grad, step_size):
        l2 = np.linalg.norm(grad[vidx])
        if l2 > 1e-2:   # cap to 1% of edge length per step
            grad[vidx] = 1e-2 / l2

        for vidx, vertex in mesh.vertices.items():
            if getattr(vertex, 'fixed', False):
                continue
            # plain descent
            vertex.position -= step_size * grad[vidx]
            # re-project if constrained
            if hasattr(vertex, 'constraint'):
                vertex.position = vertex.constraint.project_position(vertex.position)

    def __repr__(self): ...


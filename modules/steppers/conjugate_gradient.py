# modules/steppers/conjugate_gradient.py
import numpy as np
from .base import BaseStepper

import numpy as np
from .base import BaseStepper

class ConjugateGradient(BaseStepper):
    def __init__(self, restart_interval=10, precondition=False):
        self.prev_grad = {}
        self.prev_dir = {}
        self.restart_interval = restart_interval
        self.iter_count = 0
        self.precondition = precondition

    def reset(self):
        self.prev_grad.clear()
        self.prev_dir.clear()
        self.iter_count = 0

    def step(self, mesh, grad, step_size):
        for vidx, vertex in mesh.vertices.items():
            if getattr(vertex, 'fixed', False):
                continue

            g = grad[vidx]

            if self.precondition:
                g = g / (np.linalg.norm(g) + 1e-8)

            restart = False
            if vidx not in self.prev_grad:
                d = -g
            else:
                prev_g = self.prev_grad[vidx]
                prev_d = self.prev_dir[vidx]
                beta = np.dot(g, g - prev_g) / (np.dot(prev_g, prev_g) + 1e-20)  # Polak-Ribiere
                if beta < 0 or self.iter_count % self.restart_interval == 0:
                    d = -g  # restart
                    restart = True
                else:
                    d = -g + beta * prev_d

            d /= (np.linalg.norm(d) + 1e-12)  # normalize

            # Simple line search placeholder — can be replaced
            alpha = step_size

            new_pos = vertex.position + alpha * d
            if hasattr(vertex, 'constraint'):
                new_pos = vertex.constraint.project_position(new_pos)

            vertex.position = new_pos
            self.prev_grad[vidx] = g.copy()
            self.prev_dir[vidx] = d.copy()

        self.iter_count += 1

        def __repr__(self): ...

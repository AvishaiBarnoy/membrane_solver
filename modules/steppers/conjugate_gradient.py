# modules/steppers/conjugate_gradient.py
from __future__ import annotations

import numpy as np
from typing import Callable, Dict

from geometry.entities import Mesh
from runtime.steppers.line_search import backtracking_line_search
from .base import BaseStepper

class ConjugateGradient(BaseStepper):
    """Conjugate gradient stepper with Armijo backtracking line search."""

    def __init__(
        self,
        restart_interval: int = 10,
        precondition: bool = False,
        max_iter: int = 10,
        beta: float = 0.5,
        c: float = 1e-4,
        gamma: float = 1.2,
        alpha_max_factor: float = 10.0,
    ) -> None:
        self.prev_grad: Dict[int, np.ndarray] = {}
        self.prev_dir: Dict[int, np.ndarray] = {}
        self.restart_interval = restart_interval
        self.iter_count = 0
        self.precondition = precondition
        self.max_iter = max_iter
        self.beta = beta
        self.c = c
        self.gamma = gamma
        self.alpha_max_factor = alpha_max_factor

    def reset(self):
        self.prev_grad.clear()
        self.prev_dir.clear()
        self.iter_count = 0

    def step(
        self,
        mesh: Mesh,
        grad: Dict[int, np.ndarray],
        step_size: float,
        energy_fn: Callable[[], float],
    ) -> tuple[bool, float]:
        """Take one conjugate gradient step with line search."""

        direction: Dict[int, np.ndarray] = {}

        for vidx, vertex in mesh.vertices.items():
            if getattr(vertex, "fixed", False):
                continue

            g = grad[vidx]

            if self.precondition:
                g = g / (np.linalg.norm(g) + 1e-8)

            if (
                vidx not in self.prev_grad
                or self.iter_count % self.restart_interval == 0
            ):
                d = -g
            else:
                prev_g = self.prev_grad[vidx]
                prev_d = self.prev_dir[vidx]
                beta_pr = np.dot(g, g - prev_g) / (
                    np.dot(prev_g, prev_g) + 1e-20
                )
                if beta_pr < 0:
                    d = -g
                else:
                    d = -g + beta_pr * prev_d

            d /= np.linalg.norm(d) + 1e-12
            direction[vidx] = d

        success, new_step = backtracking_line_search(
            mesh,
            direction,
            grad,
            step_size,
            energy_fn,
            max_iter=self.max_iter,
            beta=self.beta,
            c=self.c,
            gamma=self.gamma,
            alpha_max_factor=self.alpha_max_factor,
        )

        if success:
            for vidx, d in direction.items():
                self.prev_grad[vidx] = grad[vidx].copy()
                self.prev_dir[vidx] = d.copy()
            self.iter_count += 1

        return success, new_step


"""Gradient descent stepper with backtracking line search."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from geometry.entities import Mesh
from runtime.steppers.line_search import backtracking_line_search

from .base import BaseStepper


class GradientDescent(BaseStepper):
    """Perform gradient descent using Armijo backtracking."""

    def __init__(
        self,
        max_iter: int = 10,
        beta: float = 0.5,
        c: float = 1e-4,
        gamma: float = 1.2,
        alpha_max_factor: float = 10.0,
    ) -> None:
        self.max_iter = max_iter
        self.beta = beta
        self.c = c
        self.gamma = gamma
        self.alpha_max_factor = alpha_max_factor

    def step(
        self,
        mesh: Mesh,
        grad: Dict[int, np.ndarray],
        step_size: float,
        energy_fn: Callable[[], float],
        constraint_enforcer: Callable[[Mesh], None] | None = None,
    ) -> tuple[bool, float]:
        """Apply one gradient descent step with backtracking line search."""

        direction = {vidx: -g for vidx, g in grad.items()}
        return backtracking_line_search(
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
            constraint_enforcer=constraint_enforcer,
        )

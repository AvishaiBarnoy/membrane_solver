"""Gradient descent stepper with backtracking line search."""

from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from geometry.entities import Mesh
from runtime.steppers.line_search import (
    backtracking_line_search,
    strong_wolfe_line_search,
    adaptive_line_search
)

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
        )


class EnhancedGradientDescent(BaseStepper):
    """Enhanced gradient descent with multiple line search algorithms."""

    def __init__(
        self,
        line_search: str = "adaptive",
        max_iter: int = 15,
        c1: float = 1e-4,
        c2: float = 0.9,
        beta: float = 0.5,
        gamma: float = 1.2,
        alpha_max_factor: float = 10.0,
    ) -> None:
        """
        Initialize enhanced gradient descent.
        
        Parameters
        ----------
        line_search : str
            Line search algorithm: "adaptive", "strong_wolfe", "backtrack", or "simple"
        max_iter : int
            Maximum line search iterations
        c1, c2 : float
            Wolfe condition parameters
        beta, gamma : float
            Step size adjustment factors
        alpha_max_factor : float
            Maximum step size multiplier
        """
        self.line_search = line_search
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.beta = beta
        self.gamma = gamma
        self.alpha_max_factor = alpha_max_factor

    def step(
        self,
        mesh: Mesh,
        grad: Dict[int, np.ndarray],
        step_size: float,
        energy_fn: Callable[[], float],
        gradient_fn: Optional[Callable[[], Dict[int, np.ndarray]]] = None,
    ) -> tuple[bool, float]:
        """Apply one enhanced gradient descent step."""

        direction = {vidx: -g for vidx, g in grad.items()}
        
        if self.line_search == "adaptive":
            return adaptive_line_search(
                mesh, direction, grad, step_size, energy_fn, gradient_fn
            )
        elif self.line_search == "strong_wolfe":
            return strong_wolfe_line_search(
                mesh, direction, grad, step_size, energy_fn, gradient_fn,
                c1=self.c1, c2=self.c2, max_iter=self.max_iter
            )
        elif self.line_search == "backtrack":
            return adaptive_line_search(
                mesh, direction, grad, step_size, energy_fn, gradient_fn, "backtrack"
            )
        else:  # "simple"
            return backtracking_line_search(
                mesh, direction, grad, step_size, energy_fn,
                max_iter=self.max_iter, beta=self.beta, c=self.c1,
                gamma=self.gamma, alpha_max_factor=self.alpha_max_factor
            )


"""Backtracking gradient descent stepper."""
from typing import Dict, Callable

import numpy as np

from modules.steppers.base import BaseStepper
from geometry.entities import Mesh
from .line_search import backtracking_line_search


class BacktrackingGradientDescent(BaseStepper):
    """Perform gradient descent with Armijo backtracking line search."""

    def __init__(self, max_iter: int = 10, beta: float = 0.5, c: float = 1e-4,
                 gamma: float = 1.2, alpha_max_factor: float = 10.0) -> None:
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
    ):
        """Apply one gradient descent step using Armijo backtracking.

        Parameters
        ----------
        mesh : Mesh
            Mesh being optimized.
        grad : Dict[int, np.ndarray]
            Gradient for each vertex index.
        step_size : float
            Initial step size to try.
        energy_fn : Callable[[], float]
            Function returning current energy of the mesh.

        Returns
        -------
        tuple[bool, float]
            Whether the step succeeded and the updated step size.
        """
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

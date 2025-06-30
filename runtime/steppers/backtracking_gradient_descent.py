"""Backtracking gradient descent stepper."""
from typing import Dict, Callable

import numpy as np

from modules.steppers.base import BaseStepper
from geometry.entities import Mesh


class BacktrackingGradientDescent(BaseStepper):
    """Perform gradient descent with Armijo backtracking line search."""

    def __init__(self, max_iter: int = 10, beta: float = 0.5, c: float = 1e-4,
                 gamma: float = 1.2, alpha_max_factor: float = 10.0) -> None:
        self.max_iter = max_iter
        self.beta = beta
        self.c = c
        self.gamma = gamma
        self.alpha_max_factor = alpha_max_factor

    def step(self, mesh: Mesh, grad: Dict[int, np.ndarray], step_size: float,
             energy_fn: Callable[[], float]):
        """Apply one descent step with backtracking.

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
        original_positions = {
            vidx: v.position.copy()
            for vidx, v in mesh.vertices.items()
            if not getattr(v, "fixed", False)
        }

        E0 = energy_fn()
        grad_norm_sq = sum(np.dot(g, g) for g in grad.values())
        if grad_norm_sq < 1e-20:
            print("[DEBUG] Gradient norm too small; skipping step.")
            return False, step_size

        alpha = step_size
        alpha_max = self.alpha_max_factor * step_size

        for _ in range(self.max_iter):
            for vidx, vertex in mesh.vertices.items():
                if getattr(vertex, "fixed", False):
                    continue
                vertex.position[:] = original_positions[vidx] - alpha * grad[vidx]
                if hasattr(vertex, "constraint"):
                    vertex.position[:] = vertex.constraint.project_position(
                        vertex.position
                    )

            E_trial = energy_fn()
            if E_trial <= E0 - self.c * alpha * grad_norm_sq:
                print(f"[DEBUG] Line search accepted step size {alpha:.3e}")
                new_step = min(alpha * self.gamma, alpha_max)
                return True, new_step

            alpha *= self.beta

        print(
            f"[DEBUG] Line search failed to satisfy Armijo after {self.max_iter} iterations. Reverting."
        )
        for vidx, vertex in mesh.vertices.items():
            if getattr(vertex, "fixed", False):
                continue
            vertex.position[:] = original_positions[vidx]

        print("[DIAGNOSTIC] Zero-step detected: no trial step reduced energy.")
        print(f"[DIAGNOSTIC] Current step_size = {step_size:.2e}")
        return False, step_size

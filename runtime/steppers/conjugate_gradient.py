# modules/steppers/conjugate_gradient.py
from __future__ import annotations

from typing import Callable, Dict

import numpy as np

from geometry.entities import Mesh
from runtime.steppers.line_search import (
    backtracking_line_search,
    backtracking_line_search_array,
)

from .base import BaseStepper


class ConjugateGradient(BaseStepper):
    """Conjugate gradient stepper with Armijo backtracking line search."""

    def __init__(
        self,
        restart_interval: int = 10,
        precondition: bool = False,
        max_iter: int = 10,
        beta: float = 0.7,
        c: float = 1e-4,
        gamma: float = 1.5,
        alpha_max_factor: float = 10.0,
    ) -> None:
        self.prev_grad: Dict[int, np.ndarray] = {}
        self.prev_dir: Dict[int, np.ndarray] = {}
        self.prev_grad_arr: np.ndarray | None = None
        self.prev_dir_arr: np.ndarray | None = None
        self.prev_vertex_ids: tuple[int, ...] | None = None
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
        self.prev_grad_arr = None
        self.prev_dir_arr = None
        self.prev_vertex_ids = None
        self.iter_count = 0

    def step(
        self,
        mesh: Mesh,
        grad: Dict[int, np.ndarray],
        step_size: float,
        energy_fn: Callable[[], float],
        constraint_enforcer: Callable[[Mesh], None] | None = None,
    ) -> tuple[bool, float]:
        """Take one conjugate gradient step with line search."""

        if isinstance(grad, np.ndarray):
            mesh.build_position_cache()
            vertex_ids = tuple(mesh.vertex_ids.tolist())
            if self.prev_vertex_ids is None or self.prev_vertex_ids != vertex_ids:
                self.prev_grad_arr = None
                self.prev_dir_arr = None
                self.prev_vertex_ids = vertex_ids
                self.iter_count = 0

            direction_arr = np.zeros_like(grad)
            for row, vidx in enumerate(vertex_ids):
                if getattr(mesh.vertices[vidx], "fixed", False):
                    continue
                g = grad[row]
                if self.precondition:
                    g = g / (np.linalg.norm(g) + 1e-8)

                if (
                    self.prev_grad_arr is None
                    or self.iter_count % self.restart_interval == 0
                ):
                    d = -g
                else:
                    prev_g = self.prev_grad_arr[row]
                    prev_d = self.prev_dir_arr[row]
                    beta_pr = np.dot(g, g - prev_g) / (np.dot(prev_g, prev_g) + 1e-20)
                    if beta_pr < 0:
                        d = -g
                    else:
                        d = -g + beta_pr * prev_d

                direction_arr[row] = d

            success, new_step = backtracking_line_search_array(
                mesh,
                direction_arr,
                grad,
                step_size,
                energy_fn,
                vertex_ids,
                max_iter=self.max_iter,
                beta=self.beta,
                c=self.c,
                gamma=self.gamma,
                alpha_max_factor=self.alpha_max_factor,
                constraint_enforcer=constraint_enforcer,
            )

            if success:
                self.prev_grad_arr = grad.copy()
                self.prev_dir_arr = direction_arr.copy()
                self.iter_count += 1

            return success, new_step

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
                beta_pr = np.dot(g, g - prev_g) / (np.dot(prev_g, prev_g) + 1e-20)
                if beta_pr < 0:
                    d = -g
                else:
                    d = -g + beta_pr * prev_d

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
            constraint_enforcer=constraint_enforcer,
        )

        if success:
            for vidx, d in direction.items():
                self.prev_grad[vidx] = grad[vidx].copy()
                self.prev_dir[vidx] = d.copy()
            self.iter_count += 1

        # Return the step size suggested by the line search, even on failure,
        # so callers can shrink toward a zero-step cutoff if needed.
        return success, new_step

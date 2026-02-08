"""Quasi-Newton BFGS stepper with backtracking line search."""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from geometry.entities import Mesh
from runtime.steppers.line_search import (
    backtracking_line_search,
    backtracking_line_search_array,
)

from .base import BaseStepper


class BFGS(BaseStepper):
    """BFGS stepper using a dense inverse-Hessian approximation.

    This is intended for moderate-sized problems; it resets when the vertex
    set changes (e.g., after refinement).
    """

    def __init__(
        self,
        max_iter: int = 10,
        beta: float = 0.7,
        c: float = 1e-4,
        gamma: float = 1.5,
        alpha_max_factor: float = 10.0,
    ) -> None:
        self.max_iter = max_iter
        self.beta = beta
        self.c = c
        self.gamma = gamma
        self.alpha_max_factor = alpha_max_factor
        self._prev_x: np.ndarray | None = None
        self._prev_grad: np.ndarray | None = None
        self._H_inv: np.ndarray | None = None
        self._prev_vids: Tuple[int, ...] | None = None

    def reset(self) -> None:
        self._prev_x = None
        self._prev_grad = None
        self._H_inv = None
        self._prev_vids = None

    def _collect_state(
        self, mesh: Mesh, grad: Dict[int, np.ndarray]
    ) -> tuple[Tuple[int, ...], np.ndarray, np.ndarray]:
        vids = tuple(sorted(vid for vid, v in mesh.vertices.items() if not v.fixed))
        x = np.zeros(3 * len(vids), dtype=float)
        g = np.zeros_like(x)
        for i, vid in enumerate(vids):
            pos = mesh.vertices[vid].position
            x[3 * i : 3 * i + 3] = pos
            g[3 * i : 3 * i + 3] = grad.get(vid, np.zeros(3))
        return vids, x, g

    def step(
        self,
        mesh: Mesh,
        grad: Dict[int, np.ndarray] | np.ndarray,
        step_size: float,
        energy_fn: Callable[[], float],
        constraint_enforcer: Callable[[Mesh], None] | None = None,
    ) -> tuple[bool, float, float]:
        """Take one BFGS step with line search."""

        if isinstance(grad, np.ndarray):
            mesh.build_position_cache()
            fixed_mask = mesh.fixed_mask
            movable_rows = np.flatnonzero(~fixed_mask)
            vids = tuple(mesh.vertex_ids[movable_rows].tolist())
            positions = mesh.positions_view()
            x = positions[movable_rows].reshape(-1).copy()
            g = grad[movable_rows].reshape(-1).copy()
        else:
            vids, x, g = self._collect_state(mesh, grad)

        if self._prev_vids is not None and self._prev_vids != vids:
            self.reset()

        if self._H_inv is None:
            self._H_inv = np.eye(len(x), dtype=float)

        if self._prev_x is not None and self._prev_grad is not None:
            s = x - self._prev_x
            y = g - self._prev_grad
            ys = float(np.dot(y, s))
            if ys > 1e-12:
                rho = 1.0 / ys
                I_n = np.eye(len(x), dtype=float)
                V = I_n - rho * np.outer(s, y)
                self._H_inv = V @ self._H_inv @ V.T + rho * np.outer(s, s)
            else:
                # If curvature condition fails, reset the approximation.
                self._H_inv = np.eye(len(x), dtype=float)

        direction_vec = -self._H_inv.dot(g)
        if isinstance(grad, np.ndarray):
            direction_arr = np.zeros_like(grad)
            if movable_rows.size:
                direction_arr[movable_rows] = direction_vec.reshape(-1, 3)
            success, new_step, accepted_energy = backtracking_line_search_array(
                mesh,
                direction_arr,
                grad,
                step_size,
                energy_fn,
                mesh.vertex_ids,
                max_iter=self.max_iter,
                beta=self.beta,
                c=self.c,
                gamma=self.gamma,
                alpha_max_factor=self.alpha_max_factor,
                constraint_enforcer=constraint_enforcer,
            )
        else:
            direction: Dict[int, np.ndarray] = {}
            for i, vid in enumerate(vids):
                direction[vid] = direction_vec[3 * i : 3 * i + 3]

            success, new_step, accepted_energy = backtracking_line_search(
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
            self._prev_x = x
            self._prev_grad = g
            self._prev_vids = vids

        return success, new_step, float(accepted_energy)

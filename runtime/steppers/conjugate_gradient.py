# modules/steppers/conjugate_gradient.py
from __future__ import annotations

import numpy as np
from typing import Callable, Dict, Optional

from geometry.entities import Mesh
from runtime.steppers.line_search import (
    backtracking_line_search,
    strong_wolfe_line_search,
    adaptive_line_search
)
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


class EnhancedConjugateGradient(BaseStepper):
    """Enhanced conjugate gradient with improved line search algorithms."""

    def __init__(
        self,
        restart_interval: int = 10,
        precondition: bool = False,
        line_search: str = "adaptive",
        max_iter: int = 15,
        c1: float = 1e-4,
        c2: float = 0.1,  # Lower c2 for CG
        beta_formula: str = "polak_ribiere",
        adaptive_restart: bool = True,
    ) -> None:
        """
        Initialize enhanced conjugate gradient.
        
        Parameters
        ----------
        restart_interval : int
            Fixed restart interval (used if adaptive_restart=False)
        precondition : bool
            Whether to precondition gradients
        line_search : str
            Line search algorithm: "adaptive", "strong_wolfe", "backtrack", or "simple"
        max_iter : int
            Maximum line search iterations
        c1, c2 : float
            Wolfe condition parameters (c2 should be lower for CG)
        beta_formula : str
            Beta computation: "polak_ribiere", "fletcher_reeves", or "hestenes_stiefel"
        adaptive_restart : bool
            Whether to use adaptive restart criteria
        """
        self.prev_grad: Dict[int, np.ndarray] = {}
        self.prev_dir: Dict[int, np.ndarray] = {}
        self.restart_interval = restart_interval
        self.iter_count = 0
        self.precondition = precondition
        self.line_search = line_search
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.beta_formula = beta_formula
        self.adaptive_restart = adaptive_restart

    def reset(self):
        self.prev_grad.clear()
        self.prev_dir.clear()
        self.iter_count = 0

    def _compute_beta(self, g: np.ndarray, prev_g: np.ndarray, prev_d: np.ndarray) -> float:
        """Compute beta coefficient using different formulas."""
        g_norm_sq = np.dot(g, g)
        prev_g_norm_sq = np.dot(prev_g, prev_g)
        
        if prev_g_norm_sq < 1e-20:
            return 0.0
        
        if self.beta_formula == "fletcher_reeves":
            return g_norm_sq / prev_g_norm_sq
        elif self.beta_formula == "hestenes_stiefel":
            y = g - prev_g
            numerator = np.dot(g, y)
            denominator = np.dot(prev_d, y)
            return numerator / denominator if abs(denominator) > 1e-20 else 0.0
        else:  # "polak_ribiere" (default)
            return np.dot(g, g - prev_g) / prev_g_norm_sq

    def _should_restart(self, g: np.ndarray, prev_g: np.ndarray, direction: np.ndarray) -> bool:
        """Determine if CG should restart."""
        if not self.adaptive_restart:
            return self.iter_count % self.restart_interval == 0
        
        # Restart if we've lost conjugacy (Powell restart criterion)
        if np.abs(np.dot(g, prev_g)) > 0.2 * np.dot(g, g):
            return True
        
        # Restart if direction is not sufficiently downhill
        if np.dot(g, direction) > -0.1 * np.dot(g, g):
            return True
        
        # Fixed interval restart as backup
        return self.iter_count % self.restart_interval == 0

    def step(
        self,
        mesh: Mesh,
        grad: Dict[int, np.ndarray],
        step_size: float,
        energy_fn: Callable[[], float],
        gradient_fn: Optional[Callable[[], Dict[int, np.ndarray]]] = None,
    ) -> tuple[bool, float]:
        """Take one enhanced conjugate gradient step."""

        direction: Dict[int, np.ndarray] = {}

        for vidx, vertex in mesh.vertices.items():
            if getattr(vertex, "fixed", False):
                continue

            g = grad[vidx]

            if self.precondition:
                g = g / (np.linalg.norm(g) + 1e-8)

            # Determine if we should restart
            if (vidx not in self.prev_grad or 
                self._should_restart(g, self.prev_grad.get(vidx, g), 
                                   self.prev_dir.get(vidx, -g))):
                d = -g
            else:
                prev_g = self.prev_grad[vidx]
                prev_d = self.prev_dir[vidx]
                
                beta = self._compute_beta(g, prev_g, prev_d)
                
                # Ensure beta is non-negative (restart if negative)
                if beta < 0:
                    d = -g
                else:
                    d = -g + beta * prev_d

            # Normalize direction
            d /= np.linalg.norm(d) + 1e-12
            direction[vidx] = d

        # Use enhanced line search
        if self.line_search == "adaptive":
            success, new_step = adaptive_line_search(
                mesh, direction, grad, step_size, energy_fn, gradient_fn
            )
        elif self.line_search == "strong_wolfe":
            success, new_step = strong_wolfe_line_search(
                mesh, direction, grad, step_size, energy_fn, gradient_fn,
                c1=self.c1, c2=self.c2, max_iter=self.max_iter
            )
        elif self.line_search == "backtrack":
            success, new_step = adaptive_line_search(
                mesh, direction, grad, step_size, energy_fn, gradient_fn, "backtrack"
            )
        else:  # "simple"
            success, new_step = backtracking_line_search(
                mesh, direction, grad, step_size, energy_fn,
                max_iter=self.max_iter, c=self.c1
            )

        if success:
            for vidx, d in direction.items():
                self.prev_grad[vidx] = grad[vidx].copy()
                self.prev_dir[vidx] = d.copy()
            self.iter_count += 1
        else:
            # Reset on failure to avoid accumulating bad directions
            self.reset()

        return success, new_step


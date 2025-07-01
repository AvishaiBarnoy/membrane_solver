# modules/steppers/base.py
"""Abstract base class for optimization steppers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Dict

import numpy as np

class BaseStepper(ABC):
    """Base interface for classes performing optimization steps."""

    @abstractmethod
    def step(
        self,
        mesh,
        grad: Dict[int, np.ndarray],
        step_size: float,
        energy_fn: Callable[[], float],
    ) -> tuple[bool, float]:
        """Advance ``mesh`` along ``grad`` with a given ``step_size``.

        Parameters
        ----------
        mesh : Mesh
            The mesh being optimized.
        grad : Dict[int, np.ndarray]
            Gradient for each vertex index.
        step_size : float
            Proposed step size.
        energy_fn : Callable[[], float]
            Function returning the current energy of ``mesh``.

        Returns
        -------
        tuple[bool, float]
            A flag indicating if the step was accepted and the updated step
            size to use on the next iteration.
        """

    def __repr__(self) -> str:  # pragma: no cover - simple utility
        params = ", ".join(f"{k}={v!r}" for k, v in vars(self).items())
        return f"{self.__class__.__name__}({params})"


# modules/steppers/base.py
from abc import ABC, abstractmethod
import numpy as np

class BaseStepper(ABC):
    @abstractmethod
    def step(self, mesh, grad: dict[int, np.ndarray], step_size: float):
        """
        Perform one update of mesh.vertices based on grad and step_size.
        Update mesh.vertices in place, moving each free vertex
        according to its gradient and step_size.
        """
        ...

    def __repr__(self): ...

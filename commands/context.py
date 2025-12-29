from dataclasses import dataclass, field

from geometry.entities import Mesh
from runtime.minimizer import Minimizer
from runtime.steppers.base import BaseStepper


@dataclass
class CommandContext:
    """Holds the shared state for the simulation session."""

    mesh: Mesh
    minimizer: Minimizer
    stepper: BaseStepper
    should_exit: bool = False
    history: list[str] = field(default_factory=list)

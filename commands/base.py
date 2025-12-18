from abc import ABC, abstractmethod
from typing import List


class Command(ABC):
    """Abstract base class for all simulation commands."""

    @abstractmethod
    def execute(self, context, args: List[str]) -> None:
        """
        Execute the command.

        Args:
            context: The CommandContext object holding shared state.
            args: A list of string arguments passed to the command.
        """
        pass

"""Custom exception types for the membrane solver."""

from __future__ import annotations


class MembraneSolverError(Exception):
    """Base class for domain-specific errors."""


class InvalidEdgeIndexError(MembraneSolverError):
    """Raised when attempting to access an edge with an invalid index."""

    def __init__(self, index: int, message: str | None = None) -> None:
        if message is None:
            message = (
                f"Edge index {index} is invalid. "
                "Edge IDs are 1-based; use negative values only for orientation."
            )
        super().__init__(message)
        self.index = index


__all__ = ["MembraneSolverError", "InvalidEdgeIndexError"]

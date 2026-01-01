"""Custom exception types for the membrane solver."""

from __future__ import annotations

from typing import Any


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


class BodyOrientationError(MembraneSolverError):
    """Raised when facets belonging to a body are not oriented consistently."""

    def __init__(
        self,
        message: str,
        *,
        body_index: int | None = None,
        edge_index: int | None = None,
        facet_indices: tuple[int, int] | None = None,
        mesh: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.body_index = body_index
        self.edge_index = edge_index
        self.facet_indices = facet_indices
        self.mesh = mesh


__all__ = ["MembraneSolverError", "InvalidEdgeIndexError", "BodyOrientationError"]

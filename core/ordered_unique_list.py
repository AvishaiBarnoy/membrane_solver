from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar, overload

T = TypeVar("T")


class OrderedUniqueList(list[T]):
    """List that preserves insertion order while preventing duplicates.

    This is used for module-name lists that historically behaved like sets in
    parts of the codebase (via ``add``/``update``), but where deterministic
    ordering is important for reproducibility.
    """

    def __init__(self, iterable: Iterable[T] | None = None):
        super().__init__()
        if iterable is not None:
            self.update(iterable)

    def add(self, item: T) -> None:
        """Set-like alias for ``append`` (no duplicates)."""
        self.append(item)

    def update(self, items: Iterable[T]) -> None:
        """Set-like bulk add (no duplicates, preserves first-seen order)."""
        for item in items:
            self.append(item)

    def append(self, item: T) -> None:  # type: ignore[override]
        if item in self:
            return
        super().append(item)

    def extend(self, items: Iterable[T]) -> None:  # type: ignore[override]
        for item in items:
            self.append(item)

    def insert(self, index: int, item: T) -> None:  # type: ignore[override]
        if item in self:
            return
        super().insert(index, item)

    @overload
    def __getitem__(self, index: int) -> T: ...

    @overload
    def __getitem__(self, index: slice) -> "OrderedUniqueList[T]": ...

    def __getitem__(self, index):  # type: ignore[override]
        out = super().__getitem__(index)
        if isinstance(index, slice):
            return OrderedUniqueList(out)
        return out

    def copy(self) -> "OrderedUniqueList[T]":
        return OrderedUniqueList(self)

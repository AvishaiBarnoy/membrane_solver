"""Pytest configuration and test categorization.

We keep the codebase's existing flat `tests/` layout, but categorize tests into
`unit`, `regression`, and `e2e` via markers so CI can run targeted subsets.
"""

from __future__ import annotations

import pathlib

import pytest


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-apply test category markers based on filename conventions."""
    for item in items:
        path = pathlib.Path(str(item.fspath))
        name = path.name.lower()

        if "benchmark" in name:
            item.add_marker(pytest.mark.benchmark)
            continue

        if "e2e" in name or "end_to_end" in name:
            item.add_marker(pytest.mark.e2e)
            continue

        if "regression" in name:
            item.add_marker(pytest.mark.regression)
            continue

        item.add_marker(pytest.mark.unit)

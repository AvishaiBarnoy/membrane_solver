"""Pytest configuration and test categorization.

We keep the codebase's existing flat `tests/` layout, but categorize tests into
`unit`, `regression`, and `e2e` via markers so CI can run targeted subsets.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tools.diagnostics.free_disk_profile_protocol import (
    run_free_disk_curved_bilayer_protocol,
    run_free_disk_two_stage_profile_protocol,
)


@pytest.fixture(scope="session")
def canonical_curved_protocol_result():
    """Cache the canonical curved shared-rim protocol run across test files."""
    return run_free_disk_curved_bilayer_protocol()


@pytest.fixture(scope="session")
def canonical_profile_protocol_result():
    """Cache the two-stage profile protocol run across test files."""
    return run_free_disk_two_stage_profile_protocol()


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

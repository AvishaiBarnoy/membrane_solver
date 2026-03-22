from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import pytest_collection_modifyitems


class _FakeMarker:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeItem:
    def __init__(self, path: str, marks: list[str] | None = None) -> None:
        self.fspath = path
        self._marks = [_FakeMarker(name) for name in (marks or [])]

    def iter_markers(self):
        return iter(self._marks)

    def add_marker(self, mark) -> None:
        self._marks.append(_FakeMarker(mark.name))

    @property
    def mark_names(self) -> set[str]:
        return {mark.name for mark in self._marks}


@pytest.mark.unit
def test_collection_hook_respects_explicit_regression_marker_on_e2e_file() -> None:
    item = _FakeItem(
        str(Path("tests") / "test_kozlov_free_disk_thetaB_convergence_e2e.py"),
        marks=["regression"],
    )

    pytest_collection_modifyitems(config=None, items=[item])

    assert item.mark_names == {"regression"}


@pytest.mark.unit
def test_collection_hook_respects_explicit_e2e_marker_on_regression_file() -> None:
    item = _FakeItem(
        str(
            Path("tests") / "test_kozlov_free_disk_outer_leaflet_coupling_regression.py"
        ),
        marks=["e2e"],
    )

    pytest_collection_modifyitems(config=None, items=[item])

    assert item.mark_names == {"e2e"}


@pytest.mark.unit
def test_collection_hook_still_applies_filename_based_marker_when_missing() -> None:
    item = _FakeItem(str(Path("tests") / "test_kozlov_annulus_milestone_c_e2e.py"))

    pytest_collection_modifyitems(config=None, items=[item])

    assert item.mark_names == {"e2e"}

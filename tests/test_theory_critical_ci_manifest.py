from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "manifest" / "theory_critical_pr.txt"
WORKFLOW = ROOT / ".github" / "workflows" / "CI.yml"

pytestmark = pytest.mark.unit


def _critical_nodeids() -> list[str]:
    return [
        line.strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_theory_critical_manifest_references_existing_tests() -> None:
    nodeids = _critical_nodeids()
    assert nodeids
    assert len(nodeids) == len(set(nodeids))
    for nodeid in nodeids:
        path_text, separator, test_name = nodeid.partition("::")
        assert separator == "::"
        path = ROOT / path_text
        assert path.is_file(), nodeid
        assert f"def {test_name}(" in path.read_text(encoding="utf-8"), nodeid


def test_theory_critical_manifest_is_a_blocking_pr_job() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("  theory-critical:")
    end = workflow.index("\n  regression-flat-disk:", start)
    job = workflow[start:end]

    assert "if: github.event_name == 'pull_request'" in job
    assert "continue-on-error" not in job
    assert "|| true" not in job
    assert "-o addopts=''" in job
    assert "tests/manifest/theory_critical_pr.txt" in job

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = (
    ROOT / "tests" / "manifest" / "theory_critical_audits_pr.txt",
    ROOT / "tests" / "manifest" / "theory_critical_protocols_pr.txt",
)
WORKFLOW = ROOT / ".github" / "workflows" / "CI.yml"

pytestmark = pytest.mark.unit


def _critical_nodeids() -> list[str]:
    return [
        line.strip()
        for manifest in MANIFESTS
        for line in manifest.read_text(encoding="utf-8").splitlines()
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


def test_theory_critical_manifests_are_blocking_pr_jobs() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    start = workflow.index("  theory-critical:")
    job = workflow[start : workflow.index("\n  regression-flat-disk:", start)]

    assert "if: github.event_name == 'pull_request'" in job
    assert "continue-on-error" not in job
    assert "-o addopts=''" in job
    assert 'wait "$pid" || status=1' in job
    for manifest in MANIFESTS:
        assert manifest.relative_to(ROOT).as_posix() in job

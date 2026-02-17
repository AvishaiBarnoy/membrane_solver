import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.sweep_theory_protocol_suffixes import rank_candidates

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "sweep_theory_protocol_suffixes.py"


def test_rank_candidates_is_deterministic() -> None:
    entries = [
        {
            "label": "b",
            "score": 0.2,
            "runtime_s": 4.0,
            "checks": {"valid": True},
        },
        {
            "label": "a",
            "score": 0.2,
            "runtime_s": 4.0,
            "checks": {"valid": True},
        },
        {
            "label": "bad",
            "score": 0.01,
            "runtime_s": 0.1,
            "checks": {"valid": False},
        },
    ]
    ranked = rank_candidates(entries)
    assert [x["label"] for x in ranked] == ["a", "b", "bad"]


@pytest.mark.acceptance
def test_suffix_sweep_writes_ranked_yaml_schema(tmp_path) -> None:
    out_yaml = tmp_path / "theory_protocol_sweep.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out",
            str(out_yaml),
            "--repeat",
            "1",
            "--candidate",
            "base:",
            "--candidate",
            "g1:g1",
        ],
        check=True,
        cwd=str(ROOT),
    )
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert report["meta"]["format"] == "yaml"
    assert int(report["summary"]["candidate_count"]) == 2
    assert isinstance(report["summary"]["best_candidate"], str)
    assert len(report["candidates"]) == 2
    scores = [float(c["score"]) for c in report["candidates"]]
    assert scores == sorted(scores)
    for row in report["candidates"]:
        assert set(row["ratios"].keys()) == {
            "theta_ratio",
            "elastic_ratio",
            "contact_ratio",
            "total_ratio",
        }

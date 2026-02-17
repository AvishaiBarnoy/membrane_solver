import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.audit_theory_coverage import evaluate_manifest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "audit_theory_coverage.py"
MANIFEST = ROOT / "tests" / "fixtures" / "theory_coverage_manifest.yaml"
FIXTURE = ROOT / "tests" / "fixtures" / "kozlov_1disk_3d_free_disk_theory_parity.yaml"
TEX = ROOT / "docs" / "tex" / "1_disk_3d.tex"


@pytest.mark.acceptance
def test_theory_coverage_audit_reports_all_required_present(tmp_path) -> None:
    out_yaml = tmp_path / "theory_coverage_audit.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--out",
            str(out_yaml),
            "--fail-on-required-not-present",
        ],
        check=True,
        cwd=str(ROOT),
    )
    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert report["meta"]["format"] == "yaml"
    assert int(report["summary"]["required_not_present_count"]) == 0


def test_evaluate_manifest_classifies_missing_when_checks_fail() -> None:
    manifest = {
        "items": [
            {
                "id": "x",
                "required": True,
                "theory_statement": "dummy",
                "tex_markers": ["NOT_PRESENT_TOKEN"],
                "code_refs": [{"path": "tools/audit_theory_coverage.py"}],
                "yaml_checks": [
                    {
                        "kind": "path_exists",
                        "path": "global_parameters.missing_key",
                    }
                ],
            }
        ]
    }
    geom = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    tex = TEX.read_text(encoding="utf-8")
    out = evaluate_manifest(root=ROOT, tex_text=tex, geometry=geom, manifest=manifest)
    assert out["summary"]["required_not_present_count"] == 1
    assert out["items"][0]["status"] in {"partial", "missing"}

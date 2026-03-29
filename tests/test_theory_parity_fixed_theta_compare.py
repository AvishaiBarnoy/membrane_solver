import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "theory_parity_fixed_theta_compare.py"


@pytest.mark.regression
def test_fixed_theta_compare_shows_contact_stability_and_profiled_elastic_collapse(
    tmp_path,
) -> None:
    out_yaml = tmp_path / "fixed_theta_compare.yaml"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--theta",
            "0.185",
            "--profile",
            "coarse",
            "--profile",
            "i50",
            "--profile",
            "near_edge_v1",
            "--out",
            str(out_yaml),
        ],
        check=True,
        cwd=str(ROOT),
    )

    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    rows = {row["label"]: row for row in report["rows"]}

    assert report["meta"]["theta_value"] == pytest.approx(0.185)
    assert list(rows) == ["coarse", "i50", "near_edge_v1"]

    coarse = rows["coarse"]
    i50 = rows["i50"]
    near_edge = rows["near_edge_v1"]

    coarse_contact = float(coarse["reduced_terms"]["contact_measured"])
    i50_contact = float(i50["reduced_terms"]["contact_measured"])
    near_edge_contact = float(near_edge["reduced_terms"]["contact_measured"])

    assert i50_contact == pytest.approx(coarse_contact, abs=1.0e-9)
    assert near_edge_contact == pytest.approx(coarse_contact, abs=1.0e-9)

    coarse_elastic = float(coarse["reduced_terms"]["elastic_measured"])
    i50_elastic = float(i50["reduced_terms"]["elastic_measured"])
    near_edge_elastic = float(near_edge["reduced_terms"]["elastic_measured"])

    assert coarse_elastic > i50_elastic > near_edge_elastic
    assert coarse_elastic > 1.0

    coarse_total = float(coarse["reduced_terms"]["total_measured"])
    i50_total = float(i50["reduced_terms"]["total_measured"])
    near_edge_total = float(near_edge["reduced_terms"]["total_measured"])

    assert coarse_total > i50_total > near_edge_total

    coarse_outer = float(coarse["outer_shell_geometry"]["outer_radius"])
    i50_outer = float(i50["outer_shell_geometry"]["outer_radius"])
    near_edge_outer = float(near_edge["outer_shell_geometry"]["outer_radius"])

    assert coarse_outer > i50_outer > near_edge_outer

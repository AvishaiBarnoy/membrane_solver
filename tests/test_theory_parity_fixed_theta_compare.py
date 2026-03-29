import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "theory_parity_fixed_theta_compare.py"
TAIL_TIE_TOL = 5.0e-4


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
            "default_lo",
            "--profile",
            "default",
            "--profile",
            "default_hi",
            "--out",
            str(out_yaml),
        ],
        check=True,
        cwd=str(ROOT),
    )

    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    rows = {row["label"]: row for row in report["rows"]}

    assert report["meta"]["theta_value"] == pytest.approx(0.185)
    assert list(rows) == [
        "coarse",
        "default_lo",
        "default",
        "default_hi",
    ]

    coarse = rows["coarse"]
    family_lo = rows["default_lo"]
    primary = rows["default"]
    family_hi = rows["default_hi"]

    coarse_contact = float(coarse["reduced_terms"]["contact_measured"])
    lo_contact = float(family_lo["reduced_terms"]["contact_measured"])
    primary_contact = float(primary["reduced_terms"]["contact_measured"])
    hi_contact = float(family_hi["reduced_terms"]["contact_measured"])

    assert lo_contact == pytest.approx(coarse_contact, abs=1.0e-9)
    assert primary_contact == pytest.approx(coarse_contact, abs=1.0e-9)
    assert hi_contact == pytest.approx(coarse_contact, abs=1.0e-9)

    coarse_elastic = float(coarse["reduced_terms"]["elastic_measured"])
    lo_elastic = float(family_lo["reduced_terms"]["elastic_measured"])
    primary_elastic = float(primary["reduced_terms"]["elastic_measured"])
    hi_elastic = float(family_hi["reduced_terms"]["elastic_measured"])

    assert coarse_elastic > lo_elastic > primary_elastic
    assert hi_elastic <= primary_elastic + TAIL_TIE_TOL
    assert coarse_elastic > 1.0

    coarse_total = float(coarse["reduced_terms"]["total_measured"])
    lo_total = float(family_lo["reduced_terms"]["total_measured"])
    primary_total = float(primary["reduced_terms"]["total_measured"])
    hi_total = float(family_hi["reduced_terms"]["total_measured"])

    assert coarse_total > lo_total > primary_total
    assert hi_total <= primary_total + TAIL_TIE_TOL

    coarse_outer = float(coarse["outer_shell_geometry"]["outer_radius"])
    lo_outer = float(family_lo["outer_shell_geometry"]["outer_radius"])
    primary_outer = float(primary["outer_shell_geometry"]["outer_radius"])
    hi_outer = float(family_hi["outer_shell_geometry"]["outer_radius"])

    assert coarse_outer > lo_outer > primary_outer > hi_outer

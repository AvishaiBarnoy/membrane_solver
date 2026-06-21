from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

from tools.diagnostics.parity_broad_diagnostic import (
    render_markdown_report,
    run_diagnostic,
)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "diagnostics" / "parity_broad_diagnostic.py"


def test_parity_broad_diagnostic_schema_lists_sections_and_variants() -> None:
    report = run_diagnostic(protocol=("g1",), mode="schema")

    assert report["meta"]["mode"] == "schema"
    assert report["variants"] == [
        "ghost",
        "default_current",
        "full_coupling_trace",
        "full_coupling",
        "default_no_outer_absence",
    ]
    assert report["sections"] == [
        "optimized_cases",
        "fixed_theta_cases",
        "comparison_matrix",
        "full_physics_lane_matrix",
        "observations",
    ]


def test_parity_broad_diagnostic_markdown_renders_matrix() -> None:
    report = {
        "meta": {"mode": "run", "protocol": ["g1"], "theta_values": [0.21, 0.30]},
        "comparison_matrix": [
            {
                "label": "default_current",
                "optimized_thetaB": 0.21,
                "optimized_tex_total_ratio": 1.5,
                "fixed_elastic_A_ratio": 0.25,
                "fixed_contact_B_ratio": 1.0,
                "tilt_out_quadratic": 5.0,
                "bending_tilt_out_quadratic": 0.3,
            }
        ],
        "full_physics_lane_matrix": [
            {
                "label": "full_coupling",
                "model_intent": "full_physics_candidate",
                "reference_mode": "current_geometry",
                "thetaB_value": 0.22,
                "tex_total_ratio": 1.1,
                "tex_elastic_ratio": 0.95,
                "tex_contact_ratio": 1.0,
                "base_energy_out": 2.0,
                "div_energy_out": 1.0,
                "cross_energy_out": 0.5,
                "direct_t_out": 0.08,
                "direct_phi": 0.03,
            }
        ],
        "observations": ["Current default still underperforms the no-absence control."],
        "optimized_cases": [
            {
                "label": "default_current",
                "thetaB_value": 0.21,
                "tex_total_ratio": 1.5,
                "thetaB_scan_count": 20,
                "energy_breakdown": {
                    "tilt_out": 0.05,
                    "bending_tilt_out": 0.002,
                },
                "outer_leaflet_participation": {
                    "shell_rows": {
                        "disk": {"absent": 24, "count": 24},
                        "rim": {"absent": 0, "count": 12},
                        "outer": {"absent": 0, "count": 12},
                    }
                },
            }
        ],
    }

    text = render_markdown_report(report)

    assert "# Parity Broad Diagnostic" in text
    assert "| Variant | thetaB | tex ratio |" in text
    assert "## Full-Physics Lane Matrix" in text
    assert "`default_current`" in text
    assert "Current default still underperforms" in text


def test_parity_broad_diagnostic_cli_schema_writes_yaml_and_markdown(tmp_path) -> None:
    out_yaml = tmp_path / "broad.yaml"
    out_md = tmp_path / "broad.md"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "schema",
            "--protocol",
            "g1",
            "--out",
            str(out_yaml),
            "--report-out",
            str(out_md),
        ],
        check=True,
        cwd=str(ROOT),
    )

    report = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert report["meta"]["mode"] == "schema"
    assert out_md.read_text(encoding="utf-8").startswith("# Parity Broad Diagnostic")

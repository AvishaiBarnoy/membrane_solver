import pytest

from tools.diagnostics.curved_1disk_first_two_shell_diveval_audit import (
    run_curved_1disk_first_two_shell_diveval_audit,
)


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.exhaustive
def test_curved_1disk_first_two_shell_diveval_reports_sign_comparison() -> None:
    """The coarse lane should report where the leaflet sign paths diverge."""
    report = run_curved_1disk_first_two_shell_diveval_audit()

    assert report["lane_signature"]["rim_slope_match_mode"] == "shared_rim_staggered_v1"
    assert report["lane_signature"]["bending_tilt_base_term_boundary_group_in"] == "rim"
    assert (
        report["lane_signature"]["bending_tilt_base_term_boundary_group_out"] == "rim"
    )
    assert len(report["shells"]) == 2

    for shell in report["shells"]:
        assert set(shell["subexpression_deltas"]) == {
            "div_raw_sign_matches",
            "div_signed_sign_matches",
            "div_term_sign_matches",
            "div_eval_sign_matches",
        }
        assert all(
            isinstance(value, bool) for value in shell["subexpression_deltas"].values()
        )

    assert report["first_offending_subexpression"]["call"] in {
        "sign convention application",
        "boundary-conditioned div_term branch",
        "post-div_term div_eval branch",
        "combined local expression",
    }

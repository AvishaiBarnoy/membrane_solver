import pytest

from tools.diagnostics.curved_1disk_first_two_shell_diveval_audit import (
    run_curved_1disk_first_two_shell_diveval_audit,
)


@pytest.mark.benchmark
@pytest.mark.slow
def test_curved_1disk_first_two_shell_diveval_signs_match_after_lane_fix() -> None:
    """The curved free-disk lane should no longer flip only the inner div_eval sign."""
    report = run_curved_1disk_first_two_shell_diveval_audit()

    assert report["lane_signature"]["rim_slope_match_mode"] == "shared_rim_staggered_v1"
    assert report["lane_signature"]["bending_tilt_base_term_boundary_group_in"] == "rim"
    assert (
        report["lane_signature"]["bending_tilt_base_term_boundary_group_out"] == "rim"
    )
    assert len(report["shells"]) == 2

    for shell in report["shells"]:
        assert shell["subexpression_deltas"]["div_raw_sign_matches"] is True
        assert shell["subexpression_deltas"]["div_signed_sign_matches"] is True
        assert shell["subexpression_deltas"]["div_term_sign_matches"] is True
        assert shell["subexpression_deltas"]["div_eval_sign_matches"] is True

    assert (
        report["first_offending_subexpression"]["call"] != "sign convention application"
    )

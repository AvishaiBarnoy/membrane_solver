import pytest

from tools.diagnostics.curved_1disk_first_two_shell_ingredient_audit import (
    run_curved_1disk_first_two_shell_ingredient_audit,
)


@pytest.mark.benchmark
@pytest.mark.slow
def test_curved_1disk_first_outer_shell_restores_inner_outer_base_term_parity() -> None:
    """The first free outer shell should not zero only the inner leaflet base term."""
    report = run_curved_1disk_first_two_shell_ingredient_audit()
    first_shell = report["shellwise_comparison"][0]
    base_in = float(first_shell["in"]["base_term_median"])
    base_out = float(first_shell["out"]["base_term_median"])

    assert base_in > 0.0
    assert base_in == pytest.approx(base_out, rel=1.0e-6, abs=1.0e-9)

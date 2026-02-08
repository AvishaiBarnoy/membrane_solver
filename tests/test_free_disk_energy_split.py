import copy
import os
import sys

import pytest

sys.path.insert(0, os.getcwd())

from geometry.geom_io import load_data
from tools.diagnostics.free_disk_energy_split import compute_energy_split


def test_free_disk_energy_split_matches_global_total():
    base = load_data("tests/fixtures/kozlov_free_disk_coarse_refinable.yaml")
    out = copy.deepcopy(base)

    split = compute_energy_split(base, out)
    breakdown = split["global_breakdown"]

    assert split["split_total"] == pytest.approx(
        split["global_total"],
        rel=1e-6,
        abs=1e-9,
    )

    assert (split["tilt_in_disk"] + split["tilt_in_outer"]) == pytest.approx(
        breakdown["tilt_in"], rel=1e-6, abs=1e-9
    )
    assert (
        split["bending_tilt_in_disk"] + split["bending_tilt_in_outer"]
    ) == pytest.approx(breakdown["bending_tilt_in"], rel=1e-6, abs=1e-9)

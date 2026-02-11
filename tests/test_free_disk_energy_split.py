import math

import numpy as np

from tools.diagnostics.free_disk_energy_split import compute_energy_split


def test_energy_split_uses_saved_options_without_preset_defaults():
    base = {
        "definitions": {"rim": {"rim_slope_match_group": "rim"}},
        "global_parameters": {
            "tilt_thetaB_group_in": "rim",
            "tilt_thetaB_contact_strength_in": 2.0,
            "tilt_thetaB_value": 0.5,
        },
        "energy_modules": [],
        "constraint_modules": [],
    }
    # Output has preset=rim on three vertices, but only two explicitly carry
    # rim_slope_match_group. Defaults must NOT be re-applied on load.
    out = {
        "vertices": [
            [1.0, 0.0, 0.0, {"preset": "rim", "rim_slope_match_group": "rim"}],
            [0.0, 2.0, 0.0, {"preset": "rim", "rim_slope_match_group": "rim"}],
            [0.0, 10.0, 0.0, {"preset": "rim"}],
        ],
        "edges": [[0, 1], [1, 2], [2, 0]],
        "faces": [[0, 1, 2]],
    }

    split = compute_energy_split(base, out)
    # Expected R_eff from only the two explicit rim vertices at r=1 and r=2.
    r_eff = 1.5
    expected_contact = -2.0 * math.pi * r_eff * 2.0 * 0.5
    assert np.isclose(split["contact"], expected_contact)

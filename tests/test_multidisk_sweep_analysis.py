import json
from pathlib import Path

import numpy as np

from membrane_solver.analysis.multidisk_sweep import analyze_mesh


def _two_patch_square_mesh_json() -> dict:
    # Square split into two triangles along the diagonal (0-2).
    # Faces are specified by edge indices (0-based) with optional "rN" reversal.
    return {
        "vertices": [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "edges": [
            [0, 1],
            [1, 2],
            [2, 3],
            [3, 0],
            [0, 2],
        ],
        "faces": [
            [0, 1, "r4", {"disk_patch": "top"}],
            [4, 2, 3, {"disk_patch": "bottom"}],
        ],
        "global_parameters": {
            "surface_tension": 1.0,
            "volume_constraint_mode": "none",
        },
        "instructions": [],
    }


def test_analyze_mesh_computes_energy_and_separation(tmp_path: Path):
    mesh_path = tmp_path / "case_L0.json"
    mesh_path.write_text(json.dumps(_two_patch_square_mesh_json()))

    result = analyze_mesh(
        mesh_path,
        patch_key="disk_patch",
        pair=None,  # auto-detect two patch labels
        separation="chord",
        sphere_center=np.zeros(3),
        sphere_radius=None,
        include_boundary_diagnostics=False,
    )

    metrics = result.metrics
    assert metrics["patch0"] in {"top", "bottom"}
    assert metrics["patch1"] in {"top", "bottom"}

    assert metrics["E_total"] == 1.0
    assert np.isclose(metrics["L"], np.sqrt(2.0) / 3.0)

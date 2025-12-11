import json

SAMPLE_GEOMETRY = {
    "vertices": [
        [0, 0, 0],
        [1, 0, 0],
        [1, 0, 1],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
    ],
    "edges": [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 5],
        [1, 6],
        [2, 7],
        [3, 4],
    ],
    "faces": [
        [0, 1, 2, 3],
        ["r0", 8, 5, "r9"],
        [9, 6, -10, -1],
        [-2, 10, 7, -11],
        [11, 4, -8, -3],
        [-5, -4, -7, -6],
    ],
    "bodies": {
        "faces": [[0, 1, 2, 3, 4, 5]],
        "target_volume": [1.0],
    },
    "global_parameters": {
        "surface_tension": 1.0,
        "intrinsic_curvature": 0.0,
        "bending_modulus": 0.0,
        "gaussian_modulus": 0.0,
        "volume_stiffness": 1e3,
        "volume_constraint_mode": "lagrange",
    },
    "instructions": [],
}


def write_sample_geometry(tmp_path, name="sample_geometry.json", data=None):
    """Write SAMPLE_GEOMETRY (or provided data) to tmp_path/name."""
    path = tmp_path / name
    with open(path, "w") as f:
        json.dump(data or SAMPLE_GEOMETRY, f)
    return str(path)


def cube_soft_volume_input(volume_mode: str = "penalty") -> dict:
    """Return a deep copy of the cube sample with requested volume mode."""
    import copy

    data = copy.deepcopy(SAMPLE_GEOMETRY)
    data.setdefault("global_parameters", {})
    data["global_parameters"].update(
        {
            "surface_tension": 1.0,
            "volume_constraint_mode": volume_mode,
            "volume_projection_during_minimization": True,
        }
    )
    return data

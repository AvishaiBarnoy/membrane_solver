import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from geometry.geom_io import parse_geometry
from visualization.plotting import update_live_vis


def test_live_vis_does_not_shift_axes_on_topology_redraw() -> None:
    """Live-vis should preserve axes placement when topology changes (e.g. refinement)."""
    data = {
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
            [0, 1, 4],
            [4, 2, 3],
        ],
        "global_parameters": {"surface_tension": 0.0},
        "instructions": [],
    }
    mesh = parse_geometry(data)

    state = update_live_vis(mesh, state=None, title="Step 0", color_by="tilt_mag")
    ax = state["ax"]
    pos0 = tuple(float(v) for v in ax.get_position().bounds)

    mesh._topology_version = getattr(mesh, "_topology_version", 0) + 1
    state = update_live_vis(mesh, state=state, title="Refine 1/1", color_by="tilt_mag")
    ax = state["ax"]
    pos1 = tuple(float(v) for v in ax.get_position().bounds)

    assert np.allclose(pos0, pos1, atol=1e-6)

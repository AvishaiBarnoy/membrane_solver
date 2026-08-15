from __future__ import annotations

import numpy as np

from geometry.entities import Mesh, Vertex


def _mesh() -> Mesh:
    mesh = Mesh()
    mesh.vertices = {0: Vertex(0, np.zeros(3))}
    mesh.build_position_cache()
    return mesh


def test_named_mutation_hooks_keep_geometry_topology_field_epochs_distinct():
    mesh = _mesh()
    start = (mesh._version, mesh._topology_version, mesh._tilts_version)

    mesh.touch_geometry()
    assert (mesh._version, mesh._topology_version, mesh._tilts_version) == (
        start[0] + 1,
        start[1],
        start[2],
    )
    mesh.touch_topology()
    mesh.touch_tilts()
    assert (mesh._version, mesh._topology_version, mesh._tilts_version) == (
        start[0] + 1,
        start[1] + 1,
        start[2] + 1,
    )


def test_fixed_flag_hooks_do_not_touch_geometry_epoch():
    mesh = _mesh()
    geometry_version = mesh._version
    mesh.touch_fixed_flags()
    mesh.touch_tilt_fixed_flags()
    assert mesh._version == geometry_version
    assert mesh._fixed_flags_version == 1
    assert mesh._tilt_fixed_flags_version == 1

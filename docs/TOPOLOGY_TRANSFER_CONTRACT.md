# Topology Transfer Contract

Status: **active — refinement inventory**
Owner: `runtime/refinement.py`

| State family | Current refinement rule | Primary coverage |
|---|---|---|
| Vertex position and tilts | midpoint/centroid interpolation | `tests/test_refinement.py`, `tests/test_tilt_validation.py` |
| Fixed and leaflet fixed flags | inherit only when all relevant parents are fixed | `tests/test_tilt_validation.py` |
| Vertex constraints | merge shared circle/plane constraints without changing precedence | `tests/test_refinement.py`, `tests/test_refinement_disk_interface_tag_propagation_regression.py` |
| Presets and disk tags | inherit only matching parent intent; apply existing preset definitions | `tests/test_refinement_preserves_presets.py` |
| Edge/facet/body options | copied from the relevant parent; child facets adjust target-area bookkeeping | `tests/test_refinement.py` |
| Interface/rim groups | propagate matching disk-interface and rigid-disk tags | disk-interface regression tests |
| Epochs and topology | rebuild loops and commit topology/cache state after construction | `tests/test_topology_invariants_regression.py` |

No extraction may merge option inheritance with topology construction or change
parent-precedence behavior. The first code slice, if needed, is one pure option
policy family with direct characterization tests.

# Diagnostic Report Schema Contract

Status: **complete — maintained compatibility contract**
Owners: `tools/reproduce_theory_parity.py` and `tools/diagnostics/`

Diagnostic tools may be refactored internally only when they preserve:

- stable top-level report sections such as `meta`, `metrics`, diagnostics, and
  lane-specific payloads;
- metric names, numeric units, tolerance interpretation, and fixture labels;
- YAML key insertion order (`sort_keys=False`) where artifacts are consumed;
- CLI output paths, exit status, and state-file semantics;
- observational behavior: diagnostics must not alter the simulation trajectory
  they measure.

Serialization or pure report formatting is the lowest-risk extraction boundary,
but only when the complete production-and-test cluster gets smaller or gains a
demonstrable shared contract. Protocol execution, metric collection, and
scientific comparisons require separate baseline artifact validation.

The Phase 5 broad helper extraction was rejected on total-size grounds. Future
diagnostic changes must continue to honor this contract; it does not authorize
protocol, metric, tolerance, or scientific-baseline changes.

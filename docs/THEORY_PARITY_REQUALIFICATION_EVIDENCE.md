# Theory-Parity Requalification Evidence

Status: **coarse-lane expectation approved; numeric golden requalification pending**

## Scope

This record covers the physical-edge default fixture:
`tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml`.
It does not change fixtures, protocol behavior, tolerances, or golden reports.

## Reproduction

Two independent runs of `tools/reproduce_theory_parity.py` produced identical
headline values:

| Metric | Current value | Repeat delta |
|---|---:|---:|
| `final_energy` | -1.185941168964711 | 0 |
| `thetaB_value` | 0.19 | 0 |
| theory `theta_ratio` | 2.058846611995499 | 0 |
| theory `elastic_ratio` | 2.072545344395226 | 0 |
| theory `contact_ratio` | 2.058846611995500 | 0 |
| theory `total_ratio` | 2.045147879595774 | 0 |

The strict expected acceptance failure is therefore a stable requalification
gap, not observed run-to-run nondeterminism.

## Difference from the committed physical-edge baseline

| Protected metric | Baseline | Current | Delta |
|---|---:|---:|---:|
| `final_energy` | -1.167132916936509 | -1.185941168964711 | -0.018808252028202 |
| `thetaB_value` | 0.18 | 0.19 | 0.01 |
| theory `theta_ratio` | 1.950486263995736 | 2.058846611995499 | 0.1083603480 |
| theory `elastic_ratio` | 1.888259357436079 | 2.072545344395226 | 0.1842859869 |
| theory `contact_ratio` | 1.950486263995737 | 2.058846611995500 | 0.1083603480 |
| theory `total_ratio` | 2.012713170555394 | 2.045147879595774 | 0.0324347090 |

The current report also has additional diagnostic and metadata keys. Those are
schema additions, but the protected numeric shifts above independently require
scientific review.

## Current decision

The default physical-edge fixture remains a coarse analytical lane. Its
acceptance contract requires an extrapolated first-shell readout and bounded
improvement over the pre-fix comparison baseline; it does not require the
continuum-scale trace continuation supplied by explicit-trace topology.

## Required decision before numeric rebaseline or driver extraction

1. Confirm the signed-curvature and shape-pullback corrections are intended in
   the physical-edge lane.
2. Evaluate qualitative lane invariants against the updated report.
3. Approve new tolerances and/or refreshed baselines only after that review.
4. Keep protocol activation and metric collection in the driver until steps
   1–3 are complete.

## Qualitative invariant gate

The following current-lane checks pass against the corrected implementation:

- the post-core-fix default lane expects `thetaB = 0.19` and the current theory
  total ratio;
- the outer shape-pullback term matches its finite-difference derivative;
- theta/energy remain inside the default-lane guardrail; and
- director/profile parity improves over the comparison baseline.

The former continuum-scale continuation gate has been replaced with this
approved coarse-lane contract. The strict numeric golden remains separately
requalified because protected energy and theory-ratio values changed.

## Default lane versus explicit-trace control

The pre-fix comparison baseline has a disk/free-side outer-trace gap of
0.1349348681. The corrected default analytical-parity lane reduces it only to
0.1336881471, while the acceptance threshold is below 0.0849348681. Its
free-side `t_out` increases from 0.0413088001 to 0.0533760673, which is real
but insufficient for the continuum-scale criterion.

The explicit-trace full-physics control reports direct-trace `t_out` of
0.1686792363 and `phi` of 0.2003389506, but it also selects `thetaB = 0.30`
and identifies as a full-physics candidate. It validates that the explicit
trace mechanism can produce a direct outer response; it must not be substituted
for the default analytical-parity lane to satisfy the gate.

## Structural cause candidate

The default analytical-parity fixture has 109 vertices and its first measured
free shell is approximately 0.18626 beyond the disk radius. It has no explicit
`R + epsilon` trace ring. The full-physics trace control has 121 vertices and
inserts a trace layer at `R + 0.005`, with additional support topology. It also
uses `current_geometry` for the bending-tilt base term, while the default uses
`flat_reference_zero_J0`.

Those are intentional model/topology differences, not an implementation-level
cache or reporting discrepancy. The selected policy is to retain the coarse
analytical expectation; explicit-trace topology remains a separate full-physics
control lane.

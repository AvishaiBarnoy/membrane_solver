# Theory-Parity Numeric Rebaseline Manifest

Status: **draft — no baseline or tolerance change authorized**

## Purpose

Specify the minimum evidence required to refresh the physical-edge default
numeric golden after the approved coarse analytical-lane acceptance contract.
This document is a handoff artifact, not approval to modify fixtures or
baselines.

## Candidate artifact

- Current fixture:
  `tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml`
- Existing golden:
  `tests/fixtures/theory_parity_physical_edge_default_baseline.yaml`
- Producer:
  `tools/reproduce_theory_parity.py`

## Confirmed deterministic candidate values

| Metric | Candidate value |
|---|---:|
| `final_energy` | -1.185941168964711 |
| `thetaB_value` | 0.19 |
| theory `theta_ratio` | 2.058846611995499 |
| theory `elastic_ratio` | 2.072545344395226 |
| theory `contact_ratio` | 2.058846611995500 |
| theory `total_ratio` | 2.045147879595774 |

## Required approval and validation

1. Confirm these values represent the intended signed-curvature and
   shape-pullback model.
2. Preserve the default lane's approved coarse extrapolated-response contract.
3. Regenerate the golden from the named fixture, with no manual value edits.
4. Review all changed metric keys, including new diagnostics/metadata fields.
5. Set tolerances from documented numerical repeatability, not broadening them
   to mask model disagreement.
6. Run the physical-edge baseline acceptance file with its default marker
   filter disabled, plus the qualitative coarse-lane checks.

## Explicit exclusions

- Do not change the explicit-trace full-physics control fixture.
- Do not delete or relax the direct-response tests for that control.
- Do not alter protocol ordering, theory-lane selection, or report serialization
  as part of the rebaseline.

# Feature Contract: Curved 1-Disk Transition-Gradient Regularity Fix

## Goal
Replace the remaining singular shared-rim support-transition derivative path
with a geometry-derived discretization that preserves fixed-theta shape
propagation without making the theta sweep too soft or too stiff.

## Non-goals
- Do not rescale energies, tune coefficients, add hidden weights, or fit to
  `docs/1_disk_3d.tex`.
- Do not change theta scan scheduling, contact strength, shell-2 target
  direction, or public config by default.
- Do not remove the support-transition one-ring shape metric fix unless a
  diagnostic proves it is no longer needed.
- Do not claim full theory reproduction until selected theta, log/K1 profiles,
  and energy split all pass independently.

## User-visible behavior
- Fixed-theta short relaxation keeps the physical positive outer log-height
  slope.
- Selected `thetaB` does not regress to `0.06` and does not run away to the
  current high branch near `0.22-0.26`.
- Near-rim half split, outward shell-2 target direction, and `z`-only shape
  updates remain preserved.
- Runtime energy and diagnostic split totals continue to reconcile.

## Public interfaces to add/change
- No public CLI/config/API changes by default.
- Runtime changes must stay internal to the curved free-disk
  `shared_rim_staggered_v1` lane.
- Diagnostic reports may add transition derivative regularity fields.

## Data model / file format changes
- None.

## Key invariants
- Any transition derivative ownership must be geometry/control-volume derived,
  not selected from theory agreement.
- Artificial support rows must not dominate accepted shape updates.
- Transition energy handling must not create a low-theta or high-theta branch by
  deleting or over-preserving a singular local derivative.
- Non-curved and non-`shared_rim_staggered_v1` lanes remain behaviorally
  unchanged.

## Failure modes & error handling
- If a candidate makes fixed-theta shape propagation positive but shifts
  selected theta to the high edge, reject it as too soft.
- If a candidate restores transition triangle energy but returns selected theta
  to the low branch, reject it as too stiff.
- If diagnostic energy reconciliation regresses, revert before testing profile
  fits.

## Rejected candidates
- Full transition-triangle exclusion removes the low branch and fixes the
  support-gradient direction, but makes the reduced energy too soft and selects
  the current high branch.
- Support-row-only and rim/support-row-only shape metric projections leave the
  accepted update support-dominated and give the wrong log-slope sign.
- Support-row corner redistribution and radial-overlap area scaling are both
  too stiff; the transition derivative remains too large even when geometric
  area is small.
- One-leaflet transition exclusion splits are not sufficient: excluding only the
  outer or only the inner leaflet prevents the high branch, but leaves the
  system too stiff and pushes candidate ordering back toward the low branch.
- Keeping only physical outer-edge transition triangles for the outer leaflet
  while excluding all artificial support-transition triangles for the inner
  leaflet is the first non-fitted candidate that avoids both the low branch and
  the high branch in focused theta ordering. It does not remove scalar
  transition-energy concentration, so validation must focus on gradient
  regularity and reduced-energy ordering rather than claiming scalar support
  energy is gone.

## Test plan
- Acceptance-first tests:
  - selected theta stays between the previous low branch and current high branch
    without targeting the TeX optimum directly;
  - fixed-theta one-step log slope remains positive;
  - transition-band projected gradient no longer dominates accepted updates;
  - runtime module totals reconcile with diagnostic splits.
- Diagnostics:
  - `python -m tools.diagnostics.curved_1disk_shape_direction_audit --horizon 1`
  - `python -m tools.diagnostics.curved_1disk_transition_band_ownership_audit`
  - `python -m tools.diagnostics.curved_1disk_energy_control_volume_audit --theta 0.1845693593`
  - `python -m tools.diagnostics.curved_1disk_theory_benchmark`
- Focused tests:
  - `pytest -q tests/test_curved_1disk_support_gradient_fix_acceptance.py tests/test_curved_1disk_transition_band_ownership_audit.py`
  - `pytest -q -m benchmark tests/test_curved_1disk_theory_benchmark.py tests/test_curved_1disk_energy_control_volume_audit.py`

## Performance considerations
- This affects hot energy/gradient assembly and minimizer direction selection;
  run the representative curved 1-disk benchmark diagnostics before review.

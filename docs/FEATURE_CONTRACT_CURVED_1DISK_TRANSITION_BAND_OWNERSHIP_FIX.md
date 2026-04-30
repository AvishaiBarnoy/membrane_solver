# Feature Contract: Curved 1-Disk Transition-Band Ownership Fix

## Goal
Correct the curved free-disk shared-rim transition band so artificial interface
support rows/triangles no longer carry nearly all outer-membrane elastic energy
and shape-gradient ownership.

## Non-goals
- Do not rescale energies, tune coefficients, add calibration factors, or add
  hidden weights to force agreement with `docs/1_disk_3d.tex`.
- Do not change theta scan scheduling, contact strength, shell-2 target
  direction, or public config by default.
- Do not remove the shared-rim support ring or hard-freeze its shape gradient;
  that candidate already regressed selected `thetaB` to `0.06`.
- Do not claim full theory reproduction until fixed-theta profile fits and
  selected-theta behavior both pass.

## User-visible behavior
- At fixed `thetaB ~= 0.1845693593`, short curved free-disk relaxation should
  move the outer log-height slope in the physical trumpet direction.
- Transition-band attributed outer elastic fraction should fall from the current
  diagnostic value of `~0.983` because the artificial support band should not
  represent almost the entire outer membrane.
- Transition-band projected shape-gradient fraction should fall from the current
  diagnostic value of `~0.9996`.
- Selected `thetaB` must not regress to the low `0.06` branch; movement in
  selected theta must follow from corrected ownership, not a tuning rule.
- Existing near-rim half split, outward shell-2 target direction, and `z`-only
  curved free-disk shape update invariants remain preserved.

## Public interfaces to add/change
- No public CLI/config/API changes by default.
- Diagnostic reports may add transition-band ownership fields if needed.
- Runtime changes must be internal to the curved free-disk
  `shared_rim_staggered_v1` lane.

## Data model / file format changes
- None.

## Key invariants
- Runtime module totals must continue to reconcile with diagnostic energy
  breakdowns.
- Shape-gradient module sums must reconcile with the full projected gradient.
- Artificial support-band handling must be geometry/control-volume derived, not
  chosen from theory-fit coefficients.
- Non-curved and non-`shared_rim_staggered_v1` lanes must remain behaviorally
  unchanged.

## Failure modes & error handling
- If a candidate improves fixed-theta log slope but selected `thetaB` returns to
  `0.06`, reject it and keep the failing acceptance test as evidence.
- If energy reconciliation regresses, revert the runtime change before further
  tuning.
- If transition-band ownership remains dominant after the change, do not claim a
  fix; continue with a narrower module-level ownership audit.

## Rejected candidates
- Redistributing artificial support-row corner area to the non-support corners
  keeps the transition triangles physically active, but preserves too much of
  the oversized transition control volume. In the fixed-theta sweep it makes the
  elastic response too stiff and pushes candidate ordering back toward the low
  branch. A later control-volume fix must use a geometry-derived radial
  ownership width, not a fitted area fraction.
- Scaling support-transition triangle control areas by their radial overlap with
  the physical outer sliver is also too stiff. The scaled area is small, but the
  transition gradients remain singularly large, so this is not just an area
  accounting problem.

## Test plan
- Acceptance-first tests:
  - transition-band outer elastic fraction drops below the current `~0.983`;
  - transition-band projected gradient fraction drops below the current
    `~0.9996`;
  - fixed-theta one-step/short-relaxation outer log slope gets the physical sign;
  - selected `thetaB` stays above `0.06`;
  - shared-rim target direction, near-rim split, and `z`-only update invariants
    remain green.
- Diagnostics:
  - `python -m tools.diagnostics.curved_1disk_transition_band_ownership_audit`
  - `python -m tools.diagnostics.curved_1disk_shape_direction_audit`
  - `python -m tools.diagnostics.curved_1disk_energy_control_volume_audit --theta 0.1845693593`
- Focused tests:
  - `pytest -q tests/test_curved_1disk_transition_band_ownership_audit.py tests/test_curved_1disk_support_gradient_fix_acceptance.py`
  - `pytest -q -o addopts='' tests/test_curved_1disk_shared_rim_fix_acceptance.py`

## Performance considerations
- Any energy/gradient ownership change affects hot numerical paths and must be
  validated with the representative curved 1-disk benchmark diagnostics above.

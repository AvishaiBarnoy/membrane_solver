# Feature Contract: Curved 1-Disk Shape Propagation Fix

## Goal
Make the fixed-theta curved free-disk lane propagate the TeX outer trumpet and
K1 tilt profile beyond the shared-rim support shells.

## Non-goals
- Do not rescale energies or add calibration factors.
- Do not change theta scan scheduling, contact strength, or public config by default.
- Do not revisit shell-2 target direction unless diagnostics show it regressed.
- Do not claim full `docs/1_disk_3d.tex` reproduction until theta selection and
  profile fits both pass.

## User-visible behavior
- At fixed `thetaB ~= 0.1845693593`, the outer height log fit has the TeX sign
  and order-one slope ratio through the benchmark log window.
- The outer leaflet K1 fit has lambda close to the TeX decay length and low
  leaflet mismatch in the outer window.
- The near-rim split and outward shell-2 target direction remain preserved.
- Selected `thetaB` moves only as a consequence of the profile fix, not from a
  new tuning knob.

## Public interfaces to add/change
- No public config or CLI changes are planned.
- Diagnostic tests may add explicit fixed-theta profile-lock assertions.

## Data model / file format changes
- None.

## Key invariants
- Shared-rim target direction remains physically outward.
- Disk rim `thetaB` remains imposed through the existing boundary/contact path.
- Runtime energy module totals continue to reconcile with the Gate A diagnostic
  split before and after the shape fix.

## Failure modes & error handling
- If free outer shells remain flat, keep the acceptance tests xfailed and inspect
  constraint/projection gauges before changing energy terms.
- If profile propagation improves but energy reconciliation regresses, stop and
  restore the Gate A invariant before continuing.

## Test plan
- Acceptance-first xfailed tests for fixed-theta outer log-height and K1 profile
  fidelity.
- Existing shared-rim acceptance tests for target direction, near-rim split, and
  selected-theta movement.
- Benchmark diagnostics: `curved_1disk_miss_diagnosis`,
  `curved_1disk_theory_benchmark`, and `curved_1disk_energy_control_volume_audit`.

## Performance considerations
- The intended fix may touch constraint/projection or shape-relaxation paths and
  must be validated with the existing benchmark-marked curved 1-disk tests.

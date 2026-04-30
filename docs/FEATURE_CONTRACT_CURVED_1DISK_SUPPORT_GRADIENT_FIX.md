# Feature Contract: Curved 1-Disk Support-Gradient Fix

## Goal
Fix the curved free-disk shape solve so accepted fixed-theta shape updates are
not dominated by shared-rim support-shell `z` gradients when a valid outer
trumpet descent direction exists.

## Non-goals
- Do not rescale energies, add calibration factors, hidden weights, or tune
  coefficients to `docs/1_disk_3d.tex`.
- Do not change theta scan scheduling, contact strength, or public config by
  default.
- Do not change shell-2 target direction, near-rim split semantics, or energy
  ownership unless a new diagnostic proves those regressed.
- Do not claim full theory reproduction until fixed-theta profile fits and
  selected-theta behavior are both validated.

## User-visible behavior
- At fixed `thetaB ~= 0.1845693593`, accepted shape updates should retain a
  meaningful component along the outer log/trumpet direction instead of being
  almost entirely near-support-shell motion.
- The outer height log profile should move in the physical trumpet direction
  over short fixed-theta relaxations.
- Existing shared-rim invariants remain preserved: outward shell-2 target,
  near-rim half split, `z`-only curved free-disk shape updates, and Gate A
  energy reconciliation.

## Public interfaces to add/change
- No public CLI/config/API changes by default.
- Diagnostic-only helpers may expose support-gradient, shell-normalized, or
  area-weighted direction metrics.
- Runtime changes, if needed, must be internal to the curved free-disk lane and
  must not introduce new user-facing tuning knobs.

## Data model / file format changes
- None.

## Key invariants
- Runtime module totals must reconcile with diagnostic energy deltas before and
  after the fix.
- Constraint enforcement must not reset free outer `z` perturbations or mutate
  unrelated tilt fields during minimize line search.
- Any metric/preconditioner or support-shell handling must be derived from
  geometry/control-volume structure, not from matching theory coefficients.
- Existing non-curved and non-`shared_rim_staggered_v1` lanes must remain
  behaviorally unchanged.

## Failure modes & error handling
- If support-shell suppression improves alignment but breaks near-rim split or
  target direction, stop and keep the failing test as evidence.
- If area/shell-normalized probes improve alignment only diagnostically but do
  not improve accepted runtime updates, do not claim a fix; classify the
  blocker as solver/path conditioning.
- If fixed-theta profile improves while energy reconciliation regresses, revert
  the runtime change and restore the reconciliation invariant.

## Rejected candidates
- Hard-freezing the support-transition one-ring removes the immediate support
  dominance and gives the fixed-theta log slope the physical sign, but it
  regresses selected `thetaB` to `0.06`; keep the acceptance checks as xfailed
  evidence, not as a runtime fix.
- Uniformly damping the transition-band gradient preserves selected `thetaB`
  only when the damping is weak enough that support-shell domination and the
  negative short-relaxation log slope remain.
- Re-projecting after the curved lane's `z`-only shape projection preserves the
  selected-theta ordering but does not change the support-dominated accepted
  update. A z-only KKT variant increases support alignment.

## Test plan
- Add acceptance-first tests that currently fail or remain xfailed for:
  - accepted update alignment: log/trumpet component increases relative to the
    current `~0.03` one-step cosine;
  - near-support domination decreases from the current `~0.58` one-step cosine;
  - fixed-theta outer log slope gets the physical sign over short relaxation;
  - shared-rim target direction, near-rim half split, and `z`-only shape updates
    do not regress.
- Run diagnostics:
  - `python -m tools.diagnostics.curved_1disk_trumpet_descent_audit`
  - `python -m tools.diagnostics.curved_1disk_shape_direction_audit`
  - `python -m tools.diagnostics.curved_1disk_shape_propagation_blocker`
  - `python -m tools.diagnostics.curved_1disk_energy_control_volume_audit --theta 0.1845693593`
- Run focused tests:
  - `pytest -q tests/test_curved_1disk_trumpet_descent_audit.py tests/test_curved_1disk_shape_direction_audit.py`
  - `pytest -q -o addopts='' tests/test_curved_1disk_shared_rim_fix_acceptance.py`

## Performance considerations
- Any runtime fix that changes shape metric, preconditioning, projection, or
  support-shell ownership affects numerical behavior and must include the
  representative curved 1-disk benchmark diagnostics above.

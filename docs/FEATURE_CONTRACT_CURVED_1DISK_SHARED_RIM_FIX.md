# Feature Contract: Curved 1-Disk Shared-Rim Fix

## Goal

Make the curved single-caveolin free-membrane lane reproduce the qualitative
`docs/1_disk_3d.tex` tensionless target by correcting shared-rim target
construction and outer shape propagation.

## Non-goals

- Do not add tuning knobs, calibration factors, or hidden scale corrections.
- Do not change unrelated rim-matching modes.
- Do not claim full TeX parity until the benchmark lock criteria pass.
- Do not refactor general minimizer or mesh data layout unless a focused test
  proves it is required for this fix.

## Acceptance Criteria

- For fixed `thetaB ~= 0.1845693593`, the shell-2 target direction has positive
  cosine with the physical outward radial direction.
- Near-rim split remains theory-consistent: `phi ~= thetaB/2`,
  `theta_in ~= thetaB/2`, and `theta_out ~= thetaB/2`.
- Outer height propagates beyond the first support shell and has nonzero
  log-window slope with the expected sign.
- The selected curved-branch `thetaB` moves upward from the current low branch
  (`0.06`) without relying on energy rescaling.
- Existing known-miss diagnostics either pass or update their expected call to
  the next remaining physical bottleneck.

## Interfaces (Public API / CLI / Config)

No new user-facing config is planned.

The existing curved free-disk use of
`rim_slope_match_mode = "shared_rim_staggered_v1"` may change internally so
that:

- shape secant uses the physical rim and first free shell;
- tilt continuation targets the next unconstrained shell with an outward local
  radial direction at that target shell;
- outer geometry is free to relax except for intended far-boundary gauges.

## Data Model Changes

None expected.

## Key Invariants

- Disk rim thetaB remains imposed by the existing thetaB boundary path.
- The physical rim remains the reference for the 1D-style continuation law.
- Shape-side and tilt-side shell selection must be explicit in diagnostics.
- Far-boundary constraints may fix gauges but must not suppress the trumpet
  profile in the free outer membrane.

## Failure Modes / Error Handling

- If shell data cannot identify rim, first free shell, and next unconstrained
  shell, diagnostics should fail with an assertion explaining the missing shell.
- If target direction construction produces an inward radial direction, the
  dedicated audit must report that before minimization results are trusted.
- If the fix improves target direction but not shape propagation, stop and
  isolate shape constraints/projection before changing energy terms.

## Test Plan

- Acceptance tests:
  - fixed-theta shell-2 outward target direction
  - fixed-theta nonzero outer log-window height slope
  - near-rim split preservation
  - selected thetaB moves above the current low branch
- Unit tests:
  - shell selection chooses the intended physical rim, first free shell, and
    next unconstrained shell
  - target radial direction is computed from target-shell geometry and points
    outward
- Regression tests:
  - existing diagnostic CLIs continue to emit ranked calls
  - unrelated rim-matching modes keep their current behavior

## Performance Notes

This touches constraint/shape behavior in a numerical path.  Include the
affected curved one-disk benchmark in validation.  No broad performance
benchmark is required unless implementation changes hot-loop assembly or
minimizer internals.

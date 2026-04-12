# Stage A Complete-Support Design Note

## Purpose

This document defines the diagnostics/specification track for a future
complete-support prototype in the Stage A lane. It is not a prototype and does
not change current solver behavior.

The current local-patching stream stops at commit `0580736` on the Stage A
level-2 follow-up branch. That checkpoint reduced the residual level-2 outer
artifact, but did not remove the shell-scale pattern. Any further work should
move to a principled support-model design track rather than adding more
Stage-A-local stabilizers.

## Problem Statement

The remaining Stage A level-2 artifact is tied to a partial-support band in the
outer leaflet support model:

- the residual shell pattern appears on the outer shells near
  `r ≈ 0.9659 / 0.9745 / 0.99998`
- the coefficient jump that seeds it occurs one band inward, where
  `A_eff / A_vor ≈ 0.50–0.57`
- on those inward rows, kept triangle participation drops from `6` to `3`

The current effective-area redistribution is energy/gradient-consistent for the
implemented discrete model, but it is not refinement-stable on this partial
support transition.

## What “Complete-Support” Means

“Complete-support” does **not** mean blindly restoring full Voronoi area
everywhere.

It means introducing a **refinement-stable, energy/gradient-consistent local
support formulation** for rows whose effective support is only partially
represented by the kept triangle fan.

The future model must:

1. preserve scalar-energy / gradient consistency
2. avoid shell-scale coefficient jumps caused solely by refinement of partial
   support
3. remain local to the intended support transition
4. reduce to the current geometric/Voronoi behavior on full-support rows
5. leave the lower-refinement Stage A branch materially unchanged

## Non-Goals

- No new Stage-A-local stabilizers
- No direct patching of the derived `ratio = A_eff / A_vor`
- No deep-relaxation schedule exploration
- No solver-flow redesign
- No blanket “set `A_eff = A_vor`” replacement

## Current Regression Targets

Any future prototype must preserve the currently accepted behavior of:

- Stage A level `0`
- Stage A level `1`
- the flat KH lane

In particular:

- Stage A levels `0` and `1` must remain materially stable in
  `theta_mean`, `phi_mean`, `zmax`, total energy, and energy split
- the current provisional Stage A level-1 benchmark reference schedule remains
  `g3 -> r -> g5`
- targeted flat KH benchmark/theory regressions must remain green

## Prototype Validation Targets

The later prototype should be validated against the current residual Stage A
level-2 artifact only:

1. pre-update seed on the shells near
   `r ≈ 0.9659 / 0.9745 / 0.99998`
2. first `g1` after the second refine
3. final level-2 relaxed branch

Measured outputs should include:

- shellwise `dE/dz`
- shellwise `A_eff`, `A_vor`, and support participation
- `theta_mean`, `phi_mean`, `zmax`, total energy, and energy split

## Prototype Success / Failure Rule

The future complete-support prototype passes the design gate only if all of the
following are true:

1. the level-2 shell-scale `+ / - / +` seed on the target outer shells is
   materially reduced or removed
2. the first-`g1` alternating shell response is materially reduced without
   merely shifting the pattern to adjacent shells
3. Stage A levels `0` and `1` remain materially unchanged
4. the flat KH lane does not regress

If the prototype only produces another modest reduction of the same shell-scale
pattern, or simply moves it inward/outward while keeping the same qualitative
artifact, then the prototype fails and the support formulation must be revised
before further behavior changes are proposed.

## Implementation Plan

### PR 1: Diagnostics / Specification

This PR.

- document the requirements of the future support model
- define the prototype success/failure rule
- lock the regression targets that must remain stable

### PR 2: Prototype Support Model

Implement one principled complete-support formulation for the outer leaflet
support/area model near partial-support bands.

Constraints:

- keep scalar energy and gradients internally consistent
- scope the change to the support formulation itself
- avoid lane-specific behavioral knobs

### PR 3: Validation / Refinement

Only if needed after the prototype:

- tighten tests and diagnostics
- compare the prototype directly against the current residual level-2 artifact
- decide whether the new support model should replace the temporary local
  stabilization lineage

## Recommended Next Step

Use this note as the contract for the future prototype stream. Do not implement
the prototype on this branch.

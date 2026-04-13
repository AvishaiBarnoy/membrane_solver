# Stage A Outer Support-Model Redesign Plan

## Purpose

This document defines the redesign-track plan for a broader outer leaflet
support formulation in the Stage A lane.

It is intentionally **not** a prototype and does not change solver behavior.
The goal is to replace the current partial-support handling with a
refinement-stable outer support model that remains fully energy/gradient
consistent without relying on Stage-A-local behavioral stabilizers.

## Why This Redesign Track Exists

The local patching stream stopped at commit `0580736`, which modestly reduced
the residual level-2 outer-shell artifact but did not remove it.

The complete-support prototype stream then tested a small family of local
completion rules on top of the current kept-area model. Those prototypes were
informative, but they did not pass the design gate.

The next step is therefore a broader support-model reformulation rather than
another local completion tweak.

## Problem Statement

The residual Stage A level-2 artifact remains localized near the outer shell
band around:

- `r ≈ 0.9659 / 0.9745 / 0.99998`

The seed is created one band inward, where the outer leaflet support becomes
partial:

- `A_eff / A_vor ≈ 0.50–0.57`
- kept triangle participation drops from `6` to `3`

The resulting coefficient jump is then amplified by the outer
`bending_tilt_out` `grad_linear` channel.

The redesign target is therefore the **outer support formulation itself**, not
the derived `ratio` factor and not another Stage-A-local stabilizer.

## Why the Previous Local-Completion Family Was Insufficient

- `blend_half`-style local stabilization reduced the artifact only modestly and
  did not remove the shell-scale pattern.
- Angular-support completion restored missing support too aggressively and
  amplified the outer positive `grad_linear` lobes.
- `local_density_matched_support_v1` stayed monotone and capped, but still
  raised the partial-support band enough to worsen the outer lobes.
- `bounded_density_matched_support_v2` landed in a conservative completion
  range, but still failed the gate because the first-`g1` shell pattern was not
  materially collapsed.
- All tested variants were still based on **completing the current kept-area
  model locally**, so they inherited the same underlying shell-scale transition
  rather than replacing it.

## What the Redesigned Support Model Must Preserve

### 1. Energy/gradient consistency

The scalar energy, tilt-side derivatives, and shape-gradient paths must all use
the same support model. The redesign must not patch a derived factor in only
one gradient channel.

### 2. Lower-level validated behavior

The redesign must preserve the currently accepted behavior of:

- Stage A level `0`
- Stage A level `1`
- the flat KH lane

In practical terms, that means:

- no material regression in `theta_mean`, `phi_mean`, `zmax`, total energy, or
  energy split for Stage A levels `0` and `1`
- no regression in the targeted flat KH benchmark/theory tests
- no change to the current provisional Stage A level-1 benchmark schedule
  (`g3 -> r -> g5`)

### 3. Benchmark structure

The redesign must be validated against the same Stage A level-2 artifact gates
already established. This stream should not expand into deep-relaxation branch
exploration or unrelated solver changes.

## What the Redesigned Support Model Must Not Do

- It must **not recreate shell-scale coefficient jumps** under refinement.
- It must **not rely on local completion of the current kept-area model** as
  its core mechanism.
- It must **not require Stage-A-local behavioral stabilizers** to suppress the
  resulting artifact.
- It must **not** be a disguised `A_eff = A_vor` restoration everywhere.
- It must **not** solve the artifact by changing only a derived quantity such
  as `A_eff / A_vor` in one gradient channel.

## Candidate Reformulation Direction

The redesign should treat partial outer support as its own local support model,
not as “missing area” to be filled on top of the current kept-area shares.

That suggests a new formulation built around:

- a local outer-support control volume or support measure defined directly from
  the outer leaflet’s intended geometric support
- a refinement-stable transition between full-support and partial-support rows
- direct use of that support measure in both the scalar energy and its
  gradients

In other words, the new model should define the support quantity natively,
rather than trying to infer it by completing the current redistributed area.

## Validation Gates

Any later prototype from this redesign track must be judged by the same gates:

1. remove or materially collapse the level-2 shell-scale seed on
   `r ≈ 0.9659 / 0.9745 / 0.99998`
2. materially reduce the first-`g1` alternating shell response without merely
   shifting it
3. preserve Stage A levels `0` and `1`
4. avoid regressing the flat KH lane

If a prototype only produces another modest reduction of the same shell-scale
pattern, it fails this redesign gate.

## Implementation Plan

### PR 1: Redesign specification

This PR.

- define the requirements of the broader outer support-model reformulation
- record why the local-completion family was insufficient
- define what the new model must preserve and must not do

### PR 2: First redesign prototype

Implement one new outer support formulation that is:

- refinement-stable
- energy/gradient-consistent
- not based on local completion of the current kept-area model

### PR 3: Prototype validation

Validate that prototype against:

- the level-2 shell seed
- the first `g1` after second refine
- final Stage A levels `0` and `1`
- the flat KH lane

## Recommended Next Step

Use this note as the starting contract for the broader outer support-model
reformulation. Do not implement the redesign prototype on this branch.

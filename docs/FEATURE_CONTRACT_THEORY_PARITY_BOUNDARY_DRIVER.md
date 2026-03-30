# Feature Contract: Theory-Parity Field-Driven Boundary for thetaB

## Goal
Replace the scalar `tilt_thetaB_value` feedback loop with a theory-aligned, field-based mechanism that natively drives the boundary tilt (`thetaB`) while selecting the correct nontrivial branch. The design will allow `thetaB` to emerge naturally from the local field balance (measured at `r=R`) while still injecting the required contact work/boundary driving force to prevent the state from collapsing to the trivial zero branch. This will be achieved via either a blended inner targeting rule or a reduced-DOF (boundary mode) path for `thetaB`.

## Non-goals
- Solving the full parity profile match (height and free-side tilts at all infinity ranges) in a single update.
- Introducing mesh refinement, additional trace/support shells, or altering mesh density.
- Tweaking existing unguided scalar-projection heuristics (e.g. `rim_slope_match_out` raw row-swapping) without a theoretical field-based justification.
- Modifying or degrading the backward compatibility of the `legacy_coarse` pipeline.

## Acceptance Criteria
- **Field-Driven Driving:** The global runtime flag for scalar `thetaB` optimization (`tilt_thetaB_value`) decoupled from the free interface law for the scaffold lane. The boundary driving must be supplied by the field gradient itself (e.g., explicit contact gradient on the boundary `tilt_in` rows) or a blended cross-interface constraint.
- **Nontrivial Branch Stabilization:** The solution for the scaffold lane (`kozlov_1disk_3d_free_disk_theory_parity_physical_edge_scaffold_gapfill_eps005_n3_release.yaml`) does not collapse to `thetaB ≈ 0` and avoids the excessive elastic blow-up branch.
- **Diagnostics Accuracy:** `thetaB` is reported as a measured output from the system (disk-side boundary `tilt_in` at `r=R`), decoupled from optimization feedback.
- **Seamless Operator Integration:** The modified interface operator (`rim_slope_match_out.py`) and contact module accurately balance boundary work with gradient propagation on the boundary nodes.

## Interfaces (Public API / CLI / Config)
- **Configuration Toggle:** A new setting or scaffold-driven trigger to disable the independent scalar minimization path (`tilt_thetaB_optimize: false`).
- **Minimizer Hooks:** Conditional bypass logic in `runtime/minimizer.py` for `thetaB` optimization when in the new field-driven mode.
- **Reporting Metrics:** Updates in `tools/reproduce_theory_parity.py` to ensure `metrics.thetaB_value` faithfully reflects the final measured tilt.

## Data Model Changes
- The scalar `tilt_thetaB_value` is effectively retired as an independent global DOF on the developmental scaffold lanes. It transforms into an observable derived metric, with system physics strictly driven by node-level `tilt_in` fields at the rim.

## Key Invariants
- `legacy_coarse` parity tests must remain unaffected.
- The `physical_edge_default` parity guardrails must pass without breaking.
- Any new node-level energy derivatives introduced (e.g., to replace the pure scalar contact work) must precisely match the corresponding energy formula.

## Failure Modes / Error Handling
- **Trivial Zero Branch Collapse:** If the proposed boundary gradients lack sufficient magnitude to offset elastic forces, the system will collapse back to zero. Acceptance tests will strictly enforce an empirically expected range for `thetaB > 0` and check against TeX ratios.
- **Elastic Energy Blow-Up:** Over-constraining the inner target or contact gradient might recreate the known bad branch (massive elastic overshoot). This must be mitigated, and guarded by explicit bounds on the `tex_elastic_ratio`.

## Test Plan
- **Acceptance tests:**
  - Add/modify a long-schedule test in `tests/test_theory_parity_against_tex_acceptance.py` validating that the scaffold gapfill fixture converges to a stable, non-zero `thetaB` *without* scalar feedback.
- **Unit tests:**
  - Verification of the new gradient injection or blended target matching logic to ensure derivatives correctly map to the explicit boundary rows.
- **Regression tests:**
  - Full parity test suite executions to verify `test_physical_edge_default_keeps_theta_and_energy_in_guardrail_while_fixing_interface` still resolves correctly.

## Performance Notes
- All new gradient computations or interface blends must be computationally lightweight avoiding full system global matrix recalculations. They will execute at node-level during regular minimizer steps and must not compromise the developmental testing cycle cadence.

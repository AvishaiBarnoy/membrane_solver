# Overnight Theory-Parity Improvement Plan

## Summary

The active parity stream is the scaffold/free-side boundary-driver work described
in `docs/THEORY_PARITY_CHECKPOINT.md`. The current default physical-edge lane
keeps good `thetaB` and total-energy parity, but the scaffold lane can collapse
to the zero branch once scalar `thetaB` feedback is bypassed. The overnight goal
is to restore a nontrivial scaffold branch through field-level boundary driving,
then tighten the scaffold interface split only if the restored branch remains
bounded.

Keep these invariants:

- Do not change KH/TeX parameter ratios.
- Do not promote scaffold behavior into `physical_edge_default`.
- Do not delete or rewrite theory fixtures.
- Do not start with more shell insertion, row swapping, or target fitting.

## Stage 0: Baseline

Run the focused baseline checks before editing:

```bash
pytest -q -o addopts='' \
  tests/test_rim_slope_match_out_constraint.py::test_physical_edge_scaffold_shape_projection_moves_trace_shell_height \
  tests/test_rim_slope_match_out_constraint.py::test_physical_edge_scaffold_uses_trace_layer_for_operator_tilt_rows

pytest -q -o addopts='' \
  tests/test_theory_parity_against_tex_acceptance.py::test_scaffold_gapfill_base_long_schedule_activates_outer_leaflet_without_repair \
  tests/test_theory_parity_against_tex_acceptance.py::test_physical_edge_default_keeps_theta_and_energy_in_guardrail_while_fixing_interface
```

Expected starting point on this branch: the default-lane guardrail passes, while
the scaffold gapfill test may fail by collapsing to near-zero outer activity.

## Stage 1: Field-Level Contact Work

Add an opt-in contact work mode in `modules/energy/tilt_thetaB_contact_in.py`:

- `tilt_thetaB_contact_work_mode: scalar` is the default and preserves existing
  scalar contact work, `E = -2*pi R_eff gamma thetaB`.
- `tilt_thetaB_contact_work_mode: field_linear` uses the measured boundary field,
  `theta_mean = sum(w_i * dot(t_in_i, r_hat_i)) / sum(w_i)`, and computes
  `E = -2*pi R_eff gamma theta_mean`.
- In `field_linear`, add the exact tilt gradient
  `-(2*pi R_eff gamma / sum(w_i)) * w_i * r_hat_i`.
- Leave `tilt_thetaB_contact_penalty_mode: legacy` behavior intact.

Enable `field_linear` only on
`tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_scaffold_gapfill_eps005_n3_release.yaml`.
Keep the fixed-`d` scaffold fixture as a dead-branch control unless a later
stage explicitly changes it.

## Stage 2: Scaffold Branch Criteria

Update scaffold acceptance only after Stage 1 is measured:

- Gap-filled release scaffold should be the active branch-restoration lane.
- Fixed-`d` scaffold may remain a known dead-branch control.
- `metrics.thetaB_value` must remain measured from disk-side `tilt_in` at `R`.
- The gap-filled scaffold should show nonzero `bending_tilt_out`, nonzero
  `tilt_out`, nonzero direct interface `phi`, and bounded TeX ratios.

Do not loosen `physical_edge_default` guardrails.

## Stage 3: Projector Follow-Up, Only If Needed

If field-level contact restores a nonzero scaffold branch but leaves the
interface split too large, add a scaffold-only projector mode in
`modules/constraints/rim_slope_match_out.py`:

- `rim_slope_match_scaffold_projector_mode: continuity_v2`
- Preserve the residual-gated projection cadence.
- Use `continuity_target = theta_disk - t_in_rad`.
- Use `phi_target = 0.5 * (phi_current + continuity_target)`.
- Use `t_out_target = phi_target`.

Enable this only on the gap-filled scaffold fixture after Stage 1 results prove
that the remaining blocker is work allocation rather than zero-branch collapse.

## Validation

Run after Stage 1:

```bash
pytest -q -o addopts='' tests/test_tilt_thetaB_contact_in_pure_contact_regression.py
pytest -q -o addopts='' tests/test_rim_slope_match_out_constraint.py
pytest -q -o addopts='' \
  tests/test_theory_parity_against_tex_acceptance.py::test_scaffold_gapfill_base_long_schedule_activates_outer_leaflet_without_repair \
  tests/test_theory_parity_against_tex_acceptance.py::test_scaffold_gapfill_reports_measured_thetaB_from_disk_boundary \
  tests/test_theory_parity_against_tex_acceptance.py::test_physical_edge_default_keeps_theta_and_energy_in_guardrail_while_fixing_interface
```

Final run before handoff:

```bash
pytest -q -o addopts='' \
  tests/test_theory_parity_against_tex_acceptance.py \
  tests/test_reproduce_theory_parity_acceptance.py
```

Record the final scaffold values for `thetaB_value`, direct `t_in`, direct
`t_out`, direct `phi`, `bending_tilt_out`, `tilt_out`, elastic ratio, and total
ratio in `docs/THEORY_PARITY_CHECKPOINT.md`.

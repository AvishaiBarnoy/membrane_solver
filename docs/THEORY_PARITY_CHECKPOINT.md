# Theory Parity Checkpoint

## Current State
- The active reproducer is [tools/reproduce_theory_parity.py](/Users/User/github/membrane_solver/tools/reproduce_theory_parity.py).
- The coarse fixture [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml) is now treated as `legacy_coarse` only.
- The active parity-development fixture is [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml).
- Coarse reports now measure interface geometry from the first local shell outside the physical disk edge `R = 7/15`, while preserving the legacy solver shell radii as secondary diagnostics.

## Legacy Coarse Lane
- Current coarse report:
  - `thetaB = 0.09`
  - `final_energy = -0.56989`
  - `tex total_ratio = 0.49139`
  - measured local-shell geometry:
    - `rim_radius ≈ 0.46667`
    - `outer_radius ≈ 0.72784`
  - legacy solver radii still reported for reference:
    - `legacy_solver_rim_radius ≈ 0.98296`
    - `legacy_solver_outer_radius ≈ 1.965`
- Conclusion: fixing the coarse report geometry did not materially improve the coarse solver lane; it remains a regression anchor only.

## Physical-Edge Default Lane
- The new tracked default lane is [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml).
- Current default report:
  - `thetaB = 0.18`
  - `final_energy = -1.15885`
  - `tex total_ratio = 0.99921`
  - `tex elastic_ratio = 0.95128`
  - `rim_radius ≈ 0.46667`
  - `outer_radius ≈ 0.65755`
  - `phi_mean ≈ 0.00376`
  - `phi_over_half_theta ≈ 0.04174`
- New TeX-facing one-sided trace diagnostics:
  - `disk_theta_at_R ≈ 0.18000`
  - `disk_t_in_at_R ≈ 0.17624`
  - `outer_t_out_trace_at_R+ ≈ 0.04131`
  - `phi_trace_at_R+ ≈ 0.09029`
  - `disk_theta_at_R - 2 phi_trace_at_R+ ≈ -5.8e-4`
  - `disk_t_in_at_R - outer_t_out_trace_at_R+ ≈ 0.13493`
- New outer-profile parity diagnostics:
  - `phi_profile_rel_rmse ≈ 1.913`
  - `z_profile_rel_rmse ≈ 0.999`
- New geometry-vs-tilt split:
  - `outer_geometry_trace_at_R+ ≈ 0.09029`
  - `outer_t_out_trace_at_R+ ≈ 0.04131`
  - `outer_geometry_vs_tilt_trace_gap ≈ 0.04898`
- This lane is derived from the same physical-edge construction as the earlier `near_edge_v1` reference, but is now tracked as the generic bottom-up default rather than as a one-off named fix.
- Current kept interface-side improvement:
  - the physical-edge law now pairs the first outer shell to disk-boundary rows by explicit nearest azimuth (`rim_rows_for_disk`) instead of relying on independently ordered rings
  - a second-shell-supported composition was tested and produced the same behavior on the current family, so it was not kept as a separate runtime change
  - the local-shell builder now uses an order-preserving cyclic azimuth match when adjacent rings have equal counts; this cleans up pair regularity but does not materially change the current parity metrics
  - the TeX-facing diagnostics PR is merged:
    - same-point one-sided interface traces at `R`
    - outer-profile parity against the TeX field
    - explicit geometry-vs-tilt trace separation on the outer side

## Physical-Edge Family
- The profile helper in [tools/theory_parity_interface_profiles.py](/Users/User/github/membrane_solver/tools/theory_parity_interface_profiles.py) now defines the generic default family:
  - `default_lo = (0.776, 2.68)`
  - `default = (0.772, 2.66)`
  - `default_hi = (0.771, 2.655)`
- Current optimized sweep on this branch:
  - `default_lo`: `thetaB = 0.18`, `tex total_ratio = 0.99152`, `outer_radius ≈ 0.65960`
  - `default`: `thetaB = 0.18`, `tex total_ratio = 0.99921`, `outer_radius ≈ 0.65755`
  - `default_hi`: `thetaB = 0.18`, `tex total_ratio = 1.00110`, `outer_radius ≈ 0.65704`
  - `coarse`: `thetaB = 0.09`, `tex total_ratio = 0.49139`, `outer_radius ≈ 0.72784`
- Current fixed-`thetaB = 0.185` comparison:
  - `coarse`: `elastic = 2.13375`, `contact = -2.32493`, `total = -0.19119`, `phi_over_half_theta ≈ 0.00643`
  - `default_lo`: `elastic = 0.96211`, `contact = -2.32493`, `total = -1.36283`, `phi_over_half_theta ≈ 0.14277`
  - `default`: `elastic = 0.91989`, `contact = -2.32493`, `total = -1.40505`, `phi_over_half_theta ≈ 0.16956`
  - `default_hi`: `elastic = 0.91984`, `contact = -2.32493`, `total = -1.40509`, `phi_over_half_theta ≈ 0.17208`
- Practical conclusion:
  - contact remains effectively fixed across the physical-edge family
  - optimized parity remains smooth across `lo / primary / hi` and is slightly tighter around the TeX target than the previous baseline set
  - fixed-`thetaB` elastic terms still vary smoothly with near-edge geometry
  - the new one-sided trace diagnostics show that the geometric outer slope trace at `R+` is close to the TeX relation `phi_* = theta_B / 2`
  - but the free-membrane-side leaflet traces and outer height profile are still not near the TeX continuation law
  - the outer-side mismatch is now clearly split:
    - what we get right:
      - `thetaB`
      - total energy
      - geometric slope trace `phi(R+)`
    - what we still get wrong:
      - free-side `t_in(R+)` is effectively zero instead of `thetaB / 2`
      - outer `tilt_out` trace at `R+`
      - outer height/profile `z(r)`
  - the family remains in a non-pathological regime and can be used as the active parity-development base

## Symmetry Breaking
- The physical-edge parity reproducer still needs an explicit symmetry-breaking kick to leave the flat symmetric branch, but it no longer needs that kick for the full run.
- In [tools/reproduce_theory_parity.py](/Users/User/github/membrane_solver/tools/reproduce_theory_parity.py), `_activate_local_outer_shell_for_parity(...)` applies a small `z` bump (`parity_physical_edge_z_bump`, defaulting to `DEFAULT_PHYSICAL_EDGE_Z_BUMP`) to the first local outer shell when its height is too close to zero.
- Current reproducer behavior:
  - `parity_physical_edge_z_bump = 0.0` now really means no kick
  - the default physical-edge path uses the configured bump only to leave the symmetric branch, then releases it to `0.0` immediately after the first protocol step
  - releasing the bump after the first step preserves the current good branch exactly on the default lane
- Diagnostic result from the current default family:
  - with the current kick (`1e-3`):
    - `default`: `thetaB = 0.18`, `tex total_ratio = 0.99921`, `outer_t_out_trace_at_R+ ≈ 0.04131`, `trace_gap ≈ 0.04898`
  - with a tiny kick (`1e-12`):
    - `default`: `thetaB = 0.18`, `tex total_ratio = 0.96974`, `outer_t_out_trace_at_R+ ≈ 0.01008`, `trace_gap ≈ 0.07461`
  - with zero kick (`0.0`) for the full run:
    - `default`: `thetaB = 0.17`, `tex total_ratio = 0.93973`, `phi_trace(R+) ≈ 0.0`, `outer_t_out_trace(R+) ≈ 0.0`
  - with transient release (`1e-3` until `g10`, then `0.0`):
    - `default`: `thetaB = 0.18`, `tex total_ratio = 0.99921`, `phi_trace(R+) ≈ 0.09029`, `outer_t_out_trace(R+) ≈ 0.04131`
- Shell-level interpretation on `default`:
  - current kick:
    - first-shell geometric secant `≈ 0.00376`
    - first-shell `tilt_out ≈ 0.00378`
    - extrapolated trace `t_out(R+) ≈ 0.04131`
  - tiny kick:
    - first-shell geometric secant `≈ 0.00057`
    - first-shell `tilt_out ≈ 0.00058`
    - extrapolated trace `t_out(R+) ≈ 0.01008`
- Conclusion:
  - the current near-TeX physical-edge parity result is still branch-selection dependent
  - a persistent kick is not required once the branch is selected; a transient kick after initialization is enough on the current default lane
  - fully kick-free recovery of the good branch remains open work

## Rejected Next-Step Candidate
- A follow-up runtime pass tested whether the remaining gap could be reduced by changing only how the physical-edge outer `tilt_out` field is represented/read near `R+`, while keeping the geometric `phi` law unchanged.
- Two candidate directions were rejected:
  - replacing the physical-edge law with a one-sided boundary-trace target built from the first two outer shells
  - changing only the outer-tilt row/weight representation to an extrapolated trace
- Outcome:
  - both directions made the outer-tilt trace worse rather than better
  - the first one also drove the family to a bad branch (`thetaB ≈ 0.25`, `tex total_ratio ≈ 1.74`)
  - the second kept energy/theta roughly unchanged but collapsed `outer_t_out_trace_at_R+` toward zero, increasing the geometry-vs-tilt gap
- Conclusion:
  - the remaining gap is not fixed by swapping outer rows/weights in `rim_slope_match_out`
  - the next higher-signal target is how the outer `tilt_out` field itself is formed/regularized near the first two shells outside `R`

## Rejected Resolution-Only Candidate
- A follow-up mesh-construction pass tested whether the bad free-side trace split could be fixed by moving the first outer shell closer to `R` while keeping the current physics unchanged.
- First observation:
  - current default lane:
    - `outer_radius ≈ 0.65755`
    - `delta_r ≈ 0.19089`
    - decay length is `1 / lambda ≈ 0.06667`
    - so the first outer shell sits about `2.86` decay lengths away from `R`
- One-ring tightening alone was not viable:
  - when the first shell was pushed inward with the current law, parity destabilized badly
  - `t_in(R+)` remained near zero
  - `phi(R+)` and total parity worsened instead of improving
- A more principled three-ring refined family was then tried by moving the first two outer rings inward together.
- Outcome:
  - all three refined cases timed out before producing a usable parity report
  - so “better radial resolution with the current coupling” is not a practical fix by itself
- Conclusion:
  - under-resolution near `R+` is part of the problem
  - but the remaining mismatch is not a pure mesh-spacing issue
  - the free-side leaflet continuity/coupling law is still the more likely dominant gap

## Performance
- Exact reproducer-path benchmark, current branch:
  - `legacy_coarse`: `13.380 s`
  - `physical_edge_default`: `19.956 s`

## Benchmark Split
- `legacy_anchor` means the historical internal benchmark based on summed leaflet moduli:
  - `kappa = bending_modulus_in + bending_modulus_out`
  - `kappa_t = tilt_modulus_in + tilt_modulus_out`
  - this produces `thetaB_star ≈ 0.0923`
- `tex_benchmark` means the TeX convention in [docs/1_disk_3d.tex](/Users/User/github/membrane_solver/docs/1_disk_3d.tex):
  - `kappa = 1`
  - `kappa_t = 225`
  - `R = 7/15`
  - `hΔε/a = 4.286`
  - this produces `thetaB_star ≈ 0.185`

## Active Direction
- Keep `legacy_coarse` only as a regression/history anchor.
- Use `physical_edge_default` plus the `default_lo / default / default_hi` family as the active parity-development path.
- Evaluate future operator changes only against the physical-edge family and treat improvements to the coarse lane as incidental, not primary.
- The next real gap is no longer total energy parity; it is the missing TeX match in the free-membrane-side leaflet traces and outer height profile.
- Current free-side trace picture on the default lane:
  - `t_in(R+) ≈ -0.00132`
  - `t_out(R+) ≈ 0.04131`
  - `phi(R+) ≈ 0.09029`
- Director audit on the same default lane:
  - reconstructed disk-side director at `R-`: `~0.17624`
  - reconstructed free-side inner director at `R+`: `~0.08897`
  - reconstructed free-side outer director at `R+`: `~0.04131`
  - so the current gap is not only a tilt-decomposition issue; director continuity itself still appears to be broken in the discrete solution
- The most promising next stream is now continuity/coupling work on the free side of the interface:
  - inspect how `tilt_in` and `tilt_out` are formed/regularized on the first two outer shells
  - do not start with more mesh squeezing or more `rim_slope_match_out` row/weight edits
  - keep the current physical-edge family as the evaluation set while changing the free-side leaflet coupling

## Experimental Trace-Ring Attempt
- A diagnostics-only discretization experiment inserted an explicit free-side trace ring at `R + ε` with `ε ≈ 0.0333` (`trace_radius = 0.50` in current scaled units).
- Two variants were tested:
  - `trace_ring_free_geometry`
  - `trace_ring_planar_geometry`
- First outcome:
  - the initial local ring insertion broke refinement because the annulus faces were not rewritten consistently
- After rewriting the local annulus faces, the refined trace-ring variants did run through the full parity protocol.
- Refined outcome:
  - baseline current default:
    - `thetaB ≈ 0.18`
    - TeX `total_ratio ≈ 0.999`
  - free-geometry refined trace ring:
    - `thetaB ≈ 0.30`
    - TeX `total_ratio ≈ 1.584`
    - `phi(R+)` turns negative (`≈ -0.0427`)
    - director continuity becomes much worse (`disk_vs_free_inner_director_gap ≈ 0.427`)
  - planar-geometry refined trace ring:
    - `thetaB ≈ 0.30`
    - TeX `total_ratio ≈ 1.876`
    - `phi(R+)` collapses to `0`
    - director continuity remains very poor (`disk_vs_free_inner_director_gap ≈ 0.347`)
- Reassessment after fixing the actual construction:
  - the earlier `no_refine` runs were not valid tests of the intended idea because the inserted `R+ε` ring was free to drift radially in `[x,y]`
  - the trace-layer builder now pins that ring to the target circle, so the intended ordering is preserved:
    - `R ≈ 0.46667`
    - `R+ε ≈ 0.47667`
    - first free-side shell `R+ ≈ 0.48667` in the `no_refine` runs
  - corrected `no_refine` result with pinned `R+ε`:
    - `thetaB ≈ 0.32`
    - TeX `total_ratio ≈ 3.426`
    - `disk_t_in(R) ≈ 0.320`
    - `t_in(R+) ≈ -0.00166`
    - `t_out(R+) ≈ 0`
    - `phi(R+) ≈ 0`
  - corrected refined+retagged result with pinned `R+ε` and extra minimization:
    - `thetaB ≈ 0.21`
    - TeX `total_ratio ≈ 2.308`
    - `disk_t_in(R) ≈ 0.210`
    - `t_in(R+) ≈ 0.0095`
    - `t_out(R+) ≈ 0`
    - `phi(R+) ≈ 0`
  - both free and planar variants collapse onto the same bad branch once the ring is pinned radially; the planar option does not rescue parity
- Conclusion:
  - once the `R+ε` layer is actually kept at the intended radius, the trace-layer construction still does not solve parity
  - both `no_refine` and refined-retagged variants collapse to a bad branch with effectively zero free-side geometry trace
  - even with full refinement, adding a trace ring is not a viable parity fix in the current edge-only mesh/protocol
  - the planar ghost version is especially not acceptable as a production direction
  - interpreting the discrete disk rim as `[R, R+ε]` does not rescue the current formulation
  - if a trace layer is revisited later, it likely needs a more principled mesh/discretization treatment rather than a local ring insertion on the current topology

## Ghost-shell Recheck

- Revisited the outer ghost-shell idea as a discretization device, not as a `thetaB`-targeting trick.
- Current path:
  - use [tools/theory_parity_interface_profiles.py](/Users/User/github/membrane_solver/tools/theory_parity_interface_profiles.py) `build_trace_ring_fixture(...)`
  - keep the inserted ring pinned to a fixed circle via `pin_to_circle`
  - report direct shell values in [tools/reproduce_theory_parity.py](/Users/User/github/membrane_solver/tools/reproduce_theory_parity.py) under `metrics.diagnostics.interface_shell_at_R_plus_epsilon`
- The old extrapolated block `metrics.diagnostics.interface_traces_at_R` remains in place so the direct shell and traced values can be compared on the same run.

- Direct-shell acceptance coverage:
  - tracked fixture:
    - [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_ghost_eps005.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_ghost_eps005.yaml)
  - modular scaffold fixture for future near-`R` experiments:
    - [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_scaffold_eps005_n3_d005.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_scaffold_eps005_n3_d005.yaml)
    - built via `build_outer_shell_scaffold_fixture(...)` with:
      - trace shell at `R+ε`
      - `outer_shells=3`
      - `outer_shells_d=0.05`
    - shell radii:
      - `R ≈ 0.46667`
      - `R+ε ≈ 0.47167`
      - first support shell `≈ 0.52167`
- [tests/test_theory_parity_against_tex_acceptance.py](/Users/User/github/membrane_solver/tests/test_theory_parity_against_tex_acceptance.py)
    - `test_physical_edge_ghost_shell_reports_direct_outer_interface_shell`
    - `test_physical_edge_ghost_shell_fixture_relieves_known_bad_branch`
    - `test_physical_edge_ghost_shell_improves_direct_interface_behavior_over_bad_branch`
  - this confirms that when `parity_trace_layer_radius` is present:
    - the local shell builder picks the inserted shell as the first free shell
    - direct `t_in/t_out` match the first-shell quantities used by the old trace diagnostics
    - the parity report switches its `interface_primary_readout` source to `direct_trace_layer`

- Exact parity-path benchmark and observed behavior for the current fixed ghost shell with `ε = 0.005`:
  - baseline `physical_edge_default`:
    - runtime: `~18.02 s`
    - `thetaB = 0.18`
    - TeX `total_ratio ≈ 0.99921`
    - traced `t_in(R+) ≈ -0.00132`
    - traced `t_out(R+) ≈ 0.04131`
    - traced `phi(R+) ≈ 0.09029`
  - ghost-shell lane with direct `R+ε` shell:
    - runtime: `~10.87 s`
    - `thetaB ≈ 0.21`
    - TeX `total_ratio ≈ 1.101`
    - primary readout source: `direct_trace_layer`
    - direct `t_in(R+ε) ≈ 0.00672`
    - direct `t_out(R+ε) ≈ 0.00394`
    - direct outer-side slope `phi(R+ε) ≈ 1.76e-4`
    - direct free-inner vs free-outer director gap `≈ 0.00295`
    - traced `t_in(R+) ≈ 0.00682`
    - traced `t_out(R+) ≈ 0.00404`
    - traced `phi(R+) ≈ -4.46e-6`

- Interpretation:
  - the inserted shell removes the ambiguity about where the outer-side value is being read
  - the current operator change partially relieves the worst shell collapse:
    - `tilt_out` on the direct shell is now nonzero
    - the free-inner vs free-outer director mismatch on the direct shell is much smaller than in the original bad branch
    - TeX `total_ratio` is closer to `1` than the old ghost-shell collapse (`~1.155 -> ~1.101`)
  - but parity is still not fixed:
    - direct `phi(R+ε)` remains far too small
    - direct `t_in(R+ε)` remains well below the expected continuation scale
    - disk-to-free-inner continuity is still poor
  - so the ghost shell is now a useful diagnostic lane with partial operator relief, not a solved theory-parity lane
  - the new multi-shell scaffold builder is now available, but the `n=3, d=0.05` scaffold is not yet promoted as the active parity lane because the full parity solve on that fixture is currently unstable

## Current Scaffold / Operator Handoff

This section is the current handoff target for future parity work. It captures
the active scaffold fixtures, the exact test/run paths, the operator changes
already tried, and the latest measured behavior.

### Active files for parity work

- Reproducer / report path:
  - [tools/reproduce_theory_parity.py](/Users/User/github/membrane_solver/tools/reproduce_theory_parity.py)
    - active report source for:
      - `metrics.diagnostics.interface_traces_at_R`
      - `metrics.diagnostics.interface_shell_at_R_plus_epsilon`
      - `metrics.diagnostics.interface_primary_readout`
      - `metrics.diagnostics.trace_error_split`
- Fixture builders / shell scaffolds:
  - [tools/theory_parity_interface_profiles.py](/Users/User/github/membrane_solver/tools/theory_parity_interface_profiles.py)
    - `build_trace_ring_fixture(...)`
    - `build_outer_shell_scaffold_fixture(...)`
    - `build_gap_filled_outer_shell_scaffold_fixture(...)`
- Main theory-facing constraint/operator module:
  - [modules/constraints/rim_slope_match_out.py](/Users/User/github/membrane_solver/modules/constraints/rim_slope_match_out.py)
- Free-side tilt magnitude modules touched by the current scaffold work:
  - [modules/energy/tilt_in.py](/Users/User/github/membrane_solver/modules/energy/tilt_in.py)
  - [modules/energy/tilt_out.py](/Users/User/github/membrane_solver/modules/energy/tilt_out.py)
- Runtime cadence hook touched by scaffold projection work:
  - [runtime/minimizer.py](/Users/User/github/membrane_solver/runtime/minimizer.py)
- Primary acceptance suite:
  - [tests/test_theory_parity_against_tex_acceptance.py](/Users/User/github/membrane_solver/tests/test_theory_parity_against_tex_acceptance.py)
- Focused regression/unit suites:
  - [tests/test_rim_slope_match_out_constraint.py](/Users/User/github/membrane_solver/tests/test_rim_slope_match_out_constraint.py)
  - [tests/test_tilt_leaflet_pure.py](/Users/User/github/membrane_solver/tests/test_tilt_leaflet_pure.py)
  - [tests/test_local_interface_shells_unit.py](/Users/User/github/membrane_solver/tests/test_local_interface_shells_unit.py)
  - [tests/test_theory_parity_interface_sweep.py](/Users/User/github/membrane_solver/tests/test_theory_parity_interface_sweep.py)

### Active fixtures and what they mean

- Baseline production comparison lane:
  - [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml)
  - still the reference lane for `thetaB` and total-ratio guardrails
- Direct-shell ghost lane:
  - [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_ghost_eps005.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_ghost_eps005.yaml)
  - meaning:
    - disk boundary at `R`
    - explicit interface shell at `R+ε` with `ε = 0.005`
    - no extra support shells beyond that diagnostic layer
- Fixed-`d` scaffold lane:
  - [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_scaffold_eps005_n3_d005.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_scaffold_eps005_n3_d005.yaml)
  - meaning:
    - trace shell at `R+ε`
    - support shells at `R+ε+kd`, `k = 1..3`, with `d = 0.05`
- Gap-filled release scaffold lane:
  - [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_scaffold_gapfill_eps005_n3_release.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_scaffold_gapfill_eps005_n3_release.yaml)
  - meaning:
    - trace shell at `R+ε`
    - support shells distributed across the full gap to the original first free ring
    - one nearby unpinned release ring before the original far free ring
  - this is the main scaffold diagnostic lane for current operator work

### Scaffold geometry that future agents should keep in mind

- Disk radius:
  - `R = 7/15 ≈ 0.4666667`
- Trace-shell radius used in current scaffold work:
  - `R+ε ≈ 0.4716667`
- Fixed-`d` scaffold shell radii:
  - `0.4716667`, `0.5216667`, `0.5716667`, `0.6216667`, then the old far ring at `0.772`
- Gap-filled release scaffold shell radii:
  - trace shell `≈ 0.4716667`
  - support shells `≈ 0.5317333`, `0.5918`, `0.6518667`
  - release ring `≈ 0.7119333`
  - original old free ring `≈ 0.772`
- Important topology note:
  - the scaffold builder must compact edges after rebuilding the annulus strip
  - otherwise stale long disk-to-old-rim spokes remain in the mesh and produce large collision counts
  - after edge compaction, the scaffold lanes return to the same collision count as the zero-shell ghost lane (`12` vertex-edge collisions), instead of the earlier `48`

### Test and run entry points

- Normal exact parity reproducer:
  - [tools/reproduce_theory_parity.py](/Users/User/github/membrane_solver/tools/reproduce_theory_parity.py)
- Acceptance helper path:
  - [tests/test_theory_parity_against_tex_acceptance.py](/Users/User/github/membrane_solver/tests/test_theory_parity_against_tex_acceptance.py)
    - `_build_context(...)`
    - `_run_protocol_with_parity_activation(...)`
    - `_collect_report_from_context(...)`
- Default parity protocol:
  - imported in the acceptance file as `DEFAULT_PROTOCOL`
- Long scaffold interface protocol:
  - defined in [tests/test_theory_parity_against_tex_acceptance.py](/Users/User/github/membrane_solver/tests/test_theory_parity_against_tex_acceptance.py) as `LONG_INTERFACE_PROTOCOL`
  - exact sequence:
    - `g40`
    - `r`
    - `V5`
    - `g100`
    - then repeated tilt/energy passes:
      - `V1`, `energy` repeated many times
      - `V5`, `energy` repeated many times
      - `V10`, `energy` repeated three times
- Long repair suffix:
  - `LONG_INTERFACE_REPAIR_SUFFIX = ("u", "V2", "cg", "g20", "energy")`
- The long scaffold protocol is the main diagnostic sequence for current free-side work allocation debugging.

### Acceptance tests that matter right now

- Ghost/direct-shell diagnostics:
  - [tests/test_theory_parity_against_tex_acceptance.py](/Users/User/github/membrane_solver/tests/test_theory_parity_against_tex_acceptance.py)
    - `test_physical_edge_ghost_shell_reports_direct_outer_interface_shell`
    - `test_physical_edge_ghost_shell_fixture_relieves_known_bad_branch`
    - `test_physical_edge_ghost_shell_improves_direct_interface_behavior_over_bad_branch`
- Long scaffold branch characterization:
  - [tests/test_theory_parity_against_tex_acceptance.py](/Users/User/github/membrane_solver/tests/test_theory_parity_against_tex_acceptance.py)
    - `test_scaffold_long_interface_schedule_stays_on_inner_leaflet_only_branch`
    - `test_scaffold_gapfill_base_long_schedule_activates_outer_leaflet_without_repair`
    - `test_scaffold_long_interface_repair_turns_on_outer_leaflet_only_by_blowing_up_elastic_energy`
- Default-lane guardrails:
  - [tests/test_theory_parity_against_tex_acceptance.py](/Users/User/github/membrane_solver/tests/test_theory_parity_against_tex_acceptance.py)
    - `test_physical_edge_default_reports_trace_resolution_and_operator_split`
    - `test_physical_edge_default_keeps_theta_and_energy_in_guardrail_while_fixing_interface`
- Focused operator/unit tests:
  - [tests/test_rim_slope_match_out_constraint.py](/Users/User/github/membrane_solver/tests/test_rim_slope_match_out_constraint.py)
    - scaffold trace-shell row selection
    - scaffold shape/tilt projection semantics
  - [tests/test_tilt_leaflet_pure.py](/Users/User/github/membrane_solver/tests/test_tilt_leaflet_pure.py)
    - derived interface-shell weights for `tilt_in` and `tilt_out`
  - [tests/test_local_interface_shells_unit.py](/Users/User/github/membrane_solver/tests/test_local_interface_shells_unit.py)
    - local-shell builder behavior for trace/support shell selection
  - [tests/test_theory_parity_interface_sweep.py](/Users/User/github/membrane_solver/tests/test_theory_parity_interface_sweep.py)
    - fixture/scaffold generation regression coverage

### Current operator / discretization changes already in the branch

- [modules/energy/tilt_in.py](/Users/User/github/membrane_solver/modules/energy/tilt_in.py)
  and [modules/energy/tilt_out.py](/Users/User/github/membrane_solver/modules/energy/tilt_out.py)
  now apply explicit trace-layer row weights rather than treating the `R+ε`
  shell as a full bulk annulus or as a zero-weight dummy shell.
- The current interface-shell weight is derived from the actual radial fraction
  represented by the trace layer:
  - `shell_fraction = (rim_radius - disk_radius) / (outer_radius - disk_radius)`
  - row weight uses `sqrt(shell_fraction)`
- [modules/constraints/rim_slope_match_out.py](/Users/User/github/membrane_solver/modules/constraints/rim_slope_match_out.py)
  now supports scaffold-specific `tilt_rows` and trace-shell targeting in the
  physical-edge mode.
- The same module also now contains scaffold-only hard projection logic:
  - `enforce_constraint(...)`
  - this is a discrete interface-shell projector for the explicit `R+ε` shell
  - it currently performs a local joint update of:
    - trace-shell height / secant `phi`
    - trace-shell outer radial tilt `t_out`
- [runtime/minimizer.py](/Users/User/github/membrane_solver/runtime/minimizer.py)
  has a scaffold-only cadence hook during leaflet tilt relaxation:
  - it calls the hard shape projection only at tilt-block boundaries
  - it now gates that call on the current residual size from
    `matching_residual_diagnostics(...)`
  - the hard projection is skipped when the current rim mismatch is already small

### What was tried and what happened

- Dead branch under the long scaffold schedule:
  - both scaffold lanes originally stayed on:
    - `bending_tilt_out ≈ 0`
    - `tilt_out ≈ 0`
    - `phi(R+ε) ≈ 0`
  - this meant the outer leaflet was effectively not participating
- Repair branch under `LONG_INTERFACE_REPAIR_SUFFIX`:
  - the outer leaflet could be turned on only by appending:
    - `u`, `V2`, `cg`, `g20`, `energy`
  - but that produced a bad high-elastic branch:
    - positive total energy
    - very large elastic ratio
    - unacceptable parity
- Unconditional aggressive shape projection:
  - broke the dead branch
  - but drove the scaffold lane into a high-elastic/wrong-sign branch
- Exact joint local interface projector without residual gating:
  - improved `|t_in|` vs `|t_out|` mismatch significantly
  - but still over-drove the elastic part of the system
- Residual-gated scaffold projector:
  - current best overall balance so far
  - keeps the outer channel active
  - avoids the previous large elastic overshoot
  - but still does not satisfy the shell parity target

### Latest measured scaffold result

These are the latest useful numbers for the main scaffold lane:
- fixture:
  - [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_scaffold_gapfill_eps005_n3_release.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_scaffold_gapfill_eps005_n3_release.yaml)
- protocol:
  - `LONG_INTERFACE_PROTOCOL`
- result:
  - `t_in(R+ε) ≈ -0.07272`
  - `t_out(R+ε) ≈ 0.23136`
  - `||t_in|-|t_out|| ≈ 0.15864`
  - `bending_tilt_out ≈ 0.39011`
  - `tilt_out ≈ 0.25613`
  - internal energy `≈ 1.75537`
  - external work `≈ -4.90121`
  - total energy `≈ -3.14584`
- interpretation:
  - the dead branch is no longer the main issue
  - the large elastic blow-up branch is also no longer the main issue
  - the remaining issue is a work-allocation split:
    - `t_out` still stays materially larger than `|t_in|`
    - the local projector is still carrying too much of the old outer-tilt state forward

### Latest failed inner-targeting experiment

- A follow-up scaffold-only experiment changed `physical_edge_staggered_v1` so
  scaffold lanes no longer routed the scalar-`thetaB` inner law through disk
  rows when `rim_slope_match_thetaB_param` is active.
- Instead, the inner constraint was evaluated on the free-side interface shell.
- Outcome on the gap-filled release scaffold lane under `LONG_INTERFACE_PROTOCOL`:
  - `thetaB ≈ 0.54`
  - `t_in(R+ε) ≈ 0.54`
  - `t_out(R+ε) ≈ 5.6e-08`
  - `phi(R+ε) ≈ 5.6e-08`
  - `bending_tilt_out ≈ 7.5e-4`
  - `tilt_out ≈ 7.5e-5`
- Interpretation:
  - this did not fix the parity split
  - it simply flipped the scaffold lane onto a different bad branch:
    - oversized `thetaB`
    - oversized free-side `t_in`
    - collapsed outer geometry and `t_out`
- Practical conclusion:
  - pure disk-side inner targeting is wrong
  - pure free-side inner targeting is also wrong
  - the next candidate, if the inner law stays in this module, is a scaffold-only
    blended inner targeting rule rather than either extreme

### Current disk/free profile diagnosis

- On the same failed inner-targeting branch, the radial means show that the
  inside-disk profile is not the expected smooth `I1`-type increase:
  - several inner rings still have small negative mean `t_in`
  - near the edge the profile is non-monotone and jumps sharply:
    - `r ≈ 0.45597`: `t_in ≈ 0.0814`
    - `r = R ≈ 0.46667`: `t_in ≈ -0.1883`
    - `r = R+ε ≈ 0.47167`: `t_in ≈ 0.54`
- So the current scaffold branch does not preserve the desired disk-side
  `tilt_in` behavior.
- On the free side:
  - `t_in` does decay after `R+ε`, but from the wrong branch:
    - `R+ε ≈ 0.47167`: `t_in ≈ 0.54`
    - `0.53173`: `t_in ≈ 0.1575`
    - `0.59180`: `t_in ≈ 0.0441`
    - `0.65187`: `t_in ≈ 0.0170`
  - `t_out` remains essentially zero everywhere outside
- So the current branch is not a physically acceptable “grow on `[0,R]`, decay on
  `[R+ε,\infty)`” realization; it is an inner-only oversized-`thetaB` branch.

### Important diagnostic finding: direct-shell `phi` convention bug

- The earlier direct-shell report under
  `metrics.diagnostics.interface_shell_at_R_plus_epsilon.phi_secant_at_R_plus_epsilon`
  was using the shell-to-next-shell slope.
- The constraint module itself uses the disk-to-trace-shell secant over `ε`.
- This mismatch created a false impression that the scaffold lane had a sign
  inconsistency between reported `phi(R+ε)` and the constraint’s own residuals.
- [tools/reproduce_theory_parity.py](/Users/User/github/membrane_solver/tools/reproduce_theory_parity.py)
  was corrected so the direct-shell report now uses the same disk-to-trace-shell
  secant as the constraint.
- After that fix:
  - the scaffold report and `matching_residual_diagnostics(...)` agree on the
    interface `phi`
  - the remaining issue is genuinely branch selection / work allocation, not a
    reporting-only discrepancy

### Current main suspicion on the oversized-`thetaB` scaffold branch

- The gap-filled scaffold fixture still has:
  - `tilt_thetaB_optimize: true`
  - `tilt_thetaB_optimize_every: 1`
  - `rim_slope_match_thetaB_param: tilt_thetaB_value`
- So the scaffold lane is still running the scalar-`thetaB` optimizer every
  iteration while the scaffold interface law is also reading that same scalar.
- This is now a primary suspect for why the scaffold branch can inflate
  `thetaB` to `~0.39` and then `~0.54` under different targeting choices.
- Future diagnosis should isolate this explicitly before more operator surgery:
  - compare the scaffold lane with `tilt_thetaB_optimize` disabled
  - then compare with optimization enabled but a fixed `thetaB` seed
  - only then decide whether the branch problem is mainly:
    - the scalar `thetaB` optimizer
    - the interface projector
    - or their interaction

### New modeling direction under investigation

- The current scaffold diagnostics now support a stronger theory-facing conclusion:
  on parity/scaffold lanes, `thetaB` should likely be treated as a measured
  boundary value, not as an optimized scalar that feeds back into the interface law.
- In the continuum theory used in [docs/1_disk_3d.tex](/Users/User/github/membrane_solver/docs/1_disk_3d.tex):
  - `thetaB` is the disk-side boundary tilt measured at `r = R`
  - it is an output of the field solution
  - it is not an independently optimized degree of freedom
- The current discrete scaffold path originally did the opposite:
  - [runtime/minimizer.py](/Users/User/github/membrane_solver/runtime/minimizer.py)
    could optimize `tilt_thetaB_value`
  - [modules/constraints/rim_slope_match_out.py](/Users/User/github/membrane_solver/modules/constraints/rim_slope_match_out.py)
    then read that same scalar through `rim_slope_match_thetaB_param`
  - so the interface law was driven by an optimizer-controlled scalar rather than
    by the measured disk boundary field
- That decoupling experiment has now been implemented on scaffold lanes:
  - scaffold `rim_slope_match_out` no longer uses `theta_scalar`
  - scaffold report `metrics.thetaB_value` is now measured from disk-side
    `tilt_in` at `r = R`
  - scaffold `tilt_thetaB_optimize` is bypassed in
    [runtime/minimizer.py](/Users/User/github/membrane_solver/runtime/minimizer.py)
- The result is decisive:
  - the scaffold lane collapses to the trivial zero branch
  - measured `thetaB` becomes approximately zero
  - `t_in(R+ε)`, `t_out(R+ε)`, and `phi(R+ε)` all become approximately zero
- This is not just a numerical accident. The structural reason is now clear:
  - with `tilt_thetaB_contact_penalty_mode: off`, the contact module
    [modules/energy/tilt_thetaB_contact_in.py](/Users/User/github/membrane_solver/modules/energy/tilt_thetaB_contact_in.py)
    contributes only the scalar work term `-2π R_eff γ θ_B`
  - in that mode it adds no field gradient to `tilt_in`
  - so once scaffold lanes stop using the optimized scalar `thetaB` as the
    interface driver, the field equations lose the only nontrivial boundary forcing
  - the zero branch then becomes the natural minimum of the remaining field solve
- Controlled scaffold runs now imply:
  - optimizer on with scalar feedback:
    - nontrivial branches exist, but `thetaB` can inflate to `~0.39` or `~0.54`
    - work allocation is wrong
  - optimizer off / measured-`thetaB` only:
    - branch collapses to zero because the scalar contact work no longer drives
      the field at all
- Working hypothesis:
  - the problem is deeper than “report `thetaB` correctly”
  - the current parity/scaffold formulation has no theory-aligned mechanism that
    both:
    - lets `thetaB` emerge from the solved field
    - and still injects the correct nontrivial boundary driving without falling
      back to scalar-feedback branch selection
- Current recommended investigation order:
  1. stop revisiting scalar-decoupling; that part is done
  2. design a scaffold-only branch-selection / boundary-driving mechanism that
     comes from the field formulation itself rather than scalar `thetaB` feedback
  3. likely candidates to evaluate next:
     - a blended inner-targeting rule between disk-side and free-side rows
     - a theory-aligned reduced-DOF / reduced-energy path where `thetaB` enters
       as a solved boundary mode rather than a runtime scalar optimizer
- This is now the highest-signal theory-facing next step, and it should be
  treated as a modeling change rather than a small local projector tweak.

### Default-lane guardrail status

- The default lane remains the production comparison lane.
- The scaffold work should not be promoted there yet.
- Keep using:
  - [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_default.yaml)
  - plus the acceptance guardrails in
    [tests/test_theory_parity_against_tex_acceptance.py](/Users/User/github/membrane_solver/tests/test_theory_parity_against_tex_acceptance.py)
- The current scaffold operator work is still development-only and should be evaluated first on the scaffold lane before any promotion.

### Recommended next fixes

- Highest-signal next direction:
  - keep the residual-gated scaffold projector
  - change the local interface projector so it depends less on the current
    outer radial tilt carry-over and more on the inner continuity target
    `theta_disk - t_in`
- Do not start with:
  - more shell insertion
  - more `rim_slope_match_out` row-swapping
  - target-function fitting
  - new physics terms or ad hoc penalties
- The next likely useful change is still in:
  - [modules/constraints/rim_slope_match_out.py](/Users/User/github/membrane_solver/modules/constraints/rim_slope_match_out.py)
  - not in the scaffold builder
- Any future runtime cadence change must still be benchmarked on the exact
  parity path, because this is a performance-sensitive solver path.

### Minimal command/test set to resume work

When a future agent resumes this stream, the minimal useful set is:

- Read:
  - [docs/THEORY_PARITY_CHECKPOINT.md](/Users/User/github/membrane_solver/docs/THEORY_PARITY_CHECKPOINT.md)
  - [docs/1_disk_3d.tex](/Users/User/github/membrane_solver/docs/1_disk_3d.tex)
  - [modules/constraints/rim_slope_match_out.py](/Users/User/github/membrane_solver/modules/constraints/rim_slope_match_out.py)
  - [runtime/minimizer.py](/Users/User/github/membrane_solver/runtime/minimizer.py)
  - [tools/reproduce_theory_parity.py](/Users/User/github/membrane_solver/tools/reproduce_theory_parity.py)
  - [tools/theory_parity_interface_profiles.py](/Users/User/github/membrane_solver/tools/theory_parity_interface_profiles.py)
- Run first:
  - `pytest -q -o addopts='' tests/test_rim_slope_match_out_constraint.py::test_physical_edge_scaffold_shape_projection_moves_trace_shell_height tests/test_rim_slope_match_out_constraint.py::test_physical_edge_scaffold_uses_trace_layer_for_operator_tilt_rows`
  - `pytest -q -o addopts='' tests/test_theory_parity_against_tex_acceptance.py::test_scaffold_gapfill_base_long_schedule_activates_outer_leaflet_without_repair`
  - `pytest -q -o addopts='' tests/test_theory_parity_against_tex_acceptance.py::test_physical_edge_default_keeps_theta_and_energy_in_guardrail_while_fixing_interface`
- Then, if changing shell generation:
  - `pytest -q -o addopts='' tests/test_local_interface_shells_unit.py tests/test_theory_parity_interface_sweep.py`
- Then benchmark the exact reproducer path on the scaffold fixture and the default fixture before claiming progress.

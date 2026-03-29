# Theory Parity Checkpoint

## Current State
- The active fixed-lane parity fixture is [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml).
- The current reproducer is [tools/reproduce_theory_parity.py](/Users/User/github/membrane_solver/tools/reproduce_theory_parity.py).
- The coarse legacy lane still uses the physical disk ring at `R = 7/15` for `tilt_thetaB_group_in: disk`, but computes the outer slope channel through tagged `rim` and `outer` groups, with the `rim` shell sitting at radius `1.0`.
- The reconciled runtime now also supports `rim_slope_match_mode: physical_edge_staggered_v1`, which builds the outer slope channel from local shells just outside the physical disk edge.
- Current report values on this branch are approximately:
  - `thetaB_value = 0.09`
  - `elastic_measured = 0.56177`
  - `contact_measured = -1.13105`
  - `final_energy = -0.56928`

## Reconcile Source
- The later parity work lives in the separate worktree `/private/tmp/membrane_solver_tiltin_parity`.
- That worktree contains the validated intermediate and near-edge lanes, but they are not yet reconciled into this repo checkout.
- The first reusable artifact ported from that worktree is [tools/theory_parity_interface_profiles.py](/Users/User/github/membrane_solver/tools/theory_parity_interface_profiles.py), which captures the named near-edge shell profiles:
  - `i50 = (0.8, 2.8)`
  - `i60 = (0.76, 2.6)`
  - `near_edge_v1 = (0.76, 2.6)`

## Validated Lanes In The Parity Worktree
- `i50`
  - `thetaB = 0.17`
  - `final_energy = -1.13080`
  - `total_ratio = 0.97503`
  - `outer_radius ≈ 0.67382`
- `i60`
  - `thetaB = 0.19`
  - `final_energy = -1.23705`
  - `total_ratio = 1.06664`
  - `outer_radius ≈ 0.64790`
- `near_edge_v1`
  - `thetaB = 0.18`
  - `final_energy = -1.15545`
  - `total_ratio = 0.99628`
  - `outer_radius ≈ 0.65588`

These results came from the parity worktree reproducer, not from the current repo state. They remain useful parity-history checkpoints, but not current acceptance targets here.

## Reconciled Tracked Lanes In This Repo
- The runtime slice required to run named physical-edge lanes is now in this repo.
- Tracked fixtures now exist for:
  - [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_i50_interface.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_i50_interface.yaml)
  - [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_i60_interface.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_i60_interface.yaml)
  - [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_near_edge_v1.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_near_edge_v1.yaml)
- Current local baselines for those tracked fixtures are:
  - `i50_interface_v1`
    - `thetaB = 0.17`
    - `final_energy = -1.12596`
    - `tex total_ratio = 0.97086`
    - `outer_radius ≈ 0.67365`
  - `i60_interface_v1`
    - `thetaB = 0.19`
    - `final_energy = -1.23225`
    - `tex total_ratio = 1.06250`
    - `outer_radius ≈ 0.64749`
  - `near_edge_v1`
    - `thetaB = 0.18`
    - `final_energy = -1.15545`
    - `tex total_ratio = 0.99628`
    - `outer_radius ≈ 0.65588`
- After reconciling the scalar-disk inner enforcement path in [rim_slope_match_out.py](/Users/User/github/membrane_solver/modules/constraints/rim_slope_match_out.py), the local tracked lanes are now much closer to the old parity worktree and are useful acceptance targets again.

## Reconciled Diagnostics In This Repo
- Added reusable diagnostics:
  - [theory_parity_interface_sweep.py](/Users/User/github/membrane_solver/tools/theory_parity_interface_sweep.py)
  - [theory_parity_fixed_theta_compare.py](/Users/User/github/membrane_solver/tools/theory_parity_fixed_theta_compare.py)
- Current local interface sweep result:
  - best label: `i60`
  - `i60`: `thetaB = 0.18`, TeX `total_ratio = 0.99628`, `outer_radius ≈ 0.65588`
  - `near_edge_v1`: `thetaB = 0.18`, TeX `total_ratio = 0.99628`, `outer_radius ≈ 0.65588`
  - `i50`: `thetaB = 0.17`, TeX `total_ratio = 0.92539`, `outer_radius ≈ 0.67647`
  - `coarse`: `thetaB = 0.09`, TeX `total_ratio = 0.49086`, `outer_radius ≈ 1.96540`
- Current fixed-`thetaB = 0.185` comparison:
  - `coarse`: `elastic = 2.14084`, `contact = -2.32493`, `total = -0.18409`, `outer_radius ≈ 1.96582`
  - `i50`: `elastic = 1.26263`, `contact = -2.32493`, `total = -1.06230`, `outer_radius ≈ 0.67647`
  - `near_edge_v1`: `elastic = 1.00471`, `contact = -2.32493`, `total = -1.32022`, `outer_radius ≈ 0.62866`
- Practical conclusion from the current repo state:
  - the parity-protocol helpers in [reproduce_theory_parity.py](/Users/User/github/membrane_solver/tools/reproduce_theory_parity.py) materially change the measured parity state
  - the scalar-`thetaB` disk-side enforcement path in [rim_slope_match_out.py](/Users/User/github/membrane_solver/modules/constraints/rim_slope_match_out.py) was a real missing runtime slice; after reconciling it, the tracked physical-edge lanes moved back near the old parity-worktree behavior
  - coarse legacy runs now expose a finite outer shell instead of `NaN`, and the coarse lane is back to a non-collapsed `thetaB ≈ 0.09`
  - contact remains stable across the profiled fixed-`thetaB` lanes
  - the recovered inner-divergence operator, parity-protocol helpers, and scalar-disk enforcement slice all materially change parity behavior
  - follow-up diff inspection against `/private/tmp/membrane_solver_tiltin_parity/tools/reproduce_theory_parity.py` shows the remaining unported changes there are now mostly diagnostics, audits, and report-schema expansion rather than more protocol-loop behavior
  - the repo is now close enough to the old parity worktree that the next step should be principled generalization from `near_edge_v1` / `i60`, not more blind reconcile patches

## Performance
- Exact reproducer-path benchmark, 3 runs each:
  - coarse lane: `10.347 s` average
  - `i50` lane: `24.467 s` average

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

## What This PR Slice Does
- Keeps the existing legacy lane intact for regression tracking.
- Makes the report explicit by writing both `metrics.tex_benchmark` and `metrics.legacy_anchor`.
- Leaves `metrics.theory` as a compatibility alias for the legacy anchor in this slice so existing tools keep working.
- Moves TeX acceptance to the explicit benchmark fields instead of the legacy alias.

## Next Implementation Slices
- Reconcile the remaining runtime/operator pieces needed to recover the old parity-worktree behavior on top of the now-tracked lane assets.
- Use the fixed-`thetaB` compare to identify which operator/runtime changes restore the old worktree ordering without changing contact.
- Re-evaluate the inner-disk operator changes only after the physical-edge runtime path has been reconciled.

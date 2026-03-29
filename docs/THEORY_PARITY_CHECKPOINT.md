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
- This lane is derived from the same physical-edge construction as the earlier `near_edge_v1` reference, but is now tracked as the generic bottom-up default rather than as a one-off named fix.
- Current kept interface-side improvement:
  - the physical-edge law now pairs the first outer shell to disk-boundary rows by explicit nearest azimuth (`rim_rows_for_disk`) instead of relying on independently ordered rings
  - a second-shell-supported composition was tested and produced the same behavior on the current family, so it was not kept as a separate runtime change
  - the local-shell builder now uses an order-preserving cyclic azimuth match when adjacent rings have equal counts; this cleans up pair regularity but does not materially change the current parity metrics

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
  - but the outer-leaflet trace and outer height profile are still not near the TeX continuation law
  - the family remains in a non-pathological regime and can be used as the active parity-development base

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
- The next real gap is no longer total energy parity; it is the missing TeX match in the outer-leaflet trace and outer height profile.

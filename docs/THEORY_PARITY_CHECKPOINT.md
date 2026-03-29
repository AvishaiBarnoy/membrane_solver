# Theory Parity Checkpoint

## Current State
- The active reproducer is [tools/reproduce_theory_parity.py](/Users/User/github/membrane_solver/tools/reproduce_theory_parity.py).
- The coarse fixture [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity.yaml) is now treated as `legacy_coarse` only.
- The active parity-development fixture is [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_primary.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_primary.yaml).
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

## Physical-Edge Primary Lane
- The new tracked primary lane is [tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_primary.yaml](/Users/User/github/membrane_solver/tests/fixtures/kozlov_1disk_3d_free_disk_theory_parity_physical_edge_primary.yaml).
- Current primary report:
  - `thetaB = 0.19`
  - `final_energy = -1.18843`
  - `tex total_ratio = 1.02472`
  - `tex elastic_ratio = 1.03413`
  - `rim_radius ≈ 0.46667`
  - `outer_radius ≈ 0.65143`
  - `phi_mean ≈ 0.00450`
  - `phi_over_half_theta ≈ 0.04737`
- This lane is derived from the same physical-edge construction as the earlier `near_edge_v1` reference, but is now tracked as the generic development base rather than as a one-off named fix.

## Physical-Edge Family
- The profile helper in [tools/theory_parity_interface_profiles.py](/Users/User/github/membrane_solver/tools/theory_parity_interface_profiles.py) now defines the generic family:
  - `physical_edge_family_lo = (0.78, 2.7)`
  - `physical_edge_primary_v1 = (0.76, 2.6)`
  - `physical_edge_family_hi = (0.758, 2.6)`
- Current optimized sweep on this branch:
  - `physical_edge_family_lo`: `thetaB = 0.18`, `tex total_ratio = 0.98657`, `outer_radius ≈ 0.66166`
  - `physical_edge_primary_v1`: `thetaB = 0.19`, `tex total_ratio = 1.02472`, `outer_radius ≈ 0.65143`
  - `physical_edge_family_hi`: `thetaB = 0.19`, `tex total_ratio = 1.02947`, `outer_radius ≈ 0.65041`
  - `coarse`: `thetaB = 0.09`, `tex total_ratio = 0.49139`, `outer_radius ≈ 0.72784`
- Current fixed-`thetaB = 0.185` comparison:
  - `coarse`: `elastic = 2.13375`, `contact = -2.32493`, `total = -0.19119`, `phi_over_half_theta ≈ 0.00643`
  - `physical_edge_family_lo`: `elastic = 0.98336`, `contact = -2.32493`, `total = -1.34157`, `phi_over_half_theta ≈ 0.13575`
  - `physical_edge_primary_v1`: `elastic = 0.96742`, `contact = -2.32493`, `total = -1.35751`, `phi_over_half_theta ≈ 0.21415`
  - `physical_edge_family_hi`: `elastic = 0.96153`, `contact = -2.32493`, `total = -1.36341`, `phi_over_half_theta ≈ 0.21802`
- Practical conclusion:
  - contact remains effectively fixed across the physical-edge family
  - elastic decreases smoothly as the near-edge shell is tightened
  - the family is now in a non-pathological regime and can be used as the active parity-development base

## Performance
- Exact reproducer-path benchmark, current branch:
  - `legacy_coarse`: `11.865 s`
  - `physical_edge_primary_v1`: `19.439 s`

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
- Use `physical_edge_primary_v1` plus the `lo / primary / hi` family as the active parity-development path.
- Evaluate future operator changes only against the physical-edge family and treat improvements to the coarse lane as incidental, not primary.

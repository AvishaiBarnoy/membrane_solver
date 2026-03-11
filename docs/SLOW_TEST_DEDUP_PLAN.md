# Slow Test Dedup Plan

## Goal

Reduce the runtime of the explicit slow suite

```bash
pytest -q -o addopts='' -m "acceptance or benchmark or e2e or slow or script"
```

without dropping unique behavior coverage.

Current observed runtime:

- fast default suite: about 72s
- explicit slow suite: about 70 minutes

## Current state

The explicit slow suite is now green. The remaining problem is overlap, not
correctness.

The dominant cost comes from repeated calls to the same benchmark and script
entry points:

- `run_flat_disk_one_leaflet_benchmark(...)`
- `run_flat_disk_kh_term_audit(...)`
- `run_flat_disk_kh_term_audit_refine_sweep(...)`
- `tools/reproduce_theory_parity.py`
- `tools/diagnostics/physics_sweep.py`

## Inventory by behavior cluster

### 1. Flat benchmark path

Primary overlapping files:

- `/Users/User/github/membrane_solver/tests/test_flat_disk_one_leaflet_benchmark_e2e.py`
- `/Users/User/github/membrane_solver/tests/test_flat_disk_one_leaflet_optimize_preset_benchmark.py`
- `/Users/User/github/membrane_solver/tests/test_flat_disk_one_leaflet_optimize_mode_tradeoff_benchmark.py`
- `/Users/User/github/membrane_solver/tests/test_flat_disk_tilt_mass_mode_benchmark.py`
- `/Users/User/github/membrane_solver/tests/test_flat_disk_one_leaflet_curved_lane_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_reproduce_flat_disk_one_leaflet_acceptance.py`

Observed overlap:

- `test_flat_disk_one_leaflet_benchmark_e2e.py` is the largest concentration of
  repeated benchmark calls.
- Several smaller benchmark files exercise the same benchmark entry point with
  narrow parameter changes.

Recommended main slow-suite retention:

- keep one broad benchmark/E2E module:
  - `/Users/User/github/membrane_solver/tests/test_flat_disk_one_leaflet_benchmark_e2e.py`
- keep one acceptance snapshot module:
  - `/Users/User/github/membrane_solver/tests/test_reproduce_flat_disk_one_leaflet_acceptance.py`
- keep curved-lane acceptance separately because it covers a distinct lane:
  - `/Users/User/github/membrane_solver/tests/test_flat_disk_one_leaflet_curved_lane_acceptance.py`

Recommended nightly/exhaustive candidates:

- `/Users/User/github/membrane_solver/tests/test_flat_disk_one_leaflet_optimize_preset_benchmark.py`
- `/Users/User/github/membrane_solver/tests/test_flat_disk_one_leaflet_optimize_mode_tradeoff_benchmark.py`
- `/Users/User/github/membrane_solver/tests/test_flat_disk_tilt_mass_mode_benchmark.py`

Reason:

- these are parameter-sensitivity checks on the same benchmark driver
- they are useful, but they overlap heavily with the broad benchmark module

### 2. KH term audit path

Primary overlapping files:

- `/Users/User/github/membrane_solver/tests/test_flat_disk_kh_term_audit_regression.py`
- `/Users/User/github/membrane_solver/tests/test_flat_disk_kh_region_parity_regression.py`
- `/Users/User/github/membrane_solver/tests/test_flat_disk_kh_partition_ablation_regression.py`

Observed overlap:

- `test_flat_disk_kh_term_audit_regression.py` already covers most audit
  semantics directly.
- The smaller region/partition regression files mostly validate wrapper logic
  around the same audit path.

Recommended main slow-suite retention:

- keep:
  - `/Users/User/github/membrane_solver/tests/test_flat_disk_kh_term_audit_regression.py`

Recommended nightly/exhaustive candidates:

- `/Users/User/github/membrane_solver/tests/test_flat_disk_kh_region_parity_regression.py`
- `/Users/User/github/membrane_solver/tests/test_flat_disk_kh_partition_ablation_regression.py`

Reason:

- wrapper-level overlap with the main KH audit regression file

### 3. Theory parity / theory CLI path

Primary overlapping files:

- `/Users/User/github/membrane_solver/tests/test_reproduce_theory_parity_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_parity_against_tex_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_parity_trend.py`
- `/Users/User/github/membrane_solver/tests/test_reproduce_theory_parity_fixed_polish.py`
- `/Users/User/github/membrane_solver/tests/test_theory_equivalence_audit_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_scaling_audit_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_mesh_convergence_audit_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_coverage_audit_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_protocol_suffix_sweep_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_parity_drift_triage.py`
- `/Users/User/github/membrane_solver/tests/test_theory_parity_expansion_exploratory.py`

Observed overlap:

- many of these shell out to the same theory CLI and inspect adjacent artifact
  families
- they are all valuable, but not all need to live in the main slow path

Recommended main slow-suite retention:

- `/Users/User/github/membrane_solver/tests/test_reproduce_theory_parity_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_parity_against_tex_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_parity_trend.py`

Reason:

- together they cover:
  - baseline artifact correctness
  - TeX target comparison
  - trend artifact generation

Recommended nightly/exhaustive candidates:

- `/Users/User/github/membrane_solver/tests/test_reproduce_theory_parity_fixed_polish.py`
- `/Users/User/github/membrane_solver/tests/test_theory_equivalence_audit_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_scaling_audit_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_mesh_convergence_audit_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_coverage_audit_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_protocol_suffix_sweep_acceptance.py`
- `/Users/User/github/membrane_solver/tests/test_theory_parity_drift_triage.py`
- `/Users/User/github/membrane_solver/tests/test_theory_parity_expansion_exploratory.py`

### 4. CLI / script hygiene

Primary files:

- `/Users/User/github/membrane_solver/tests/test_cli_end_to_end.py`
- `/Users/User/github/membrane_solver/tests/test_cli_benchmark_end_to_end.py`
- `/Users/User/github/membrane_solver/tests/test_visualization_hygiene.py`
- `/Users/User/github/membrane_solver/tests/test_tilt_benchmark_runner.py`
- `/Users/User/github/membrane_solver/tests/test_profile_macro_hotspots.py`
- `/Users/User/github/membrane_solver/tests/test_physics_sweep_inventory.py`

Observed overlap:

- these are mostly unique script entry points
- overlap is lower here than in benchmark/theory modules

Recommended main slow-suite retention:

- keep all for now except where exact CLI overlap is proven later

### 5. Kozlov E2E family

Primary files:

- `/Users/User/github/membrane_solver/tests/test_kozlov_*.py`
- `/Users/User/github/membrane_solver/tests/test_bending_tilt_leaflet_e2e.py`
- `/Users/User/github/membrane_solver/tests/test_single_leaflet_curvature_induction.py`
- `/Users/User/github/membrane_solver/tests/test_tilt_source_decay_e2e.py`

Observed overlap:

- there is likely overlap within the Kozlov free-disk family
- but that needs a second pass by scenario, not just by filename

Recommended action:

- defer Kozlov dedup to a separate pass
- do not reclassify these yet without scenario-level review

## Proposed tier structure

### Fast default

Already implemented:

```bash
pytest -q
```

### Main slow suite

Keep one representative test per unique expensive behavior:

```bash
pytest -q -o addopts='' -m "(acceptance or benchmark or e2e or slow or script) and not exhaustive"
```

after reclassifying the overlapping modules above.

### Nightly / exhaustive

Add a new marker, for example `exhaustive`, for overlap-heavy tests that still
have value but should not stay in the main slow path.

Candidate first moves:

- optimize-preset benchmark module
- optimize-mode tradeoff benchmark module
- tilt-mass-mode benchmark module
- theory parity auxiliary audit modules
- KH audit wrapper regressions

## Next dedup PR

1. Add an `exhaustive` marker in `pytest.ini`.
2. Move the clearly overlapping benchmark/theory modules above from the main
   slow suite to the exhaustive tier.
3. Re-run:
   - `pytest -q`
   - `pytest -q -o addopts='' -m "(acceptance or benchmark or e2e or slow or script) and not exhaustive"`
   - optional exhaustive collection sanity check:
     `pytest -q -o addopts='' -m "exhaustive"`
4. Measure the runtime reduction.

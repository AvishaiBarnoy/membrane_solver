# Test Taxonomy and Suite Contract

Status: **active**
Owner: `tests/conftest.py` and `pytest.ini`

## Classification rule

Every collected test has at least one category marker.

1. An explicit category marker is authoritative:
   `unit`, `regression`, `e2e`, `acceptance`, `benchmark`, `slow`, `script`, or
   `exhaustive`.
2. If no explicit category is present, `pytest_collection_modifyitems` assigns
   one deterministic fallback from the filename:
   - `benchmark` in the filename → `benchmark`;
   - `e2e` or `end_to_end` → `e2e`;
   - `regression` → `regression`;
   - otherwise → `unit`.
3. A test may additionally carry a cost marker such as `slow`, `script`, or
   `exhaustive`; an explicit cost marker prevents fallback classification.

This gives complete collection classification without relying on a manually
maintained 300-file list. The source-to-test map in `subsystems.yaml` remains a
separate, intentionally partial impact map for high-risk work.

## Suite selection

| Suite | Command | Purpose |
|---|---|---|
| Default | `pytest -q` | fast confidence suite; excludes acceptance, benchmark, e2e, slow, script, and exhaustive tests |
| Focused subsystem | `pytest -q <paths>` | change-manifest validation |
| Main slow | `pytest -q -o addopts='' -m "(acceptance or benchmark or e2e or slow or script) and not exhaustive"` | representative expensive behavior |
| Exhaustive | `pytest -q -o addopts='' -m exhaustive` | overlap-heavy/nightly scenarios |
| Complete collection | `pytest --collect-only -q -o addopts=''` | taxonomy and discovery health |

## Explicit non-default workflows

The following verified subprocess or diagnostic workflows are explicitly kept
out of the default suite:

| Test path | Markers | Rationale |
|---|---|---|
| `tests/test_tilt_benchmark_runner.py` | `slow`, `script` | invokes the benchmark-runner subprocess over mesh fixtures |
| `tests/test_profile_macro_hotspots.py` | `slow`, `script` | invokes profiling subprocess and writes profiling artifacts |
| `tests/test_scaffold_energy_imbalance_audit.py` | `slow`, `script` | runs scaffold diagnostic/reproducer workflows |
| CLI schema test in `tests/test_thetaB_cadence_relaxation_audit.py` | `slow`, `script` | invokes diagnostic CLI and writes YAML/Markdown artifacts |
| Curved 1-disk diagnostic/audit modules | `slow` | theory-facing diagnostics whose workflows are not fast default confidence checks |
| Curved 1-disk support and curved-disk parity acceptance modules | `acceptance`, `slow` | scientific acceptance behavior kept out of default selection |
| Flat-disk benchmark, KH audit, rim-fidelity, and curved sweep modules | `slow` plus applicable regression/e2e/benchmark markers | timing baseline identified these as expensive scientific workflows |
| Shape-pullback audit in `tests/test_free_one_disc_shape_parity_audit.py` | `slow` (function level) | finite-difference geometry audit took more than 97 seconds; the two pure helper tests remain default |

## Maintenance rules

- Add explicit markers whenever a test's behavior differs from filename
  classification, especially for subprocess, artifact-producing, expensive, or
  scientific-acceptance tests.
- Keep a fast assertion in a mixed slow module unmarked only when it belongs in
  the default suite; otherwise mark at function level.
- Do not use marker changes to hide a failing test. Selection changes require a
  baseline result and a reason in the relevant manifest.
- Add high-risk source/test relationships to `subsystems.yaml`; do not infer a
  test is irrelevant because it is absent from that map.
- Reassess default runtime after every addition to this explicit non-default
  list.

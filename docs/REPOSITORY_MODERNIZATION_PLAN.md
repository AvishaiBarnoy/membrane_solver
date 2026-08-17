# Repository Modernization Plan

Status: **structural modernization complete; scientific requalification separate**

This is the sole repository-level authority for modernization scope, ownership,
phase status, and remaining work. Detailed files below are operational contracts,
not additional plans.

## Objective

Reduce the context and rediscovery cost of repository work while preserving
solver behavior, scientific lanes, mutation order, cache epochs, and report
schemas. Prefer a small cohesive module over either a monolith or many one-use
helpers. Count production code, tests, and governance together when evaluating
an extraction.

## Guardrails

- Preserve behavior unless a separately authorized change has a failing test
  first.
- Keep `theory_parity_lane`, `shared_rim_staggered_v1`, physical-edge, and
  explicit-trace branches intact.
- Keep numerical optimization in dense NumPy arrays; topology remains
  object-oriented.
- `Mesh` owns geometry, topology, row-order, and fixed-field mutation epochs.
- Do not combine structural compaction with fixture, tolerance, or scientific
  expectation changes.
- Reject an extraction when its complete production-and-test cluster grows
  without a concrete navigation, correctness, or runtime benefit.

## Subsystem routing

| Subsystem | Primary owners | Focused validation |
|---|---|---|
| Test selection | `pytest.ini`, `tests/conftest.py`, `tests/manifest/` | `tests/test_pytest_marker_classification_unit.py` |
| Minimization lifecycle | `runtime/minimizer.py`, `runtime/minimizer_helpers.py`, `runtime/steppers/` | minimizer, rollback, and reduced-line-search tests |
| Geometry/cache | `geometry/mesh.py`, `geometry/cache_checks.py`, `runtime/energy_context.py` | cache, mutation-hook, and energy-context tests |
| Tilt relaxation | `runtime/steppers/tilt_relaxation.py`, `runtime/projections/tilt.py` | tilt solve-mode, projection, and energy-guard tests |
| Energy/constraints | `runtime/evaluation_manager.py`, `modules/energy/`, `modules/constraints/` | manager, directional-derivative, and module parity tests |
| Topology/IO | `geometry/io_readers.py`, `geometry/polygon_triangulation.py`, `runtime/refinement.py` | geometry IO and refinement tests |
| Diagnostics/theory | `tools/diagnostics/`, `tools/reproduce_theory_parity.py` | diagnostic unit tests plus explicitly selected scientific lanes |

Use `tests/manifest/subsystems.yaml` for the compact source-to-test impact map
and `tests/manifest/TAXONOMY.md` for suite commands.

## Active contracts

| Contract | Authority |
|---|---|
| `docs/CACHE_DEPENDENCY_TABLE.md` | cache keys, epochs, and mutation ownership |
| `docs/ENERGY_CONSTRAINT_API_CONTRACT.md` | dense-array and compatibility dispatch boundaries |
| `docs/PARAMETER_COMPATIBILITY_INVENTORY.md` | aliases, defaults, and retirement gates |
| `docs/TILT_SOLVE_MODE_MATRIX.md` | tilt modes and mutation boundaries |
| `docs/TOPOLOGY_TRANSFER_CONTRACT.md` | refinement metadata transfer |
| `docs/DIAGNOSTIC_NECESSITY_INVENTORY.md` | diagnostic/tool retention decisions |
| `docs/DIAGNOSTIC_REPORT_SCHEMA_CONTRACT.md` | report schema stability |
| `docs/THEORY_PARITY_WORKFLOW_CONTRACT.md` | scientific workflow and lane ownership |
| `docs/THEORY_PARITY_REQUALIFICATION_EVIDENCE.md` | current scientific evidence |
| `docs/THEORY_PARITY_REBASELINE_MANIFEST.md` | candidate numeric rebaseline requiring approval |

Historical or speculative documents are context only. If they conflict with
this plan or an active contract, the active authority wins.

## Phase status

### Phase 1 — repository plan

Complete. Established goals, guardrails, architecture boundaries, and a
measured baseline.

### Phase 2 — subsystem plan

Complete. Established ownership and source-to-test routing for the seven active
subsystems above.

### Phase 3 — per-change manifests and low-risk extraction

Complete. Added test taxonomy, compatibility/cache/tilt/topology contracts,
input normalization, rejected-step policy boundaries, report serialization
inventory, and initial low-risk extractions. Temporary per-change manifests
were compacted into this phase record and the active contracts.

### Phase 4 — subsystem convergence

Complete. Clarified cache epochs, trial-array cache behavior, energy-only array
APIs, topology/IO ownership, minimizer state snapshots, and tilt policy/problem
data. Characterization-only decisions were retained where extraction would
fragment a cohesive lifecycle.

Key reviewed commits:

- `5ceed05` — native energy-only module APIs;
- `df8be0d` — parser/refinement ownership;
- `e93024b` — cache and module dispatch contracts;
- `1b30fa4` — minimizer state and tilt policies.

### Phase 5 — integration and scientific separation

Structural integration is complete. The accumulated tree was reviewed by
logical change set, focused tests were run, and only coherent artifacts were
committed:

- `ddc9994` removed two obsolete tools (73 lines);
- `4f1d68d` classified expensive test lanes and activated `exhaustive`;
- an oversized diagnostics extraction was rejected after its full cluster
  measured net +2,949 lines;
- `e2e2d39` replaced 2,516 lines of draft planning documents with a 657-line
  authority/contract set;
- `0b67fec` added focused input validation for the retained runtime probe.

Fixture and theory-acceptance changes remain unstaged until scientific
requalification is explicitly authorized.

Scientific requalification is not required to close structural integration. It
must remain a separate change set governed by the theory workflow contract and
requalification evidence.

## Change protocol

For each future structural change:

1. identify one owner and one behavior boundary;
2. record expected total line/file delta;
3. run focused tests before editing;
4. extract or compact without changing call order or mutation ownership;
5. rerun focused tests and import/collection checks;
6. remove superseded code only after the new owner is exercised;
7. reject or revise the change if the full cluster fails its stated benefit.

Do not create a permanent manifest for routine work. Update an active contract
only when its maintained boundary changes.

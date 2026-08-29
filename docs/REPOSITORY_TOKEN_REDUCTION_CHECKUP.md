# Repository Token-Reduction Checkup

Status: **test/CI detour closed; resume subsystem convergence**
Date: 2026-08-29

## Objective

Reduce the context and rediscovery cost of repository work while preserving
solver behavior and scientific evidence. Count production code, tests, and
governance together. Prefer bounded, cohesive, net-neutral or net-negative
changes over broad decomposition.

## Current baseline

- 551 tracked Python files and 126,949 Python lines.
- 42 Markdown files and 6,548 Markdown lines before this checkup.
- Largest areas: tests 46,303 lines; tools 38,149; modules 19,389; runtime
  10,659; geometry 5,088; commands 1,634.
- Since the Phase 4 branch point (`caca279`), current `main` is net 4,876 lines
  smaller: 3,556 additions and 8,432 deletions.
- The largest remaining context hotspots are diagnostic drivers, followed by
  `runtime/minimizer.py`, `runtime/steppers/tilt_relaxation.py`,
  `runtime/refinement.py`, and `geometry/mesh.py`.

## Phase reconciliation

| Phase | Current status on `main` | Decision |
|---|---|---|
| 1 — repository plan | Complete in substance through `AGENTS.md`, repository routing, guardrails, and measured baselines. | Do not restore the old long plan. |
| 2 — subsystem plan | Sufficient but intentionally compact. The detailed historical YAML map never merged. | Add routing only after repeated search cost is observed. |
| 3 — per-change manifests and low-risk extraction | Complete. Evaluation, projection, IO, topology, minimizer-helper, and diagnostic boundaries are present. | Continue using focused manifests transiently; do not accumulate permanent manifests. |
| 4 — subsystem convergence | **Incomplete on `main`.** Its historical helper-heavy implementation remained on a stale divergent branch. | Re-evaluate against the current tree; do not cherry-pick it wholesale. |
| 5 — integration and scientific separation | Complete. Fast, PR, background, exhaustive, and theory lanes are separated. | Frozen except under the reopen criteria in `SLOW_TEST_DEDUP_PLAN.md`. |
| 6 — retirement and ownership cleanup | Complete. Obsolete tools, diagnostic wrappers, prototypes, and duplicate disk/rim-source implementations were removed or consolidated. | Preserve the retained scientific lanes. |

The historical `codex/fix-seven-theory-audits` branch is evidence, not an
integration source. It contains useful experiments but predates later theory
fixes and repository compaction. Several proposed Phase 4 slices increased the
full production-and-test cluster despite improving locality.

## Active phase: 4R — convergence re-evaluation

Evaluate remaining hotspots in this order:

1. `runtime/minimizer.py` and rejected-step/scaffold ownership;
2. `runtime/steppers/tilt_relaxation.py` policy versus numerical loop ownership;
3. `runtime/refinement.py` parsing, policy, and topology ownership;
4. the three largest retained diagnostic drivers.

For each slice:

1. identify one owner and behavior boundary;
2. measure files and lines that a typical change must inspect;
3. use existing tests unless a real coverage gap is found;
4. preserve call order, mutation ownership, cache epochs, and report schemas;
5. reject the slice if it adds fragmentation or governance without a concrete
   navigation, correctness, or runtime benefit;
6. remeasure the complete production-test-documentation cluster.

Stop after three accepted slices and repeat this checkup. Do not return to
repository-wide test cleanup during Phase 4R.

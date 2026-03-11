# Validation Workflow

Use this file before changing runtime behavior, tests, or benchmarks for KH-related work.

## Acceptance-First

- Translate the requested behavior into an acceptance or integration test before editing runtime flow.
- If the change is local to a discrete operator or energy term, add a unit/regression test for the local invariant as well.
- Prefer failing tests that expose sign, tangency, or operator-consistency errors directly.

## Minimum Behavioral Checks

- Tangency preservation: projected tilt vectors satisfy `t · n ~= 0`.
- Null behavior: flat or divergence-free cases do not create spurious KH curvature response.
- Sign behavior: changing the sign of a source or tilt pattern changes the coupled response consistently.
- Leaflet parity: inner/outer terms do not accidentally share arrays, flags, or boundary conventions.

## Benchmark Guidance

Start from `docs/TILT_BENCHMARKS.md` rather than inventing a new benchmark family. Reuse those cases and expectations when possible:

- flat strip or annulus decay tests for the KH screening length,
- source-reinforcement and cancellation cases,
- coupled-shape response cases,
- divergence-free regression cases.

## Performance Guardrails

If the change touches a tilt hot path, solver, operator, or buffer layout, run the repo's performance harness:

`python tools/tilt_perf_guardrails.py --pin-threads --warmups 1 --runs 5 --output-json benchmarks/outputs/tilt_perf_baseline.json`

and compare candidate changes with:

`python tools/tilt_perf_guardrails.py --pin-threads --warmups 1 --runs 5 --baseline-json benchmarks/outputs/tilt_perf_baseline.json --output-json benchmarks/outputs/tilt_perf_candidate.json`

Use the exact affected simulation path, not a synthetic proxy.

## When to Stop and Re-check

- You cannot tell whether a file uses `J`, `2H`, or another curvature normalization.
- A proposed change would alter normal orientation or leaflet sign conventions.
- A new term appears to mix KH with another membrane model.
- The implementation requires loops over mesh entities inside a hot energy path.

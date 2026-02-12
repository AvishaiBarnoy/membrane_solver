# Cache Context Draft (Option B)

## Goal

Make energy evaluation and diagnostics purely observational by removing mutable
geometry/energy caches from `Mesh` and instead storing them in an explicit
context object that is created per evaluation or per minimization run.

This is the long-term follow-up to Option A (run diagnostics on a copy) which
restores determinism without requiring a deep refactor.

## Problem Statement

Today, some geometry and energy routines cache derived quantities (triangle
rows, curvature operators, vertex areas, etc.) in `Mesh`-owned fields keyed off
version counters. This is a practical performance optimization, but it has an
important consequence:

- Calling a "read-only" diagnostic (energy breakdown, curvature stats, etc.)
  can mutate `mesh` state (populate caches, update cache versions, touch
  scratch arrays).
- If the debug path runs additional diagnostics, the minimization trajectory
  can change due to cache mutation, conditional branches, or module ordering
  effects.

Option A mitigates this by running diagnostics on a copy. Option B targets the
root cause: make evaluation state explicit and disposable.

## Proposed Interfaces

### `GeometryCache`

A lightweight container for derived geometry arrays associated with a
particular `(mesh topology, positions)` state. It holds arrays such as:

- triangle row cache (triangle->vertex row indices)
- per-triangle areas and normals
- cotangent weights / Laplace-Beltrami operator inputs
- mixed Voronoi / barycentric vertex areas
- mean curvature and related intermediate arrays

Key properties:

- It does not own or modify `Mesh`.
- It uses `mesh._version` and `mesh._vertex_ids_version` to validate reuse.
- It may keep references to SoA views (positions, row maps) but must rebuild if
  the mesh versions change.

### `EnergyContext`

An evaluation workspace that bundles:

- `GeometryCache`
- reusable scratch buffers (e.g., dense gradient arrays)
- evaluation flags (debug counters, profiling hooks)

Suggested signature patterns:

```python
E = module.compute_energy_array(mesh, global_params, resolver,
                                *, positions, index_map, ctx)

E = module.compute_energy_and_gradient_array(mesh, global_params, resolver,
                                             *, positions, index_map,
                                             grad_arr, ctx)
```

Implementation detail: `ctx` should be optional at first (modules ignore it
unless they need cached geometry).

## Invariants

- Energy/gradient evaluation must not mutate `Mesh` (other than explicitly
  allowed operations like tilt projection, which should be considered part of
  "state update" rather than "evaluation").
- Diagnostics must be observational: running them cannot change the next
  minimization step outcome (given identical RNG state and floating point
  settings).
- Cache reuse must be correct under:
  - position updates without topology changes
  - topology changes (refinement/equiangulation) that bump mesh versions

## Migration Strategy (Small PRs)

1. Add `runtime/energy_context.py` with `EnergyContext` and `GeometryCache`.
2. Thread an optional `ctx` through `EnergyModuleManager` compute routines.
3. Move the highest-impact caches first:
   - triangle row cache / SoA index map scratch
   - curvature operator components used by bending energies
4. Update selected hot modules to consume `ctx.geometry_cache` instead of
   calling `mesh.triangle_row_cache()` (or other mesh-owned caches).
5. Once coverage is broad enough, deprecate mesh-owned caches or keep them
   only for interactive tooling where mutability is acceptable.

Each PR that touches hot-loop computation must include a representative
benchmark per `AGENTS.md`.

## Diagnostics Behavior (After Option B)

- Debug diagnostics can run on the same `Mesh` instance, but with a fresh
  `EnergyContext` so all caches/scratch are isolated.
- Minimization can re-use a single `EnergyContext` across iterations for speed,
  while still preserving the invariant that `Mesh` itself is not mutated by
  evaluation.

## Risks

- This is a broader refactor that will touch many call sites.
- Care is needed to avoid accidental sharing of scratch arrays across
  evaluations (thread safety / reentrancy).
- Benchmarks are required to ensure no performance regression in bending/tilt
  dominated workloads.

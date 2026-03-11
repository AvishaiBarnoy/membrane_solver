---
name: surface-geometry
description: Use for repository tasks that implement, review, debug, benchmark, or explain the triangle-mesh discrete geometry operators used by this codebase. Trigger on requests about cotangent Laplace-Beltrami operators, mean-curvature vectors, Gaussian curvature from angle defects, mixed Voronoi or dual areas, triangle P1 basis gradients, cotan derivatives, normal/tangent projection, curvature normalization, or geometry-cache/sign issues in `geometry/` and geometry-dependent energy code. Do not use for generic geometry, meshing, optimization, or membrane-model questions unless they explicitly depend on these discrete surface-geometry conventions.
---

# Surface Geometry

Use this skill to keep geometry operators on triangle meshes consistent with the discretizations already used in this repository.

Focus on discrete surface geometry, not membrane constitutive modeling. For KH/Helfrich physics decisions, pair this skill with `membrane-physics`.

## Workflow

1. Identify the geometry object being changed.
   Distinguish curvature evaluation, cotan/Laplace-Beltrami operators, tangent/normal projection, per-triangle basis gradients, or geometry derivatives.
2. Load only the needed reference file.
   Read [references/discrete-geometry-core.md](references/discrete-geometry-core.md) for the core discrete operators and convention traps.
   Read [references/repo-map.md](references/repo-map.md) for where those operators live in this codebase.
   Read [references/validation-workflow.md](references/validation-workflow.md) before editing geometry-sensitive runtime code.
3. Check normalization and orientation before editing.
   Do not assume `H`, `J`, `2H`, signed curvature, or normal orientation without verifying the local file and tests.
4. Keep the implementation in the repo's vectorized geometry pattern.
   Gather dense arrays, evaluate batched operators, and scatter results back without per-entity Python loops in hot paths.
5. Validate the operator and its derivatives.
   Add regression tests for sign, invariance, and finite-difference or directional-derivative agreement when gradients are touched.

## Default Decisions

- Prefer the repo's cotangent-based discrete Laplace-Beltrami and mixed-area curvature machinery over ad hoc alternatives.
- Preserve ambient 3D vector formulations unless an existing local-basis implementation is already correct and tested.
- Treat curvature comments, tests, and energies as a single convention bundle; if one changes, inspect the others.
- Use cached geometry only when mesh versioning rules still hold.
- If a task is really about membrane constitutive terms rather than geometry operators, switch to or combine with `membrane-physics`.

## When To Update This Skill

Update this skill whenever the repository changes the preferred or required discrete-geometry contract that Codex should follow. In practice, update it when any of the following changes:

- curvature discretization or normalization,
- cotan/Laplace-Beltrami operator family,
- dual-area or Gaussian-curvature convention,
- tangent/normal representation rules,
- geometry-cache or versioning assumptions that affect operator usage,
- required geometry regression tests, derivative checks, or benchmark workflow.

Do not update the skill for local refactors that preserve the same geometry algorithm and validation contract.

## Trigger Examples

- "Check whether this cotan Laplacian uses the same mean-curvature normalization as the bending energy."
- "Add a regression test for Gaussian curvature from angle defects on a triangulated sphere."
- "Review this `grad_cotan` implementation for sign or degeneracy mistakes."
- "Map the continuum Laplace-Beltrami formula onto the operators in `geometry/curvature.py`."
- "Debug why tangent projection or vertex normals are drifting during optimization."

## References

- [references/discrete-geometry-core.md](references/discrete-geometry-core.md): core triangle-mesh operators, definitions, and convention traps.
- [references/repo-map.md](references/repo-map.md): where geometry operators, caches, and derivative paths live in this repo.
- [references/validation-workflow.md](references/validation-workflow.md): test and benchmark expectations for geometry-side changes.

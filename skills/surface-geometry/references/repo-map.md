# Repo Map

Use this file when you need to know where the discrete geometry conventions already live in the codebase.

## Primary Anchors

- `geometry/curvature.py`:
  vectorized curvature fields, cotan-weight machinery, mixed areas, angle defects, and curvature caches.
- `geometry/bending_derivatives.py`:
  geometry derivative helpers such as `grad_cotan` and triangle-area gradients.
- `modules/energy/bending.py`:
  bending energy and geometry-gradient paths that depend on cotan and curvature conventions.
- `modules/energy/bending_tilt.py` and `modules/energy/bending_tilt_leaflet.py`:
  geometry-sensitive coupling terms that reuse cotan-gradient machinery.
- `geometry/tilt_operators.py`:
  triangle P1 basis gradients and divergence operators relevant when geometry and vector fields interact.
- `docs/EVOLVER_COMPARISON.md`:
  repo-level explanation of how the bending discretization differs from Surface Evolver.

## Practical Mapping

- If the task is about mean curvature on vertices, start from `geometry/curvature.py`.
- If the task is about analytic derivatives of cotangents, triangle areas, or geometry terms in bending, start from `geometry/bending_derivatives.py` and the energy modules that consume it.
- If the task mixes geometry with tangent vector fields, check `geometry/tilt_operators.py` before inventing another surface-gradient formula.
- If the task is about caching or performance, inspect mesh versioning and existing geometry caches before changing recomputation behavior.

## Common Failure Modes

- Mixing integrated and pointwise curvature quantities.
- Comparing a `|H|` implementation against a signed-curvature theory expression.
- Forgetting the repo's normalization difference relative to Surface Evolver-style `(1/R1 + 1/R2)^2`.
- Touching cotan-gradient code without a finite-difference check.
- Recomputing expensive geometry in inner loops despite an existing cache/version guard.

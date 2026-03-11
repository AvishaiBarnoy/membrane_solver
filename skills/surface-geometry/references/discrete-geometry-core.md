# Discrete Geometry Core

Use this file when the task depends on the mathematical meaning of a discrete surface operator or on matching a continuum identity to the repo's triangle-mesh discretization.

## Core Operators

- Cotangent Laplace-Beltrami:
  use cotangent weights on triangle meshes to discretize `Delta_s x` and related surface operators.
- Mean curvature vector:
  in the common cotan formulation, `Delta_s x = -2 H n` up to the repo's chosen normalization and sign conventions.
- Mixed Voronoi area / dual area:
  use per-vertex dual areas to convert integrated quantities into pointwise fields.
- Gaussian curvature:
  use angle defect divided by dual area for pointwise `K`.
- Triangle P1 basis gradients:
  use barycentric basis gradients for piecewise-linear scalar/vector fields and their surface divergence/gradient evaluations.

## Convention Traps

- `H`, `J`, and `2H` are often mixed. Do not assume the local file uses the same symbol as the paper or benchmark text.
- Mean curvature can be represented as:
  - a signed scalar,
  - a non-negative magnitude,
  - a mean-curvature vector `H n`,
  - or a total curvature based on `k1 + k2`.
- Principal curvatures derived from `H` and `K` depend on the chosen normalization of `H`.
- A code path that uses curvature magnitude may intentionally avoid dependence on global facet orientation; do not "fix" that without checking downstream uses.

## Triangle-Mesh Consequences

- Degenerate or near-degenerate triangles can blow up cotangents, curvature estimates, and their gradients.
- Orientation changes can flip signed normals and any signed-curvature quantity derived from them.
- Derivatives of cotangents and triangle areas are often the most fragile part of the implementation; check them separately from the forward operator.

## Implementation Defaults

- Prefer batched ambient 3D formulas over per-vertex local-coordinate loops.
- Use `np.add.at` or direct array scattering for accumulated vertex quantities.
- Reuse cached geometry data only when mesh/version invariants are satisfied.
- Keep comments explicit about whether an array is integrated, pointwise, signed, unsigned, or vector-valued.

# KH Core

Use this file when the task depends on continuum-model identities, sign conventions, or what is and is not part of the Kozlov-Hamm extension.

## Definitions

- `N`: unit surface normal.
- `n`: lipid/director orientation.
- `t = n / (n·N) - N`: tilt vector; for small tilt it is tangent to the surface and `|t| = tan(theta)`.
- `b`: curvature tensor of the dividing surface.
- `t'`: surface derivative / tilt tensor.
- `b_tilde = b - t'`: effective curvature tensor.
- `J_tilde = tr(b_tilde) = J - div_s(t)`: effective total curvature.
- `K_tilde = det(b_tilde)`: effective Gaussian curvature.

## Core KH Statement

For a monolayer with small tilt and slow variation, bending and tilt-splay enter through the same kinematics. The practical consequence is:

- Do not model KH as pure Helfrich bending plus an arbitrary curvature-splay coupling coefficient.
- Treat curvature and tilt-splay through the effective curvature quantities above.

The paper's monolayer density can be written as:

`f = (kappa / 2) * (J_tilde - J_s)^2 + kappa_bar * K_tilde + (kappa_theta / 2) * |t|^2`

Expanded schematically for the mean-curvature part:

`f_mean ~ (kappa / 2) * (J - div_s(t) - J_s)^2`

## Invariants to Preserve

- The minus sign in `J_tilde = J - div_s(t)` is fixed by the kinematics.
- Constant tilt magnitude contributes through the tilt penalty `|t|^2`; it does not couple linearly to bending at this order.
- Varying tilt and bending share the same bending moduli in the KH construction.
- If Gaussian curvature is not represented explicitly in a discrete term, do not silently reinterpret the remaining model as full KH; call it a reduced KH-like term.

## Implementation Consequences

- Keep tilt tangent to the surface; normal components are non-physical bookkeeping errors in this model.
- Prefer ambient 3D tangent vectors over ad hoc 2D local-coordinate storage unless a correct connection/transport treatment already exists.
- When porting formulas to code, map `div_s(t)` to the discrete surface divergence of the tangent tilt field on the current mesh.

## Leaflet Notes

- For bilayers, treat inner and outer leaflet tilt fields separately.
- Do not assume the same sign convention for both leaflets without checking the repo's current normal orientation and test baselines.
- If a requested change mixes KH monolayer physics with bilayer constraints or boundary anchoring, make the coupling assumption explicit.

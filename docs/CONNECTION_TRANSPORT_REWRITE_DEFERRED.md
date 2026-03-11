# Connection-Aware Tilt Rewrite (Deferred)

## Status

Deferred. The connection-aware tilt rewrite is correct on the covered unit and
flat-guard surfaces, but it does not improve curved parity enough to justify
rollout and is still too slow on the representative curved lane.

Current default remains:

- `tilt_transport_model = "ambient_v1"`

The connection-aware path remains benchmark-only:

- `tilt_transport_model = "connection_v1"`

## Why this was attempted

The curved `free_z` / KH lane shows a transport/coupling problem beyond the
rim:

- flat strict-KH still passes
- the inner disk `I1` profile inside `r <= R` is realized correctly
- the parity blocker appears in transport beyond the rim
- the original implementation compares and differentiates ambient 3D tilt
  vectors, then projects back to tangent planes after updates

That made a connection-aware tangent-field rewrite the next credible
discretization upgrade.

## What was implemented

### PR 1: Transport infrastructure

Added:

- [`/Users/User/github/membrane_solver/geometry/tangent_transport.py`](/Users/User/github/membrane_solver/geometry/tangent_transport.py)

Helpers added:

- `minimal_rotation_transport(...)`
- `transport_vectors(...)`
- `transport_vertex_tilts_to_triangle_planes(...)`
- `edge_transport_pairs(...)`

Status:

- unit-tested
- no runtime behavior change

### PR 2: Connection-aware smoothness

Updated:

- [`/Users/User/github/membrane_solver/modules/energy/tilt_smoothness.py`](/Users/User/github/membrane_solver/modules/energy/tilt_smoothness.py)
- [`/Users/User/github/membrane_solver/modules/energy/tilt_smoothness_in.py`](/Users/User/github/membrane_solver/modules/energy/tilt_smoothness_in.py)
- [`/Users/User/github/membrane_solver/modules/energy/tilt_smoothness_out.py`](/Users/User/github/membrane_solver/modules/energy/tilt_smoothness_out.py)

Added mode:

- `tilt_transport_model = "ambient_v1" | "connection_v1"`

Behavior:

- `connection_v1` replaces ambient edgewise tilt differences with
  minimal-rotation transport between tangent planes

Status:

- correct on planar-equivalence and energy/gradient tests
- no curved parity improvement at smoke

### PR 3: Connection-aware divergence

Updated:

- [`/Users/User/github/membrane_solver/geometry/tilt_operators.py`](/Users/User/github/membrane_solver/geometry/tilt_operators.py)
- [`/Users/User/github/membrane_solver/modules/energy/bending_tilt_leaflet.py`](/Users/User/github/membrane_solver/modules/energy/bending_tilt_leaflet.py)

Behavior:

- for `connection_v1`, vertex tilts are transported into the triangle tangent
  plane before P1 divergence evaluation
- effective tilt gradients are rotated back to ambient storage coordinates

Status:

- correct on planar-equivalence and directional-derivative tests
- no curved section-parity improvement at smoke
- slight energy-parity worsening at smoke

### PR 4: Connection-aware inner splay/twist

Updated:

- [`/Users/User/github/membrane_solver/modules/energy/tilt_splay_twist_in.py`](/Users/User/github/membrane_solver/modules/energy/tilt_splay_twist_in.py)

Behavior:

- for `connection_v1`, the dominant inner differential stack now transports
  inner tilts into triangle tangent planes before divergence/curl evaluation
- local tilt gradients are rotated back to ambient coordinates

Status:

- correct on planar-equivalence and directional-derivative tests
- still no curved section-parity improvement at smoke

### PR 5: Geometry-versioned caching

Added cache:

- [`/Users/User/github/membrane_solver/geometry/tangent_transport.py`](/Users/User/github/membrane_solver/geometry/tangent_transport.py)
  - `triangle_plane_transport_data(...)`

Cache policy:

- caches geometry-only transport data:
  - triangle normals
  - vertex-to-triangle rotations
  - transpose rotations
- cache is used only when:
  - `positions` shares memory with live mesh positions
  - `tri_rows` is the live mesh triangle-row cache
- copied / perturbed arrays bypass the cache intentionally

Status:

- cache correctness tests pass
- runtime still too slow on representative curved lane

## Benchmarks that stopped rollout

### Curved smoke

Ambient baseline:

- `seconds = 16.54`
- `theta_factor = 1.1811703875`
- `energy_factor = 1.3645931364`
- `disk_ratio_v2 = 1.0743388980`
- `outer_near_ratio_v2 = 4.8708438482`
- `outer_far_ratio_v2 = 6.1153467642`

Connection-aware with cache:

- `seconds = 72.23`
- `theta_factor = 1.1811703875`
- `energy_factor = 1.3955380226`
- `disk_ratio_v2 = 1.0743388980`
- `outer_near_ratio_v2 = 4.8708438482`
- `outer_far_ratio_v2 = 6.1153467642`

Interpretation:

- about `4.4x` slower than ambient
- no section-parity improvement
- slightly worse energy parity

### Representative curved `p5`

Ambient baseline:

- `seconds = 262.85`
- `theta_factor = 1.0124317607`
- `energy_factor = 1.0211486229`
- `disk_ratio_v2 = 0.5850885103`
- `outer_near_ratio_v2 = 1.2973722991`
- `outer_far_ratio_v2 = 0.9598956812`

Connection-aware with cache:

- did not complete within an additional 240 seconds after the ambient baseline
  finished

Interpretation:

- still too slow for the representative curved lane
- not worth promoting without a stronger parity benefit

## Flat-guard status

These stayed green throughout the staged rewrite:

- [`/Users/User/github/membrane_solver/tests/test_flat_disk_kh_term_audit_regression.py`](/Users/User/github/membrane_solver/tests/test_flat_disk_kh_term_audit_regression.py)
- [`/Users/User/github/membrane_solver/tests/test_flat_disk_one_leaflet_benchmark_e2e.py`](/Users/User/github/membrane_solver/tests/test_flat_disk_one_leaflet_benchmark_e2e.py)
- [`/Users/User/github/membrane_solver/tests/test_reproduce_flat_disk_one_leaflet_acceptance.py`](/Users/User/github/membrane_solver/tests/test_reproduce_flat_disk_one_leaflet_acceptance.py)

So the rewrite is flat-safe on the covered guard suite. The blocker is curved
runtime/parity value, not flat regression.

## Why rollout was stopped

The rewrite was stopped for two independent reasons:

1. It did not improve the blocked curved section-parity metrics.
2. It remained too slow on the real curved lane, even after caching.

That means the current connection-aware implementation is not justified as the
next active parity path.

## If we return to this later

Treat it as a separate performance-first project, not as an immediate parity
fix.

Recommended sequence:

1. Reduce hot-path cost further before re-evaluating parity:
   - cache more of the transported basis action
   - remove repeated `einsum` / rotation work where possible
   - consider compiled kernels for the transported operators
2. Re-benchmark:
   - curved smoke
   - representative curved `p5`
3. Only continue if both are true:
   - runtime becomes acceptable
   - curved section parity improves materially

## Current recommended direction instead

Return to the curved operator/model path directly:

- continue diagnosing the disk/near-band parity blocker
- focus on the curved KH operator/model side rather than the transport rewrite
- use the current ambient implementation as the active baseline

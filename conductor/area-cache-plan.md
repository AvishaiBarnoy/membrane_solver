# Tilt Area Caching Optimization Plan

## Objective
Reuse the existing cached vertex areas from `mesh.barycentric_vertex_areas()` in the dual-leaflet tilt relaxation loop, rather than recomputing them on every step.

## Key Files & Context
- **Tilt Relaxation Loop**: `runtime/steppers/tilt_relaxation.py`
  - Function: `TiltRelaxationManager.relax_leaflet_tilts`
  - Current Behavior: Calls `_tilt_vertex_areas_from_triangles` for both inner and outer leaflets on every invocation.

## Implementation Steps
1. **Inner Leaflet Optimization**:
   - In `relax_leaflet_tilts`, replace `tilt_vertex_areas_in = _tilt_vertex_areas_from_triangles(...)` with `tilt_vertex_areas_in = mesh.barycentric_vertex_areas(positions=positions)`.
   - This eliminates a redundant $O(N)$ area recomputation for the inner leaflet, pulling directly from the mesh's curvature/area cache.

2. **Outer Leaflet Fast-Path**:
   - The outer leaflet area computation respects an `absent_mask_out`.
   - Add a fast-path condition: `if not np.any(absent_mask_out): tilt_vertex_areas_out = tilt_vertex_areas_in`.
   - Keep the fallback computation only for the rare cases where outer vertices are actively masked.

## Verification & Testing
- Run `pytest -q tests/test_tilt_vertex_area_cache_matches_module_regression.py` (if it exists) or general tilt regression tests to ensure numerical parity.
- The `barycentric_vertex_areas` implementation is mathematically identical to `_tilt_vertex_areas_from_triangles`, so this change is perfectly safe and zero-overhead.

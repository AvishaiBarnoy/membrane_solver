# Cache Dependency Table

Status: **active — core geometry caches inventoried**
Owner: `geometry/mesh.py`, `geometry/cache_checks.py`,
`geometry/cache_writes.py`, and `runtime/energy_context.py`

This table documents the dependency tuples already enforced by the current
cache predicates and context methods. It does not add an invalidation engine or
change cache behavior. A cache is reusable only when every named dependency is
unchanged.

| Cache / value | Owner and writer | Dependencies | Invalidated by | Primary coverage |
|---|---|---|---|---|
| Mesh position SoA view | `Mesh.build_position_cache` | mesh geometry version, vertex-ID order | position mutation, vertex rebind | `tests/test_caching.py`, `tests/test_energy_context.py` |
| Mesh triangle rows | `Mesh` triangle-row builder | facet-loop version, vertex-ID order | facet-loop rebuild, vertex rebind; not coordinate-only mutation | `tests/test_caching.py`, `tests/test_energy_context.py` |
| Triangle areas and unnormalized normals | `store_triangle_area_normals_cache` | active position-cache identity, mesh geometry version | coordinate mutation or a noncanonical position array | `tests/test_geometry_cache_checks.py`, `tests/test_geometry_cache_writes.py` |
| Vertex normals | `store_vertex_normals_cache` | active position-cache identity, mesh geometry version, facet-loop version | coordinate mutation, facet-loop rebuild, or noncanonical positions | `tests/test_geometry_cache_checks.py`, `tests/test_geometry_cache_writes.py` |
| Barycentric vertex areas | `store_barycentric_vertex_areas_cache` | mesh geometry version, facet-loop version, expected vertex count | coordinate mutation, facet-loop rebuild, or row-count change | `tests/test_geometry_cache_checks.py`, `tests/test_geometry_cache_writes.py`, `tests/test_energy_context.py` |
| P1 triangle shape gradients | `store_p1_triangle_grad_cache` | mesh geometry version, facet-loop version, all four payload arrays | coordinate mutation, facet-loop rebuild, or missing payload | `tests/test_geometry_cache_checks.py`, `tests/test_geometry_cache_writes.py`, `tests/test_energy_context.py` |
| Context SoA views | `EnergyContext.geometry.soa_views` | mesh geometry version, vertex-ID order | coordinate mutation or vertex rebind | `tests/test_energy_context.py` |
| Context triangle rows | `EnergyContext.geometry.triangle_rows` | facet-loop version, vertex-ID order | facet-loop rebuild or vertex rebind; not coordinate-only mutation | `tests/test_energy_context.py` |
| Context triangle geometry | `EnergyContext.geometry.triangle_areas_and_normals` | canonical mesh-position identity plus mesh geometry version, facet-loop version, vertex-ID order | coordinate mutation, facet-loop rebuild, vertex rebind, or a noncanonical trial-position array | `tests/test_energy_context.py` |
| Context barycentric areas | `EnergyContext.geometry.barycentric_vertex_areas` | canonical mesh-position identity plus mesh geometry version, facet-loop version, vertex-ID order | coordinate mutation, facet-loop rebuild, vertex rebind, or a noncanonical trial-position array | `tests/test_energy_context.py` |
| Context P1 gradients | `EnergyContext.geometry.p1_triangle_shape_gradients` | canonical mesh-position identity plus mesh geometry version, facet-loop version, vertex-ID order | coordinate mutation, facet-loop rebuild, vertex rebind, or a noncanonical trial-position array | `tests/test_energy_context.py` |
| Total surface area / shape gradient | `geometry.mesh_computations` | mesh geometry version; gradient cache additionally requires active position-cache identity | geometry mutation; a noncanonical position array bypasses only the gradient cache | `tests/test_caching.py` |
| Curvature data / cotangent weights | `geometry.curvature.compute_curvature_data` | active geometry-cache position identity, mesh geometry version, facet-loop version | coordinate mutation, facet-loop rebuild, or a noncanonical position array | `tests/test_caching.py`, `tests/test_refactor_compatibility.py`, `tests/test_bending_effective_areas_raw_cache_unit.py` |
| Tilt fixed masks | `runtime.minimizer_helpers.get_cached_tilt_fixed_mask` | tilt-fixed-flags version, vertex-ID version, current vertex count | fixed-flag mutation, vertex-ID rebind, or vertex-count mismatch | `tests/test_minimizer_helpers.py`, `tests/test_mesh_mutation_hooks_unit.py` |
| Mesh fixed mask | `Mesh.fixed_mask` | fixed-flags version, vertex-ID version, current vertex count | fixed-flag mutation, vertex-ID rebind, or vertex-count mismatch | `tests/test_caching.py`, `tests/test_geometry_cache_checks.py` |
| Triangle-plane transport data | `geometry.tangent_transport.triangle_plane_transport_data` | live position/triangle-row identity, mesh geometry version, topology version, cache tag | coordinate/topology mutation, noncanonical arrays, or a different tag | `tests/test_tangent_transport.py` |
| Frozen geometry eligibility | `Mesh._geometry_cache_active` / `geometry.cache_checks.geometry_freeze_cache_active` | live position-cache identity and geometry epoch, or active freeze depth with geometry/facet-loop epochs and frozen-array identity | geometry/facet-loop mutation, freeze exit, or a different position array | `tests/test_geometry_cache_checks.py`, `tests/test_caching.py`, `tests/test_energy_context.py` |
| Tilt / inner / outer dense views | `Mesh.tilts_view`, `tilts_in_view`, `tilts_out_view` | dense `(N, 3)` shape, cached vertex count, vertex-ID version | vertex-count/ID-row rebind or incompatible cache shape | `tests/test_geometry_cache_checks.py`, `tests/test_tilt_leaflet_pure.py` |
| Connectivity maps | `Mesh.build_connectivity_maps` | topology version, vertex/edge/facet counts | topology mutation or a collection-size mismatch | `tests/test_geometry_cache_checks.py`, `tests/test_connectivity_caching.py` |
| Boundary vertex IDs | `Mesh.boundary_vertex_ids` | topology version | topology mutation | `tests/test_geometry_cache_checks.py`, `tests/test_connectivity_caching.py` |

## Mutation ownership

`Mesh` remains the mutation authority. The table intentionally distinguishes:

- geometry version: coordinate-dependent values;
- facet-loop version: facet-row/connectivity-dependent values;
- vertex-ID version: dense-row binding values;
- topology version: adjacency-only values, documented separately from these
  geometry kernels.

Field-only mutations such as tilt updates must not be treated as coordinate or
topology invalidations unless the owning cache explicitly depends on that
field. This distinction preserves cache reuse across tilt-only optimization.

## Next inventory slice

All currently known mesh/runtime cache families have a dependency row. This
table is still not permission to broaden invalidation or centralize writes;
such a change would require a separate behavior-preserving manifest and
mutation-specific characterization tests.
`geometry.cache_checks.field_mask_cache_valid()` and
`geometry.cache_checks.geometry_freeze_cache_active()` are shared by their
already-characterized owners only; they do not invalidate or write caches.
`geometry.cache_checks.vector_field_cache_needs_rebind()` similarly only
centralizes the dense-field row-binding predicate.
`geometry.cache_checks.topology_cache_valid()` and
`geometry.cache_checks.connectivity_cache_valid()` only centralize existing
topology cache checks; construction and boundary traversal remain in `Mesh`.

Explicit trial-position arrays passed to `EnergyContext` geometry methods are
computed fresh and never read or populate mesh-bound context geometry caches.
This preserves reusable cache performance for the canonical mesh SoA view while
preventing one trial geometry from leaking into another at the same mesh epoch.

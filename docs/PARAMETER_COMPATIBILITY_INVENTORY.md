# Parameter and Compatibility Inventory

Status: **active inventory; no deprecations approved**
Repository authority: `docs/REPOSITORY_MODERNIZATION_PLAN.md`.

## Reading this inventory

`GlobalParameters` provides a small set of canonical defaults, while YAML and
module option readers accept many additional parameters. An item recorded here
is supported or under investigation; it is not obsolete merely because it is a
fallback, alias, or legacy path.

## Canonical global defaults

Source: `core/parameters/global_parameters.py`.

| Family | Canonical keys |
|---|---|
| Surface/volume | `surface_tension`, `volume_stiffness`, `volume_constraint_mode`, `volume_projection_during_minimization`, `volume_tolerance` |
| Shape stepping | `max_zero_steps`, `step_size_floor`, `step_size`, `step_size_mode` |
| Bending | `intrinsic_curvature`, `bending_modulus`, `bending_energy_model`, `bending_gradient_mode`, `gaussian_modulus` |
| Tilt solver | `tilt_solver`, `tilt_cg_preconditioner` |
| Mesh repair | `mesh_quality_auto_repair_enabled`, `mesh_quality_auto_repair_every`, `mesh_quality_aspect_threshold`, `mesh_quality_aspect_percentile`, `mesh_quality_max_repair_passes` |

`GlobalParameters` also supports both dictionary-style access (`get`, `set`,
`update`) and attribute access for known default keys. Preserve both until an
explicit compatibility manifest establishes callers and migration policy.

## Dynamic high-impact parameter families

These families are intentionally not treated as a complete schema yet. The
listed owners are the first place to define a canonical contract.

| Family | Examples | Primary owners |
|---|---|---|
| Theory/benchmark lanes | `theory_parity_lane`, `benchmark_geometry_lane`, `benchmark_parameterization` | `runtime/minimizer.py`, theory checkpoint |
| Shape fallback/reduced line search | `shape_scaffold_rejected_step_fallback`, `line_search_reduced_*` | `runtime/minimizer.py` |
| Tilt relaxation | `tilt_solve_mode`, `tilt_step_size`, `tilt_tol`, `tilt_inner_steps`, `tilt_coupled_steps`, `tilt_cg_*`, `tilt_projection_*` | `runtime/steppers/tilt_relaxation.py` |
| Theta-B optimization | `tilt_thetaB_*`, `rim_slope_match_thetaB_param` | `runtime/tilt_optimization.py`, theta-B constraints |
| Rim/interface/scaffold | `rim_slope_match_*`, `parity_trace_layer_radius`, `parity_outer_shells` | rim constraints, `runtime/interface_validation.py` |
| Leaflet presence | `leaflet_*_absence_mode`, absence presets | `modules/energy/leaflet_presence.py` |
| Tilt material/selection | `tilt_modulus_*`, `tilt_mass_mode*`, `tilt_*_exclude_shared_rim*`, `tilt_rim_source_*` | `modules/energy/tilt_params.py`, rim source modules |
| Bending numerical controls | `spontaneous_curvature`, `bending_fd_eps`, leaflet moduli | `modules/energy/bending_params.py`, bending modules |
| Constraint/geometry targets | `target_surface_area`, `perimeter_constraints`, `rigid_disk_*`, `pin_to_plane_*` | constraint modules, `geometry/io_readers.py` |

## Verified aliases and compatibility behavior

| Category | Accepted form | Canonical / behavior | Owner |
|---|---|---|---|
| Constraint name | `pin_surface_group_to_shape` | normalized to `pin_to_plane` | `geometry/input_normalization.py` |
| Pin-to-plane keys | `pin_surface_group_to_shape_{mode,group,normal,point}` | normalized to corresponding `pin_to_plane_*` keys; canonical input wins when both exist | `geometry/input_normalization.py` |
| Bending curvature | `spontaneous_curvature` | preferred when present; otherwise `intrinsic_curvature` | `modules/energy/bending_params.py` |
| Tilt modulus typo | `tilt_modolus_{in,out}` | fallback if `tilt_modulus_{in,out}` is absent | `modules/energy/tilt_params.py` and disk-target modules |
| Tilt mass mode | `tilt_mass_mode` | fallback for leaflet-specific `tilt_mass_mode_{leaflet}` | `modules/energy/tilt_params.py` |
| Shared-rim exclusion | historical leaflet-suffixed and out-side variants | fallback behind canonical `tilt_{leaflet}_exclude_shared_rim_outer_rows` | `modules/energy/tilt_params.py` |
| Bending gradient mode | `fd` | normalized to `finite_difference` | `modules/energy/bending_params.py` |
| Theta-B penalty mode | `on`, `true`, `1` | normalized to `legacy` penalty mode | `modules/energy/tilt_thetaB_contact_in.py` |
| Manager method | `ConstraintModuleManager.get_constraint` | backward-compatible alias of `get_module` | `runtime/constraint_manager.py` |
| Entity imports | `geometry.entities` | backward-compatible re-export facade | `geometry/entities.py` |
| Gradient API | legacy dict gradients | supported through manager/minimizer adapters | `runtime/constraint_manager.py`, `runtime/minimizer.py` |

## Fallbacks requiring separate decisions

| Fallback / legacy behavior | Primary owner | Required evidence before alteration or removal |
|---|---|---|
| Volume post-step projection | IO, minimizer, volume constraint | fixture scan, constraint behavior tests, scientific owner decision |
| Rejected shape-step trace-z fallback | `runtime/minimizer.py` | activation path tests and parity-lane review |
| Tilt CG-to-GD fallback | tilt relaxation manager | solve-mode/fallback matrix and convergence baseline |
| Polygon/nontriangle energy paths | surface/body modules | supported-domain decision and topology coverage |
| Legacy theta-B penalty | theta-B energy/constraint modules | theory acceptance and replacement contract |
| Single-frame inclusion operator | inclusion constraint helpers | multi-rim behavior tests and owner decision |

## Retirement protocol

Do not remove an item until a dedicated change manifest records:

1. all repository call sites and fixtures using it;
2. external/support policy and deprecation window, if applicable;
3. canonical replacement and precedence when both forms appear;
4. focused compatibility tests before and after migration;
5. theory-lane review where the item can influence scientific results;
6. rollback path.

The next inventory expansion should extract dynamic parameter keys automatically
from consumers, then manually assign type, units, default, validation, and lane
ownership. That expansion must not infer that a dynamically read key is safe to
remove.

# runtime/minimizer.py

import inspect
import logging
from typing import Callable, Dict, List, Optional

import numpy as np

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.entities import Mesh
from runtime.constraint_manager import ConstraintModuleManager
from runtime.diagnostics.audit import (
    check_gauss_bonnet,
    log_accepted_step_stats,
    log_debug_energy_context,
    log_energy_consistency,
    log_energy_phase,
    log_lagrange_tangency_check,
    log_step_direction_stats,
)
from runtime.diagnostics.gauss_bonnet import GaussBonnetMonitor
from runtime.energy_context import EnergyContext
from runtime.energy_manager import EnergyModuleManager
from runtime.equiangulation import equiangulate_iteration
from runtime.evaluation_manager import EvaluationManager
from runtime.interface_validation import validate_disk_interface_topology
from runtime.leaflet_validation import validate_leaflet_absence_topology
from runtime.minimizer_helpers import (
    build_reduced_line_search_energy_fn,
    build_reduced_line_search_trial_energy_fn,
    get_cached_tilt_fixed_mask,
)
from runtime.projections.curved_disk import project_curved_free_disk_shape_dofs
from runtime.projections.tilt import (
    project_tilts_to_tangent_array,
)
from runtime.steppers.base import BaseStepper
from runtime.steppers.tilt_relaxation import TiltRelaxationManager

logger = logging.getLogger("membrane_solver")


class Minimizer:
    """Coordinate the optimization loop for a mesh."""

    def __init__(
        self,
        mesh: Mesh,
        global_params: GlobalParameters,
        stepper: BaseStepper,
        energy_manager: EnergyModuleManager,
        constraint_manager: ConstraintModuleManager,
        energy_modules: Optional[List[str]] = None,
        constraint_modules: Optional[List[str]] = None,
        step_size: float = 1e-3,
        tol: float = 1e-6,
        quiet: bool = False,
    ) -> None:
        self.mesh = mesh
        self.global_params = global_params
        self.energy_manager = energy_manager
        self.constraint_manager = constraint_manager
        self.stepper = stepper
        self.step_size = step_size
        self.tol = tol
        self.quiet = quiet
        self.max_zero_steps = int(global_params.get("max_zero_steps", 10))
        self.step_size_floor = float(global_params.get("step_size_floor", 1e-8))

        # Use provided module lists or fall back to those defined on the mesh
        module_list = (
            energy_modules if energy_modules is not None else mesh.energy_modules
        )
        self.energy_module_names = list(module_list)
        self.energy_modules = [
            self.energy_manager.get_module(mod) for mod in module_list
        ]
        self._validate_energy_modules_array()

        constraint_list = (
            constraint_modules
            if constraint_modules is not None
            else mesh.constraint_modules
        )
        self.constraint_modules = [
            self.constraint_manager.get_constraint(constraint)
            for constraint in constraint_list
        ]
        self._has_enforceable_constraints = any(
            hasattr(mod, "enforce_constraint") for mod in self.constraint_modules
        )

        logger.debug(f"Loaded energy modules: {self.energy_manager.modules.keys()}")
        logger.debug(f"Mesh energy_modules: {self.mesh.energy_modules}")

        self.param_resolver = ParameterResolver(global_params)
        self._gauss_bonnet_monitor: GaussBonnetMonitor | None = None
        self._tilt_fixed_mask_cache: np.ndarray | None = None
        self._tilt_fixed_mask_version: int = -1
        self._tilt_fixed_mask_vertex_version: int = -1
        self._tilt_fixed_mask_in_cache: np.ndarray | None = None
        self._tilt_fixed_mask_in_version: int = -1
        self._tilt_fixed_mask_in_vertex_version: int = -1
        self._tilt_fixed_mask_out_cache: np.ndarray | None = None
        self._tilt_fixed_mask_out_version: int = -1
        self._tilt_fixed_mask_out_vertex_version: int = -1
        self._soa_cache_version: int = -1
        self._soa_cache_vertex_version: int = -1
        self._soa_positions: np.ndarray | None = None
        self._soa_index_map: Dict[int, int] | None = None
        self._soa_grad_dummy: np.ndarray | None = None
        self._last_mesh_op_tilt_constraints_enforced: bool = False
        self._energy_context: EnergyContext | None = None
        self._module_accepts_ctx: dict[int, bool] = {}
        self._module_energy_array_spec: dict[
            int, tuple[bool, bool, frozenset[str] | None]
        ] = {}
        self._stepper_accepts_trial_energy_fn: bool | None = None
        self._last_tilt_projection_stats: dict[str, float | int | str] = {
            "projection_cadence": "per_step",
            "projection_interval": 1,
            "projection_apply_count": 0,
            "tilt_projection_norm_loss_outer_far": 0.0,
        }
        self._last_inner_coupled_update_mode_stats: dict[
            str, float | int | str | bool
        ] = {
            "enabled": False,
            "mode": "off",
            "candidate_row_count": 0,
            "capped_row_count": 0,
            "rim_row_count": 0,
            "cap_magnitude": 0.0,
        }

        self._evaluation_manager = EvaluationManager(
            mesh=self.mesh,
            global_params=self.global_params,
            param_resolver=self.param_resolver,
            energy_modules=self.energy_modules,
            energy_module_names=self.energy_module_names,
            energy_context_fn=self.energy_context,
            experimental_energy_scale_fn=self._experimental_energy_scale_for_module,
        )

        self._tilt_relaxation_manager = TiltRelaxationManager(
            param_resolver=self.param_resolver,
            energy_context_fn=self.energy_context,
            compute_energy_and_tilt_gradient_array_fn=self._compute_energy_and_tilt_gradient_array,
            compute_energy_array_with_tilts_fn=self._compute_energy_array_with_tilts,
            compute_energy_and_leaflet_tilt_gradients_array_fn=self._compute_energy_and_leaflet_tilt_gradients_array,
            compute_tilt_dependent_energy_with_leaflet_tilts_fn=self._compute_tilt_dependent_energy_with_leaflet_tilts,
            set_leaflet_tilts_from_arrays_fast_fn=self._set_leaflet_tilts_from_arrays_fast,
            triangle_rows_fn=self._triangle_rows,
            tilt_fixed_mask_fn=self._tilt_fixed_mask,
            tilt_fixed_mask_in_fn=self._tilt_fixed_mask_in,
            tilt_fixed_mask_out_fn=self._tilt_fixed_mask_out,
        )

    def reset_soa_caches(self) -> None:
        """Clear cached SOA views after mesh-replacing operations."""
        self._soa_cache_version = -1
        self._soa_cache_vertex_version = -1
        self._soa_positions = None
        self._soa_index_map = None
        self._soa_grad_dummy = None
        self._tilt_fixed_mask_cache = None
        self._tilt_fixed_mask_version = -1
        self._tilt_fixed_mask_vertex_version = -1
        self._tilt_fixed_mask_in_cache = None
        self._tilt_fixed_mask_in_version = -1
        self._tilt_fixed_mask_in_vertex_version = -1
        self._tilt_fixed_mask_out_cache = None
        self._tilt_fixed_mask_out_version = -1
        self._tilt_fixed_mask_out_vertex_version = -1
        self._energy_context = None
        self._module_energy_array_spec = {}
        self._module_accepts_ctx = {}
        self._stepper_accepts_trial_energy_fn = None

    def _sync_evaluation_manager(self) -> None:
        """Keep the extracted evaluator aligned with mutable minimizer state."""
        self._evaluation_manager.mesh = self.mesh
        self._evaluation_manager.global_params = self.global_params
        self._evaluation_manager.param_resolver = self.param_resolver
        self._evaluation_manager.energy_modules = self.energy_modules
        self._evaluation_manager.energy_module_names = self.energy_module_names

    def _validate_energy_modules_array(self) -> None:
        """Ensure energy modules support the array API required by minimization."""
        for module in self.energy_modules:
            if not hasattr(module, "compute_energy_and_gradient_array"):
                name = getattr(module, "__name__", repr(module))
                raise TypeError(
                    f"Energy module {name} lacks compute_energy_and_gradient_array; "
                    "dict fallbacks are not supported in the minimization loop."
                )

    def _use_stage_a_joint_tilt_projection(self) -> bool:
        """Enable coupled shape+tilt projection only for the Stage A emergent lane."""
        lane = str(self.global_params.get("theory_parity_lane") or "").strip().lower()
        return lane == "stage_a_emergent"

    def energy_context(self) -> EnergyContext:
        """Return a reusable evaluation context bound to current mesh versions."""
        if self._energy_context is None:
            self._energy_context = EnergyContext()
        self._energy_context.ensure_for_mesh(self.mesh)
        return self._energy_context

    def _stepper_supports_trial_energy_fn(self) -> bool:
        """Return whether the active stepper accepts ``trial_energy_fn``."""
        if self._stepper_accepts_trial_energy_fn is None:
            accepts = False
            step_fn = getattr(self.stepper, "step", None)
            if step_fn is not None:
                try:
                    sig = inspect.signature(step_fn)
                    accepts = "trial_energy_fn" in sig.parameters
                except (TypeError, ValueError):
                    accepts = False
            self._stepper_accepts_trial_energy_fn = accepts
        return bool(self._stepper_accepts_trial_energy_fn)

    def _call_module_array(self, module, **kwargs):
        """Call module array API with explicit ctx and graceful fallback."""
        key = id(module)
        accepts_ctx = self._module_accepts_ctx.get(key)
        if accepts_ctx is None:
            accepts_ctx = False
            fn = getattr(module, "compute_energy_and_gradient_array", None)
            if fn is not None:
                try:
                    sig = inspect.signature(fn)
                    if "ctx" in sig.parameters:
                        accepts_ctx = True
                    else:
                        accepts_ctx = any(
                            p.kind is inspect.Parameter.VAR_KEYWORD
                            for p in sig.parameters.values()
                        )
                except (TypeError, ValueError):
                    accepts_ctx = False
            self._module_accepts_ctx[key] = accepts_ctx

        if accepts_ctx:
            return module.compute_energy_and_gradient_array(
                self.mesh,
                self.global_params,
                self.param_resolver,
                ctx=self.energy_context(),
                **kwargs,
            )
        return module.compute_energy_and_gradient_array(
            self.mesh,
            self.global_params,
            self.param_resolver,
            **kwargs,
        )

    def _call_module_energy_array(self, module, **kwargs):
        """Call energy-only array API while honoring per-module signatures."""
        fn = getattr(module, "compute_energy_array")
        key = id(module)
        spec = self._module_energy_array_spec.get(key)
        if spec is None:
            accepts_resolver = False
            accepts_ctx = False
            accepted_kwargs: frozenset[str] | None = frozenset()
            try:
                sig = inspect.signature(fn)
                params = sig.parameters
                accepts_resolver = "param_resolver" in params
                accepts_var_kwargs = any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
                accepts_ctx = "ctx" in params or accepts_var_kwargs
                if accepts_var_kwargs:
                    accepted_kwargs = None
                else:
                    accepted_kwargs = frozenset(
                        name
                        for name, param in params.items()
                        if param.kind
                        in (
                            inspect.Parameter.POSITIONAL_OR_KEYWORD,
                            inspect.Parameter.KEYWORD_ONLY,
                        )
                    )
            except (TypeError, ValueError):
                accepted_kwargs = None
            spec = (accepts_resolver, accepts_ctx, accepted_kwargs)
            self._module_energy_array_spec[key] = spec

        accepts_resolver, accepts_ctx, accepted_kwargs = spec
        call_kwargs = kwargs
        if accepted_kwargs is not None:
            call_kwargs = {
                name: value for name, value in kwargs.items() if name in accepted_kwargs
            }
        if accepts_ctx:
            call_kwargs = dict(call_kwargs)
            call_kwargs["ctx"] = self.energy_context()

        if accepts_resolver:
            return fn(self.mesh, self.global_params, self.param_resolver, **call_kwargs)
        return fn(self.mesh, self.global_params, **call_kwargs)

    @staticmethod
    def _coerce_energy_value(energy_value) -> float:
        """Normalize scalar- or array-valued module energies to a float total."""
        energy_arr = np.asarray(energy_value, dtype=float)
        if energy_arr.ndim == 0:
            return float(energy_arr)
        return float(np.sum(energy_arr))

    def _soa_views(self) -> tuple[np.ndarray, Dict[int, int], np.ndarray]:
        """Return cached SoA views for positions, index map, and a scratch buffer."""
        ctx = self.energy_context()
        positions, index_map = ctx.geometry.soa_views(self.mesh)
        grad_dummy = ctx.scratch_array(
            "soa_grad_dummy", shape=positions.shape, dtype=positions.dtype
        )
        return positions, index_map, grad_dummy

    def _triangle_rows(self) -> tuple[np.ndarray | None, list[int]]:
        """Return triangle rows/facets via energy context cache."""
        return self.energy_context().geometry.triangle_rows(self.mesh)

    def _tilt_fixed_mask(self) -> np.ndarray:
        """Return a boolean mask for vertices whose tilt is clamped.

        The mask is in ``mesh.vertex_ids`` row order.
        """
        mask, flags_version, vertex_version = get_cached_tilt_fixed_mask(
            mesh=self.mesh,
            flag_attr="tilt_fixed",
            cached_mask=self._tilt_fixed_mask_cache,
            cached_flags_version=self._tilt_fixed_mask_version,
            cached_vertex_version=self._tilt_fixed_mask_vertex_version,
        )
        self._tilt_fixed_mask_cache = mask
        self._tilt_fixed_mask_version = flags_version
        self._tilt_fixed_mask_vertex_version = vertex_version
        return mask

    def _tilt_fixed_mask_in(self) -> np.ndarray:
        """Return a boolean mask for vertices whose inner-leaflet tilt is clamped."""
        mask, flags_version, vertex_version = get_cached_tilt_fixed_mask(
            mesh=self.mesh,
            flag_attr="tilt_fixed_in",
            cached_mask=self._tilt_fixed_mask_in_cache,
            cached_flags_version=self._tilt_fixed_mask_in_version,
            cached_vertex_version=self._tilt_fixed_mask_in_vertex_version,
        )
        self._tilt_fixed_mask_in_cache = mask
        self._tilt_fixed_mask_in_version = flags_version
        self._tilt_fixed_mask_in_vertex_version = vertex_version
        return mask

    def _tilt_fixed_mask_out(self) -> np.ndarray:
        """Return a boolean mask for vertices whose outer-leaflet tilt is clamped."""
        mask, flags_version, vertex_version = get_cached_tilt_fixed_mask(
            mesh=self.mesh,
            flag_attr="tilt_fixed_out",
            cached_mask=self._tilt_fixed_mask_out_cache,
            cached_flags_version=self._tilt_fixed_mask_out_version,
            cached_vertex_version=self._tilt_fixed_mask_out_vertex_version,
        )
        self._tilt_fixed_mask_out_cache = mask
        self._tilt_fixed_mask_out_version = flags_version
        self._tilt_fixed_mask_out_vertex_version = vertex_version
        return mask

    def _uses_leaflet_tilts(self) -> bool:
        """Return True when any loaded module depends on tilt_in/tilt_out."""
        return any(
            getattr(module, "USES_TILT_LEAFLETS", False)
            for module in self.energy_modules
        )

    def _uses_vertex_tilts(self) -> bool:
        """Return True when any loaded module depends on the single tilt field."""
        return any(
            getattr(module, "USES_TILT", False) for module in self.energy_modules
        )

    def _experimental_energy_scale_for_module(self, module_name: str) -> float:
        """Return module-specific experimental scale for curved-theta ablation."""
        mode = (
            str(
                self.global_params.get("curved_theta_objective_ablation_mode", "off")
                or "off"
            )
            .strip()
            .lower()
        )
        if mode == "off":
            return 1.0
        if mode != "inner_outer_rescaled":
            raise ValueError(
                "curved_theta_objective_ablation_mode must be 'off' or "
                "'inner_outer_rescaled'."
            )
        if (
            str(self.global_params.get("benchmark_geometry_lane") or "flat_pinned")
            .strip()
            .lower()
            != "free_z"
        ):
            return 1.0
        if (
            str(self.global_params.get("benchmark_parameterization") or "legacy")
            .strip()
            .lower()
            != "kh_physical"
        ):
            return 1.0

        inner_scale = float(
            self.global_params.get("curved_theta_objective_ablation_inner_scale", 1.0)
            or 1.0
        )
        outer_scale = float(
            self.global_params.get("curved_theta_objective_ablation_outer_scale", 1.0)
            or 1.0
        )
        contact_scale = float(
            self.global_params.get("curved_theta_objective_ablation_contact_scale", 1.0)
            or 1.0
        )
        if inner_scale <= 0.0 or outer_scale <= 0.0 or contact_scale <= 0.0:
            raise ValueError("curved theta objective ablation scales must be > 0.")

        name = str(module_name)
        if name in {
            "tilt_in",
            "bending_tilt_in",
            "tilt_splay_twist_in",
            "tilt_smoothness_in",
        }:
            return float(inner_scale)
        if name in {
            "tilt_out",
            "bending_tilt_out",
            "tilt_smoothness_out",
            "tilt_rim_source_out",
            "tilt_disk_target_out",
        }:
            return float(outer_scale)
        if name == "tilt_thetaB_contact_in":
            return float(contact_scale)
        return 1.0

    def _line_search_energy_fn(self) -> Callable[[], float]:
        """Return an energy callback for steppers (optionally reduced over tilts).

        When ``global_parameters.line_search_reduced_energy`` is enabled, the
        callback performs a short inner tilt relaxation (positions frozen) prior
        to evaluating the energy.

        IMPORTANT: When reduced-energy is enabled, the line-search routine must
        snapshot tilts *after* evaluating the baseline energy and restore them
        on rejected trial steps. Otherwise, accepted trial positions can end up
        paired with stale tilts.
        """

        def _projected_energy() -> float:
            # Keep objective evaluations consistent with the post-step state:
            # the minimizer always projects stored tilts to the local tangent
            # plane after accepting a step. Without this, the per-step energy
            # printed from the accepted line-search trial can differ wildly
            # from the energy reported after the iteration completes.
            self.mesh.project_tilts_to_tangent()
            return float(self.compute_energy())

        if not bool(self.global_params.get("line_search_reduced_energy", False)):
            return _projected_energy

        reduced_steps = int(
            self.global_params.get("line_search_reduced_tilt_inner_steps", 0) or 0
        )
        if reduced_steps <= 0:
            return _projected_energy

        return build_reduced_line_search_energy_fn(
            mesh=self.mesh,
            global_params=self.global_params,
            reduced_steps=reduced_steps,
            uses_leaflet_tilts=self._uses_leaflet_tilts(),
            projected_energy_fn=_projected_energy,
            compute_energy_fn=self.compute_energy,
            relax_tilts_fn=self._relax_tilts,
            relax_leaflet_tilts_fn=self._relax_leaflet_tilts,
            set_leaflet_tilts_fast_fn=self._set_leaflet_tilts_from_arrays_fast,
            logger_obj=logger,
        )

    def _line_search_trial_energy_fn(
        self,
    ) -> Callable[[np.ndarray], float] | None:
        """Return a pure-array trial-energy callback for eligible line searches."""
        if self._has_enforceable_constraints:
            return None

        uses_leaflet_tilts = self._uses_leaflet_tilts()
        uses_vertex_tilts = self._uses_vertex_tilts()
        if uses_leaflet_tilts and uses_vertex_tilts:
            return None

        reduced_steps = int(
            self.global_params.get("line_search_reduced_tilt_inner_steps", 0) or 0
        )
        reduced_energy = bool(
            self.global_params.get("line_search_reduced_energy", False)
        )
        if reduced_energy and reduced_steps > 0:

            def _projected_energy() -> float:
                self.mesh.project_tilts_to_tangent()
                return float(self.compute_energy())

            def _trial_raw_energy(positions: np.ndarray) -> float:
                if uses_leaflet_tilts:
                    return float(
                        self._compute_energy_array_with_leaflet_tilts(
                            positions=positions,
                            tilts_in=self.mesh.tilts_in_view(),
                            tilts_out=self.mesh.tilts_out_view(),
                        )
                    )
                if uses_vertex_tilts:
                    return float(
                        self._compute_total_energy_array_with_tilts(
                            positions=positions,
                            tilts=self.mesh.tilts_view(),
                        )
                    )
                return float(self._compute_energy_array_total(positions=positions))

            def _trial_projected_energy(positions: np.ndarray) -> float:
                if uses_leaflet_tilts:
                    normals = self.mesh.vertex_normals(positions)
                    tilts_in = project_tilts_to_tangent_array(
                        self.mesh.tilts_in_view(), normals
                    )
                    tilts_out = project_tilts_to_tangent_array(
                        self.mesh.tilts_out_view(), normals
                    )
                    self._set_leaflet_tilts_from_arrays_fast(tilts_in, tilts_out)
                    return float(
                        self._compute_energy_array_with_leaflet_tilts(
                            positions=positions,
                            tilts_in=tilts_in,
                            tilts_out=tilts_out,
                        )
                    )

                if uses_vertex_tilts:
                    normals = self.mesh.vertex_normals(positions)
                    tilts = project_tilts_to_tangent_array(
                        self.mesh.tilts_view(), normals
                    )
                    self.mesh.set_tilts_from_array(tilts)
                    return float(
                        self._compute_total_energy_array_with_tilts(
                            positions=positions, tilts=tilts
                        )
                    )

                return float(self._compute_energy_array_total(positions=positions))

            return build_reduced_line_search_trial_energy_fn(
                mesh=self.mesh,
                global_params=self.global_params,
                reduced_steps=reduced_steps,
                uses_leaflet_tilts=uses_leaflet_tilts,
                projected_energy_fn=_projected_energy,
                compute_energy_fn=self.compute_energy,
                projected_energy_at_positions_fn=_trial_projected_energy,
                compute_energy_at_positions_fn=_trial_raw_energy,
                relax_tilts_fn=self._relax_tilts,
                relax_leaflet_tilts_fn=self._relax_leaflet_tilts,
                set_leaflet_tilts_fast_fn=self._set_leaflet_tilts_from_arrays_fast,
                logger_obj=logger,
            )

        if uses_leaflet_tilts:

            def _trial_projected_leaflet_energy(positions: np.ndarray) -> float:
                normals = self.mesh.vertex_normals(positions)
                tilts_in = project_tilts_to_tangent_array(
                    self.mesh.tilts_in_view(), normals
                )
                tilts_out = project_tilts_to_tangent_array(
                    self.mesh.tilts_out_view(), normals
                )
                return float(
                    self._compute_energy_array_with_leaflet_tilts(
                        positions=positions,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                    )
                )

            return _trial_projected_leaflet_energy

        if uses_vertex_tilts:

            def _trial_projected_tilt_energy(positions: np.ndarray) -> float:
                normals = self.mesh.vertex_normals(positions)
                tilts = project_tilts_to_tangent_array(self.mesh.tilts_view(), normals)
                return float(
                    self._compute_total_energy_array_with_tilts(
                        positions=positions, tilts=tilts
                    )
                )

            return _trial_projected_tilt_energy

        return lambda positions: float(
            self._compute_energy_array_total(positions=positions)
        )

    def _compute_energy_array_total(self, *, positions: np.ndarray) -> float:
        """Compute total energy for fixed positions and the current mesh tilt state."""
        self._sync_evaluation_manager()
        return self._evaluation_manager.compute_energy_array_total(positions=positions)

    def _compute_total_energy_array_with_tilts(
        self,
        *,
        positions: np.ndarray,
        tilts: np.ndarray,
    ) -> float:
        """Compute total energy for fixed positions and a projected tilt field."""
        self._sync_evaluation_manager()
        return self._evaluation_manager.compute_total_energy_array_with_tilts(
            positions=positions,
            tilts=tilts,
        )

    def _compute_energy_array_with_tilts(
        self,
        *,
        positions: np.ndarray,
        tilts: np.ndarray,
    ) -> float:
        """Compute tilt-dependent energy for fixed ``positions``/``tilts``.

        Uses the array API when available and passes ``tilts`` opportunistically
        (falling back when a module does not accept tilt arguments).
        """
        self._sync_evaluation_manager()
        return self._evaluation_manager.compute_energy_array_with_tilts(
            positions=positions,
            tilts=tilts,
        )

    def _compute_energy_and_tilt_gradient_array(
        self,
        *,
        positions: np.ndarray,
        tilts: np.ndarray,
        tilt_grad_arr: np.ndarray,
    ) -> float:
        """Compute tilt-dependent energy and accumulate dense tilt gradient."""
        self._sync_evaluation_manager()
        return self._evaluation_manager.compute_energy_and_tilt_gradient_array(
            positions=positions,
            tilts=tilts,
            tilt_grad_arr=tilt_grad_arr,
        )

    def _compute_energy_array_with_leaflet_tilts(
        self,
        *,
        positions: np.ndarray,
        tilts_in: np.ndarray,
        tilts_out: np.ndarray,
        grad_dummy: np.ndarray | None = None,
    ) -> float:
        """Compute total energy for fixed positions and leaflet tilt arrays."""
        self._sync_evaluation_manager()
        return self._evaluation_manager.compute_energy_array_with_leaflet_tilts(
            positions=positions,
            tilts_in=tilts_in,
            tilts_out=tilts_out,
            grad_dummy=grad_dummy,
        )

    def _compute_tilt_dependent_energy_with_leaflet_tilts(
        self,
        *,
        positions: np.ndarray,
        tilts_in: np.ndarray,
        tilts_out: np.ndarray,
        grad_dummy: np.ndarray | None = None,
        tilt_vertex_areas_in: np.ndarray | None = None,
        tilt_vertex_areas_out: np.ndarray | None = None,
    ) -> float:
        """Compute energy of tilt-dependent modules only (positions frozen).

        This is used inside inner-loop tilt relaxation. Shape-only energy terms
        are constant when positions are frozen, so dropping them preserves
        backtracking accept/reject decisions while avoiding extra work.
        """
        self._sync_evaluation_manager()
        return (
            self._evaluation_manager.compute_tilt_dependent_energy_with_leaflet_tilts(
                positions=positions,
                tilts_in=tilts_in,
                tilts_out=tilts_out,
                grad_dummy=grad_dummy,
                tilt_vertex_areas_in=tilt_vertex_areas_in,
                tilt_vertex_areas_out=tilt_vertex_areas_out,
            )
        )

    def _compute_energy_and_leaflet_tilt_gradients_array(
        self,
        *,
        positions: np.ndarray,
        tilts_in: np.ndarray,
        tilts_out: np.ndarray,
        tilt_in_grad_arr: np.ndarray,
        tilt_out_grad_arr: np.ndarray,
        tilt_vertex_areas_in: np.ndarray | None = None,
        tilt_vertex_areas_out: np.ndarray | None = None,
        grad_dummy: np.ndarray | None = None,
        tilt_only: bool = False,
    ) -> float:
        """Compute total energy and accumulate leaflet tilt gradients."""
        index_map = self.mesh.vertex_index_to_row
        if grad_dummy is None:
            grad_dummy = np.zeros_like(positions)
        else:
            grad_dummy.fill(0.0)
        tilt_in_grad_arr.fill(0.0)
        tilt_out_grad_arr.fill(0.0)
        total_energy = 0.0

        for name, module in zip(self.energy_module_names, self.energy_modules):
            scale = self._experimental_energy_scale_for_module(str(name))
            # Fast path for the pure tilt magnitude penalties: when positions
            # are frozen (tilt relaxation inner loop), precomputed vertex areas
            # avoid repeated triangle cross-products.
            if (
                name == "tilt_in"
                and tilt_vertex_areas_in is not None
                and getattr(module, "USES_TILT_LEAFLETS", False)
            ):
                k_tilt = float(self.param_resolver.get(None, "tilt_modulus_in") or 0.0)
                if k_tilt != 0.0:
                    sq = np.einsum("ij,ij->i", tilts_in, tilts_in)
                    total_energy += float(
                        float(scale) * 0.5 * k_tilt * np.sum(sq * tilt_vertex_areas_in)
                    )
                    tilt_in_grad_arr += (
                        float(scale) * k_tilt * tilts_in * tilt_vertex_areas_in[:, None]
                    )
                continue

            if (
                name == "tilt_out"
                and tilt_vertex_areas_out is not None
                and getattr(module, "USES_TILT_LEAFLETS", False)
            ):
                k_tilt = float(self.param_resolver.get(None, "tilt_modulus_out") or 0.0)
                if k_tilt != 0.0:
                    sq = np.einsum("ij,ij->i", tilts_out, tilts_out)
                    total_energy += float(
                        float(scale) * 0.5 * k_tilt * np.sum(sq * tilt_vertex_areas_out)
                    )
                    tilt_out_grad_arr += (
                        float(scale)
                        * k_tilt
                        * tilts_out
                        * tilt_vertex_areas_out[:, None]
                    )
                continue

            if hasattr(module, "compute_energy_and_gradient_array"):
                grad_arg = (
                    None
                    if tilt_only and getattr(module, "USES_TILT_LEAFLETS", False)
                    else grad_dummy
                )
                in_before = None
                out_before = None
                if abs(float(scale) - 1.0) > 1.0e-15:
                    in_before = tilt_in_grad_arr.copy()
                    out_before = tilt_out_grad_arr.copy()
                try:
                    E_mod = self._call_module_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=grad_arg,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                        tilt_in_grad_arr=tilt_in_grad_arr,
                        tilt_out_grad_arr=tilt_out_grad_arr,
                    )
                except TypeError:
                    E_mod = self._call_module_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=grad_arg,
                    )
                if in_before is not None and out_before is not None:
                    in_delta = tilt_in_grad_arr - in_before
                    out_delta = tilt_out_grad_arr - out_before
                    tilt_in_grad_arr[:] = in_before + (float(scale) * in_delta)
                    tilt_out_grad_arr[:] = out_before + (float(scale) * out_delta)
                total_energy += float(scale) * float(E_mod)
                continue

            res = module.compute_energy_and_gradient(
                self.mesh, self.global_params, self.param_resolver
            )
            if not isinstance(res, tuple) or len(res) < 2:
                raise ValueError(
                    f"Unexpected return from energy module {module}: {res!r}"
                )
            total_energy += float(scale) * float(res[0])

        return float(total_energy)

    def _relax_tilts(
        self,
        *,
        positions: np.ndarray,
        mode: str,
    ) -> None:
        """Relax vertex tilt vectors according to the configured solve mode."""
        self._tilt_relaxation_manager.relax_tilts(
            mesh=self.mesh,
            global_params=self.global_params,
            positions=positions,
            mode=mode,
        )

    def _relax_leaflet_tilts(
        self,
        *,
        positions: np.ndarray,
        mode: str,
    ) -> None:
        """Relax inner/outer leaflet tilt vectors according to solve mode."""
        self._tilt_relaxation_manager.relax_leaflet_tilts(
            mesh=self.mesh,
            global_params=self.global_params,
            constraint_manager=self.constraint_manager,
            constraint_modules=self.constraint_modules,
            positions=positions,
            mode=mode,
        )
        self._last_tilt_projection_stats = (
            self._tilt_relaxation_manager.last_tilt_projection_stats
        )
        self._last_inner_coupled_update_mode_stats = (
            self._tilt_relaxation_manager.last_inner_coupled_update_mode_stats
        )

    @staticmethod
    def _tilt_vertex_areas_from_triangles(
        *, n_vertices: int, tri_rows: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Return barycentric per-vertex areas based on triangle areas."""
        from runtime.steppers.tilt_relaxation import (
            _tilt_vertex_areas_from_triangles as _impl,
        )

        return _impl(n_vertices=n_vertices, tri_rows=tri_rows, positions=positions)

    def refresh_modules(self):
        """Re-load energy and constraint modules from the current mesh state."""
        # Refresh energy modules
        self.energy_module_names = list(self.mesh.energy_modules)
        self.energy_modules = [
            self.energy_manager.get_module(mod) for mod in self.mesh.energy_modules
        ]
        # Refresh constraint modules
        self.constraint_modules = [
            self.constraint_manager.get_constraint(constraint)
            for constraint in self.mesh.constraint_modules
        ]
        self._has_enforceable_constraints = any(
            hasattr(mod, "enforce_constraint") for mod in self.constraint_modules
        )
        self._sync_evaluation_manager()
        self.reset_soa_caches()
        logger.info(
            f"Minimizer modules refreshed: {len(self.energy_modules)} energy, {len(self.constraint_modules)} constraint."
        )

    def __repr__(self):
        msg = f"""### MINIMIZER ###
MESH:\t {self.mesh}
GLOBAL PARAMETERS:\t {self.global_params}
STEPPER:\t {self.stepper}
STEP SIZE:\t {self.step_size}
############"""
        return msg

    def compute_energy_and_gradient_array(self):
        """Return total energy and dense gradient array for the current mesh."""
        positions, index_map, _ = self._soa_views()
        grad_arr = np.zeros_like(positions)

        total_energy = 0.0
        for module in self.energy_modules:
            # Use fast array path
            E_mod = self._call_module_array(
                module,
                positions=positions,
                index_map=index_map,
                grad_arr=grad_arr,
            )
            total_energy += E_mod

        # Apply constraint modifications to the shape gradient. When leaflet
        # tilts are active, joint shape+tilt constraints should project the
        # shape block against the full coupled manifold while leaving the
        # tilt block at zero in this outer shape step.
        if (
            self._uses_leaflet_tilts()
            and self._use_stage_a_joint_tilt_projection()
            and hasattr(
                self.constraint_manager, "apply_joint_gradient_modifications_array"
            )
        ):
            tilt_in_grad = np.zeros_like(grad_arr)
            tilt_out_grad = np.zeros_like(grad_arr)
            self._compute_energy_and_leaflet_tilt_gradients_array(
                positions=positions,
                tilts_in=self.mesh.tilts_in_view(),
                tilts_out=self.mesh.tilts_out_view(),
                tilt_in_grad_arr=tilt_in_grad,
                tilt_out_grad_arr=tilt_out_grad,
                tilt_only=True,
            )
            self.constraint_manager.apply_joint_gradient_modifications_array(
                grad_arr,
                tilt_in_grad,
                tilt_out_grad,
                self.mesh,
                self.global_params,
                positions=positions,
                tilts_in=self.mesh.tilts_in_view(),
                tilts_out=self.mesh.tilts_out_view(),
            )
        elif hasattr(self.constraint_manager, "apply_gradient_modifications_array"):
            self.constraint_manager.apply_gradient_modifications_array(
                grad_arr, self.mesh, self.global_params
            )

        # Always zero gradients for fixed vertices in the array pipeline.
        fixed_mask = self.mesh.fixed_mask
        if fixed_mask.shape[0] == grad_arr.shape[0]:
            grad_arr[fixed_mask] = 0.0

        return total_energy, grad_arr

    def compute_energy_and_gradient(self):
        """Return total energy and gradient for the current mesh."""

        # Use array backend for performance
        total_energy, grad_arr = self.compute_energy_and_gradient_array()

        # Convert back to dictionary for legacy components if needed
        # (Though we prefer staying in array space as long as possible)
        grad = self._grad_arr_to_dict(grad_arr)

        # Apply constraint modifications to the gradient (dictionary-based path)
        # Note: compute_energy_and_gradient_array already handled array modifications
        # but if we are here, we might need the dict modifications if array one was missing.
        if not hasattr(self.constraint_manager, "apply_gradient_modifications_array"):
            self.constraint_manager.apply_gradient_modifications(
                grad, self.mesh, self.global_params
            )
            self._zero_fixed_gradients(grad)

        log_lagrange_tangency_check(self, grad)

        return total_energy, grad

    def compute_energy_and_gradient_dict(self):
        """Return total energy and dict gradient using legacy module APIs.

        This intentionally bypasses the dense-array pipeline and is useful for
        regression tests comparing dict and array assembly.
        """
        total_energy = 0.0
        grad: Dict[int, np.ndarray] = {
            idx: np.zeros(3, dtype=float) for idx in self.mesh.vertices
        }

        for module in self.energy_modules:
            res = module.compute_energy_and_gradient(
                self.mesh, self.global_params, self.param_resolver
            )
            if isinstance(res, tuple) and len(res) >= 2:
                E_mod = res[0]
                g_mod = res[1]
            else:
                raise ValueError(
                    f"Unexpected return from energy module {module}: {res!r}"
                )

            total_energy += float(E_mod)
            for vidx, gvec in g_mod.items():
                if vidx in grad:
                    grad[vidx] += gvec

        self.constraint_manager.apply_gradient_modifications(
            grad, self.mesh, self.global_params
        )
        self._zero_fixed_gradients(grad)
        return float(total_energy), grad

    def compute_energy(self):
        """Compute the total energy using the loaded modules."""
        positions, _, _ = self._soa_views()
        return float(self._compute_energy_array_total(positions=positions))

    def compute_energy_breakdown(self) -> Dict[str, float]:
        """Return per-module energy contributions for the current mesh."""
        # Diagnostic breakdowns should be evaluated from a clean geometry cache.
        # This avoids carrying stale curvature-derived intermediates from prior
        # minimization iterations into report-only code paths.
        self.mesh._curvature_cache = {}
        self.mesh._curvature_version = -1
        positions, index_map, grad_dummy = self._soa_views()
        breakdown: Dict[str, float] = {}

        for name, module in zip(self.energy_module_names, self.energy_modules):
            scale = self._experimental_energy_scale_for_module(str(name))
            if hasattr(module, "compute_energy_and_gradient_array"):
                grad_dummy.fill(0.0)
                E_mod = self._call_module_array(
                    module,
                    positions=positions,
                    index_map=index_map,
                    grad_arr=grad_dummy,
                )
            else:
                E_mod, _ = module.compute_energy_and_gradient(
                    self.mesh,
                    self.global_params,
                    self.param_resolver,
                    compute_gradient=False,
                )
            breakdown[name] = float(scale) * float(E_mod)
        return breakdown

    def _grad_arr_to_dict(self, grad_arr: np.ndarray) -> Dict[int, np.ndarray]:
        """Convert a dense gradient array into a sparse dict keyed by vertex id."""
        return {
            vid: grad_arr[row].copy()
            for row, vid in enumerate(self.mesh.vertex_ids)
            if np.any(grad_arr[row])
        }

    def _zero_fixed_gradients(self, grad: Dict[int, np.ndarray]) -> None:
        """Helper to zero out gradients for fixed vertices in dict format."""
        for vidx, vertex in self.mesh.vertices.items():
            if getattr(vertex, "fixed", False):
                grad[vidx][:] = 0.0

    def project_constraints(self, grad: Dict[int, np.ndarray] | np.ndarray) -> None:
        """Project gradients onto the feasible set defined by constraints."""

        if isinstance(grad, np.ndarray):
            self.project_constraints_array(grad)
            return

        # Legacy dict path
        self._zero_fixed_gradients(grad)
        # Add projection for non-fixed constraints if needed here

    def project_constraints_array(self, grad_arr: np.ndarray) -> None:
        """Array-based variant of ``project_constraints``.

        The array pipeline uses KKT projection for hard constraints. This hook
        only enforces fixed vertices for compatibility with legacy call sites.
        """
        vertex_ids = self.mesh.vertex_ids
        for row, vidx in enumerate(vertex_ids):
            if getattr(self.mesh.vertices[int(vidx)], "fixed", False):
                grad_arr[row] = 0.0

    def _enforce_constraints(self, mesh: Mesh | None = None):
        """Invoke all constraint modules on the current mesh."""
        if not self._has_enforceable_constraints:
            return

        target_mesh = mesh if mesh is not None else self.mesh
        self.constraint_manager.enforce_all(
            target_mesh,
            global_params=self.global_params,
            context="minimize",
        )
        # Some constraints act on the tilt field only (e.g., rim-matching
        # projections). Enforce them here as well so the post-step state is
        # consistent without needing an extra tilt-relaxation pass.
        if hasattr(self.constraint_manager, "enforce_tilt_constraints"):
            self.constraint_manager.enforce_tilt_constraints(
                target_mesh, global_params=self.global_params
            )

    def _update_scalar_params(self) -> None:
        """Allow energy modules to update global scalar parameters."""
        thetaB_opt = bool(self.global_params.get("tilt_thetaB_optimize", False))
        for name, module in zip(self.energy_module_names, self.energy_modules):
            # When optimizing thetaB as a global scalar DOF, skip the local
            # thetaB closed-form update (it only minimizes the contact+penalty
            # term, not the full reduced energy after tilt relaxation).
            if thetaB_opt and name == "tilt_thetaB_contact_in":
                continue
            if hasattr(module, "update_scalar_params"):
                try:
                    module.update_scalar_params(
                        self.mesh, self.global_params, self.param_resolver
                    )
                except TypeError:
                    module.update_scalar_params(self.mesh, self.global_params)

    def _set_leaflet_tilts_from_arrays_fast(
        self, tilts_in: np.ndarray, tilts_out: np.ndarray
    ) -> None:
        """Update leaflet tilt caches from dense arrays without per-vertex scatter."""
        tilts_in_arr = np.asarray(tilts_in, dtype=float)
        tilts_out_arr = np.asarray(tilts_out, dtype=float)
        in_view = self.mesh.tilts_in_view()
        out_view = self.mesh.tilts_out_view()
        if tilts_in_arr.shape != in_view.shape or tilts_out_arr.shape != out_view.shape:
            raise ValueError("tilt arrays must match mesh leaflet tilt view shapes")
        in_view[:] = tilts_in_arr
        out_view[:] = tilts_out_arr
        self.mesh.touch_tilts_in()
        self.mesh.touch_tilts_out()

    def _optimize_thetaB_scalar(self, *, tilt_mode: str, iteration: int) -> None:
        """Optionally optimize the scalar thetaB by sampling reduced energies.

        This treats thetaB as a global scalar degree of freedom and updates it
        by comparing the total energy after a partial tilt relaxation for a few
        candidate thetaB values.
        """
        mode_match = (
            str(self.global_params.get("rim_slope_match_mode") or "").strip().lower()
        )
        trace_radius = self.global_params.get("parity_trace_layer_radius")
        outer_shells = int(self.global_params.get("parity_outer_shells", 0) or 0)
        if (
            mode_match == "physical_edge_staggered_v1"
            and trace_radius is not None
            and outer_shells > 0
        ):
            return
        if not bool(self.global_params.get("tilt_thetaB_optimize", False)):
            return

        every = int(self.global_params.get("tilt_thetaB_optimize_every", 10) or 10)
        if every <= 0:
            every = 1
        if int(iteration) % every != 0:
            return

        delta = float(self.global_params.get("tilt_thetaB_optimize_delta", 0.02) or 0.0)
        if delta <= 0.0:
            return

        # Warm-start from the current relaxed state.
        base_thetaB = float(self.global_params.get("tilt_thetaB_value") or 0.0)
        base_tin = self.mesh.tilts_in_view().copy(order="F")
        base_tout = self.mesh.tilts_out_view().copy(order="F")

        # Use a smaller inner relaxation budget for the thetaB scan to keep the
        # optimization cheap, but still responsive.
        orig_inner_steps = self.global_params.get("tilt_inner_steps", None)
        scan_steps = int(
            self.global_params.get("tilt_thetaB_optimize_inner_steps", 20) or 20
        )
        if scan_steps < 1:
            scan_steps = 1
        self.global_params.set("tilt_inner_steps", scan_steps)

        # Guard threshold prevents thetaB scan candidates from diverging.
        scan_guard_factor = float(
            self.global_params.get("tilt_relax_energy_guard_factor", 0.0) or 0.0
        )

        def eval_candidate(thetaB_val: float) -> tuple[float, np.ndarray, np.ndarray]:
            self.global_params.set("tilt_thetaB_value", float(thetaB_val))
            self._set_leaflet_tilts_from_arrays_fast(base_tin, base_tout)
            # Relax tilts only; shape is handled by the main loop.
            self._relax_leaflet_tilts(
                positions=self.mesh.positions_view(), mode=tilt_mode
            )
            e = float(self.compute_energy())
            # If the candidate's energy is much worse than the baseline,
            # discard it to prevent tilt divergence from corrupting
            # the selection.
            if scan_guard_factor > 0.0:
                scan_threshold = max(
                    float(
                        self.global_params.get("tilt_relax_energy_guard_min", 1e-4)
                        or 1e-4
                    ),
                    abs(e0) * scan_guard_factor,
                )
                if e > scan_threshold:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(
                            "thetaB scan: thetaB=%.6g candidate E=%.6g "
                            "exceeds guard threshold %.6g; discarding.",
                            thetaB_val,
                            e,
                            scan_threshold,
                        )
                    # Return sentinel: restore base tilts and flag as bad.
                    self._set_leaflet_tilts_from_arrays_fast(base_tin, base_tout)
                    return (float("inf"), base_tin, base_tout)
            return (
                e,
                self.mesh.tilts_in_view().copy(order="F"),
                self.mesh.tilts_out_view().copy(order="F"),
            )

        try:
            e0 = float(self.compute_energy())
            e_minus, tin_minus, tout_minus = eval_candidate(base_thetaB - delta)
            e_plus, tin_plus, tout_plus = eval_candidate(base_thetaB + delta)
        finally:
            # Restore the user's configured inner step budget.
            if orig_inner_steps is None:
                self.global_params.unset("tilt_inner_steps")
            else:
                self.global_params.set("tilt_inner_steps", orig_inner_steps)

        # Pick the best among the sampled points (cheap coordinate descent).
        best = min(
            [
                (e0, base_thetaB, base_tin, base_tout),
                (e_minus, base_thetaB - delta, tin_minus, tout_minus),
                (e_plus, base_thetaB + delta, tin_plus, tout_plus),
            ],
            key=lambda x: x[0],
        )

        best_e, best_thetaB, best_tin, best_tout = best
        # Guard: if the best candidate is worse than the starting state,
        # roll back to the original thetaB and tilts to prevent the
        # optimizer from pushing the system into an unstable configuration
        # (e.g. after mesh refinement when the tilt eigenvalue spectrum
        # has changed).
        if best_e > e0:
            self.global_params.set("tilt_thetaB_value", float(base_thetaB))
            self._set_leaflet_tilts_from_arrays_fast(base_tin, base_tout)
            if logger.isEnabledFor(logging.WARNING):
                logger.warning(
                    "thetaB optimize: i=%d rollback (best E %.6g > base E %.6g); "
                    "keeping thetaB=%.6g.",
                    int(iteration),
                    best_e,
                    e0,
                    base_thetaB,
                )
        else:
            self.global_params.set("tilt_thetaB_value", float(best_thetaB))
            self._set_leaflet_tilts_from_arrays_fast(best_tin, best_tout)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "thetaB optimize: i=%d thetaB %.6g -> %.6g (E %.6g -> %.6g)",
                int(iteration),
                base_thetaB,
                float(self.global_params.get("tilt_thetaB_value") or 0.0),
                e0,
                best_e,
            )

    def enforce_constraints_after_mesh_ops(self, mesh: Mesh | None = None):
        """Enforce constraints after discrete mesh operations."""
        if not self._has_enforceable_constraints:
            return

        target_mesh = mesh if mesh is not None else self.mesh
        self.constraint_manager.enforce_all(
            target_mesh,
            global_params=self.global_params,
            context="mesh_operation",
        )
        if hasattr(self.constraint_manager, "enforce_tilt_constraints"):
            self.constraint_manager.enforce_tilt_constraints(
                target_mesh, global_params=self.global_params
            )
            self._last_mesh_op_tilt_constraints_enforced = True
        else:
            self._last_mesh_op_tilt_constraints_enforced = False
        # Constraint modules mutate vertex positions/tilts in-place; bump the
        # mesh version so cached SoA views (positions/tilts/curvature) are not
        # stale after mesh operations like refinement/averaging.
        target_mesh.increment_version()

    def _triangle_aspect_percentile(self, percentile: float = 90.0) -> float:
        """Return triangle aspect-ratio percentile ``h_max/h_min`` for current mesh."""
        tri_rows, _ = self.mesh.triangle_row_cache()
        if tri_rows is None or len(tri_rows) == 0:
            return float("nan")
        pos = self.mesh.positions_view()
        tri = pos[np.asarray(tri_rows, dtype=int)]
        e01 = np.linalg.norm(tri[:, 0] - tri[:, 1], axis=1)
        e12 = np.linalg.norm(tri[:, 1] - tri[:, 2], axis=1)
        e20 = np.linalg.norm(tri[:, 2] - tri[:, 0], axis=1)
        h_max = np.maximum.reduce([e01, e12, e20])
        h_min = np.minimum.reduce([e01, e12, e20])
        ratio = h_max / np.maximum(h_min, 1e-18)
        return float(np.percentile(ratio, float(percentile)))

    def _maybe_auto_mesh_quality_repair(self, *, iteration: int) -> bool:
        """Optionally run bounded mesh-quality repair via equiangulation passes."""
        if not bool(self.global_params.get("mesh_quality_auto_repair_enabled", False)):
            return False
        every = int(self.global_params.get("mesh_quality_auto_repair_every", 0) or 0)
        if every <= 0 or ((int(iteration) + 1) % every) != 0:
            return False

        threshold = float(
            self.global_params.get("mesh_quality_aspect_threshold", 0.0) or 0.0
        )
        if threshold <= 0.0:
            return False
        perc = float(
            self.global_params.get("mesh_quality_aspect_percentile", 90.0) or 90.0
        )
        max_passes = int(
            self.global_params.get("mesh_quality_max_repair_passes", 1) or 1
        )
        if max_passes <= 0:
            return False

        aspect_p = self._triangle_aspect_percentile(percentile=perc)
        if not np.isfinite(aspect_p) or aspect_p <= threshold:
            return False

        changed_any = False
        for _ in range(max_passes):
            new_mesh, changed = equiangulate_iteration(self.mesh)
            if not bool(changed):
                break
            self.mesh = new_mesh
            self.enforce_constraints_after_mesh_ops(self.mesh)
            self.mesh.project_tilts_to_tangent()
            self.mesh.increment_version()
            changed_any = True
            aspect_p = self._triangle_aspect_percentile(percentile=perc)
            if not np.isfinite(aspect_p) or aspect_p <= threshold:
                break
        if changed_any:
            reset = getattr(self.stepper, "reset", None)
            if callable(reset):
                reset()
        return changed_any

    def minimize(
        self, n_steps: int = 1, callback: Optional[Callable[["Mesh", int], None]] = None
    ):
        """Run the optimization loop for ``n_steps`` iterations."""
        validate_leaflet_absence_topology(self.mesh, self.global_params)
        validate_disk_interface_topology(self.mesh, self.global_params)
        zero_step_counter = 0
        step_success = True

        if n_steps <= 0:
            E, grad = self.compute_energy_and_gradient()
            self._enforce_constraints()
            log_energy_consistency(self, "no_steps")
            E = float(self.compute_energy())
            return {
                "energy": E,
                "gradient": grad,
                "mesh": self.mesh,
                "step_success": True,
                "iterations": 0,
                "terminated_early": True,
            }

        if self._has_enforceable_constraints:
            self.enforce_constraints_after_mesh_ops(self.mesh)
            self.mesh.project_tilts_to_tangent()
            self.mesh.increment_version()

        check_gauss_bonnet(self)
        last_grad_arr = None
        last_state_energy: float | None = None
        for i in range(n_steps):
            if callback:
                callback(self.mesh, i)

            self._update_scalar_params()
            log_debug_energy_context(self, i)

            # Tilt solve modes are evaluated before the shape convergence check so
            # that fixed-geometry runs can still relax the tilt field.
            tilt_mode = self.global_params.get("tilt_solve_mode", "fixed")
            if self._uses_leaflet_tilts():
                # Guard against tilt-only relaxation spikes on fixed geometry.
                guard_factor = float(
                    self.global_params.get("tilt_relax_energy_guard_factor", 0.0) or 0.0
                )
                guard_min = float(
                    self.global_params.get("tilt_relax_energy_guard_min", 0.0) or 0.0
                )
                if guard_factor > 0.0:
                    pre_energy = float(self.compute_energy())
                    pre_tin = self.mesh.tilts_in_view().copy(order="F")
                    pre_tout = self.mesh.tilts_out_view().copy(order="F")
                    threshold = max(guard_min, abs(pre_energy) * guard_factor)
                    max_retries = int(
                        self.global_params.get("tilt_relax_energy_guard_retries", 4)
                        or 4
                    )
                    orig_tilt_step = float(
                        self.global_params.get("tilt_step_size", 0.0) or 0.0
                    )
                    trial_step = orig_tilt_step
                    accepted = False
                    for attempt in range(max_retries + 1):
                        self._relax_leaflet_tilts(
                            positions=self.mesh.positions_view(), mode=tilt_mode
                        )
                        post_energy = float(self.compute_energy())
                        if post_energy <= threshold:
                            accepted = True
                            self.mesh.project_tilts_to_tangent()
                            self.mesh.increment_version()
                            break
                        # Roll back and retry with a smaller tilt step.
                        self._set_leaflet_tilts_from_arrays_fast(pre_tin, pre_tout)
                        trial_step *= 0.5
                        self.global_params.set("tilt_step_size", trial_step)
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(
                                "Tilt guard retry %d/%d: E %.6g -> %.6g "
                                "(threshold %.6g); trying tilt_step=%.3e",
                                attempt + 1,
                                max_retries,
                                pre_energy,
                                post_energy,
                                threshold,
                                trial_step,
                            )
                    # Restore the original tilt step size.
                    self.global_params.set("tilt_step_size", orig_tilt_step)
                    if not accepted:
                        self._set_leaflet_tilts_from_arrays_fast(pre_tin, pre_tout)
                        if logger.isEnabledFor(logging.WARNING):
                            logger.warning(
                                "Tilt relaxation energy spike: %.6g -> %.6g "
                                "(threshold %.6g). Rolling back tilts for "
                                "iteration %d after %d retries.",
                                pre_energy,
                                post_energy,
                                threshold,
                                i,
                                max_retries,
                            )
                else:
                    self._relax_leaflet_tilts(
                        positions=self.mesh.positions_view(), mode=tilt_mode
                    )
            else:
                self._relax_tilts(positions=self.mesh.positions_view(), mode=tilt_mode)

            self._update_scalar_params()
            if self._uses_leaflet_tilts():
                self._optimize_thetaB_scalar(tilt_mode=str(tilt_mode), iteration=i)

            # Use array-based path for main loop
            E, grad_arr = self.compute_energy_and_gradient_array()
            self.project_constraints_array(grad_arr)
            project_curved_free_disk_shape_dofs(self.mesh, self.global_params, grad_arr)
            last_grad_arr = grad_arr

            if logger.isEnabledFor(logging.DEBUG):
                log_energy_phase(i, "pre_step", E)
                log_step_direction_stats(i, grad_arr)

            # check convergence by gradient norm
            grad_norm = float(np.linalg.norm(grad_arr))
            if grad_norm < self.tol:
                logger.debug("Converged: gradient norm below tolerance.")
                logger.info(f"Converged in {i} iterations; |∇E|={grad_norm:.3e}")
                log_energy_consistency(self, "converged")
                return {
                    "energy": E,
                    "gradient": self._grad_arr_to_dict(grad_arr),
                    "mesh": self.mesh,
                    "step_success": True,
                    "iterations": i + 1,
                    "terminated_early": True,
                }
            logger.debug("Iteration %d: |∇E|=%.3e", i, grad_norm)

            step_mode = str(
                self.global_params.get("step_size_mode", "adaptive") or "adaptive"
            ).lower()
            fixed_step = float(
                self.global_params.get("step_size", self.step_size) or self.step_size
            )
            step_size_in = fixed_step if step_mode == "fixed" else self.step_size
            energy_fn = self._line_search_energy_fn()
            trial_energy_fn = self._line_search_trial_energy_fn()
            reduced_flag = (
                bool(self.global_params.get("line_search_reduced_energy", False))
                and int(
                    self.global_params.get("line_search_reduced_tilt_inner_steps", 0)
                    or 0
                )
                > 0
            )
            # Signal to the line-search routine that energy_fn may mutate tilts
            # during trial evaluations (reduced objective), and optionally
            # switch to an accept/reject rule that uses post-relax reduced
            # energies (rather than Armijo using the partial gradient).
            setattr(self.mesh, "_line_search_reduced_energy", reduced_flag)
            if reduced_flag:
                accept_rule = str(
                    self.global_params.get("line_search_reduced_accept_rule", "armijo")
                    or "armijo"
                )
                setattr(self.mesh, "_line_search_reduced_accept_rule", accept_rule)

            try:
                step_kwargs = {}
                if self._stepper_supports_trial_energy_fn():
                    step_kwargs["trial_energy_fn"] = trial_energy_fn
                step_success, self.step_size, accepted_energy = self.stepper.step(
                    self.mesh,
                    grad_arr,
                    step_size_in,
                    energy_fn,
                    constraint_enforcer=self._enforce_constraints
                    if self._has_enforceable_constraints
                    else None,
                    **step_kwargs,
                )
            finally:
                if hasattr(self.mesh, "_line_search_reduced_energy"):
                    delattr(self.mesh, "_line_search_reduced_energy")
                if hasattr(self.mesh, "_line_search_reduced_accept_rule"):
                    delattr(self.mesh, "_line_search_reduced_accept_rule")
            last_state_energy = float(accepted_energy)
            if logger.isEnabledFor(logging.DEBUG):
                # Do not probe energy here: additional debug-only energy
                # evaluations can mutate caches and perturb the optimization
                # trajectory.
                log_energy_phase(i, "post_step", float(last_state_energy))
            # Keep any stored 3D tilt field tangent to the updated surface.
            self.mesh.project_tilts_to_tangent()
            self.mesh.increment_version()
            if not self.quiet:
                # User-visible step diagnostics should report the same total
                # energy definition as `energy` / final summaries.
                total_area = self.mesh.compute_total_surface_area()
                # Clear curvature cache before reporting to avoid leaking
                # stale tilt-dependent intermediates across iterations.
                self.mesh._curvature_cache = {}
                self.mesh._curvature_version = -1
                reported_energy = float(self.compute_energy())
                print(
                    f"Step {i:4d}: Area = {total_area:.5f}, Energy = {reported_energy:.5f}, Step Size  = {step_size_in:.2e}"
                )
            if logger.isEnabledFor(logging.DEBUG):
                # Keep this phase marker without re-evaluating energy.
                log_energy_phase(i, "post_step_tilt_project", float("nan"))
            if step_mode == "fixed":
                # Keep the cross-iteration step size constant, but still allow
                # the line search to backtrack within each iteration for
                # stability.
                self.step_size = fixed_step

            check_gauss_bonnet(self)
            if not step_success:
                if self.step_size <= self.step_size_floor:
                    zero_step_counter += 1
                    if zero_step_counter >= self.max_zero_steps:
                        logger.info(
                            "Terminating early after %d consecutive zero-steps "
                            "with step size <= %.2e.",
                            zero_step_counter,
                            self.step_size_floor,
                        )
                        log_energy_consistency(self, "terminated_early")
                        return {
                            "energy": float(self.compute_energy()),
                            "gradient": self._grad_arr_to_dict(last_grad_arr)
                            if last_grad_arr is not None
                            else {},
                            "mesh": self.mesh,
                            "step_success": False,
                            "iterations": i + 1,
                            "terminated_early": True,
                        }
                else:
                    zero_step_counter = 0
                reset = getattr(self.stepper, "reset", None)
                if callable(reset):
                    reset()
            else:
                zero_step_counter = 0

                log_accepted_step_stats(
                    self,
                    iteration=i,
                    E_before=E,
                    E_accepted=float(last_state_energy)
                    if last_state_energy is not None
                    else float("nan"),
                    step_size=self.step_size,
                )

                mode = self.global_params.get("volume_constraint_mode", "lagrange")

                proj_flag = self.global_params.get(
                    "volume_projection_during_minimization", True
                )
                vol_tol = float(self.global_params.get("volume_tolerance", 1e-3))
                if mode == "lagrange" and not proj_flag and self.mesh.bodies:
                    max_rel_violation = 0.0
                    for body in self.mesh.bodies.values():
                        target = body.target_volume
                        if target is None:
                            target = body.options.get("target_volume")
                        if target is None:
                            continue
                        current = body.compute_volume(self.mesh)
                        denom = max(abs(target), 1.0)
                        rel = abs(current - target) / denom
                        if rel > max_rel_violation:
                            max_rel_violation = rel

                    if max_rel_violation > vol_tol:
                        logger.debug(
                            "Volume drift %.3e exceeds tolerance %.3e; "
                            "applying geometric volume projection.",
                            max_rel_violation,
                            vol_tol,
                        )
                        self.enforce_constraints_after_mesh_ops(self.mesh)
                        self.mesh.project_tilts_to_tangent()
                        self.mesh.increment_version()
                        if logger.isEnabledFor(logging.DEBUG):
                            # Avoid debug-only energy probes here as well.
                            log_energy_phase(i, "post_step_constraints", float("nan"))
                        reset = getattr(self.stepper, "reset", None)
                        if callable(reset):
                            reset()

                self._maybe_auto_mesh_quality_repair(iteration=i)

        if self._has_enforceable_constraints:
            # One final projection improves cross-platform determinism for hard
            # constraints (e.g. fixed volume) without impacting the line search
            # acceptance logic.
            self.constraint_manager.enforce_all(
                self.mesh,
                global_params=self.global_params,
                context="finalize",
            )
            self.mesh.project_tilts_to_tangent()
            # Finalize projections mutate positions/tilts in-place; bump the
            # mesh version so subsequent energy evaluations rebuild geometry-
            # versioned caches from the finalized state.
            self.mesh.increment_version()

        log_energy_consistency(self, "finalize")
        self.mesh._curvature_cache = {}
        self.mesh._curvature_version = -1
        final_energy = float(self.compute_energy())
        return {
            "energy": final_energy,
            "gradient": self._grad_arr_to_dict(last_grad_arr)
            if last_grad_arr is not None
            else {},
            "mesh": self.mesh,
            "step_success": step_success,
            "iterations": n_steps,
            "terminated_early": False,
        }

# runtime/minimizer.py

import copy
import logging
from typing import Callable, Dict, List, Optional

import numpy as np

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.entities import Mesh, _fast_cross
from modules.energy.leaflet_presence import (
    leaflet_absent_vertex_mask,
    leaflet_present_triangle_mask,
)
from runtime.constraint_manager import ConstraintModuleManager
from runtime.diagnostics.gauss_bonnet import GaussBonnetMonitor
from runtime.energy_context import EnergyContext
from runtime.energy_manager import EnergyModuleManager
from runtime.interface_validation import validate_disk_interface_topology
from runtime.leaflet_validation import validate_leaflet_absence_topology
from runtime.steppers.base import BaseStepper

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
        self._last_mesh_op_tilt_constraints_enforced: bool | None = None
        self._energy_context: EnergyContext | None = None

    def _validate_energy_modules_array(self) -> None:
        """Ensure energy modules support the array API required by minimization."""
        for module in self.energy_modules:
            if not hasattr(module, "compute_energy_and_gradient_array"):
                name = getattr(module, "__name__", repr(module))
                raise TypeError(
                    f"Energy module {name} lacks compute_energy_and_gradient_array; "
                    "dict fallbacks are not supported in the minimization loop."
                )

    def energy_context(self) -> EnergyContext:
        """Return a reusable evaluation context bound to current mesh versions."""
        if self._energy_context is None:
            self._energy_context = EnergyContext()
        self._energy_context.ensure_for_mesh(self.mesh)
        # Internal bridge while modules are incrementally migrated to
        # context-aware cache access.
        setattr(self.mesh, "_active_energy_context", self._energy_context)
        return self._energy_context

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

    def _log_debug_energy_context(self, iteration: int) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        breakdown = self._diagnostic_energy_breakdown()
        thetaB_contact = float(breakdown.get("tilt_thetaB_contact_in", 0.0))
        thetaB_val = float(self.global_params.get("tilt_thetaB_value") or 0.0)
        line_search_reduced = bool(
            self.global_params.get("line_search_reduced_energy", False)
        )
        line_search_steps = int(
            self.global_params.get("line_search_reduced_tilt_inner_steps", 0) or 0
        )
        guard_factor = float(
            self.global_params.get("tilt_relax_energy_guard_factor", 0.0) or 0.0
        )
        tilt_enforced = self._last_mesh_op_tilt_constraints_enforced
        logger.debug(
            "Debug energy context i=%d: tilt_thetaB_contact_in=%.6g "
            "tilt_thetaB_value=%.6g line_search_reduced=%s "
            "line_search_steps=%d tilt_relax_guard=%.6g "
            "tilt_constraints_enforced=%s",
            iteration,
            thetaB_contact,
            thetaB_val,
            line_search_reduced,
            line_search_steps,
            guard_factor,
            tilt_enforced,
        )

    def _diagnostic_snapshot(self) -> dict:
        """Capture mutable state that must not change in debug diagnostics."""
        snapshot: dict = {
            "global_params": dict(self.global_params.to_dict()),
            "tilts": None,
            "tilts_in": None,
            "tilts_out": None,
        }
        if self._uses_leaflet_tilts():
            snapshot["tilts_in"] = self.mesh.tilts_in_view().copy(order="F")
            snapshot["tilts_out"] = self.mesh.tilts_out_view().copy(order="F")
        else:
            snapshot["tilts"] = self.mesh.tilts_view().copy(order="F")
        return snapshot

    def _diagnostic_restore(self, snapshot: dict) -> None:
        """Restore mutable state after debug diagnostics."""
        if snapshot.get("tilts_in") is not None:
            if not np.array_equal(self.mesh.tilts_in_view(), snapshot["tilts_in"]):
                self.mesh.set_tilts_in_from_array(snapshot["tilts_in"])
            if not np.array_equal(self.mesh.tilts_out_view(), snapshot["tilts_out"]):
                self.mesh.set_tilts_out_from_array(snapshot["tilts_out"])
        elif snapshot.get("tilts") is not None:
            if not np.array_equal(self.mesh.tilts_view(), snapshot["tilts"]):
                self.mesh.set_tilts_from_array(snapshot["tilts"])
        if snapshot.get("global_params") != self.global_params.to_dict():
            self.global_params._params = dict(snapshot["global_params"])

    def _run_debug_diagnostic(self, fn):
        """Run a diagnostic without mutating the live simulation state.

        Diagnostics are evaluated on a deep-copied mesh/parameters to guarantee
        observational behavior even when energy modules populate caches.
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return fn()

        # Only handle bound methods on this Minimizer instance. For other
        # callables, fall back to the snapshot/restore guard.
        fn_self = getattr(fn, "__self__", None)
        fn_name = getattr(fn, "__name__", None)
        if fn_self is not self or not isinstance(fn_name, str):
            snapshot = self._diagnostic_snapshot()
            try:
                return fn()
            finally:
                self._diagnostic_restore(snapshot)

        mesh_copy = self.mesh.copy()
        gp_copy = copy.deepcopy(self.global_params)
        mesh_copy.global_parameters = gp_copy
        diag = Minimizer(
            mesh_copy,
            gp_copy,
            self.stepper,
            self.energy_manager,
            self.constraint_manager,
            energy_modules=list(getattr(mesh_copy, "energy_modules", [])),
            constraint_modules=list(getattr(mesh_copy, "constraint_modules", [])),
            step_size=float(self.step_size),
            tol=float(self.tol),
            quiet=True,
        )

        if fn_name == "compute_energy":
            return diag.compute_energy()
        if fn_name == "compute_energy_breakdown":
            return diag.compute_energy_breakdown()
        if fn_name == "compute_energy_and_gradient_array":
            return diag.compute_energy_and_gradient_array()

        # Fallback: if this is another Minimizer method that takes no args,
        # prefer calling it on the copied Minimizer instance.
        target = getattr(diag, fn_name, None)
        if callable(target):
            try:
                return target()
            except TypeError:
                pass

        # Last resort: snapshot/restore guard on the live state.
        snapshot = self._diagnostic_snapshot()
        try:
            return fn()
        finally:
            self._diagnostic_restore(snapshot)

    def _diagnostic_energy(self) -> float:
        """Energy computation for diagnostics (debug-safe)."""
        return float(self._run_debug_diagnostic(self.compute_energy))

    def _diagnostic_energy_breakdown(self) -> Dict[str, float]:
        """Energy breakdown for diagnostics (debug-safe)."""
        return self._run_debug_diagnostic(self.compute_energy_breakdown)

    def _tilt_fixed_mask(self) -> np.ndarray:
        """Return a boolean mask for vertices whose tilt is clamped.

        The mask is in ``mesh.vertex_ids`` row order.
        """
        flags_version = self.mesh._tilt_fixed_flags_version
        vertex_version = self.mesh._vertex_ids_version
        if (
            self._tilt_fixed_mask_cache is not None
            and self._tilt_fixed_mask_version == flags_version
            and self._tilt_fixed_mask_vertex_version == vertex_version
            and len(self._tilt_fixed_mask_cache) == len(self.mesh.vertices)
        ):
            return self._tilt_fixed_mask_cache
        mask = np.array(
            [
                bool(getattr(self.mesh.vertices[int(vid)], "tilt_fixed", False))
                for vid in self.mesh.vertex_ids
            ],
            dtype=bool,
        )
        self._tilt_fixed_mask_cache = mask
        self._tilt_fixed_mask_version = flags_version
        self._tilt_fixed_mask_vertex_version = vertex_version
        return mask

    def _tilt_fixed_mask_in(self) -> np.ndarray:
        """Return a boolean mask for vertices whose inner-leaflet tilt is clamped."""
        flags_version = self.mesh._tilt_fixed_flags_version
        vertex_version = self.mesh._vertex_ids_version
        if (
            self._tilt_fixed_mask_in_cache is not None
            and self._tilt_fixed_mask_in_version == flags_version
            and self._tilt_fixed_mask_in_vertex_version == vertex_version
            and len(self._tilt_fixed_mask_in_cache) == len(self.mesh.vertices)
        ):
            return self._tilt_fixed_mask_in_cache
        mask = np.array(
            [
                bool(getattr(self.mesh.vertices[int(vid)], "tilt_fixed_in", False))
                for vid in self.mesh.vertex_ids
            ],
            dtype=bool,
        )
        self._tilt_fixed_mask_in_cache = mask
        self._tilt_fixed_mask_in_version = flags_version
        self._tilt_fixed_mask_in_vertex_version = vertex_version
        return mask

    def _tilt_fixed_mask_out(self) -> np.ndarray:
        """Return a boolean mask for vertices whose outer-leaflet tilt is clamped."""
        flags_version = self.mesh._tilt_fixed_flags_version
        vertex_version = self.mesh._vertex_ids_version
        if (
            self._tilt_fixed_mask_out_cache is not None
            and self._tilt_fixed_mask_out_version == flags_version
            and self._tilt_fixed_mask_out_vertex_version == vertex_version
            and len(self._tilt_fixed_mask_out_cache) == len(self.mesh.vertices)
        ):
            return self._tilt_fixed_mask_out_cache
        mask = np.array(
            [
                bool(getattr(self.mesh.vertices[int(vid)], "tilt_fixed_out", False))
                for vid in self.mesh.vertex_ids
            ],
            dtype=bool,
        )
        self._tilt_fixed_mask_out_cache = mask
        self._tilt_fixed_mask_out_version = flags_version
        self._tilt_fixed_mask_out_vertex_version = vertex_version
        return mask

    @staticmethod
    def _project_tilts_to_tangent_array(
        tilts: np.ndarray, normals: np.ndarray
    ) -> np.ndarray:
        """Project a dense tilt array into the vertex tangent planes."""
        dot = np.einsum("ij,ij->i", tilts, normals)
        return tilts - dot[:, None] * normals

    @staticmethod
    def _project_tilts_axisymmetric_about_center(
        *,
        positions: np.ndarray,
        tilts: np.ndarray,
        normals: np.ndarray,
        center: np.ndarray,
        axis: np.ndarray,
        fixed_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Project a tilt field into the axisymmetric (radial) subspace.

        This is a theory-mode helper used to compare against the axisymmetric
        continuum model in `docs/tex/1_disk_3d.tex`.

        The projection keeps only the tangent-plane radial component about
        `center`, where the in-plane radial direction is defined by removing
        the component along `axis`.
        """
        center = np.asarray(center, dtype=float).reshape(3)
        axis = np.asarray(axis, dtype=float).reshape(3)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-15:
            axis = np.array([0.0, 0.0, 1.0], dtype=float)
        else:
            axis = axis / axis_norm

        r_vec = positions - center[None, :]
        r_vec = r_vec - np.einsum("ij,j->i", r_vec, axis)[:, None] * axis[None, :]
        r_len = np.linalg.norm(r_vec, axis=1)
        good = r_len > 1e-12

        r_hat = np.zeros_like(r_vec)
        r_hat[good] = r_vec[good] / r_len[good][:, None]

        # Use the radial direction projected into each vertex tangent plane.
        r_dir = r_hat - np.einsum("ij,ij->i", r_hat, normals)[:, None] * normals
        r_norm = np.linalg.norm(r_dir, axis=1)
        good &= r_norm > 1e-12

        r_dir_unit = np.zeros_like(r_dir)
        r_dir_unit[good] = r_dir[good] / r_norm[good][:, None]

        proj = np.zeros_like(tilts)
        amp = np.einsum("ij,ij->i", tilts, r_dir_unit)
        proj[good] = amp[good][:, None] * r_dir_unit[good]

        if fixed_mask is not None and np.any(fixed_mask):
            proj[fixed_mask] = tilts[fixed_mask]
        return proj

    def _uses_leaflet_tilts(self) -> bool:
        """Return True when any loaded module depends on tilt_in/tilt_out."""
        return any(
            getattr(module, "USES_TILT_LEAFLETS", False)
            for module in self.energy_modules
        )

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

        uses_leaflet = self._uses_leaflet_tilts()
        gp = self.global_params

        override_keys = ("tilt_inner_steps", "tilt_coupled_steps", "tilt_cg_max_iters")
        override_present = {key: (key in gp) for key in override_keys}
        override_old = {key: gp.get(key) for key in override_keys}

        def _restore_overrides() -> None:
            for key in override_keys:
                if override_present.get(key, False):
                    gp.set(key, override_old[key])
                else:
                    gp.unset(key)

        def energy_fn() -> float:
            tilt_mode = str(gp.get("tilt_solve_mode", "fixed") or "fixed")
            mode_norm = tilt_mode.strip().lower()
            if mode_norm in ("", "none", "off", "false", "fixed"):
                return float(_projected_energy())

            positions = self.mesh.positions_view()
            try:
                gp.set("tilt_inner_steps", reduced_steps)
                gp.set("tilt_coupled_steps", reduced_steps)
                gp.set("tilt_cg_max_iters", reduced_steps)

                if uses_leaflet:
                    self._relax_leaflet_tilts(positions=positions, mode=tilt_mode)
                else:
                    self._relax_tilts(positions=positions, mode=tilt_mode)

                return float(_projected_energy())
            finally:
                _restore_overrides()

        return energy_fn

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
        index_map = self.mesh.vertex_index_to_row
        grad_dummy = np.zeros_like(positions)
        total_energy = 0.0

        for module in self.energy_modules:
            if not getattr(module, "USES_TILT", False):
                continue
            if hasattr(module, "compute_energy_array"):
                try:
                    E_mod = module.compute_energy_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                        tilts=tilts,
                    )
                except TypeError:
                    E_mod = module.compute_energy_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                    )
                total_energy += float(E_mod)
                continue

            if hasattr(module, "compute_energy_and_gradient_array"):
                try:
                    E_mod = module.compute_energy_and_gradient_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=grad_dummy,
                        tilts=tilts,
                        tilt_grad_arr=None,
                    )
                except TypeError:
                    try:
                        E_mod = module.compute_energy_and_gradient_array(
                            self.mesh,
                            self.global_params,
                            self.param_resolver,
                            positions=positions,
                            index_map=index_map,
                            grad_arr=grad_dummy,
                            tilts=tilts,
                        )
                    except TypeError:
                        E_mod = module.compute_energy_and_gradient_array(
                            self.mesh,
                            self.global_params,
                            self.param_resolver,
                            positions=positions,
                            index_map=index_map,
                            grad_arr=grad_dummy,
                        )
                total_energy += float(E_mod)
                continue

            # Legacy dict modules (typically tilt-independent): energy-only path.
            try:
                E_mod, _ = module.compute_energy_and_gradient(
                    self.mesh,
                    self.global_params,
                    self.param_resolver,
                    compute_gradient=False,
                )
            except TypeError:
                E_mod, _ = module.compute_energy_and_gradient(
                    self.mesh, self.global_params, self.param_resolver
                )
            total_energy += float(E_mod)

        return float(total_energy)

    def _compute_energy_and_tilt_gradient_array(
        self,
        *,
        positions: np.ndarray,
        tilts: np.ndarray,
        tilt_grad_arr: np.ndarray,
    ) -> float:
        """Compute tilt-dependent energy and accumulate dense tilt gradient."""
        index_map = self.mesh.vertex_index_to_row
        grad_dummy = np.zeros_like(positions)
        tilt_grad_arr.fill(0.0)
        total_energy = 0.0

        for module in self.energy_modules:
            if not getattr(module, "USES_TILT", False):
                continue
            if hasattr(module, "compute_energy_and_gradient_array"):
                try:
                    E_mod = module.compute_energy_and_gradient_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=grad_dummy,
                        tilts=tilts,
                        tilt_grad_arr=tilt_grad_arr,
                    )
                except TypeError:
                    try:
                        E_mod = module.compute_energy_and_gradient_array(
                            self.mesh,
                            self.global_params,
                            self.param_resolver,
                            positions=positions,
                            index_map=index_map,
                            grad_arr=grad_dummy,
                            tilts=tilts,
                        )
                    except TypeError:
                        E_mod = module.compute_energy_and_gradient_array(
                            self.mesh,
                            self.global_params,
                            self.param_resolver,
                            positions=positions,
                            index_map=index_map,
                            grad_arr=grad_dummy,
                        )
                total_energy += float(E_mod)
                continue

            # Dict fallback: accept modules that optionally return a tilt gradient.
            res = module.compute_energy_and_gradient(
                self.mesh, self.global_params, self.param_resolver
            )

            if not isinstance(res, tuple) or len(res) < 2:
                raise ValueError(
                    f"Unexpected return from energy module {module}: {res!r}"
                )

            total_energy += float(res[0])
            if len(res) >= 3 and res[2] is not None:
                g_tilt = res[2]
                for vidx, gvec in g_tilt.items():
                    row = index_map.get(int(vidx))
                    if row is not None:
                        tilt_grad_arr[row] += gvec

        return float(total_energy)

    def _compute_energy_array_with_leaflet_tilts(
        self,
        *,
        positions: np.ndarray,
        tilts_in: np.ndarray,
        tilts_out: np.ndarray,
        grad_dummy: np.ndarray | None = None,
    ) -> float:
        """Compute total energy for fixed positions and leaflet tilt arrays."""
        index_map = self.mesh.vertex_index_to_row
        if grad_dummy is None:
            grad_dummy = np.zeros_like(positions)
        else:
            grad_dummy.fill(0.0)
        total_energy = 0.0

        for module in self.energy_modules:
            if hasattr(module, "compute_energy_array"):
                try:
                    E_mod = module.compute_energy_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                    )
                except TypeError:
                    E_mod = module.compute_energy_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                    )
                total_energy += float(E_mod)
                continue

            if hasattr(module, "compute_energy_and_gradient_array"):
                try:
                    E_mod = module.compute_energy_and_gradient_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=grad_dummy,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                        tilt_in_grad_arr=None,
                        tilt_out_grad_arr=None,
                    )
                except TypeError:
                    E_mod = module.compute_energy_and_gradient_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=grad_dummy,
                    )
                total_energy += float(E_mod)
                continue

            try:
                E_mod, _ = module.compute_energy_and_gradient(
                    self.mesh,
                    self.global_params,
                    self.param_resolver,
                    compute_gradient=False,
                )
            except TypeError:
                E_mod, _ = module.compute_energy_and_gradient(
                    self.mesh, self.global_params, self.param_resolver
                )
            total_energy += float(E_mod)

        return float(total_energy)

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
        index_map = self.mesh.vertex_index_to_row
        if grad_dummy is None:
            grad_dummy = np.zeros_like(positions)
        else:
            grad_dummy.fill(0.0)
        total_energy = 0.0

        for name, module in zip(self.energy_module_names, self.energy_modules):
            if not getattr(module, "USES_TILT_LEAFLETS", False):
                continue

            # Fast path for pure tilt magnitude penalties.
            if name == "tilt_in" and tilt_vertex_areas_in is not None:
                k_tilt = float(self.param_resolver.get(None, "tilt_modulus_in") or 0.0)
                if k_tilt != 0.0:
                    sq = np.einsum("ij,ij->i", tilts_in, tilts_in)
                    total_energy += float(
                        0.5 * k_tilt * np.sum(sq * tilt_vertex_areas_in)
                    )
                continue
            if name == "tilt_out" and tilt_vertex_areas_out is not None:
                k_tilt = float(self.param_resolver.get(None, "tilt_modulus_out") or 0.0)
                if k_tilt != 0.0:
                    sq = np.einsum("ij,ij->i", tilts_out, tilts_out)
                    total_energy += float(
                        0.5 * k_tilt * np.sum(sq * tilt_vertex_areas_out)
                    )
                continue

            if hasattr(module, "compute_energy_array"):
                try:
                    E_mod = module.compute_energy_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                    )
                except TypeError:
                    # Some tilt modules ignore passed tilts and read from mesh.
                    E_mod = module.compute_energy_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                    )
                total_energy += float(E_mod)
                continue

            if hasattr(module, "compute_energy_and_gradient_array"):
                try:
                    E_mod = module.compute_energy_and_gradient_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=None,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                        tilt_in_grad_arr=None,
                        tilt_out_grad_arr=None,
                    )
                except TypeError:
                    # Some tilt modules ignore passed tilts and read from mesh.
                    E_mod = module.compute_energy_and_gradient_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=None,
                    )
                total_energy += float(E_mod)
                continue

            # Legacy dict modules are rare here; fall back to full energy.
            # (Inner-loop performance comes from the array modules.)
            E_full = self._compute_energy_array_with_leaflet_tilts(
                positions=positions,
                tilts_in=tilts_in,
                tilts_out=tilts_out,
                grad_dummy=grad_dummy,
            )
            return float(E_full)

        return float(total_energy)

    @staticmethod
    def _tilt_vertex_areas_from_triangles(
        *, n_vertices: int, tri_rows: np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """Return barycentric per-vertex areas based on triangle areas."""
        tri_pos = positions[tri_rows]
        v0 = tri_pos[:, 0, :]
        v1 = tri_pos[:, 1, :]
        v2 = tri_pos[:, 2, :]
        areas = 0.5 * np.linalg.norm(_fast_cross(v1 - v0, v2 - v0), axis=1)
        vertex_areas = np.zeros(n_vertices, dtype=float)
        a3 = areas / 3.0
        np.add.at(vertex_areas, tri_rows[:, 0], a3)
        np.add.at(vertex_areas, tri_rows[:, 1], a3)
        np.add.at(vertex_areas, tri_rows[:, 2], a3)
        return vertex_areas

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
                        0.5 * k_tilt * np.sum(sq * tilt_vertex_areas_in)
                    )
                    tilt_in_grad_arr += (
                        k_tilt * tilts_in * tilt_vertex_areas_in[:, None]
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
                        0.5 * k_tilt * np.sum(sq * tilt_vertex_areas_out)
                    )
                    tilt_out_grad_arr += (
                        k_tilt * tilts_out * tilt_vertex_areas_out[:, None]
                    )
                continue

            if hasattr(module, "compute_energy_and_gradient_array"):
                grad_arg = (
                    None
                    if tilt_only and getattr(module, "USES_TILT_LEAFLETS", False)
                    else grad_dummy
                )
                try:
                    E_mod = module.compute_energy_and_gradient_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=grad_arg,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                        tilt_in_grad_arr=tilt_in_grad_arr,
                        tilt_out_grad_arr=tilt_out_grad_arr,
                    )
                except TypeError:
                    E_mod = module.compute_energy_and_gradient_array(
                        self.mesh,
                        self.global_params,
                        self.param_resolver,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=grad_arg,
                    )
                total_energy += float(E_mod)
                continue

            res = module.compute_energy_and_gradient(
                self.mesh, self.global_params, self.param_resolver
            )
            if not isinstance(res, tuple) or len(res) < 2:
                raise ValueError(
                    f"Unexpected return from energy module {module}: {res!r}"
                )
            total_energy += float(res[0])

        return float(total_energy)

    def _relax_tilts(
        self,
        *,
        positions: np.ndarray,
        mode: str,
    ) -> None:
        """Relax vertex tilt vectors according to the configured solve mode."""
        mode_norm = str(mode or "").strip().lower()
        if mode_norm in ("", "none", "off", "false", "fixed"):
            return

        if mode_norm not in ("nested", "coupled"):
            logger.warning("Unknown tilt_solve_mode=%r; treating as 'fixed'.", mode)
            return

        step_size = float(self.global_params.get("tilt_step_size", 0.0) or 0.0)
        if step_size <= 0.0:
            return

        tol = float(self.global_params.get("tilt_tol", 0.0) or 0.0)
        if tol <= 0.0:
            tol = 0.0

        if mode_norm == "nested":
            n_inner = int(self.global_params.get("tilt_inner_steps", 0) or 0)
        else:
            n_inner = int(
                self.global_params.get(
                    "tilt_coupled_steps", self.global_params.get("tilt_inner_steps", 0)
                )
                or 0
            )
        if n_inner <= 0:
            return

        solver = (
            str(self.global_params.get("tilt_solver", "gd") or "gd").strip().lower()
        )
        if solver not in ("gd", "cg"):
            logger.warning("Unknown tilt_solver=%r; using gradient descent.", solver)
            solver = "gd"
        if solver == "cg":
            max_iters = int(self.global_params.get("tilt_cg_max_iters", n_inner) or 0)
            if max_iters <= 0:
                return
        else:
            max_iters = n_inner

        fixed_mask = self._tilt_fixed_mask()
        has_free = bool(np.any(~fixed_mask))
        if not has_free:
            return

        with self.mesh.geometry_freeze(positions):
            tilts = self.mesh.tilts_view().copy(order="F")
            normals = self.mesh.vertex_normals(positions)
            tilts = self._project_tilts_to_tangent_array(tilts, normals)
            tilt_fixed_vals = tilts[fixed_mask].copy() if np.any(fixed_mask) else None

            tilt_grad = np.zeros_like(tilts)
            preconditioner = None
            if solver == "cg":
                preconditioner = (
                    str(
                        self.global_params.get("tilt_cg_preconditioner", "jacobi")
                        or "jacobi"
                    )
                    .strip()
                    .lower()
                )
                if preconditioner in ("none", "off", "false"):
                    preconditioner = None

            if solver == "gd":
                for _ in range(max_iters):
                    E0 = self._compute_energy_and_tilt_gradient_array(
                        positions=positions, tilts=tilts, tilt_grad_arr=tilt_grad
                    )
                    if np.any(fixed_mask):
                        tilt_grad[fixed_mask] = 0.0

                    gnorm = float(np.linalg.norm(tilt_grad[~fixed_mask]))
                    if gnorm == 0.0:
                        break
                    if tol > 0.0 and gnorm < tol:
                        break

                    step = step_size
                    accepted = False
                    for _bt in range(12):
                        trial = tilts - step * tilt_grad
                        trial = self._project_tilts_to_tangent_array(trial, normals)
                        if tilt_fixed_vals is not None:
                            trial[fixed_mask] = tilt_fixed_vals
                        E1 = self._compute_energy_array_with_tilts(
                            positions=positions, tilts=trial
                        )
                        if E1 <= E0:
                            tilts = trial
                            accepted = True
                            break
                        step *= 0.5
                        if step < 1e-16:
                            break

                    if not accepted:
                        break
            else:
                M_inv = None
                if preconditioner == "jacobi":
                    M_inv = self._build_tilt_cg_preconditioner(
                        positions=positions,
                        index_map=self.mesh.vertex_index_to_row,
                        fixed_mask=fixed_mask,
                    )

                E0 = self._compute_energy_and_tilt_gradient_array(
                    positions=positions, tilts=tilts, tilt_grad_arr=tilt_grad
                )
                if np.any(fixed_mask):
                    tilt_grad[fixed_mask] = 0.0
                gnorm = float(np.linalg.norm(tilt_grad[~fixed_mask]))
                if gnorm == 0.0 or (tol > 0.0 and gnorm < tol):
                    self.mesh.set_tilts_from_array(tilts)
                    return

                residual = -tilt_grad
                if M_inv is not None:
                    z_vec = residual * M_inv[:, None]
                else:
                    z_vec = residual
                direction = z_vec.copy()
                rz_old = float(np.sum(residual * z_vec))

                for _ in range(max_iters):
                    if gnorm == 0.0:
                        break
                    if tol > 0.0 and gnorm < tol:
                        break

                    step = step_size
                    accepted = False
                    for _bt in range(12):
                        trial = tilts + step * direction
                        trial = self._project_tilts_to_tangent_array(trial, normals)
                        if tilt_fixed_vals is not None:
                            trial[fixed_mask] = tilt_fixed_vals
                        E1 = self._compute_energy_array_with_tilts(
                            positions=positions, tilts=trial
                        )
                        if E1 <= E0:
                            tilts = trial
                            E0 = E1
                            accepted = True
                            break
                        step *= 0.5
                        if step < 1e-16:
                            break

                    if not accepted:
                        break

                    E0 = self._compute_energy_and_tilt_gradient_array(
                        positions=positions, tilts=tilts, tilt_grad_arr=tilt_grad
                    )
                    if np.any(fixed_mask):
                        tilt_grad[fixed_mask] = 0.0

                    gnorm = float(np.linalg.norm(tilt_grad[~fixed_mask]))
                    if gnorm == 0.0 or (tol > 0.0 and gnorm < tol):
                        break

                    residual = -tilt_grad
                    if M_inv is not None:
                        z_vec = residual * M_inv[:, None]
                    else:
                        z_vec = residual
                    rz_new = float(np.sum(residual * z_vec))
                    if rz_old == 0.0:
                        break
                    beta = rz_new / rz_old
                    direction = z_vec + beta * direction
                    rz_old = rz_new

        self.mesh.set_tilts_from_array(tilts)

    def _build_tilt_cg_preconditioner(
        self,
        *,
        positions: np.ndarray,
        index_map: Dict[int, int],
        fixed_mask: np.ndarray,
    ) -> np.ndarray:
        """Return a Jacobi preconditioner for the single-tilt CG solve."""
        n_vertices = len(self.mesh.vertex_ids)
        diag = np.zeros(n_vertices, dtype=float)

        k_tilt = float(self.param_resolver.get(None, "tilt_rigidity") or 0.0)
        if k_tilt != 0.0:
            tri_rows, _ = self._triangle_rows()
            if tri_rows is not None and len(tri_rows) > 0:
                areas = self.energy_context().geometry.triangle_areas(
                    self.mesh, positions
                )
                if areas is not None and len(areas) == len(tri_rows):
                    vertex_areas = np.zeros(n_vertices, dtype=float)
                    area_thirds = areas / 3.0
                    np.add.at(vertex_areas, tri_rows[:, 0], area_thirds)
                    np.add.at(vertex_areas, tri_rows[:, 1], area_thirds)
                    np.add.at(vertex_areas, tri_rows[:, 2], area_thirds)
                    diag += k_tilt * vertex_areas

        k_smooth = float(
            self.param_resolver.get(None, "tilt_smoothness_rigidity") or 0.0
        )
        if k_smooth != 0.0:
            from geometry.curvature import compute_curvature_data

            _k_vecs, _areas, weights, tri_rows = compute_curvature_data(
                self.mesh, positions, index_map
            )
            if weights is not None and tri_rows is not None and len(tri_rows) > 0:
                c0 = weights[:, 0]
                c1 = weights[:, 1]
                c2 = weights[:, 2]
                factor = 0.5 * k_smooth
                np.add.at(diag, tri_rows[:, 0], factor * (c1 + c2))
                np.add.at(diag, tri_rows[:, 1], factor * (c2 + c0))
                np.add.at(diag, tri_rows[:, 2], factor * (c0 + c1))

        diag = np.where(diag > 1e-12, diag, 1.0)
        diag[fixed_mask] = 1.0
        return 1.0 / diag

    def _build_leaflet_tilt_cg_preconditioner(
        self,
        *,
        positions: np.ndarray,
        index_map: Dict[int, int],
        fixed_mask_in: np.ndarray,
        fixed_mask_out: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return Jacobi preconditioners for leaflet tilt CG solves."""
        n_vertices = len(self.mesh.vertex_ids)
        diag_in = np.zeros(n_vertices, dtype=float)
        diag_out = np.zeros(n_vertices, dtype=float)

        k_in = float(self.param_resolver.get(None, "tilt_modulus_in") or 0.0)
        k_out = float(self.param_resolver.get(None, "tilt_modulus_out") or 0.0)
        if k_in != 0.0 or k_out != 0.0:
            tri_rows, _ = self._triangle_rows()
            if tri_rows is not None and len(tri_rows) > 0:
                areas = self.energy_context().geometry.triangle_areas(
                    self.mesh, positions
                )
                if areas is not None and len(areas) == len(tri_rows):
                    vertex_areas = np.zeros(n_vertices, dtype=float)
                    area_thirds = areas / 3.0
                    np.add.at(vertex_areas, tri_rows[:, 0], area_thirds)
                    np.add.at(vertex_areas, tri_rows[:, 1], area_thirds)
                    np.add.at(vertex_areas, tri_rows[:, 2], area_thirds)
                    if k_in != 0.0:
                        diag_in += k_in * vertex_areas
                    if k_out != 0.0:
                        diag_out += k_out * vertex_areas

        k_smooth_in = float(
            self.param_resolver.get(None, "bending_modulus_in")
            or self.param_resolver.get(None, "bending_modulus")
            or 0.0
        )
        k_smooth_out = float(
            self.param_resolver.get(None, "bending_modulus_out")
            or self.param_resolver.get(None, "bending_modulus")
            or 0.0
        )
        if k_smooth_in != 0.0 or k_smooth_out != 0.0:
            from geometry.curvature import compute_curvature_data

            _k_vecs, _areas, weights, tri_rows = compute_curvature_data(
                self.mesh, positions, index_map
            )
            if weights is not None and tri_rows is not None and len(tri_rows) > 0:
                c0 = weights[:, 0]
                c1 = weights[:, 1]
                c2 = weights[:, 2]
                if k_smooth_in != 0.0:
                    factor_in = 0.5 * k_smooth_in
                    np.add.at(diag_in, tri_rows[:, 0], factor_in * (c1 + c2))
                    np.add.at(diag_in, tri_rows[:, 1], factor_in * (c2 + c0))
                    np.add.at(diag_in, tri_rows[:, 2], factor_in * (c0 + c1))
                if k_smooth_out != 0.0:
                    factor_out = 0.5 * k_smooth_out
                    np.add.at(diag_out, tri_rows[:, 0], factor_out * (c1 + c2))
                    np.add.at(diag_out, tri_rows[:, 1], factor_out * (c2 + c0))
                    np.add.at(diag_out, tri_rows[:, 2], factor_out * (c0 + c1))

        diag_in = np.where(diag_in > 1e-12, diag_in, 1.0)
        diag_out = np.where(diag_out > 1e-12, diag_out, 1.0)
        diag_in[fixed_mask_in] = 1.0
        diag_out[fixed_mask_out] = 1.0
        return 1.0 / diag_in, 1.0 / diag_out

    def _relax_leaflet_tilts(
        self,
        *,
        positions: np.ndarray,
        mode: str,
    ) -> None:
        """Relax inner/outer leaflet tilt vectors according to solve mode."""
        mode_norm = str(mode or "").strip().lower()
        if mode_norm in ("", "none", "off", "false", "fixed"):
            return

        if mode_norm not in ("nested", "coupled"):
            logger.warning("Unknown tilt_solve_mode=%r; treating as 'fixed'.", mode)
            return

        step_size = float(self.global_params.get("tilt_step_size", 0.0) or 0.0)
        if step_size <= 0.0:
            return

        tol = float(self.global_params.get("tilt_tol", 0.0) or 0.0)
        if tol <= 0.0:
            tol = 0.0

        if mode_norm == "nested":
            n_inner = int(self.global_params.get("tilt_inner_steps", 0) or 0)
        else:
            n_inner = int(
                self.global_params.get(
                    "tilt_coupled_steps", self.global_params.get("tilt_inner_steps", 0)
                )
                or 0
            )
        if n_inner <= 0:
            return

        solver = (
            str(self.global_params.get("tilt_solver", "gd") or "gd").strip().lower()
        )
        if solver not in ("gd", "cg"):
            logger.warning("Unknown tilt_solver=%r; using gradient descent.", solver)
            solver = "gd"
        if solver == "cg":
            max_iters = int(self.global_params.get("tilt_cg_max_iters", n_inner) or 0)
            if max_iters <= 0:
                return
        else:
            max_iters = n_inner

        fixed_mask_in = self._tilt_fixed_mask_in()
        fixed_mask_out = self._tilt_fixed_mask_out()
        has_free = bool(np.any(~fixed_mask_in) or np.any(~fixed_mask_out))
        if not has_free:
            return

        with self.mesh.geometry_freeze(positions):
            # Ensure tilt-only constraints (e.g. thetaB boundary projection) are
            # applied before the first tilt-gradient evaluation. Otherwise a
            # change in a scalar tilt parameter (like thetaB) can leave the tilt
            # field in a stale state (often all zeros), making the relaxation
            # no-op even though constraints require a non-zero boundary tilt.
            if hasattr(self.constraint_manager, "enforce_tilt_constraints"):
                self.constraint_manager.enforce_tilt_constraints(
                    self.mesh, global_params=self.global_params
                )

            tilts_in = self.mesh.tilts_in_view().copy(order="F")
            tilts_out = self.mesh.tilts_out_view().copy(order="F")
            normals = self.mesh.vertex_normals(positions)
            tilts_in = self._project_tilts_to_tangent_array(tilts_in, normals)
            tilts_out = self._project_tilts_to_tangent_array(tilts_out, normals)

            if bool(
                self.global_params.get("tilt_axisymmetric_about_thetaB_center", False)
            ):
                center = np.asarray(
                    self.global_params.get("tilt_thetaB_center") or [0.0, 0.0, 0.0],
                    dtype=float,
                ).reshape(3)
                axis = np.asarray(
                    self.global_params.get("tilt_thetaB_normal") or [0.0, 0.0, 1.0],
                    dtype=float,
                ).reshape(3)
                tilts_in = self._project_tilts_axisymmetric_about_center(
                    positions=positions,
                    tilts=tilts_in,
                    normals=normals,
                    center=center,
                    axis=axis,
                    fixed_mask=fixed_mask_in,
                )
                tilts_out = self._project_tilts_axisymmetric_about_center(
                    positions=positions,
                    tilts=tilts_out,
                    normals=normals,
                    center=center,
                    axis=axis,
                    fixed_mask=fixed_mask_out,
                )
            grad_dummy = np.zeros_like(positions)

            tri_rows, _ = self._triangle_rows()
            if tri_rows is None or len(tri_rows) == 0:
                return

            # Cache barycentric per-vertex areas once; positions are frozen for
            # the inner loop, so pure-tilt energies/gradients can reuse these.
            tilt_vertex_areas_in = self._tilt_vertex_areas_from_triangles(
                n_vertices=len(self.mesh.vertex_ids),
                tri_rows=tri_rows,
                positions=positions,
            )
            absent_mask_out = leaflet_absent_vertex_mask(
                self.mesh, self.global_params, leaflet="out"
            )
            tri_keep_out = leaflet_present_triangle_mask(
                self.mesh, tri_rows, absent_vertex_mask=absent_mask_out
            )
            tri_rows_out = tri_rows[tri_keep_out] if tri_keep_out.size else tri_rows
            tilt_vertex_areas_out = (
                np.zeros(len(self.mesh.vertex_ids), dtype=float)
                if tri_rows_out.size == 0
                else self._tilt_vertex_areas_from_triangles(
                    n_vertices=len(self.mesh.vertex_ids),
                    tri_rows=tri_rows_out,
                    positions=positions,
                )
            )
            grad_dummy = np.zeros_like(positions)

            tilt_fixed_vals_in = (
                tilts_in[fixed_mask_in].copy() if np.any(fixed_mask_in) else None
            )
            tilt_fixed_vals_out = (
                tilts_out[fixed_mask_out].copy() if np.any(fixed_mask_out) else None
            )

            tilt_in_grad = np.zeros_like(tilts_in)
            tilt_out_grad = np.zeros_like(tilts_out)

            def _leaflet_tilt_gradients() -> tuple[float, float]:
                E0 = self._compute_energy_and_leaflet_tilt_gradients_array(
                    positions=positions,
                    tilts_in=tilts_in,
                    tilts_out=tilts_out,
                    tilt_in_grad_arr=tilt_in_grad,
                    tilt_out_grad_arr=tilt_out_grad,
                    tilt_vertex_areas_in=tilt_vertex_areas_in,
                    tilt_vertex_areas_out=tilt_vertex_areas_out,
                    grad_dummy=grad_dummy,
                    tilt_only=True,
                )
                if hasattr(
                    self.constraint_manager, "apply_tilt_gradient_modifications_array"
                ):
                    self.constraint_manager.apply_tilt_gradient_modifications_array(
                        tilt_in_grad,
                        tilt_out_grad,
                        self.mesh,
                        self.global_params,
                        positions=positions,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                    )
                if np.any(fixed_mask_in):
                    tilt_in_grad[fixed_mask_in] = 0.0
                if np.any(fixed_mask_out):
                    tilt_out_grad[fixed_mask_out] = 0.0

                gnorm = float(
                    np.sqrt(
                        np.sum(tilt_in_grad[~fixed_mask_in] ** 2)
                        + np.sum(tilt_out_grad[~fixed_mask_out] ** 2)
                    )
                )
                return float(E0), gnorm

            preconditioner = None
            if solver == "cg":
                preconditioner = (
                    str(
                        self.global_params.get("tilt_cg_preconditioner", "jacobi")
                        or "jacobi"
                    )
                    .strip()
                    .lower()
                )
                if preconditioner in ("none", "off", "false"):
                    preconditioner = None

            if solver == "gd":
                for _ in range(max_iters):
                    E0, gnorm = _leaflet_tilt_gradients()
                    if gnorm == 0.0:
                        break
                    if tol > 0.0 and gnorm < tol:
                        break

                    step = step_size
                    accepted = False
                    for _bt in range(12):
                        trial_in = tilts_in - step * tilt_in_grad
                        trial_out = tilts_out - step * tilt_out_grad
                        trial_in = self._project_tilts_to_tangent_array(
                            trial_in, normals
                        )
                        trial_out = self._project_tilts_to_tangent_array(
                            trial_out, normals
                        )
                        if tilt_fixed_vals_in is not None:
                            trial_in[fixed_mask_in] = tilt_fixed_vals_in
                        if tilt_fixed_vals_out is not None:
                            trial_out[fixed_mask_out] = tilt_fixed_vals_out
                        E1 = self._compute_tilt_dependent_energy_with_leaflet_tilts(
                            positions=positions,
                            tilts_in=trial_in,
                            tilts_out=trial_out,
                            grad_dummy=grad_dummy,
                            tilt_vertex_areas_in=tilt_vertex_areas_in,
                            tilt_vertex_areas_out=tilt_vertex_areas_out,
                        )
                        if E1 <= E0:
                            tilts_in = trial_in
                            tilts_out = trial_out
                            accepted = True
                            break
                        step *= 0.5
                        if step < 1e-16:
                            break

                    if not accepted:
                        break

                    if hasattr(self.constraint_manager, "enforce_tilt_constraints"):
                        # Tilt constraints operate on the mesh state, so scatter the
                        # accepted tilt arrays, enforce, then re-load for continued
                        # relaxation steps.
                        self.mesh.set_tilts_in_from_array(tilts_in)
                        self.mesh.set_tilts_out_from_array(tilts_out)
                        self.constraint_manager.enforce_tilt_constraints(
                            self.mesh, global_params=self.global_params
                        )
                        tilts_in = self.mesh.tilts_in_view().copy(order="F")
                        tilts_out = self.mesh.tilts_out_view().copy(order="F")
                        tilts_in = self._project_tilts_to_tangent_array(
                            tilts_in, normals
                        )
                        tilts_out = self._project_tilts_to_tangent_array(
                            tilts_out, normals
                        )
                        if bool(
                            self.global_params.get(
                                "tilt_axisymmetric_about_thetaB_center", False
                            )
                        ):
                            tilts_in = self._project_tilts_axisymmetric_about_center(
                                positions=positions,
                                tilts=tilts_in,
                                normals=normals,
                                center=center,
                                axis=axis,
                                fixed_mask=fixed_mask_in,
                            )
                            tilts_out = self._project_tilts_axisymmetric_about_center(
                                positions=positions,
                                tilts=tilts_out,
                                normals=normals,
                                center=center,
                                axis=axis,
                                fixed_mask=fixed_mask_out,
                            )
            else:
                M_inv_in = None
                M_inv_out = None
                if preconditioner == "jacobi":
                    (
                        M_inv_in,
                        M_inv_out,
                    ) = self._build_leaflet_tilt_cg_preconditioner(
                        positions=positions,
                        index_map=self.mesh.vertex_index_to_row,
                        fixed_mask_in=fixed_mask_in,
                        fixed_mask_out=fixed_mask_out,
                    )

                E0, gnorm = _leaflet_tilt_gradients()
                if gnorm == 0.0 or (tol > 0.0 and gnorm < tol):
                    self.mesh.set_tilts_in_from_array(tilts_in)
                    self.mesh.set_tilts_out_from_array(tilts_out)
                    return

                res_in = -tilt_in_grad
                res_out = -tilt_out_grad
                if M_inv_in is not None:
                    z_in = res_in * M_inv_in[:, None]
                else:
                    z_in = res_in
                if M_inv_out is not None:
                    z_out = res_out * M_inv_out[:, None]
                else:
                    z_out = res_out

                dir_in = z_in.copy()
                dir_out = z_out.copy()
                rz_old = float(np.sum(res_in * z_in) + np.sum(res_out * z_out))

                for _ in range(max_iters):
                    if gnorm == 0.0:
                        break
                    if tol > 0.0 and gnorm < tol:
                        break

                    step = step_size
                    accepted = False
                    for _bt in range(12):
                        trial_in = tilts_in + step * dir_in
                        trial_out = tilts_out + step * dir_out
                        trial_in = self._project_tilts_to_tangent_array(
                            trial_in, normals
                        )
                        trial_out = self._project_tilts_to_tangent_array(
                            trial_out, normals
                        )
                        if tilt_fixed_vals_in is not None:
                            trial_in[fixed_mask_in] = tilt_fixed_vals_in
                        if tilt_fixed_vals_out is not None:
                            trial_out[fixed_mask_out] = tilt_fixed_vals_out
                        E1 = self._compute_tilt_dependent_energy_with_leaflet_tilts(
                            positions=positions,
                            tilts_in=trial_in,
                            tilts_out=trial_out,
                            grad_dummy=grad_dummy,
                            tilt_vertex_areas_in=tilt_vertex_areas_in,
                            tilt_vertex_areas_out=tilt_vertex_areas_out,
                        )
                        if E1 <= E0:
                            tilts_in = trial_in
                            tilts_out = trial_out
                            E0 = E1
                            accepted = True
                            break
                        step *= 0.5
                        if step < 1e-16:
                            break

                    if not accepted:
                        break

                    if hasattr(self.constraint_manager, "enforce_tilt_constraints"):
                        self.mesh.set_tilts_in_from_array(tilts_in)
                        self.mesh.set_tilts_out_from_array(tilts_out)
                        self.constraint_manager.enforce_tilt_constraints(
                            self.mesh, global_params=self.global_params
                        )
                        tilts_in = self.mesh.tilts_in_view().copy(order="F")
                        tilts_out = self.mesh.tilts_out_view().copy(order="F")
                        tilts_in = self._project_tilts_to_tangent_array(
                            tilts_in, normals
                        )
                        tilts_out = self._project_tilts_to_tangent_array(
                            tilts_out, normals
                        )
                        if bool(
                            self.global_params.get(
                                "tilt_axisymmetric_about_thetaB_center", False
                            )
                        ):
                            tilts_in = self._project_tilts_axisymmetric_about_center(
                                positions=positions,
                                tilts=tilts_in,
                                normals=normals,
                                center=center,
                                axis=axis,
                                fixed_mask=fixed_mask_in,
                            )
                            tilts_out = self._project_tilts_axisymmetric_about_center(
                                positions=positions,
                                tilts=tilts_out,
                                normals=normals,
                                center=center,
                                axis=axis,
                                fixed_mask=fixed_mask_out,
                            )

                    E0, gnorm = _leaflet_tilt_gradients()
                    if gnorm == 0.0 or (tol > 0.0 and gnorm < tol):
                        break

                    res_in = -tilt_in_grad
                    res_out = -tilt_out_grad
                    if M_inv_in is not None:
                        z_in = res_in * M_inv_in[:, None]
                    else:
                        z_in = res_in
                    if M_inv_out is not None:
                        z_out = res_out * M_inv_out[:, None]
                    else:
                        z_out = res_out
                    rz_new = float(np.sum(res_in * z_in) + np.sum(res_out * z_out))
                    if rz_old == 0.0:
                        break
                    beta = rz_new / rz_old
                    dir_in = z_in + beta * dir_in
                    dir_out = z_out + beta * dir_out
                    rz_old = rz_new

        self.mesh.set_tilts_in_from_array(tilts_in)
        self.mesh.set_tilts_out_from_array(tilts_out)
        if hasattr(self.constraint_manager, "enforce_tilt_constraints"):
            self.constraint_manager.enforce_tilt_constraints(
                self.mesh, global_params=self.global_params
            )

    def _check_gauss_bonnet(self) -> None:
        """Emit Gauss-Bonnet diagnostics if enabled."""
        if not bool(self.global_params.get("gauss_bonnet_monitor", False)):
            return

        if self._gauss_bonnet_monitor is None:
            eps_angle = float(self.global_params.get("gauss_bonnet_eps_angle", 1e-4))
            c1 = float(self.global_params.get("gauss_bonnet_c1", 1.0))
            c2 = float(self.global_params.get("gauss_bonnet_c2", 1.0))
            self._gauss_bonnet_monitor = GaussBonnetMonitor.from_mesh(
                self.mesh, eps_angle=eps_angle, c1=c1, c2=c2
            )

        report = self._gauss_bonnet_monitor.evaluate(self.mesh)
        if not report["ok"]:
            logger.warning(
                "Gauss-Bonnet drift exceeded tolerance: |ΔG|=%.3e (tol %.3e).",
                report["drift_G"],
                report["tol_G"],
            )

    def refresh_modules(self):
        """Re-load energy and constraint modules from the current mesh state."""
        # Refresh energy modules
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
            E_mod = module.compute_energy_and_gradient_array(
                self.mesh,
                self.global_params,
                self.param_resolver,
                positions=positions,
                index_map=index_map,
                grad_arr=grad_arr,
            )
            total_energy += E_mod

        # Apply constraint modifications to the gradient (e.g., Lagrange multipliers)
        if hasattr(self.constraint_manager, "apply_gradient_modifications_array"):
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

        # Optional DEBUG‑level diagnostic: in Lagrange mode the projected
        # gradient should be (numerically) tangent to each fixed‑volume
        # manifold, i.e. ⟨∇E, ∇V_body⟩ ≈ 0.
        if logger.isEnabledFor(logging.DEBUG):
            mode = self.global_params.get("volume_constraint_mode", "lagrange")
            if mode == "lagrange" and self.mesh.bodies:
                max_abs_dot = 0.0
                for body in self.mesh.bodies.values():
                    V_target = body.target_volume
                    if V_target is None:
                        V_target = body.options.get("target_volume")

                    if V_target is None:
                        continue

                    _, vol_grad = body.compute_volume_and_gradient(self.mesh)

                    dot = 0.0
                    for vidx, gVi in vol_grad.items():
                        gE = grad.get(vidx)
                        if gE is not None:
                            dot += float(np.dot(gE, gVi))
                    max_abs_dot = max(max_abs_dot, abs(dot))
                logger.debug(
                    "Lagrange tangency check: max |<∇E, ∇V>| = %.3e",
                    max_abs_dot,
                )

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
        positions, index_map, grad_dummy = self._soa_views()

        total_energy = 0.0
        for module in self.energy_modules:
            if hasattr(module, "compute_energy_and_gradient_array"):
                grad_dummy.fill(0.0)
                E_mod = module.compute_energy_and_gradient_array(
                    self.mesh,
                    self.global_params,
                    self.param_resolver,
                    positions=positions,
                    index_map=index_map,
                    grad_arr=grad_dummy,
                )
                total_energy += float(E_mod)
                continue

            E_mod, _ = module.compute_energy_and_gradient(
                self.mesh,
                self.global_params,
                self.param_resolver,
                compute_gradient=False,
            )
            total_energy += float(E_mod)
        return float(total_energy)

    def compute_energy_breakdown(self) -> Dict[str, float]:
        """Return per-module energy contributions for the current mesh."""
        positions, index_map, grad_dummy = self._soa_views()
        breakdown: Dict[str, float] = {}

        for name, module in zip(self.energy_module_names, self.energy_modules):
            if hasattr(module, "compute_energy_and_gradient_array"):
                grad_dummy.fill(0.0)
                E_mod = module.compute_energy_and_gradient_array(
                    self.mesh,
                    self.global_params,
                    self.param_resolver,
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
            breakdown[name] = float(E_mod)
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

    @staticmethod
    def _log_energy_phase(iteration: int, phase: str, energy: float) -> None:
        logger.debug("Iteration %d: %s energy=%.6f", iteration, phase, energy)

    @staticmethod
    def _log_step_direction_stats(iteration: int, grad_arr: np.ndarray) -> None:
        norm = float(np.linalg.norm(grad_arr))
        if norm <= 0.0:
            logger.debug("Iteration %d: grad_norm=0", iteration)
            return
        step_dir = -grad_arr / norm
        logger.debug(
            "Iteration %d: grad_norm=%.3e step_dir_norm=%.3e",
            iteration,
            norm,
            float(np.linalg.norm(step_dir)),
        )

    def _log_energy_consistency(self, label: str) -> None:
        if not logger.isEnabledFor(logging.DEBUG):
            return
        try:
            energy_scalar = float(self._run_debug_diagnostic(self.compute_energy))
            energy_array, _ = self._run_debug_diagnostic(
                self.compute_energy_and_gradient_array
            )
            energy_array = float(energy_array)
        except Exception as exc:
            logger.debug("Energy consistency check (%s) skipped: %s", label, exc)
            return

        logger.debug(
            "Energy consistency (%s): scalar=%.6f array=%.6f",
            label,
            energy_scalar,
            energy_array,
        )

        diff = abs(energy_scalar - energy_array)
        tol = 1e-8 * max(1.0, abs(energy_scalar), abs(energy_array))
        if diff <= tol:
            return

        try:
            breakdown = self._diagnostic_energy_breakdown()
        except Exception as exc:
            logger.debug("Energy breakdown failed during consistency check: %s", exc)
            return

        top_terms = sorted(
            breakdown.items(), key=lambda item: abs(item[1]), reverse=True
        )[:5]
        summary = ", ".join(f"{name}={value:.6f}" for name, value in top_terms)
        logger.warning(
            "Energy consistency mismatch (%s): |Δ|=%.6e (scalar=%.6f array=%.6f). "
            "Top terms: %s",
            label,
            diff,
            energy_scalar,
            energy_array,
            summary,
        )

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

    def _optimize_thetaB_scalar(self, *, tilt_mode: str, iteration: int) -> None:
        """Optionally optimize the scalar thetaB by sampling reduced energies.

        This treats thetaB as a global scalar degree of freedom and updates it
        by comparing the total energy after a partial tilt relaxation for a few
        candidate thetaB values.
        """
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

        def eval_candidate(thetaB_val: float) -> tuple[float, np.ndarray, np.ndarray]:
            self.global_params.set("tilt_thetaB_value", float(thetaB_val))
            self.mesh.set_tilts_in_from_array(base_tin)
            self.mesh.set_tilts_out_from_array(base_tout)
            # Relax tilts only; shape is handled by the main loop.
            self._relax_leaflet_tilts(
                positions=self.mesh.positions_view(), mode=tilt_mode
            )
            e = float(self.compute_energy())
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
        # Always restore the chosen candidate state (including "no change").
        self.global_params.set("tilt_thetaB_value", float(best_thetaB))
        self.mesh.set_tilts_in_from_array(best_tin)
        self.mesh.set_tilts_out_from_array(best_tout)

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "thetaB optimize: i=%d thetaB %.6g -> %.6g (E %.6g -> %.6g)",
                int(iteration),
                base_thetaB,
                best_thetaB,
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
            self._log_energy_consistency("no_steps")
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

        self._check_gauss_bonnet()
        last_grad_arr = None
        last_state_energy: float | None = None
        for i in range(n_steps):
            if callback:
                callback(self.mesh, i)

            self._update_scalar_params()
            self._log_debug_energy_context(i)

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
                    self._relax_leaflet_tilts(
                        positions=self.mesh.positions_view(), mode=tilt_mode
                    )
                    post_energy = float(self.compute_energy())
                    threshold = max(guard_min, pre_energy * guard_factor)
                    if post_energy > threshold:
                        self.mesh.set_tilts_in_from_array(pre_tin)
                        self.mesh.set_tilts_out_from_array(pre_tout)
                        if logger.isEnabledFor(logging.WARNING):
                            logger.warning(
                                "Tilt relaxation energy spike: %.6g -> %.6g (threshold %.6g). "
                                "Rolling back tilts for iteration %d.",
                                pre_energy,
                                post_energy,
                                threshold,
                                i,
                            )
                    else:
                        self.mesh.project_tilts_to_tangent()
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
            last_grad_arr = grad_arr

            if logger.isEnabledFor(logging.DEBUG):
                self._log_energy_phase(i, "pre_step", E)
                self._log_step_direction_stats(i, grad_arr)

            # check convergence by gradient norm
            grad_norm = float(np.linalg.norm(grad_arr))
            if grad_norm < self.tol:
                logger.debug("Converged: gradient norm below tolerance.")
                logger.info(f"Converged in {i} iterations; |∇E|={grad_norm:.3e}")
                self._log_energy_consistency("converged")
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
                step_success, self.step_size, accepted_energy = self.stepper.step(
                    self.mesh,
                    grad_arr,
                    step_size_in,
                    energy_fn,
                    constraint_enforcer=self._enforce_constraints
                    if self._has_enforceable_constraints
                    else None,
                )
            finally:
                if hasattr(self.mesh, "_line_search_reduced_energy"):
                    delattr(self.mesh, "_line_search_reduced_energy")
                if hasattr(self.mesh, "_line_search_reduced_accept_rule"):
                    delattr(self.mesh, "_line_search_reduced_accept_rule")
            last_state_energy = float(accepted_energy)
            if not self.quiet:
                # Compute total area only when needed for diagnostics.
                total_area = self.mesh.compute_total_surface_area()
                print(
                    f"Step {i:4d}: Area = {total_area:.5f}, Energy = {last_state_energy:.5f}, Step Size  = {step_size_in:.2e}"
                )
            if logger.isEnabledFor(logging.DEBUG):
                # Do not probe energy here: additional debug-only energy
                # evaluations can mutate caches and perturb the optimization
                # trajectory.
                self._log_energy_phase(i, "post_step", float(last_state_energy))
            # Keep any stored 3D tilt field tangent to the updated surface.
            self.mesh.project_tilts_to_tangent()
            if logger.isEnabledFor(logging.DEBUG):
                # Keep this phase marker without re-evaluating energy.
                self._log_energy_phase(i, "post_step_tilt_project", float("nan"))
            if step_mode == "fixed":
                # Keep the cross-iteration step size constant, but still allow
                # the line search to backtrack within each iteration for
                # stability.
                self.step_size = fixed_step

            self._check_gauss_bonnet()
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
                        self._log_energy_consistency("terminated_early")
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

                if logger.isEnabledFor(logging.DEBUG):
                    E_after = self.compute_energy()
                    accepted = (
                        float(last_state_energy)
                        if last_state_energy is not None
                        else float("nan")
                    )
                    diff_acc = abs(float(E_after) - accepted)
                    tol_acc = 1e-8 * max(1.0, abs(float(E_after)), abs(accepted))
                    max_rel_violation_dbg = 0.0
                    vol_msgs: list[str] = []
                    if self.mesh.bodies:
                        for body in self.mesh.bodies.values():
                            target = body.target_volume
                            if target is None:
                                target = body.options.get("target_volume")
                            if target is None:
                                continue
                            current = body.compute_volume(self.mesh)
                            denom = max(abs(target), 1.0)
                            rel = (current - target) / denom
                            max_rel_violation_dbg = max(max_rel_violation_dbg, abs(rel))
                            vol_msgs.append(
                                "body %d: V=%.6f, V0=%.6f, relΔV=%.3e"
                                % (body.index, current, target, rel)
                            )
                    logger.debug(
                        "Accepted step %d: E_before=%.6f, E_after=%.6f, "
                        "E_accepted=%.6f, |E_after-E_accepted|=%.3e (tol %.3e), "
                        "step_size=%.3e, max_relΔV=%.3e",
                        i,
                        E,
                        E_after,
                        accepted,
                        diff_acc,
                        tol_acc,
                        self.step_size,
                        max_rel_violation_dbg,
                    )
                    if vol_msgs:
                        logger.debug("Volume diagnostics: %s", "; ".join(vol_msgs))
                    if diff_acc > tol_acc:
                        try:
                            breakdown = self.compute_energy_breakdown()
                            top_terms = sorted(
                                breakdown.items(),
                                key=lambda item: abs(item[1]),
                                reverse=True,
                            )[:5]
                            summary = ", ".join(
                                f"{name}={value:.6f}" for name, value in top_terms
                            )
                            logger.warning(
                                "Accepted energy mismatch at step %d: "
                                "E_accepted=%.6f E_after=%.6f |Δ|=%.6e. Top terms: %s",
                                i,
                                accepted,
                                float(E_after),
                                diff_acc,
                                summary,
                            )
                        except Exception as exc:
                            logger.debug(
                                "Accepted energy breakdown failed at step %d: %s",
                                i,
                                exc,
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
                        if logger.isEnabledFor(logging.DEBUG):
                            # Avoid debug-only energy probes here as well.
                            self._log_energy_phase(
                                i, "post_step_constraints", float("nan")
                            )
                        reset = getattr(self.stepper, "reset", None)
                        if callable(reset):
                            reset()

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

        self._log_energy_consistency("finalize")
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

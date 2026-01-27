# runtime/minimizer.py

import logging
from typing import Callable, Dict, List, Optional

import numpy as np

from core.parameters.global_parameters import GlobalParameters
from core.parameters.resolver import ParameterResolver
from geometry.entities import Mesh
from runtime.constraint_manager import ConstraintModuleManager
from runtime.diagnostics.gauss_bonnet import GaussBonnetMonitor
from runtime.energy_manager import EnergyModuleManager
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

    def _tilt_fixed_mask(self) -> np.ndarray:
        """Return a boolean mask for vertices whose tilt is clamped.

        The mask is in ``mesh.vertex_ids`` row order.
        """
        return np.array(
            [
                bool(getattr(self.mesh.vertices[int(vid)], "tilt_fixed", False))
                for vid in self.mesh.vertex_ids
            ],
            dtype=bool,
        )

    def _tilt_fixed_mask_in(self) -> np.ndarray:
        """Return a boolean mask for vertices whose inner-leaflet tilt is clamped."""
        return np.array(
            [
                bool(getattr(self.mesh.vertices[int(vid)], "tilt_fixed_in", False))
                for vid in self.mesh.vertex_ids
            ],
            dtype=bool,
        )

    def _tilt_fixed_mask_out(self) -> np.ndarray:
        """Return a boolean mask for vertices whose outer-leaflet tilt is clamped."""
        return np.array(
            [
                bool(getattr(self.mesh.vertices[int(vid)], "tilt_fixed_out", False))
                for vid in self.mesh.vertex_ids
            ],
            dtype=bool,
        )

    @staticmethod
    def _project_tilts_to_tangent_array(
        tilts: np.ndarray, normals: np.ndarray
    ) -> np.ndarray:
        """Project a dense tilt array into the vertex tangent planes."""
        dot = np.einsum("ij,ij->i", tilts, normals)
        return tilts - dot[:, None] * normals

    def _uses_leaflet_tilts(self) -> bool:
        """Return True when any loaded module depends on tilt_in/tilt_out."""
        return any(
            getattr(module, "USES_TILT_LEAFLETS", False)
            for module in self.energy_modules
        )

    def _compute_energy_array_with_tilts(
        self,
        *,
        positions: np.ndarray,
        tilts: np.ndarray,
    ) -> float:
        """Compute total energy for a fixed ``positions``/``tilts`` state.

        Uses the array API when available and passes ``tilts`` opportunistically
        (falling back when a module does not accept tilt arguments).
        """
        index_map = self.mesh.vertex_index_to_row
        grad_dummy = np.zeros_like(positions)
        total_energy = 0.0

        for module in self.energy_modules:
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
        """Compute total energy and accumulate dense tilt gradient."""
        index_map = self.mesh.vertex_index_to_row
        grad_dummy = np.zeros_like(positions)
        tilt_grad_arr.fill(0.0)
        total_energy = 0.0

        for module in self.energy_modules:
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
    ) -> float:
        """Compute total energy for fixed positions and leaflet tilt arrays."""
        index_map = self.mesh.vertex_index_to_row
        grad_dummy = np.zeros_like(positions)
        total_energy = 0.0

        for module in self.energy_modules:
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

    def _compute_energy_and_leaflet_tilt_gradients_array(
        self,
        *,
        positions: np.ndarray,
        tilts_in: np.ndarray,
        tilts_out: np.ndarray,
        tilt_in_grad_arr: np.ndarray,
        tilt_out_grad_arr: np.ndarray,
    ) -> float:
        """Compute total energy and accumulate leaflet tilt gradients."""
        index_map = self.mesh.vertex_index_to_row
        grad_dummy = np.zeros_like(positions)
        tilt_in_grad_arr.fill(0.0)
        tilt_out_grad_arr.fill(0.0)
        total_energy = 0.0

        for module in self.energy_modules:
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
                        grad_arr=grad_dummy,
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

        fixed_mask = self._tilt_fixed_mask()
        has_free = bool(np.any(~fixed_mask))
        if not has_free:
            return

        tilts = self.mesh.tilts_view().copy(order="F")
        normals = self.mesh.vertex_normals(positions)
        tilts = self._project_tilts_to_tangent_array(tilts, normals)
        tilt_fixed_vals = tilts[fixed_mask].copy() if np.any(fixed_mask) else None

        tilt_grad = np.zeros_like(tilts)
        for _ in range(n_inner):
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

        self.mesh.set_tilts_from_array(tilts)

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

        fixed_mask_in = self._tilt_fixed_mask_in()
        fixed_mask_out = self._tilt_fixed_mask_out()
        has_free = bool(np.any(~fixed_mask_in) or np.any(~fixed_mask_out))
        if not has_free:
            return

        tilts_in = self.mesh.tilts_in_view().copy(order="F")
        tilts_out = self.mesh.tilts_out_view().copy(order="F")
        normals = self.mesh.vertex_normals(positions)
        tilts_in = self._project_tilts_to_tangent_array(tilts_in, normals)
        tilts_out = self._project_tilts_to_tangent_array(tilts_out, normals)

        tilt_fixed_vals_in = (
            tilts_in[fixed_mask_in].copy() if np.any(fixed_mask_in) else None
        )
        tilt_fixed_vals_out = (
            tilts_out[fixed_mask_out].copy() if np.any(fixed_mask_out) else None
        )

        tilt_in_grad = np.zeros_like(tilts_in)
        tilt_out_grad = np.zeros_like(tilts_out)

        for _ in range(n_inner):
            E0 = self._compute_energy_and_leaflet_tilt_gradients_array(
                positions=positions,
                tilts_in=tilts_in,
                tilts_out=tilts_out,
                tilt_in_grad_arr=tilt_in_grad,
                tilt_out_grad_arr=tilt_out_grad,
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
            if gnorm == 0.0:
                break
            if tol > 0.0 and gnorm < tol:
                break

            step = step_size
            accepted = False
            for _bt in range(12):
                trial_in = tilts_in - step * tilt_in_grad
                trial_out = tilts_out - step * tilt_out_grad
                trial_in = self._project_tilts_to_tangent_array(trial_in, normals)
                trial_out = self._project_tilts_to_tangent_array(trial_out, normals)
                if tilt_fixed_vals_in is not None:
                    trial_in[fixed_mask_in] = tilt_fixed_vals_in
                if tilt_fixed_vals_out is not None:
                    trial_out[fixed_mask_out] = tilt_fixed_vals_out
                E1 = self._compute_energy_array_with_leaflet_tilts(
                    positions=positions, tilts_in=trial_in, tilts_out=trial_out
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

        self.mesh.set_tilts_in_from_array(tilts_in)
        self.mesh.set_tilts_out_from_array(tilts_out)

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
        positions = self.mesh.positions_view()
        vertex_ids = self.mesh.vertex_ids
        index_map = self.mesh.vertex_index_to_row
        grad_arr = np.zeros_like(positions)

        total_energy = 0.0
        for module in self.energy_modules:
            if hasattr(module, "compute_energy_and_gradient_array"):
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
                continue

            # Legacy path: compute dict and scatter
            E_mod, g_mod = module.compute_energy_and_gradient(
                self.mesh, self.global_params, self.param_resolver
            )
            total_energy += E_mod
            for vidx, gvec in g_mod.items():
                row = index_map.get(vidx)
                if row is not None:
                    grad_arr[row] += gvec

        # Apply constraint modifications to the gradient (e.g., Lagrange multipliers)
        if hasattr(self.constraint_manager, "apply_gradient_modifications_array"):
            self.constraint_manager.apply_gradient_modifications_array(
                grad_arr, self.mesh, self.global_params
            )

        # Always zero gradients for fixed vertices in the array pipeline.
        for row, vidx in enumerate(vertex_ids):
            if getattr(self.mesh.vertices[int(vidx)], "fixed", False):
                grad_arr[row] = 0.0

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
        positions = self.mesh.positions_view()
        index_map = self.mesh.vertex_index_to_row
        grad_dummy = np.zeros_like(positions)

        total_energy = 0.0
        for module in self.energy_modules:
            if hasattr(module, "compute_energy_and_gradient_array"):
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
        positions = self.mesh.positions_view()
        index_map = self.mesh.vertex_index_to_row
        breakdown: Dict[str, float] = {}

        for name, module in zip(self.energy_module_names, self.energy_modules):
            if hasattr(module, "compute_energy_and_gradient_array"):
                grad_dummy = np.zeros_like(positions)
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
        for module in self.energy_modules:
            if hasattr(module, "update_scalar_params"):
                try:
                    module.update_scalar_params(
                        self.mesh, self.global_params, self.param_resolver
                    )
                except TypeError:
                    module.update_scalar_params(self.mesh, self.global_params)

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

    def minimize(
        self, n_steps: int = 1, callback: Optional[Callable[["Mesh", int], None]] = None
    ):
        """Run the optimization loop for ``n_steps`` iterations."""
        zero_step_counter = 0
        step_success = True

        if n_steps <= 0:
            E, grad = self.compute_energy_and_gradient()
            self._enforce_constraints()
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
        for i in range(n_steps):
            if callback:
                callback(self.mesh, i)

            self._update_scalar_params()

            # Tilt solve modes are evaluated before the shape convergence check so
            # that fixed-geometry runs can still relax the tilt field.
            tilt_mode = self.global_params.get("tilt_solve_mode", "fixed")
            if self._uses_leaflet_tilts():
                self._relax_leaflet_tilts(
                    positions=self.mesh.positions_view(), mode=tilt_mode
                )
            else:
                self._relax_tilts(positions=self.mesh.positions_view(), mode=tilt_mode)

            self._update_scalar_params()

            # Use array-based path for main loop
            E, grad_arr = self.compute_energy_and_gradient_array()
            self.project_constraints_array(grad_arr)
            last_grad_arr = grad_arr

            # check convergence by gradient norm
            grad_norm = float(np.linalg.norm(grad_arr))
            if grad_norm < self.tol:
                logger.debug("Converged: gradient norm below tolerance.")
                logger.info(f"Converged in {i} iterations; |∇E|={grad_norm:.3e}")
                return {
                    "energy": E,
                    "gradient": self._grad_arr_to_dict(grad_arr),
                    "mesh": self.mesh,
                    "step_success": True,
                    "iterations": i + 1,
                    "terminated_early": True,
                }
            logger.debug("Iteration %d: |∇E|=%.3e", i, grad_norm)

            if not self.quiet:
                # Compute total area only when needed for diagnostics
                total_area = self.mesh.compute_total_surface_area()
                print(
                    f"Step {i:4d}: Area = {total_area:.5f}, Energy = {E:.5f}, Step Size  = {self.step_size:.2e}"
                )

            step_mode = str(
                self.global_params.get("step_size_mode", "adaptive") or "adaptive"
            ).lower()
            fixed_step = float(
                self.global_params.get("step_size", self.step_size) or self.step_size
            )
            step_size_in = fixed_step if step_mode == "fixed" else self.step_size

            step_success, self.step_size = self.stepper.step(
                self.mesh,
                grad_arr,
                step_size_in,
                self.compute_energy,
                constraint_enforcer=self._enforce_constraints
                if self._has_enforceable_constraints
                else None,
            )
            # Keep any stored 3D tilt field tangent to the updated surface.
            self.mesh.project_tilts_to_tangent()
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
                        return {
                            "energy": E,
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
                        "step_size=%.3e, max_relΔV=%.3e",
                        i,
                        E,
                        E_after,
                        self.step_size,
                        max_rel_violation_dbg,
                    )
                    if vol_msgs:
                        logger.debug("Volume diagnostics: %s", "; ".join(vol_msgs))

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

        return {
            "energy": E,
            "gradient": self._grad_arr_to_dict(last_grad_arr)
            if last_grad_arr is not None
            else {},
            "mesh": self.mesh,
            "step_success": step_success,
            "iterations": n_steps,
            "terminated_early": False,
        }

"""Energy and gradient evaluation orchestration for minimization."""

from __future__ import annotations

import inspect
import logging
from typing import TYPE_CHECKING, Any, Callable, Dict

import numpy as np

if TYPE_CHECKING:
    from core.parameters.global_parameters import GlobalParameters
    from core.parameters.resolver import ParameterResolver
    from geometry.entities import Mesh
    from runtime.energy_context import EnergyContext

logger = logging.getLogger("membrane_solver")


class EvaluationManager:
    """Orchestrates multi-module energy and gradient evaluations."""

    def __init__(
        self,
        *,
        mesh: Mesh,
        global_params: GlobalParameters,
        param_resolver: ParameterResolver,
        energy_modules: list[Any],
        energy_module_names: list[str],
        energy_context_fn: Callable[[], EnergyContext],
        experimental_energy_scale_fn: Callable[[str], float],
    ):
        self.mesh = mesh
        self.global_params = global_params
        self.param_resolver = param_resolver
        self.energy_modules = energy_modules
        self.energy_module_names = energy_module_names
        self.energy_context_fn = energy_context_fn
        self.experimental_energy_scale_fn = experimental_energy_scale_fn

        # Cache for module API specs
        self._fn_specs: Dict[int, Any] = {}

    def _get_fn_spec(self, fn: Callable) -> Any:
        func = getattr(fn, "__func__", fn)
        key = id(func)
        spec = self._fn_specs.get(key)
        if spec is None:
            try:
                sig = inspect.signature(fn)
                params = sig.parameters
                accepts_resolver = "param_resolver" in params or "resolver" in params
                accepts_ctx = "ctx" in params or any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
                # Determine which kwargs the function explicitly accepts
                has_var_kwargs = any(
                    p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
                accepted_kwargs = (
                    frozenset(params.keys()) if not has_var_kwargs else None
                )
                spec = {
                    "sig": sig,
                    "accepts_resolver": accepts_resolver,
                    "accepts_ctx": accepts_ctx,
                    "accepted_kwargs": accepted_kwargs,
                    "has_var_kwargs": has_var_kwargs,
                }
            except (TypeError, ValueError):
                spec = {
                    "sig": None,
                    "accepts_resolver": True,
                    "accepts_ctx": False,
                    "accepted_kwargs": None,
                    "has_var_kwargs": True,
                }
            self._fn_specs[key] = spec
        return spec

    def _coerce_energy_value(self, energy_value: Any) -> float:
        energy_arr = np.asarray(energy_value, dtype=float)
        if energy_arr.ndim == 0:
            return float(energy_arr)
        return float(np.sum(energy_arr))

    def _call_fn(self, fn: Callable, **kwargs) -> Any:
        """Call module function with smart argument matching."""
        spec = self._get_fn_spec(fn)

        call_kwargs = dict(kwargs)
        if (
            spec["accepts_resolver"]
            and "param_resolver" not in call_kwargs
            and "resolver" not in call_kwargs
        ):
            # Match naming convention in the signature
            if "resolver" in spec["sig"].parameters if spec["sig"] else False:
                call_kwargs["resolver"] = self.param_resolver
            else:
                call_kwargs["param_resolver"] = self.param_resolver

        if spec["accepts_ctx"] and "ctx" not in call_kwargs:
            call_kwargs["ctx"] = self.energy_context_fn()

        # If the function doesn't take **kwargs, filter out unsupported arguments
        if spec["accepted_kwargs"] is not None:
            call_kwargs = {
                k: v for k, v in call_kwargs.items() if k in spec["accepted_kwargs"]
            }

        # Check if mesh/global_params are accepted positionally or as keywords
        if spec["sig"]:
            params = list(spec["sig"].parameters.values())
            # Project standard: first two are (mesh, global_params)
            args = []
            if len(params) > 0 and params[0].name not in call_kwargs:
                args.append(self.mesh)
            if len(params) > 1 and params[1].name not in call_kwargs:
                args.append(self.global_params)
            return fn(*args, **call_kwargs)
        else:
            return fn(self.mesh, self.global_params, **call_kwargs)

    def _call_module_array(self, module: Any, **kwargs) -> Any:
        fn = getattr(module, "compute_energy_and_gradient_array")
        return self._call_fn(fn, **kwargs)

    def _call_module_energy_array(self, module: Any, **kwargs) -> Any:
        fn = getattr(module, "compute_energy_array")
        return self._call_fn(fn, **kwargs)

    def compute_energy_array_total(self, *, positions: np.ndarray) -> float:
        """Compute total energy for fixed positions and current mesh tilts."""
        index_map = self.mesh.vertex_index_to_row
        grad_dummy = self.energy_context_fn().scratch_array(
            "energy_only_grad_dummy", shape=positions.shape, dtype=positions.dtype
        )
        total_energy = 0.0

        for name, module in zip(self.energy_module_names, self.energy_modules):
            scale = self.experimental_energy_scale_fn(str(name))
            if hasattr(module, "compute_energy_array"):
                E_mod = self._call_module_energy_array(
                    module, positions=positions, index_map=index_map
                )
                total_energy += float(scale) * self._coerce_energy_value(E_mod)
                continue

            if hasattr(module, "compute_energy_and_gradient_array"):
                grad_dummy.fill(0.0)
                E_mod = self._call_module_array(
                    module,
                    positions=positions,
                    index_map=index_map,
                    grad_arr=grad_dummy,
                )
                total_energy += float(scale) * self._coerce_energy_value(E_mod)
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
            total_energy += float(scale) * self._coerce_energy_value(E_mod)

        return float(total_energy)

    def compute_total_energy_array_with_tilts(
        self,
        *,
        positions: np.ndarray,
        tilts: np.ndarray,
    ) -> float:
        """Compute total energy for fixed positions and projected tilt field."""
        index_map = self.mesh.vertex_index_to_row
        grad_dummy = self.energy_context_fn().scratch_array(
            "energy_only_tilt_grad_dummy", shape=positions.shape, dtype=positions.dtype
        )
        total_energy = 0.0

        for name, module in zip(self.energy_module_names, self.energy_modules):
            scale = self.experimental_energy_scale_fn(str(name))
            if hasattr(module, "compute_energy_array"):
                kwargs = {"positions": positions, "index_map": index_map}
                if getattr(module, "USES_TILT", False):
                    kwargs["tilts"] = tilts
                E_mod = self._call_module_energy_array(module, **kwargs)
                total_energy += float(scale) * self._coerce_energy_value(E_mod)
                continue

            if hasattr(module, "compute_energy_and_gradient_array"):
                grad_dummy.fill(0.0)
                if getattr(module, "USES_TILT", False):
                    try:
                        E_mod = self._call_module_array(
                            module,
                            positions=positions,
                            index_map=index_map,
                            grad_arr=grad_dummy,
                            tilts=tilts,
                            tilt_grad_arr=None,
                        )
                    except TypeError:
                        try:
                            E_mod = self._call_module_array(
                                module,
                                positions=positions,
                                index_map=index_map,
                                grad_arr=grad_dummy,
                                tilts=tilts,
                            )
                        except TypeError:
                            E_mod = self._call_module_array(
                                module,
                                positions=positions,
                                index_map=index_map,
                                grad_arr=grad_dummy,
                            )
                else:
                    E_mod = self._call_module_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=grad_dummy,
                    )
                total_energy += float(scale) * self._coerce_energy_value(E_mod)
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
            total_energy += float(scale) * self._coerce_energy_value(E_mod)

        return float(total_energy)

    def compute_energy_array_with_tilts(
        self,
        *,
        positions: np.ndarray,
        tilts: np.ndarray,
    ) -> float:
        """Compute tilt-dependent energy for fixed positions/tilts."""
        index_map = self.mesh.vertex_index_to_row
        grad_dummy = np.zeros_like(positions)
        total_energy = 0.0

        module_names = self.energy_module_names
        if len(module_names) != len(self.energy_modules):
            module_names = [
                getattr(module, "__name__", module.__class__.__name__)
                for module in self.energy_modules
            ]

        for name, module in zip(module_names, self.energy_modules):
            scale = self.experimental_energy_scale_fn(str(name))
            if not getattr(module, "USES_TILT", False):
                continue
            if hasattr(module, "compute_energy_array"):
                try:
                    E_mod = self._call_module_energy_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                        tilts=tilts,
                    )
                except TypeError:
                    E_mod = self._call_module_energy_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                    )
                total_energy += float(scale) * float(E_mod)
                continue

            if hasattr(module, "compute_energy_and_gradient_array"):
                try:
                    E_mod = self._call_module_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=grad_dummy,
                        tilts=tilts,
                        tilt_grad_arr=None,
                    )
                except TypeError:
                    try:
                        E_mod = self._call_module_array(
                            module,
                            positions=positions,
                            index_map=index_map,
                            grad_arr=grad_dummy,
                            tilts=tilts,
                        )
                    except TypeError:
                        E_mod = self._call_module_array(
                            module,
                            positions=positions,
                            index_map=index_map,
                            grad_arr=grad_dummy,
                        )
                total_energy += float(scale) * float(E_mod)
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
            total_energy += float(scale) * float(E_mod)

        return float(total_energy)

    def compute_energy_and_tilt_gradient_array(
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

        module_names = self.energy_module_names
        if len(module_names) != len(self.energy_modules):
            module_names = [
                getattr(module, "__name__", module.__class__.__name__)
                for module in self.energy_modules
            ]

        for name, module in zip(module_names, self.energy_modules):
            scale = self.experimental_energy_scale_fn(str(name))
            if not getattr(module, "USES_TILT", False):
                continue
            if hasattr(module, "compute_energy_and_gradient_array"):
                grad_before = None
                if abs(float(scale) - 1.0) > 1.0e-15:
                    grad_before = tilt_grad_arr.copy()

                try:
                    E_mod = self._call_module_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=grad_dummy,
                        tilts=tilts,
                        tilt_grad_arr=tilt_grad_arr,
                    )
                except TypeError:
                    try:
                        E_mod = self._call_module_array(
                            module,
                            positions=positions,
                            index_map=index_map,
                            grad_arr=grad_dummy,
                            tilts=tilts,
                        )
                    except TypeError:
                        E_mod = self._call_module_array(
                            module,
                            positions=positions,
                            index_map=index_map,
                            grad_arr=grad_dummy,
                        )

                if grad_before is not None:
                    grad_delta = tilt_grad_arr - grad_before
                    tilt_grad_arr[:] = grad_before + (float(scale) * grad_delta)
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
            if len(res) >= 3 and res[2] is not None:
                g_tilt = res[2]
                for vidx, gvec in g_tilt.items():
                    row = index_map.get(int(vidx))
                    if row is not None:
                        tilt_grad_arr[row] += float(scale) * gvec

        return float(total_energy)

    def compute_energy_array_with_leaflet_tilts(
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

        for name, module in zip(self.energy_module_names, self.energy_modules):
            scale = self.experimental_energy_scale_fn(str(name))
            if hasattr(module, "compute_energy_array"):
                try:
                    E_mod = self._call_module_energy_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                    )
                except TypeError:
                    E_mod = self._call_module_energy_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                    )
                total_energy += float(scale) * float(E_mod)
                continue

            if hasattr(module, "compute_energy_and_gradient_array"):
                try:
                    E_mod = self._call_module_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=grad_dummy,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                        tilt_in_grad_arr=None,
                        tilt_out_grad_arr=None,
                    )
                except TypeError:
                    E_mod = self._call_module_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=grad_dummy,
                    )
                total_energy += float(scale) * float(E_mod)
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
            total_energy += float(scale) * float(E_mod)

        return float(total_energy)

    def compute_tilt_dependent_energy_with_leaflet_tilts(
        self,
        *,
        positions: np.ndarray,
        tilts_in: np.ndarray,
        tilts_out: np.ndarray,
        grad_dummy: np.ndarray | None = None,
        tilt_vertex_areas_in: np.ndarray | None = None,
        tilt_vertex_areas_out: np.ndarray | None = None,
    ) -> float:
        """Compute energy of tilt-dependent modules only (positions frozen)."""
        index_map = self.mesh.vertex_index_to_row
        if grad_dummy is None:
            grad_dummy = np.zeros_like(positions)
        else:
            grad_dummy.fill(0.0)
        total_energy = 0.0

        for name, module in zip(self.energy_module_names, self.energy_modules):
            scale = self.experimental_energy_scale_fn(str(name))
            if not getattr(module, "USES_TILT_LEAFLETS", False):
                continue

            # Fast path for pure tilt magnitude penalties.
            if name == "tilt_in" and tilt_vertex_areas_in is not None:
                k_tilt = float(self.param_resolver.get(None, "tilt_modulus_in") or 0.0)
                if k_tilt != 0.0:
                    sq = np.einsum("ij,ij->i", tilts_in, tilts_in)
                    total_energy += float(
                        float(scale) * 0.5 * k_tilt * np.sum(sq * tilt_vertex_areas_in)
                    )
                continue
            if name == "tilt_out" and tilt_vertex_areas_out is not None:
                k_tilt = float(self.param_resolver.get(None, "tilt_modulus_out") or 0.0)
                if k_tilt != 0.0:
                    sq = np.einsum("ij,ij->i", tilts_out, tilts_out)
                    total_energy += float(
                        float(scale) * 0.5 * k_tilt * np.sum(sq * tilt_vertex_areas_out)
                    )
                continue

            if hasattr(module, "compute_energy_array"):
                try:
                    E_mod = self._call_module_energy_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                        tilts_in=tilts_in,
                        tilts_out=tilts_out,
                    )
                except TypeError:
                    # Some tilt modules ignore passed tilts and read from mesh.
                    E_mod = self._call_module_energy_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                    )
                total_energy += float(scale) * float(E_mod)
                continue

            if hasattr(module, "compute_energy_and_gradient_array"):
                try:
                    E_mod = self._call_module_array(
                        module,
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
                    E_mod = self._call_module_array(
                        module,
                        positions=positions,
                        index_map=index_map,
                        grad_arr=None,
                    )
                total_energy += float(scale) * float(E_mod)
                continue

            E_full = self.compute_energy_array_with_leaflet_tilts(
                positions=positions,
                tilts_in=tilts_in,
                tilts_out=tilts_out,
                grad_dummy=grad_dummy,
            )
            return float(E_full)

        return float(total_energy)

    def compute_energy_and_leaflet_tilt_gradients_array(
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
            scale = self.experimental_energy_scale_fn(str(name))
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
                gin_before = None
                gout_before = None
                if abs(float(scale) - 1.0) > 1.0e-15:
                    gin_before = tilt_in_grad_arr.copy()
                    gout_before = tilt_out_grad_arr.copy()

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

                if gin_before is not None:
                    delta_in = tilt_in_grad_arr - gin_before
                    delta_out = tilt_out_grad_arr - gout_before
                    tilt_in_grad_arr[:] = gin_before + (float(scale) * delta_in)
                    tilt_out_grad_arr[:] = gout_before + (float(scale) * delta_out)

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

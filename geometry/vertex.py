"""Vertex entity for membrane meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

if TYPE_CHECKING:
    from geometry.mesh import Mesh


@dataclass
class Vertex:
    index: int
    position: np.ndarray
    fixed: bool = False
    options: Dict[str, Any] = field(default_factory=dict)
    tilt: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    tilt_fixed: bool = False
    tilt_in: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    tilt_out: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    tilt_fixed_in: bool = False
    tilt_fixed_out: bool = False
    _mesh: Mesh | None = field(default=None, repr=False, compare=False)
    _row: int = field(default=-1, repr=False, compare=False)

    def __getattribute__(self, name: str) -> Any:
        if name in ("tilt", "tilt_in", "tilt_out"):
            mesh = object.__getattribute__(self, "_mesh")
            row = object.__getattribute__(self, "_row")
            if mesh is not None and row >= 0:
                if name == "tilt":
                    return mesh.tilts_view()[row]
                if name == "tilt_in":
                    return mesh.tilts_in_view()[row]
                return mesh.tilts_out_view()[row]
        return object.__getattribute__(self, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("fixed", "tilt_fixed", "tilt_fixed_in", "tilt_fixed_out"):
            old = self.__dict__.get(name, None)
            object.__setattr__(self, name, value)
            if old is not None and old != value:
                mesh = self.__dict__.get("_mesh")
                if mesh is not None:
                    if name == "fixed":
                        mesh._touch_fixed_flags()
                    else:
                        mesh._touch_tilt_fixed_flags()
            return
        if name in ("tilt", "tilt_in", "tilt_out"):
            arr = np.asarray(value, dtype=float)
            if arr.shape != (3,):
                raise ValueError(f"{name} must be a 3-vector")
            mesh = self.__dict__.get("_mesh")
            row = self.__dict__.get("_row", -1)
            if mesh is not None and row >= 0:
                if name == "tilt":
                    mesh.tilts_view()[row] = arr
                elif name == "tilt_in":
                    mesh.tilts_in_view()[row] = arr
                else:
                    mesh.tilts_out_view()[row] = arr
            object.__setattr__(self, name, arr)
            return
        object.__setattr__(self, name, value)

    def copy(self):
        return Vertex(
            self.index,
            self.position.copy(),
            fixed=self.fixed,
            options=self.options.copy(),
            tilt=self.tilt.copy(),
            tilt_fixed=self.tilt_fixed,
            tilt_in=self.tilt_in.copy(),
            tilt_out=self.tilt_out.copy(),
            tilt_fixed_in=self.tilt_fixed_in,
            tilt_fixed_out=self.tilt_fixed_out,
        )

    def project_position(self, pos: np.ndarray) -> np.ndarray:
        """
        Project the given position onto the constraint, if any.
        If no constraint is defined, return the position unchanged.
        """
        if "constraint" in self.options:
            constraint = self.options["constraint"]
            return constraint.project_position(pos)
        return pos

    def project_gradient(self, grad: np.ndarray) -> np.ndarray:
        """
        Project the given gradient into the tangent space of the constraint, if any.
        If no constraint is defined, return the gradient unchanged.
        """
        if "constraint" in self.options:
            constraint = self.options["constraint"]
            return constraint.project_gradient(grad)
        return grad

    def compute_distance(self, other: Vertex) -> float:
        """
        Compute the distance to another vertex.
        """
        return np.linalg.norm(self.position - other.position)

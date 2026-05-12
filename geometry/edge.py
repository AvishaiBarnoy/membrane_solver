"""Edge entity for membrane meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict

import numpy as np

if TYPE_CHECKING:
    from geometry.mesh import Mesh


@dataclass
class Edge:
    index: int
    tail_index: int
    head_index: int
    refine: bool = True
    fixed: bool = False
    options: Dict[str, Any] = field(default_factory=dict)

    def copy(self):
        return Edge(
            self.index,
            self.tail_index,
            self.head_index,
            self.refine,
            self.fixed,
            self.options.copy(),
        )

    def reversed(self) -> Edge:
        return Edge(
            index=self.index,  # convention: reversed edge gets negative index
            tail_index=self.head_index,
            head_index=self.tail_index,
            refine=self.refine,
            fixed=self.fixed,
            options=self.options,
        )

    def compute_length(self, mesh: Mesh) -> float:
        tail = mesh.vertices[self.tail_index]
        head = mesh.vertices[self.head_index]
        return np.linalg.norm(head.position - tail.position)

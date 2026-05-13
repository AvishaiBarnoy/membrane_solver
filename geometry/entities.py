"""Orchestrator for mesh entities, re-exporting for backward compatibility."""

from __future__ import annotations

from core.parameters.global_parameters import GlobalParameters
from geometry.body import Body
from geometry.edge import Edge
from geometry.facet import Facet, _fast_cross
from geometry.mesh import Mesh, MeshError
from geometry.vertex import Vertex

__all__ = [
    "Body",
    "Edge",
    "Facet",
    "GlobalParameters",
    "Mesh",
    "MeshError",
    "Vertex",
    "_fast_cross",
]

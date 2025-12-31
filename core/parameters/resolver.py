"""Utility for resolving per-object parameters.

This small helper checks for object-specific
parameter overrides and falls back to global
parameters when none are provided.
"""

from core.parameters.global_parameters import GlobalParameters


class ParameterResolver:
    """Resolve parameters with optional per-object overrides."""

    def __init__(self, global_params: GlobalParameters):
        self.global_params = global_params

    def get(self, obj, name: str):
        """Return parameter ``name`` for ``obj`` or global default."""
        if obj is None:
            return self.global_params.get(name)
        return obj.options.get(name, self.global_params.get(name))

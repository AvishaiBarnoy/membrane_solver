# global_parameters.py


class GlobalParameters:
    def __init__(self, initial_params=None):
        """
        all parameters are defined with underscore, _, instead of spaces
        """
        # Use a dictionary to store all parameters
        self._params = {
            "surface_tension": 1.0,  # Default value
            "volume_stiffness": 1000.0,  # Default value
            # How volume constraints are enforced:
            #   "penalty"  – use soft quadratic volume energy.
            #   "lagrange" – treat volume as a hard constraint integrated via
            #                Lagrange‑style gradient projection.
            # By default we follow the Evolver‑like hard‑constraint workflow.
            "volume_constraint_mode": "lagrange",
            # Whether to apply geometric volume projection during each
            # minimization step. When set to False, the optimizer is expected
            # to handle fixed volume purely through its gradient/Lagrange
            # machinery (more Evolver‑like); when True, the legacy behaviour
            # of post‑step projection is enabled.
            "volume_projection_during_minimization": True,
            # Relative tolerance for volume drift during line search when
            # relying on Lagrange‑style constraints without geometric
            # projection. Steps that violate this tolerance are rejected.
            "volume_tolerance": 1e-3,
            "max_zero_steps": 10,
            "step_size_floor": 1e-8,
            "step_size": 1e-3,
            "intrinsic_curvature": 0.0,
            "bending_modulus": 0.0,
            "bending_energy_model": "helfrich",
            # Bending gradient implementation:
            #   "analytic" – accurate (validated vs finite differences).
            #   "approx"   – cheaper Laplacian approximation (may stall on stiff cases).
            #   "finite_difference" – debugging only (slow).
            "bending_gradient_mode": "analytic",
            "gaussian_modulus": 0.0,
        }
        # Load initial parameters if provided
        if initial_params:
            self.update(initial_params)

    def __getattr__(self, name):
        """Backwards-compatible attribute access for parameters.

        Many parts of the codebase access parameters as attributes
        (e.g. ``global_params.volume_stiffness``) while the canonical storage
        is the internal ``_params`` dict. This ensures both access patterns
        remain consistent for known parameter keys.
        """
        params = self.__dict__.get("_params")
        if params is not None and name in params:
            return params[name]
        raise AttributeError(
            f"{type(self).__name__!s} object has no attribute {name!r}"
        )

    def __setattr__(self, name, value):
        """Backwards-compatible attribute assignment for known parameter keys."""
        if name == "_params":
            object.__setattr__(self, name, value)
            return
        params = self.__dict__.get("_params")
        if params is not None and name in params:
            params[name] = value
            return
        object.__setattr__(self, name, value)

    def get(self, key, default=None):
        """Retrieve a parameter value, or return a default if not found."""
        return self._params.get(key, default)

    def set(self, key, value):
        """Set or update a parameter."""
        self._params[key] = value

    def update(self, params):
        """Update multiple parameters at once."""
        self._params.update(params)

    def __contains__(self, key):
        """Check if a parameter exists."""
        return key in self._params

    def __repr__(self):
        """String representation for debugging."""
        return f"GlobalParameters({self._params})"

    def to_dict(self):
        """Convert the parameters to a dictionary for serialization."""
        return self._params

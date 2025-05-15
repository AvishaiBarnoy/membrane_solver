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
            "step_size": 1e-4,
            "intrinsic_curvature": 0.0,
            "bending_modulus": 0.0,
            "gaussian_modulus": 0.0
        }
        # Load initial parameters if provided
        if initial_params:
            self.update(initial_params)

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

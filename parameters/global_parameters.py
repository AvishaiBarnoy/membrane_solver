# global_parameters.py

class GlobalParameters:
    def __init__(self, param_dict=None):
        """
        all parameters are defined with underscore, _, instead of spaces
        """
        param_dict = param_dict or {}

        # Set defaults, overrriden by input file if present
        self.surface_tension = param_dict.get("surface_tension", 1.0)
        self.intrinsic_curvature = param_dict.get("intrinsic_curvature", 0.0)
        self.bending_modulus = param_dict.get("bending_modulus", 0.0)
        self.gaussian_modulus = param_dict.get("gaussian_modulus", 0.0)
        self.volume_stiffness = param_dict.get("volume_stiffness", 1000.0)
        self.time_step = param_dict.get("time_step", 0.01)

        self.extra = {k: v for k, v in param_dict.items()
                      if k not in {
                          "surface_tension",
                          "intrinsic_curvature",
                          "bending_modulus",
                          "gaussian_modulus",
                          "volume_stiffness"
                      }}

    def get(self, key, default=None):
        """access extra more natively"""
        # TODO: implement this better
        return getattr(self, key, self.extra.get(key, default))

    def __repr__(self):
        return f"GlobalParameters:\n\t\
                surface_tension={self.surface_tension}\n\t\
                volume_stiffness={self.volume_stiffness}\n\t\
                bending_modulus={self.bending_modulus}\n\t\
                gaussian_modulus={self.gaussian_modulus}\n\t\
                extra={self.extra}"


# global_parameters.py

class GlobalParameters:
    def __init__(self, param_dict):
        self.surface_tension = param_dict.get("surface tension", 1.0)
        self.intrinsic_curvature = param_dict.get("intrinsic curvature", 0.0)
        self.bending_modulus = param_dict.get("bending modulus", 0.0)
        self.gaussian_modulus = param_dict.get("gaussian modulus", 0.0)
        self.volume_stiffness = param_dict.get("volume stiffness", 0.0)
        self.time_step = param_dict.get("time_step", 0.01)

        self.extra = {k: v for k, v in param_dict.items()
                      if k not in {
                          "surface tension",
                          "intrinsic curvature",
                          "bending modulus",
                          "gaussian modulus",
                          "volume stiffness"
                      }}

    def __repr__(self):
        return f"GlobalParameters:\n\t\
                surface_tension={self.surface_tension}\n\t\
                volume_stiffness={self.volume_stiffness}\n\t\
                extra={self.extra}"


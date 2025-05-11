# modules/constraints/fix_facet_area.py

class FixFacetArea:
    def __init__(self, target_area, weight=1.0):
        self.A0 = target_area
        self.w  = weight

    def compute_energy_and_gradient(self, mesh, gp, resolver):
        # volume‐style Lagrange multiplier on facet area
        ...

# registry
ALL = {
    "pin_to_plane": PinToPlane,
    "fix_facet_area": FixFacetArea,
}


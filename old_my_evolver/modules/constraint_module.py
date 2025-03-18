# modules/constraint_module.py
import numpy as np

class ConstraintModule:
    def modify_forces(self, mesh):
        """
        Example module that fixes vertex 0 by zeroing its force.
        """
        for i, vertex in enumerate(mesh.vertices):
            if i == 0:
                vertex.force = np.zeros(3, dtype=float)


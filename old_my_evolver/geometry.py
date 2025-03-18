# geometry.py
import numpy as np

class Vertex:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)
        self.force = np.zeros(3, dtype=float)
        # For conjugate gradient updates:
        self.prev_force = np.zeros(3, dtype=float)
        self.search_direction = np.zeros(3, dtype=float)
        self.initialized_cg = False  # Flag to initialize the search direction


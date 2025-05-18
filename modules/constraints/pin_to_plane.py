# modules/constraints/pin_to_plane.py

class PinToPlane:
    def __init__(self, plane_normal, plane_point):
        self.n = plane_normal
        self.p0 = plane_point

    def project_position(self, pos):
        # snap pos back onto plane
        # TODO: explain what is done here mathematically
        return pos - np.dot(pos - self.p0, self.n) * self.n

    def project_gradient(self, grad):
        # remove normal component
        # TODO: explain what is done here mathematically
        return grad - np.dot(grad, self.n) * self.n

"""
# EXAMPLE HOW TO USE:

from modules.constraints.pin_to_plane import PinToPlane

# Define a plane with normal vector and a point on the plane
plane_normal = np.array([0, 0, 1])  # z = 0 plane
plane_point = np.array([0, 0, 0])

# Create a constraint
plane_constraint = PinToPlane(plane_normal, plane_point)

# Assign the constraint to a vertex
vertex = Vertex(index=0, position=np.array([1, 1, 5]), options={'constraint': plane_constraint})

# Project the vertex position onto the plane
projected_position = vertex.project_position(vertex.position)
print("Projected Position:", projected_position)

# Project a gradient into the tangent space of the plane
gradient = np.array([1, 1, 1])
projected_gradient = vertex.project_gradient(gradient)
print("Projected Gradient:", projected_gradient)
"""

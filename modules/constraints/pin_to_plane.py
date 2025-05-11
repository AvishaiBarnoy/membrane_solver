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


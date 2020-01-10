from src.Simulator.Vector import Vector


class Ray:
    number_of_rays = 0
    origin = Vector(0, 0, 0)
    direction = Vector(0, 0, 0)
    amplitude = 0

    def __init__(self, _origin, _direction, _amplidute):

        self.origin = _origin

        self.direction = _direction

        self.setAmplitude(_amplidute)

        Ray.number_of_rays += 1

    def setAmplitude(self, _amplitude):
        self.amplitude = _amplitude

    def getAmplitude(self):
        return self.amplitude

    def getNumberOfRays(self):
        return self.number_of_rays

    def getOrigin(self):
        return self.origin

    def getDirection(self):
        return self.direction

    def setOrigin(self, _x, _y, _z):
        self.origin = Vector(_x, _y, _z)

    def setDirection(self, _dx, _dy, _dz):
        self.direction = Vector(_dx, _dy, _dz)



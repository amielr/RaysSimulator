import math

import numpy as np


class Vector:
    x, y, z = 0, 0, 0

    def __init__(self, _x, _y, _z):
        self.setX(_x)
        self.setY(_y)
        self.setZ(_z)

    def getX(self):
        return self.x

    def getY(self):
        return self.y

    def getZ(self):
        return self.z

    def setX(self, _x):
        self.x = _x

    def setY(self, _y):
        self.y = _y

    def setZ(self, _z):
        self.z = _z

    def __sub__(self, vector):
        return Vector(self.x - vector.x, self.y - vector.y, self.z - vector.z)

    def __add__(self, vector):
        return Vector(self.x + vector.x, self.y + vector.y, self.z + vector.z)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot_product(self, vector):
        v1 = [self.x, self.y, self.z]
        v2 = [vector.x, vector.y, vector.z]
        return sum((a * b) for a, b in zip(v1, v2))

    def cross(self, vector):
        crossX = self.y * vector.z - self.z * vector.y
        crossY = self.z * vector.x - self.x * vector.z
        crossZ = self.x * vector.y - self.y * vector.x
        return Vector(crossX, crossY, crossZ)

    def angle(self, vector):
        return math.acos(self.dot_product(vector) / (self.length() * vector.length()))

    def length(self):
        return math.sqrt(self.dot_product(self))

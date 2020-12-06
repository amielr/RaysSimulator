import math
import numpy as np


x, y, z = 0, 0, 0

#def __init__(self, _x, _y, _z):
#    self.setX(_x)
#    self.setY(_y)
#    self.setZ(_z)

def getX(vector):
    return vector[0]

def getY(vector):
    return vector[1]

def getZ(vector):
    return vector[2]

def setX(vector, _x):
    vector[0] = _x
    return[vector]

def setY(vector, _y):
    vector[1] = _y
    return vector

def setZ(vector, _z):
    vector[2] = _z
    return vector

# def __sub__(self, vector):
#     return np.array(self.x - vector.x, self.y - vector.y, self.z - vector.z)
#
# def __add__(self, vector):
#     return np.array(self.x + vector.x, self.y + vector.y, self.z + vector.z)
#
# def __mul__(self, scalar):
#     return np.array(self.x * scalar, self.y * scalar, self.z * scalar)

def dot_product(self, vector):
    v1 = [self.x, self.y, self.z]
    v2 = [vector.x, vector.y, vector.z]
    return sum((a * b) for a, b in zip(v1, v2))

def cross(self, vector):
    crossX = self.y * vector.z - self.z * vector.y
    crossY = self.z * vector.x - self.x * vector.z
    crossZ = self.x * vector.y - self.y * vector.x
    return np.array([[crossX], [crossY], [crossZ]])

def angle(self, vector):
    return math.acos(self.dot_product(vector) / (self.length() * vector.length()))

def length(vector):
    return math.sqrt(np.dot(vector, vector))

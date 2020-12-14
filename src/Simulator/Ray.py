from src.Simulator.Vector import *
import numpy as np
from numba import njit

number_of_rays = 0
origin = np.array([0, 0, 0], dtype=np.float_)
direction = np.array([0, 0, 0], dtype=np.float_)
amplitude = np.array([0, 0, 0], dtype=np.float_)

# def __init__(self, _origin=Vector(0, 0, 0), _direction=Vector(0, 0, 0), _amplidute=0):
#     self.origin = _origin
#
#     self.direction = _direction
#
#     self.setAmplitude(_amplidute)
#
#     #Ray.number_of_rays += 1


def setAmplitude(ray, _amplitude):
    ray[2][0] = _amplitude
    return ray


def getAmplitude(ray):
    # print("inside getAmplitude", ray[2])
    return ray[2][0]


def getNumberOfRays(self):
    return self.number_of_rays


# @njit()
def getOrigin(ray) -> np.array(3, dtype=np.float_):
    return ray[0]


# @njit()
def getDirection(ray) -> np.array(3 ,dtype=np.float_):
    return ray[1]


def setOrigin(ray, _x, _y, _z):
    ray[0] = np.array([_x, _y, _z], np.float_)
    return ray


def setDirection(ray, _dx, _dy, _dz):
    ray[1] = np.array([_dx, _dy, _dz], np.float_)
    return ray

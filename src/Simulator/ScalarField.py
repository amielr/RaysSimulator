import numpy as np


def get_rotation_matrix(angle, direction):
    angleInRadians = np.deg2rad(angle)
    angleSin = np.sin(angleInRadians)
    angleCos = np.cos(angleInRadians)

    if direction == 'x':
        return [
            [1, 0, 0],
            [0, angleCos, -angleSin],
            [0, angleSin, angleCos]
        ]

    if direction == 'y':
        return [
            [angleCos, 0, angleSin],
            [0, 1, 0],
            [-angleSin, 0, angleCos]
        ]

    return [
        [angleCos, -angleSin, 0],
        [angleSin, angleCos, 0],
        [0, 0, 1]
    ]



_xGrid, _yGrid, _zScalarField = [], [], []
_xMin, _xMax, _yMin, _yMax = 0, 0, 0, 0
#
def initiateScalarField(xGrid, yGrid, zScalarField):
     _xGrid = xGrid
     _yGrid = yGrid
     _zScalarField = zScalarField
     mirrorBorders = set_mirror_borders(xGrid, yGrid)
     return np.array([_xGrid, _yGrid, _zScalarField])

def apply_rotation(field, angle, direction):
    flattenX, flattenY, flattenZ = np.ravel(getXGrid(field)), np.ravel(getYGrid(field)), np.ravel(getZScalarField(field))
    stackFlattenXYZ = np.stack((flattenX, flattenY, flattenZ))

    rotatedXYZ = np.matmul(get_rotation_matrix(angle, direction), stackFlattenXYZ)

    _xGrid = np.reshape(rotatedXYZ[0], getXGrid(field).shape)
    _yGrid = np.reshape(rotatedXYZ[1], getYGrid(field).shape)
    _zScalarField = np.reshape(rotatedXYZ[2], getZScalarField(field).shape)
    #mirrorborders = set_mirror_borders(field)
    field = np.array([_xGrid, _yGrid, _zScalarField])
    return field

def getXGrid(field):
    return field[0]

def getYGrid(field):
    return field[1]

def getZScalarField(field):
    return field[2]


def set_mirror_borders(xGrid, yGrid):
    _xMax = xGrid.max()
    _xMin = xGrid.min()
    _yMax = yGrid.max()
    _yMin = yGrid.min()
    return [_xMax, _xMin, _yMax, _yMin]

def getMinBoundary(field):
    set_mirror_borders(field[0], field[1])
    return np.array([_xMin, _yMin])

def getMaxBoundary(field):
    set_mirror_borders(field[0], field[1])
    return np.array([_xMax, _yMax])

def add_offset(field, offset):
    #print(field)
    adjustedField = field[2] + offset
    return adjustedField

@property
def zScalarField(self):
    return self._zScalarField

@property
def xGrid(self):
    return self._xGrid

@property
def yGrid(self):
    return self._yGrid

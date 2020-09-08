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


class ScalarField:
    _xGrid, _yGrid, _zScalarField = [], [], []
    _xMin, _xMax, _yMin, _yMax = 0, 0, 0, 0

    def __init__(self, xGrid, yGrid, zScalarField):
        self._xGrid = xGrid
        self._yGrid = yGrid
        self._zScalarField = zScalarField
        self.set_mirror_borders()

    def apply_rotation(self, angle, direction):
        flattenX, flattenY, flattenZ = np.ravel(self._xGrid), np.ravel(self._yGrid), np.ravel(self._zScalarField)
        stackFlattenXYZ = np.stack((flattenX, flattenY, flattenZ))

        rotatedXYZ = np.matmul(get_rotation_matrix(angle, direction), stackFlattenXYZ)

        self._xGrid = np.reshape(rotatedXYZ[0], self._xGrid.shape)
        self._yGrid = np.reshape(rotatedXYZ[1], self._yGrid.shape)
        self._zScalarField = np.reshape(rotatedXYZ[2], self._zScalarField.shape)
        self.set_mirror_borders()

    def set_mirror_borders(self):
        self._xMax = self.xGrid.max()
        self._xMin = self.xGrid.min()
        self._yMax = self.yGrid.max()
        self._yMin = self.yGrid.min()

    def getMinBoundary(self):
        return [self._xMin, self._yMin]

    def getMaxBoundary(self):
        return [self._xMax, self._yMax]

    def add_offset(self, offset):
        self._zScalarField += offset

    @property
    def zScalarField(self):
        return self._zScalarField

    @property
    def xGrid(self):
        return self._xGrid

    @property
    def yGrid(self):
        return self._yGrid

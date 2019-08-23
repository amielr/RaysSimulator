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

    def __init__(self, xGrid, yGrid, zScalarField):
        self._xGrid = xGrid
        self._yGrid = yGrid
        self._zScalarField = zScalarField

    def apply_rotation(self, angle, direction):
        flattenX, flattenY, flattenZ = np.ravel(self._xGrid), np.ravel(self._yGrid), np.ravel(self._zScalarField)
        stackFlattenXYZ = np.stack((flattenX, flattenY, flattenZ))

        rotatedXYZ = np.matmul(get_rotation_matrix(angle, direction), stackFlattenXYZ)

        self._xGrid = np.reshape(rotatedXYZ[0], self._xGrid.shape)
        self._yGrid = np.reshape(rotatedXYZ[1], self._yGrid.shape)
        self._zScalarField = np.reshape(rotatedXYZ[2], self._zScalarField.shape)

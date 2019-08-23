import numpy as np
from scipy import interpolate
import json

from src.Simulator.ScalarField import ScalarField

with open('config.json') as config_file:
    config = json.load(config_file)

angle = config["mirrorRotationAngle"]
direction = config["mirrorRotationDirection"]


def generate_light_source():
    xGrid, yGrid = np.meshgrid(np.linspace(-10, 10, config["lightSourceDensity"]),
                               np.linspace(-10, 10, config["lightSourceDensity"]))

    pulse2d = np.where(abs(xGrid) <= 4, 1, 0) & np.where(abs(yGrid) <= 3, 1, 0)

    return np.stack((xGrid, yGrid, pulse2d))


def create_interpolated_mirror():
    xDenominator = config["xMirrorDenominator"]
    yDenominator = config["yMirrorDenominator"]

    x, y = np.meshgrid(np.linspace(-150, 150, 50), np.linspace(-150, 150, 50))
    z = (x ** 2) / xDenominator + (y ** 2) / xDenominator
    zoverlay = np.zeros(x.shape)
    z = z + zoverlay
    interpolatedMirror = interpolate.interp2d(x[0, :], y[:, 0], z, kind='cubic')

    zinterp = interpolatedMirror(x[0, :], y[:, 0])

    field = ScalarField(x, y, zinterp)
    field.apply_rotation(angle, direction)
    field._zScalarField += 450
    interpolatedMirror = interpolate.interp2d(field._xGrid[0, :], field._yGrid[:, 0], field._zScalarField, kind='cubic')
    print("shape of zrotate is:" + str(field._zScalarField.shape))

    z0 = np.zeros(zinterp.shape)
    zdist = field._zScalarField - z0
    rx = np.ravel(field._xGrid)
    ry = np.ravel(field._yGrid)
    zrotate = np.ravel(field._zScalarField)
    zdist = np.ravel(zdist)
    mirrorobject = np.stack((rx, ry, zrotate, zdist))
    mirrorobject = np.reshape(mirrorobject, (4, len(x), len(y)))

    return mirrorobject, interpolatedMirror

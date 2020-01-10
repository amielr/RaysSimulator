import numpy as np
from src.Simulator.MirrorIntersectionFunctions import max_min
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


def create_interpolated_mirror(mirrorCorrections):
    xMirrorScale = config["xMirrorScale"]
    yMirrorScale = config["yMirrorScale"]
    mirrorGridDensity = config["mirrorGridDensity"]
    mirrorDimensions = config["mirrorDimensions"]
    mirrorOffsetFromSource = config["mirrorOffsetFromSource"]

    axis = np.linspace(-mirrorDimensions, mirrorDimensions, mirrorGridDensity)

    xGrid, yGrid = np.meshgrid(axis, axis)
    mirrorBaseShape = (xGrid ** 2) / xMirrorScale + (yGrid ** 2) / yMirrorScale
    #mirrorBaseShape = xGrid + yGrid
    mirrorBaseShape = mirrorBaseShape + mirrorCorrections
    interpolatedMirrorBuilder = interpolate.interp2d(axis, axis, mirrorBaseShape, kind='cubic')

    interpolatedMirror = interpolatedMirrorBuilder(axis, axis)

    field = ScalarField(xGrid, yGrid, interpolatedMirror)
    field.apply_rotation(angle, direction)
    field.add_offset(mirrorOffsetFromSource)
    interpolatedMirrorBuilder = interpolate.interp2d(axis, axis, field._zScalarField, kind='cubic')

    z0 = np.zeros(field._zScalarField.shape)
    zdist = field._zScalarField - z0
    rx = np.ravel(field._xGrid)
    ry = np.ravel(field._yGrid)
    zrotate = np.ravel(field._zScalarField)
    zdist = np.ravel(zdist)
    mirrorBorders = np.stack((rx, ry, zrotate, zdist))
    mirrorBorders = np.reshape(mirrorBorders, (4, len(xGrid), len(yGrid)))
    mirrorBorders = max_min(mirrorBorders)

    return mirrorBorders, interpolatedMirrorBuilder

import numpy as np
from scipy import interpolate
import json

from src.Simulator.ScalarField import ScalarField

with open('config.json') as config_file:
    config = json.load(config_file)


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
    angle = config["mirrorRotationAngle"]
    direction = config["mirrorRotationDirection"]

    axis = np.linspace(-mirrorDimensions, mirrorDimensions, mirrorGridDensity)

    xGrid, yGrid = np.meshgrid(axis, axis)
    mirrorBaseShape = (xGrid ** 2) / xMirrorScale + (yGrid ** 2) / yMirrorScale
    mirrorShape = mirrorBaseShape + mirrorCorrections

    field = ScalarField(xGrid, yGrid, mirrorShape)
    field.apply_rotation(angle, direction)
    field.add_offset(mirrorOffsetFromSource)

    interpolatedMirrorBuilder = interpolate.interp2d(axis, axis, field.zScalarField, kind='cubic')
    mirrorBorders = np.array(([field.xGrid.max(), field.xGrid.min()], [field.yGrid.max(), field.yGrid.min()]))

    return mirrorBorders, interpolatedMirrorBuilder

import numpy as np
from scipy import interpolate
import json
from ..fast_interp import interp2d

import math

from src.Simulator.ScalarField import ScalarField

with open('config.json') as config_file:
    config = json.load(config_file)


def generate_light_source():
    xVec = np.linspace(-10, 10, config["lightSourceDensity"])
    yVec = np.linspace(-10, 10, config["lightSourceDensity"])
    xGrid, yGrid = np.meshgrid(xVec, yVec)

    pulse2d = np.where(abs(xGrid) <= 4, 1, 0) & np.where(abs(yGrid) <= 3, 1, 0)

    return xVec, yVec, pulse2d


def create_interpolated_mirror(mirrorCorrections):
    xMirrorScale = config["xMirrorScale"]
    yMirrorScale = config["yMirrorScale"]
    mirrorGridDensity = config["mirrorGridDensity"]
    mirrorDimensions = config["mirrorDimensions"]
    mirrorOffsetFromSource = config["mirrorOffsetFromSource"]
    angle = config["mirrorRotationAngle"]
    direction = config["mirrorRotationDirection"]
    korder = config["TaylorOrder"]

    axis, vertexDistance = np.linspace(-mirrorDimensions, mirrorDimensions, mirrorGridDensity, endpoint=False,retstep=True)

    xGrid, yGrid = np.meshgrid(axis, axis)
    # mirrorBaseShape = (xGrid ** 2) / xMirrorScale + (yGrid ** 2) / yMirrorScale
    mirrorBaseShape = np.zeros(mirrorGridDensity * mirrorGridDensity).reshape((mirrorGridDensity, mirrorGridDensity))
    mirrorShape = mirrorBaseShape

    field = ScalarField(xGrid, yGrid, mirrorShape)
    field.apply_rotation(angle, direction)
    field.add_offset(mirrorOffsetFromSource)
    field.add_offset(mirrorCorrections)

    vertexDistanceX = (field.xGrid.max()-field.xGrid.min())/(field.xGrid[0,:].size-2)
    vertexDistanceY = (field.yGrid.max()-field.yGrid.min())/(field.yGrid[:,0].size-2)
    interpolatedMirrorBuilder = interp2d(field.getMinBoundary(), field.getMaxBoundary(), [vertexDistanceX, vertexDistanceY], field.zScalarField.T, k=korder)
    # interpolatedMirrorBuilder = interpolate.interp2d(field.xGrid[0,:], field.yGrid[:,0], field.zScalarField, kind='cubic')

    mirrorBorders = field.getMinBoundary() + field.getMaxBoundary()

    return mirrorBorders, interpolatedMirrorBuilder

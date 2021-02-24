import numpy as np
from numpy import genfromtxt
import json
from ..fast_interp import interp2d, interp3d
from src.Simulator.ScalarField import *
from numba.typed import List

with open('./config.json') as config_file:
    config = json.load(config_file)


def generate_light_source():
    xVec = np.linspace(-3.75, 3.75, config["lightSourceDensity"])
    yVec = np.linspace(-2.5, 2.5, config["lightSourceDensity"])
    xGrid, yGrid = np.meshgrid(xVec, yVec)

    pulse2d = np.where(abs(xGrid) <= 2, 1, 0) & np.where(abs(yGrid) <= 1, 1, 0)
    # my_data = genfromtxt('../../Eqxtcsv.csv', delimiter=',')

    # print("we are checking generate light source", my_data)
    print(pulse2d)
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

    mirrorfield = initiateScalarField(xGrid, yGrid, mirrorShape)
    RotatedMirror = apply_rotation(mirrorfield, angle, direction)
    # print(RotatedMirror[2])
    OffsetMirror = add_offset(RotatedMirror, mirrorOffsetFromSource)
    geneticallyAdjustedMirror = add_offset(OffsetMirror, mirrorCorrections)

    vertexDistanceX = (max(getXGrid(geneticallyAdjustedMirror[0]))-min(getXGrid(geneticallyAdjustedMirror[0])))/(getXGrid(geneticallyAdjustedMirror[0]).size-2)
    vertexDistanceY = (max(getYGrid(geneticallyAdjustedMirror[1].T))-min(getYGrid(geneticallyAdjustedMirror[1].T)))/(getYGrid(geneticallyAdjustedMirror[1]).size-2)
    # print("marker")
    # print(getZScalarField(geneticallyAdjustedMirror))
    set_mirror_borders(geneticallyAdjustedMirror[0], geneticallyAdjustedMirror[1])

    vertexDetails = np.array([vertexDistanceX, vertexDistanceY])
    #vertexDetails.append(vertexDistanceX, vertexDistanceY)
    interpolatedMirrorBuilder = interp2d(getMinBoundary(), getMaxBoundary(), vertexDetails, geneticallyAdjustedMirror[2].T, k=korder)
    # interpolatedMirrorBuilder = interpolate.interp2d(field.xGrid[0,:], field.yGrid[:,0], field.zScalarField, kind='cubic')

    mirrorBorders = np.concatenate(([getMinBoundary()], [getMaxBoundary()]),axis= None)
    # print("our mirror borders are: " + str(mirrorBorders))
    return mirrorBorders, interpolatedMirrorBuilder

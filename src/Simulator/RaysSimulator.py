import math

from src.Simulator.FunctionSources import generate_light_source, create_interpolated_mirror
from src.Simulator.PlotFunctions import *
from src.Simulator.WignerFunction import wigner_transform
from src.Simulator.RayPropogation import *
import json

with open('../config.json') as config_file:
    config = json.load(config_file)


def integrate_intensity_wig(wignerobject, functionobject):
    functionreturn = []

    for x in range(0, len(wignerobject)):  # for rows and columns
        if x == 0:  # if rows then we need y values from function and x values from wigner
            direction = 1
        else:
            direction = 0  # if columns we need x values from function and y values from wigner

        for y in range(0, len(wignerobject[x])):  # iterate through all rows/columns
            ravelamplitude = np.ravel(wignerobject[x][y][0])
            ravelspace = np.ravel(wignerobject[x][y][1])  # ravel space matrix to become 1D

            uniquizedspace = np.unique(ravelspace)

            sumvec = []
            for i in np.unique(ravelspace):
                sum = ravelamplitude[np.where(ravelspace == i)].sum()
                sumvec.append(sum)

            if direction == 1:  # if rows then we need y values from function and x values from wigner
                perpvalues = np.repeat(functionobject[direction, y, 0],
                                       len(uniquizedspace))  # get the perpendicular location of space matrix
                functionreconstruct = np.stack((uniquizedspace, perpvalues, sumvec))

            else:  # if columns we need x values from function and y values from wigner
                perpvalues = np.repeat(functionobject[direction, 0, y],
                                       len(uniquizedspace))  # get the perpendicular location of space matrix
                functionreconstruct = np.stack((perpvalues, uniquizedspace, sumvec))

            functionreturn.append(functionreconstruct)

    X, Y, Z = np.empty(0, float), np.empty(0, float), np.empty(0, float)
    for i in range(len(functionreturn)):
        X = np.append(X, functionreturn[i][0])
        Y = np.append(Y, functionreturn[i][1])
        Z = np.append(Z, functionreturn[i][2])
        X.tolist()
        Y.tolist()
        Z.tolist()

    functionobjectreturn = np.stack((X, Y, Z))

    return functionobjectreturn


def ray_translated(wignerobject, dist):
    shearmatrix = np.array([[1, dist], [0, 1]])
    transformedwigner = np.empty(wignerobject.shape)

    for x in range(0, len(wignerobject)):  # for rows and columns
        for y in range(0, len(wignerobject[x])):  # iterate through all rows/columns
            ravelamplitude = np.ravel(wignerobject[x][y][0])
            ravelspace = np.ravel(wignerobject[x][y][1])  # ravel space matrix to become 1D
            ravelphase = np.ravel(wignerobject[x][y][2])  # ravel the phase matrix to become 1D
            XY = np.stack((ravelspace, ravelphase))
            transfravelxy = np.matmul(shearmatrix, XY)  # multiply matrix 2x2 raymatrix by 2xN xy matrix
            xtransfered = transfravelxy[0]  # unstack into X and Y
            ytransfered = transfravelxy[1]

            transformedwigner[x][y][0] = np.reshape(ravelamplitude, wignerobject[x][y][0].shape)
            transformedwigner[x][y][1] = np.reshape(xtransfered, wignerobject[x][y][1].shape)
            transformedwigner[x][y][2] = np.reshape(ytransfered, wignerobject[x][y][2].shape)

    return transformedwigner


def error_value_calc(screenRays):
    idealPoint = Vector(-200, 0, 180)

    aggregatedError = 0
    for ray in screenRays:
        aggregatedError += (ray.getAmplitude() * (ray.getOrigin() - idealPoint).length()) ** 2

    return math.sqrt(aggregatedError) / len(screenRays)


lightSource = generate_light_source()
zwignerFunction = wigner_transform(lightSource)
zwignerTranslatedFunction = ray_translated(zwignerFunction, 50)


def simulateMirror(mirrorCorrections):
    mirrorBorders, mirrorInterpolatedBuilder = create_interpolated_mirror(mirrorCorrections)

    # reverseFunction = integrate_intensity_wig(zwignerFunction, lightSource)
    # plot_gridata(reverseFunction)

    screenRays = ray_propogation(zwignerFunction,
                                 zwignerTranslatedFunction,
                                 lightSource,
                                 mirrorInterpolatedBuilder,
                                 mirrorBorders)

    error = error_value_calc(screenRays)

    return error

    # plot_3d_to_2d(zwignerFunction[0][midPoint][1], zwignerFunction[0][midPoint][2], zwignerFunction[0][midPoint][0])

    # plot_scatter(rayobject, 14, 15, 16)

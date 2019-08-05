import numpy as np

from numpy import linalg as LA


import time
from scipy import interpolate
from scipy.fftpack import fft
import math
from scipy.linalg import circulant
from src.FunctionSources import light_source_function, screen_function, create_interpolated_mirror
from src.PlotFunctions import *
from src.WignerFunction import wignerize_each_function
from src.RayPropogation import *


def integrate_intensity_wig(wignerobject, functionobject):
    print("we are in integrate function our x shape is")
    print(functionobject[0].shape)
    # print(x)
    wvdr = np.ravel(wignerobject)
    xr = np.ravel(functionobject[0])
    yr = np.ravel(functionobject[1])
    xr = np.ravel(wignerobject[0][2][1])

    uniquex, uniquexloc = np.unique(xr, return_inverse='true')
    uniquey, uniqueyloc = np.unique(yr, return_inverse='true')

    # print("xr")
    # print(xr)
    # print("uniquex")
    # print(uniquex)
    # print("uniquexloc")
    # print(uniquexloc)
    # print("wignerobjectspace")
    # print(wignerobject[0][2][1])

    # print("length of x")
    # print(len(xr))
    # print("length of wvd")
    # print(len(wignerobject[0]))

    # sumvec = []
    mapping = {}
    functionreturn = []

    for x in range(0, len(wignerobject)):  # for rows and columns

        if x == 0:  # if rows then we need y values from function and x values from wigner
            direction = 1
        else:
            direction = 0  # if columns we need x values from function and y values from wigner

        for y in range(0, len(wignerobject[x])):  # iterate through all rows/columns
            ravelamplitude = np.ravel(wignerobject[x][y][0])
            ravelspace = np.ravel(wignerobject[x][y][1])  # ravel space matrix to become 1D
            ravelphase = np.ravel(wignerobject[x][y][2])  # ravel the phase matrix to become 1D

            uniquizedspace = np.unique(ravelspace)

            sumvec = []
            for i in np.unique(ravelspace):
                # mapping[i] = np.where(x == i)[0]
                sum = ravelamplitude[np.where(ravelspace == i)].sum()
                # print(np.where(xr==i))
                # print("where the values correlate")
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

            print("the length of sumvec is: ")
            print(len(ravelamplitude))
            print(len(sumvec))

            # XY = np.stack((ravelspace, ravelphase))  # stack the two 1D matrices and create a 2xN matrix space and then phase
            # # print("raveled matrix shape")
            # # print(XY.shape)
            # transfravelxy = np.matmul(shearmatrix, XY)  # multiply matrix 2x2 raymatrix by 2xN xy matrix
            # xtransfered = transfravelxy[0]  # unstack into X and Y
            # ytransfered = transfravelxy[1]
            #
            # transformedwigner[x][y][0] = np.reshape(ravelamplitude,(wignerobject[x][y][0].shape))
            # transformedwigner[x][y][1] = np.reshape(xtransfered, (wignerobject[x][y][1].shape))  # reshape to original state and update object
            # transformedwigner[x][y][2] = np.reshape(ytransfered, (wignerobject[x][y][2].shape))

            # print(wignerobject[x][y])

    print("ourfunction dimensions are:")
    print(len(functionreturn))
    print(functionreturn[0].shape)

    print(functionreturn[15])
    # print(transformedwigner.shape)

    X, Y, Z = np.empty(0, float), np.empty(0, float), np.empty(0, float)
    print("our X is:")
    print(len(X))
    for i in range(len(functionreturn)):
        X = np.append(X, functionreturn[i][0])
        Y = np.append(Y, functionreturn[i][1])
        Z = np.append(Z, functionreturn[i][2])
        X.tolist()
        Y.tolist()
        Z.tolist()
    # functionobjectreturn = np.stack((X, Y, Z))
    functionobjectreturn = np.stack((X, Y, Z))
    # np.reshape(functionobjectreturn,(3,,-1))
    print("functionobjectreturn")
    print(functionobjectreturn.shape)
    # print(functionobjectreturn.shape)
    # hello = np.ravel(functionreturn)
    # print("hello")
    # print(len(hello))
    # print(hello)

    return functionobjectreturn


def projection(source, mirrorloc, mirrorfunction):
    return source


def get_threshold_locations(function):  # create matrix where the value of function in each row is above the threshold
    rows = np.where(function.any(axis=1))  # rows
    columns = np.where(function.any(axis=0))  # columns
    rowscolumns = [rows, columns]
    print(rowscolumns)
    return rowscolumns


def ray_transforms(wignerobject, dist):
    # nrows, ncolumns = wvd.shape

    refl = -2 / 1000
    shearmatrix = np.array([[1, dist], [0, 1]])
    # shearmatrix = np.array([[1,0],[refl,1]])
    # transformedwigner = []
    transformedwigner = np.empty(wignerobject.shape)

    for x in range(0, len(wignerobject)):  # for rows and columns
        for y in range(0, len(wignerobject[x])):  # iterate through all rows/columns
            ravelamplitude = np.ravel(wignerobject[x][y][0])
            ravelspace = np.ravel(wignerobject[x][y][1])  # ravel space matrix to become 1D
            ravelphase = np.ravel(wignerobject[x][y][2])  # ravel the phase matrix to become 1D
            XY = np.stack(
                (ravelspace, ravelphase))  # stack the two 1D matrices and create a 2xN matrix space and then phase
            # print("raveled matrix shape")
            # print(XY.shape)
            transfravelxy = np.matmul(shearmatrix, XY)  # multiply matrix 2x2 raymatrix by 2xN xy matrix
            xtransfered = transfravelxy[0]  # unstack into X and Y
            ytransfered = transfravelxy[1]

            transformedwigner[x][y][0] = np.reshape(ravelamplitude, (wignerobject[x][y][0].shape))
            transformedwigner[x][y][1] = np.reshape(xtransfered, (
                wignerobject[x][y][1].shape))  # reshape to original state and update object
            transformedwigner[x][y][2] = np.reshape(ytransfered, (wignerobject[x][y][2].shape))

            # print(wignerobject[x][y])

    print("our new shape is")
    print(transformedwigner.shape)
    return transformedwigner


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


def view_vectors(raysobject):
    for rayholder in raysobject:  # for each function row of rays
        print("newrow")
        for i in range(len(rayholder[0])):
            mx = rayholder[3][i] - rayholder[0][i]
            ny = rayholder[4][i] - rayholder[1][i]
            oz = rayholder[5][i] - rayholder[2][i]

            vector = np.array([mx, ny, oz])
            print(vector)

    return


def get_ray_focal_points(raysobject):
    return


def get_vector(raysobject):
    rayobjectreturn = []
    for rayholder in raysobject:  # for each function row of rays
        rayholder = np.stack(
            (rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5], rayholder[6],
             # 1* amplitude # 3* spatial origins of ray, 3* intersection point on surface
             rayholder[7], rayholder[8], rayholder[9],
             # 3* normal of the surface  1* angle between
             rayholder[9], rayholder[9], rayholder[
                 9]))  # 3* reflected rays                                                                                         #

        print("rayholder shape is:")
        print(rayholder.shape)
        for i in range(len(rayholder[0])):  # run through all elements along array
            # print("iteration number",i)
            normal = np.array([rayholder[6][i], rayholder[7][i], rayholder[8][i]])

            # r = d−2(d⋅n)n

            # rayholder[10][i] =
            # rayholder[11][i] =
            # rayholder[12][i] =
    return rayobjectreturn


def error_value_calc(raysobject, actualoutputfunction):
    accuracy = 0
    precision = 0

    idealx, idealy, idealz = -200, 0, 130
    print("our raysobject shape is in the error")
    print(len(raysobject))
    averagedist = []

    for rayholder in raysobject:  # for each function row of rays
        ytrue = np.zeros(len(rayholder[15]))
        # accuracy = accuracy_score(ytrue, rayholder[15])
        # print(accuracy)
        true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        print(ytrue.shape)
        print(rayholder[15].shape)
        y_val_true, val_pred = ytrue.reshape((-1)), rayholder[15, :].reshape((-1))

        rayholder = np.stack(
            (rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5], rayholder[6],
             # 1* Amplitude     # 3* spatial origins of ray, 3* intersection point on mirror
             rayholder[7], rayholder[8], rayholder[9], rayholder[10]
             # 3* normal of the surface  1* angle between
             , rayholder[11], rayholder[12], rayholder[13]  # 3* reflected rays
             , rayholder[14], rayholder[15], rayholder[16],  # 3* intersection points with screen
             rayholder[16],))  # distance from desired focal point
        total = 0
        for i in range(len(rayholder[0])):  # run through all elements along array

            rayholder[17][i] = math.sqrt((rayholder[14][i] - idealx) ** 2 + (rayholder[15][i] - idealy) ** 2 + (
                        rayholder[16][i] - idealz) ** 2) * abs(rayholder[0][i])
        averagedist.append(np.mean(rayholder[17]))
        print('our average distance is: ' + str(averagedist))
        # tot += ((((data[i + 1:] - data[i]) ** 2).sum(1)) ** .5).sum()
        # avg = tot / ((data.shape[0] - 1) * (data.shape[0]) / 2.)

    averagedistance = np.mean(averagedist)
    print("the average precision: " + str(averagedistance))

    #        print(average_precision_score(ytrue.flatten(), rayholder[15].flatten()))
    #     for i in range(len(rayholder[0])):          # run through all elements along array
    #         #print("iteration number",i)
    #         #print("new iteration through rayholder", i)
    #         mx = idealx-rayholder[14][i]
    #         ny = idealy-rayholder[15][i]
    #         oz = idealz-rayholder[16][i]
    #
    # print("rayholder after")
    # #print(rayobjectreturn)
    return 0


def deray_and_rebuild_xwigner(raysobject, wignerobject):
    restoredwignerobject = []

    for rayholder in raysobject:  # for each function row of rays
        # print("rayholder before")
        # print(rayholder)
        # print("new line")
        rayholder = np.stack(
            (rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5], rayholder[6],
             # 1* Amplitude     # 3* spatial origins of ray, 3* intersection point on mirror
             rayholder[7], rayholder[8], rayholder[9], rayholder[10]
             # 3* normal of the surface  1* angle between
             , rayholder[11], rayholder[12], rayholder[13]  # 3* reflected rays
             , rayholder[14], rayholder[15], rayholder[
                 16],))  # 3* intersection points with screen                                                                                   #

        # print("rayholder shape is:")
        # print(rayholder.shape)
        for i in range(len(rayholder[0])):  # run through all elements along array
            # print("iteration number",i)

            location = np.array([rayholder[14][i], rayholder[15][i], rayholder[16][i]])
            xy = rayholder[12][i] / rayholder[11][i]
            xz = rayholder[13][i] / rayholder[11][i]
            zplusy = rayholder[12] + rayholder[13]
            xyamp = rayholder[12] / zplusy
            xzamp = rayholder[13] / zplusy

            wignerblock = np.array([[rayholder[0] * xzamp], [rayholder[16]], [xz]])
            wignerblock = np.array([[rayholder[0] * xyamp], [rayholder[15]], [xy]])

            # rayholder[11][i] = r[0]
            # rayholder[12][i] = r[1]
            # rayholder[13][i] = r[2]

        restoredwignerobject.append(wignerblock)
    print("rayholder shape is:")

    restoredwignerobject = wignerobject

    return


def error_test():
    return 0


##################################################################################################################################
######################################### PROGRAM RUNS FROM HERE #################################################################
##################################################################################################################################

mirrorfunction, mirrorinterpolator = create_interpolated_mirror()
# print("the mirror distance function is")
# print(mirrorobject[2])

# plot3dto2d(mirrorobject[0],mirrorobject[1],mirrorobject[2])

lightsource = light_source_function()  # returns x , y and the Function value/amplitude

# print(xaxisline.shape)
# print(yaxisline.shape)

# plot3dto2d(functionobject[0],functionobject[1],functionobject[2])   #xyz of function input

zwignerfunction = wignerize_each_function(lightsource)  # wigner amp, space, frequency

# wignerobject = abs(wignerobject)

plot_3d_to_2d(zwignerfunction[0][5][1], zwignerfunction[0][5][2], zwignerfunction[0][5][0])  # pass in space, phase and wigner amplitude

zwignertransformedfunction = ray_transforms(zwignerfunction, 50)

# plot3dto2d(wignerobjecttrans[0][5][1],wignerobjecttrans[0][5][2],wignerobjecttrans[0][5][0])   #pass in space, phase and wigner
reversefunction = integrate_intensity_wig(zwignerfunction, lightsource)
plot_gridata(reversefunction)

rayobject = ray_propogation(zwignerfunction, zwignertransformedfunction, lightsource, mirrorinterpolator, mirrorfunction, screen_function)

error_value_calc(rayobject, 0)

print("the length of ray object is")
print(len(rayobject))
print(len(rayobject[3]))

# plot3dto2d(screenobject[0],screenobject[1],screenobject[2])
# plot3dto2d(mirrorobject[0],mirrorobject[1],mirrorobject[2])

plot_scatter(rayobject, 14, 15, 16)  # plot the locations where the reflected rays hit the screen
# plotscatter(rayobject,4,5,6)   #plot the locations where the reflected rays hit the screen

# plotquiver3dobject(rayobject,14,15,16,11,12,13) #plot the vectors at the locations wehre the reflected rays hit the screen

# plotquiver3d(rayobject[6][1], rayobject[6][2], rayobject[6][3], rayobject[6][4], rayobject[6][5], rayobject[6][6])   #where rays meet mirror
# plotquiver3d(rayobject[6][4], rayobject[6][5], rayobject[6][6], rayobject[6][11], rayobject[6][12], rayobject[6][13])    #reflected rays
# plotquiver3d(rayobject[6][4], rayobject[6][5], rayobject[6][6], rayobject[6][14], rayobject[6][15], rayobject[6][16]) #where rays meet screen  ??????

error_value_calc(rayobject, 0)

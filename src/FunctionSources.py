import numpy as np
from scipy import interpolate
import math
from numpy import sin, cos


def rotation_matrices(theta, direction):
    Rx = [[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]]
    Ry = [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
    Rz = [[cos(theta), -sin(theta), 0], [sin(theta), cos(theta), 0], [0, 0, 1]]

    if direction == 'x':
        Rreturn = Rx
    elif direction == 'y':
        Rreturn = Ry
    else:
        Rreturn = Rz

    return Rreturn


def rotate(x, y, z, angle, direction):
    ravelx = np.ravel(x)  # ravel the X matrix to become 1D
    ravely = np.ravel(y)  # ravel the Y matrix to become 1D
    ravelz = np.ravel(z)
    XYZ = np.stack((ravelx, ravely, ravelz))  # stack the two 1D matrices and create a 2xN matrix X and then Y
    # print(XY.shape)

    radian = math.radians(angle)
    transfravelxyz = np.matmul(rotation_matrices(radian, direction),
                               XYZ)  # multiply matrix 2x2 raymatrix by 2xN xy matrix
    # print(transfravelxy.shape)

    xtransfered = transfravelxyz[0]  # unstack into X and Y
    ytransfered = transfravelxyz[1]
    ztransfered = transfravelxyz[2]

    xtransfered = np.reshape(xtransfered, (x.shape))  # reshape to original state
    ytransfered = np.reshape(ytransfered, (y.shape))
    ztransfered = np.reshape(ztransfered, (z.shape))

    return (xtransfered, ytransfered, ztransfered)


def light_source_function():
    x, y = np.meshgrid(np.linspace(-10, 10, 11), np.linspace(-10, 10, 11))     # function space and parameters
    twodsquarewave = np.where(abs(x) <= 4, 1, 0) & np.where(abs(y) <= 3, 1, 0)

    # print("x function dimension is")
    # print(x.shape)
    # print("y function dimension is")
    # print(y.shape)

    raveledx, raveledy, raveledfunc = np.ravel(x), np.ravel(y), np.ravel(twodsquarewave)
    xdimension, ydimension = twodsquarewave.shape[0], twodsquarewave.shape[1]
    functiondata = np.stack((raveledx, raveledy, raveledfunc))
    reshapedfunctiondata = np.reshape(functiondata, (3, xdimension, ydimension))

    return reshapedfunctiondata  # x, z , amplitude


def screen_function():
    z, y = np.meshgrid(np.linspace(-250, -600, 81), np.linspace(-150, 150, 81))
    x = z * 0 + y * 0 - 450
    interporlatedscreenfunction = interpolate.interp2d(z[0, :], y[:, 0], x, kind='cubic')
    screenfunction = np.stack((x, y, z))
    return screenfunction, interporlatedscreenfunction


def create_interpolated_mirror():
    xdenominator = -600
    ydenominator = -600

    x, y = np.meshgrid(np.linspace(-150, 150, 50), np.linspace(-150, 150, 50))
    z = (x ** 2) / xdenominator + (y ** 2) / ydenominator
    zoverlay = np.zeros(x.shape)
    z = z + zoverlay
    interpolatedmirror = interpolate.interp2d(x[0, :], y[:, 0], z, kind='cubic')

    zinterp = interpolatedmirror(x[0, :], y[:, 0])

    x, y, zrotate = rotate(x, y, zinterp, 15, 'y')
    zrotate = zrotate + 450
    interpolatedmirror = interpolate.interp2d(x[0, :], y[:, 0], zrotate, kind='cubic')
    print("shape of zrotate is:" + str(zrotate.shape))

    z0 = np.zeros(zinterp.shape)
    zdist = zrotate - z0
    rx = np.ravel(x)
    ry = np.ravel(y)
    zrotate = np.ravel(zrotate)
    zdist = np.ravel(zdist)
    mirrorobject = np.stack((rx, ry, zrotate, zdist))
    mirrorobject = np.reshape(mirrorobject, (4, len(x), len(y)))

    return mirrorobject, interpolatedmirror

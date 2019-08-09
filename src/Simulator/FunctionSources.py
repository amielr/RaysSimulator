import numpy as np
from scipy import interpolate
import math
import json

with open('config.json') as config_file:
    config = json.load(config_file)


def rotation_matrices(angle, direction):
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


def rotate(x, y, z, angle, direction):
    flattenX, flattenY, flattenZ = np.ravel(x), np.ravel(y), np.ravel(z)
    XYZ = np.stack((flattenX, flattenY, flattenZ))

    radian = math.radians(angle)

    transfravelxyz = np.matmul(rotation_matrices(radian, direction), XYZ)

    xtransfered = transfravelxyz[0]
    ytransfered = transfravelxyz[1]
    ztransfered = transfravelxyz[2]

    xtransfered = np.reshape(xtransfered, x.shape)
    ytransfered = np.reshape(ytransfered, y.shape)
    ztransfered = np.reshape(ztransfered, z.shape)

    return xtransfered, ytransfered, ztransfered


def light_source_function():
    x, y = np.meshgrid(np.linspace(-10, 10, config["lightSourceDensity"]),
                       np.linspace(-10, 10, config["lightSourceDensity"]))

    pulse2d = np.where(abs(x) <= 4, 1, 0) & np.where(abs(y) <= 3, 1, 0)

    return np.stack((x, y, pulse2d))


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

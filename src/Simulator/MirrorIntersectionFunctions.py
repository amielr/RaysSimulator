import numpy as np
import math
from numpy import linalg as LA


def max_min(mirrorfunction):
    maxmin = np.array(([np.max(mirrorfunction[0]), np.min(mirrorfunction[0])], [np.max(mirrorfunction[1]), np.min(mirrorfunction[1])]))
    return maxmin


def dot_product(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def angle(v1, v2):
    return math.acos(dot_product(v1, v2) / (length(v1) * length(v2)))


def length(v):
    return math.sqrt(dot_product(v, v))


def return_vector_properties(allRaysFromLine, rayNumber):
    xStart = allRaysFromLine[1][rayNumber]
    xEnd = allRaysFromLine[4][rayNumber]
    yStart = allRaysFromLine[2][rayNumber]
    yEnd = allRaysFromLine[5][rayNumber]
    zStart = allRaysFromLine[3][rayNumber]
    zEnd = allRaysFromLine[6][rayNumber]

    mx = xEnd - xStart
    ny = yEnd - yStart
    oz = zEnd - zStart

    return xStart, yStart, zStart, mx, ny, oz


def is_ray_in_mirror_bounds(allRaysFromLine, i, mirrorBorders):
    if (mirrorBorders[0, 0] > allRaysFromLine[4][i] > mirrorBorders[0, 1] and
            mirrorBorders[1, 0] > allRaysFromLine[5][i] > mirrorBorders[1, 1]):
        return True
    else:
        return False


def iterate_till_error_reached(error, mirror_interp, vectorProperties):

    xPoint, yPoint, zPoint, xDirection, yDirection, zDirection = vectorProperties

    top = 1
    bottom = 0
    delta = (top - bottom) / 2
    checkValue = 10000
    checkpoint = delta

    xCheckLocation = xPoint + xDirection * checkpoint
    yCheckLocation = yPoint + yDirection * checkpoint

    zint = mirror_interp(xCheckLocation, yCheckLocation)
    while checkValue > error:
        zray = zPoint + zDirection * checkpoint

        if zray < zint:
            bottom = bottom + delta
            delta = (top - bottom) / 2
            checkpoint = bottom + delta

        else:
            top = top - delta
            delta = (top - bottom) / 2
            checkpoint = top - delta

        xCheckLocation = xPoint + xDirection * checkpoint
        yCheckLocation = yPoint + yDirection * checkpoint

        zint = mirror_interp(xCheckLocation, yCheckLocation)

        zray = zPoint + zDirection * checkpoint
        checkValue = abs(zray - zint)
    return checkpoint


def mirror_intersection_points(allRaysFromLine, rayNumber, checkPoint):

    xPoint, yPoint, zPoint, xDirection, yDirection, zDirection = return_vector_properties(allRaysFromLine, rayNumber)

    xMirrorpoint = xPoint + xDirection * checkPoint
    yMirrorpoint = yPoint + yDirection * checkPoint
    zMirrorpoint = zPoint + zDirection * checkPoint

    return xMirrorpoint, yMirrorpoint, zMirrorpoint


def planes_of_mirror_intersections(allRaysFromLine, rayNumber, mirrorInterpolator):
    dx = 0.2
    dy = 0.2

    x = allRaysFromLine[4][rayNumber]  # get the xyz coordinates of intersection with mirror
    y = allRaysFromLine[5][rayNumber]
    z = allRaysFromLine[6][rayNumber]

    p1x = x  # create triangulation points
    p1y = y + dy * np.sqrt(2)  # In order to be able to calculate reflection normal
    p2x = x + dx
    p2y = y - dy
    p3x = x - dx
    p3y = y - dy

    p1z = mirrorInterpolator(p1x, p1y)  # get equivelant z points of interpelation points
    p2z = mirrorInterpolator(p2x, p2y)
    p3z = mirrorInterpolator(p3x, p3y)

    p1 = np.array([p1x, p1y, float(p1z)])
    p2 = np.array([p2x, p2y, float(p2z)])
    p3 = np.array([p3x, p3y, float(p3z)])
    v1 = p3 - p1
    v2 = p2 - p1
    cp = np.cross(v2, v1)
    cp = cp / LA.norm(cp)
    return cp


def angle_of_ray_with_mirror(allRaysFromLine, rayNumber):
    x = allRaysFromLine[4][rayNumber]
    y = allRaysFromLine[5][rayNumber]
    z = allRaysFromLine[6][rayNumber]

    a = allRaysFromLine[7][rayNumber]
    b = allRaysFromLine[8][rayNumber]
    c = allRaysFromLine[9][rayNumber]
    v1 = np.array([x, y, z])
    v2 = np.array([a, b, c])
    allRaysFromLine[10][rayNumber] = angle(v1, v2)
    return


def reflected_vector_from_mirror(allRaysFromLine, rayNumber):
    x = allRaysFromLine[4][rayNumber] - allRaysFromLine[1][rayNumber]
    y = allRaysFromLine[5][rayNumber] - allRaysFromLine[2][rayNumber]
    z = allRaysFromLine[6][rayNumber] - allRaysFromLine[3][rayNumber]
    d = np.array([x, y, z])
    n = np.array([allRaysFromLine[7][rayNumber], allRaysFromLine[8][rayNumber], allRaysFromLine[9][rayNumber]])

    ndot = dot_product(d, n)
    r = d - 2 * (ndot * n)
    r = r / LA.norm(r)
    r = r * 100

    allRaysFromLine[11][rayNumber] = r[0]
    allRaysFromLine[12][rayNumber] = r[1]
    allRaysFromLine[13][rayNumber] = r[2]

    return r[0], r[1], r[2]

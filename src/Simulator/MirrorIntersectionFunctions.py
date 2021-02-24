import numpy as np

from src.Simulator.Vector import *
from src.Simulator.Ray import *


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
    v = np.array([mx, ny, oz])
    v = v * (1 / length(v))

    return np.array([xStart, yStart, zStart]), v


def is_ray_in_mirror_bounds(mirrorHitPoint, mirrorBorders):
    return mirrorBorders[0] < getX(mirrorHitPoint) < mirrorBorders[2] and\
           mirrorBorders[1] < getY(mirrorHitPoint) < mirrorBorders[3]


def get_ray_mirror_intersection_point(wantedError, mirror_interp, ray):
    checkpointLocation = getOrigin(ray)
    xLocation = getX(checkpointLocation)
    yLocation = getY(checkpointLocation)
    currentZ = mirror_interp(xLocation, yLocation)
    error = currentZ - getZ(checkpointLocation)

    while abs(error) > wantedError:
        checkpointLocation = checkpointLocation + getDirection(ray) * error
        currentZ = mirror_interp(getX(checkpointLocation), getY(checkpointLocation))
        error = currentZ - getZ(checkpointLocation)

    return checkpointLocation


def get_reflected_ray_from_mirror(mirrorHitPoint, mirrorInterpolator, ray):
    plane_normal = generate_plane_normal(mirrorHitPoint, mirrorInterpolator)

    reflectedRayDirection = get_reflected_direction(getDirection(ray), plane_normal)

    return reflectedRayDirection


def generate_plane_normal(mirrorHitPoint, mirrorInterpolator):
    dx = 0.2
    dy = 0.2

    x = getX(mirrorHitPoint)
    y = getY(mirrorHitPoint)

    p1x = x  # create triangulation points
    p1y = y + dy * np.sqrt(2)  # In order to be able to calculate reflection normal
    p2x = x + dx
    p2y = y - dy
    p3x = x - dx
    p3y = y - dy

    p1z = mirrorInterpolator(p1x, p1y)  # get equivelant z points of interpelation points
    p2z = mirrorInterpolator(p2x, p2y)
    p3z = mirrorInterpolator(p3x, p3y)

    p1 = np.array([p1x, p1y, p1z])
    p2 = np.array([p2x, p2y, p2z])
    p3 = np.array([p3x, p3y, p3z])
    v1 = p2 - p1
    v2 = p3 - p1
    cp = np.cross(v1, v2)
    normalizedcp = cp /np.linalg.norm(cp)

    return normalizedcp


def get_reflected_direction(direction, planeNormal):
    # ndot = direction.dot_product(planeNormal)
    ndot = np.dot(direction, planeNormal)
    reflectedRayDirection = direction - (2 * ndot) * planeNormal

    return reflectedRayDirection

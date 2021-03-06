import numpy as np

from src.Simulator.Vector import Vector


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
    v = Vector(mx, ny, oz)
    v = v * (1 / v.length())

    return Vector(xStart, yStart, zStart), v


def is_ray_in_mirror_bounds(mirrorHitPoint, mirrorBorders):
    return mirrorBorders[0, 0] > mirrorHitPoint.getX() > mirrorBorders[0, 1] and\
           mirrorBorders[1, 0] > mirrorHitPoint.getY() > mirrorBorders[1, 1]


def get_ray_mirror_intersection_point(wantedError, mirror_interp, ray):
    checkpointLocation = ray.getOrigin()
    currentZ = mirror_interp(checkpointLocation.getX(), checkpointLocation.getY())
    error = currentZ - checkpointLocation.getZ()

    while abs(error) > wantedError:
        checkpointLocation = checkpointLocation + ray.getDirection() * error
        currentZ = mirror_interp(checkpointLocation.getX(), checkpointLocation.getY())
        error = currentZ - checkpointLocation.getZ()

    return checkpointLocation


def get_reflected_ray_from_mirror(mirrorHitPoint, mirrorInterpolator, ray):
    plane_normal = generate_plane_normal(mirrorHitPoint, mirrorInterpolator)

    reflectedRayDirection = get_reflected_direction(ray.getDirection(), plane_normal)

    return reflectedRayDirection


def generate_plane_normal(mirrorHitPoint, mirrorInterpolator):
    dx = 0.2
    dy = 0.2

    x = mirrorHitPoint.getX()
    y = mirrorHitPoint.getY()

    p1x = x  # create triangulation points
    p1y = y + dy * np.sqrt(2)  # In order to be able to calculate reflection normal
    p2x = x + dx
    p2y = y - dy
    p3x = x - dx
    p3y = y - dy

    p1z = mirrorInterpolator(p1x, p1y)  # get equivelant z points of interpelation points
    p2z = mirrorInterpolator(p2x, p2y)
    p3z = mirrorInterpolator(p3x, p3y)

    p1 = Vector(p1x, p1y, p1z)
    p2 = Vector(p2x, p2y, p2z)
    p3 = Vector(p3x, p3y, p3z)
    v1 = p3 - p1
    v2 = p2 - p1
    cp = v2.cross(v1)
    cp = cp * (1 / cp.length())
    return cp


def get_reflected_direction(direction, planeNormal):
    ndot = direction.dot_product(planeNormal)
    reflectedRayDirection = direction - planeNormal * (2 * ndot)

    return reflectedRayDirection

import json
import numpy as np
from numba import njit, jit, prange, vectorize

from src.Simulator.MirrorIntersectionFunctions import *
from src.Simulator.Ray import Ray

with open('config.json') as config_file:
    config = json.load(config_file)


def line_plane_collision(planeNormal, ray, epsilon=1e-6):
    perpendicularFactor = planeNormal.getDirection().dot_product(ray.getDirection())
    if abs(perpendicularFactor) < epsilon:
        return Vector(0, 0, 0)

    w = planeNormal.getOrigin() - ray.getOrigin()
    t = planeNormal.getDirection().dot_product(w) / perpendicularFactor
    return ray.getOrigin() + ray.getDirection() * t


# @jit(parallel=True)
def build_intersections_with_mirror(rayList, mirrorInterpolator, mirrorBorders, errorValue):
    reflectedRayList = np.array([Ray() for x in range(rayList.size)])

    for rayIndex in range(rayList.size):
        ray = rayList[rayIndex]
        mirrorHitPoint = get_ray_mirror_intersection_point(errorValue, mirrorInterpolator, ray)

        if is_ray_in_mirror_bounds(mirrorHitPoint, mirrorBorders):
            reflectedRayDirection = get_reflected_ray_from_mirror(mirrorHitPoint, mirrorInterpolator, ray)

            reflectedRay = Ray(mirrorHitPoint, reflectedRayDirection, ray.getAmplitude())
            reflectedRayList[rayIndex] = reflectedRay

    # print("end mirror intersection")

    nonZeroReflectedRays = reflectedRayList[[ray.getAmplitude() > 0 for ray in reflectedRayList]]

    return nonZeroReflectedRays


# @vectorize(['Ray(Ray)'], target='cuda')
def intersect_with_mirror_parrallelly(ray, mirrorInterpolator, mirrorBorders, errorValue):
    mirrorHitPoint = get_ray_mirror_intersection_point(errorValue, mirrorInterpolator, ray)

    if is_ray_in_mirror_bounds(mirrorHitPoint, mirrorBorders):
        reflectedRayDirection = get_reflected_ray_from_mirror(mirrorHitPoint, mirrorInterpolator, ray)
        return Ray(mirrorHitPoint, reflectedRayDirection, ray.getAmplitude())

    return None


# @jit(parallel=True)
def build_intersections_with_screen(rayList, screenNormal):
    raysAtScreenList = np.array([Ray() for x in range(rayList.size)])

    for rayIndex in range(rayList.size):
        ray = rayList[rayIndex]
        screenPoints = line_plane_collision(screenNormal, ray)
        rayAtScreen = Ray(screenPoints, ray.getDirection(), ray.getAmplitude())
        raysAtScreenList[rayIndex] = rayAtScreen
    return raysAtScreenList


def ray_propogation(rayList, mirrorInterpolator, mirrorBorders):
    reflectedRayList = build_intersections_with_mirror(rayList, mirrorInterpolator, mirrorBorders,
                                                       config["mirrorErrorValue"])

    screenNormal = Ray(Vector(config["xScreenLocation"],
                              config["yScreenLocation"],
                              config["zScreenLocation"]),
                       Vector(1, 0, 0), 1)

    return build_intersections_with_screen(reflectedRayList, screenNormal)

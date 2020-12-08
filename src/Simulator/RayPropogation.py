import json
import numpy as np
from numba import njit, jit, prange, vectorize, jitclass

from src.Simulator.MirrorIntersectionFunctions import *
from src.Simulator.Ray import *

with open('../config.json') as config_file:
    config = json.load(config_file)


# @njit(parallel=True)
def line_plane_collision(planeNormal, ray, epsilon=1e-6):
    # print("we are in line plane collision")
    # print(planeNormal.dtype)
    perpendicularFactor = np.dot(getDirection(planeNormal), (getDirection(ray)))
    if abs(perpendicularFactor) < epsilon:
        return np.array([0, 0, 0])

    w = getOrigin(planeNormal) - getOrigin(ray)
    t = np.dot(getDirection(planeNormal), w) / perpendicularFactor
    return getOrigin(ray) + getDirection(ray) * t


# @njit(parallel=True)
def build_intersections_with_mirror(rayList, mirrorInterpolator, mirrorBorders, errorValue):
    reflectedRayList = []
    for x in range(len(rayList)):
        reflectedRayList.append(np.array([[origin], [direction], amplitude]))  #this could cause and error
    # print(rayList.size, "our raylist size")
    # print(len(rayList))
    # print(len(rayList[0]))
    for rayIndex in range(len(rayList)):
        ray = rayList[rayIndex-1]
        mirrorHitPoint = get_ray_mirror_intersection_point(errorValue, mirrorInterpolator, ray)

        if is_ray_in_mirror_bounds(mirrorHitPoint, mirrorBorders):
            reflectedRayDirection = get_reflected_ray_from_mirror(mirrorHitPoint, mirrorInterpolator, ray)

            reflectedRay = np.array([mirrorHitPoint], [reflectedRayDirection], getAmplitude(ray))
            reflectedRayList[rayIndex] = reflectedRay

    print("end mirror intersection")
    print(len(reflectedRayList))
    nonZeroReflectedRays = []
    for rayIndex in range(len(reflectedRayList)):
        ray = rayList[rayIndex-1]
        #print(getAmplitude(ray))
        if getAmplitude(ray) > 0:
            nonZeroReflectedRays.append(ray)
    print("our nonzeroreflectedrays: ", len(nonZeroReflectedRays))
    return nonZeroReflectedRays


# @vectorize(['Ray(Ray)'], target='cuda')
def intersect_with_mirror_parrallelly(ray, mirrorInterpolator, mirrorBorders, errorValue):
    mirrorHitPoint = get_ray_mirror_intersection_point(errorValue, mirrorInterpolator, ray)

    if is_ray_in_mirror_bounds(mirrorHitPoint, mirrorBorders):
        reflectedRayDirection = get_reflected_ray_from_mirror(mirrorHitPoint, mirrorInterpolator, ray)
        return np.array([mirrorHitPoint], [reflectedRayDirection], getAmplitude(ray))

    return None


# @njit(parallel=True)
def build_intersections_with_screen(rayList, screenNormal):
    raysAtScreenList = np.empty((3, 3, 0))
    # for x in range(len(rayList)):
    #    raysAtScreenList.append(np.array([origin], [direction], [amplitude]))

    for rayIndex in range(len(rayList)):
        ray = rayList[rayIndex]
        screenPoints = line_plane_collision(screenNormal, ray)
        gottenDirection = getDirection(ray)
        gottenAmplitude = getAmplitude(ray)
        rayAtScreen = np.array([screenPoints, gottenDirection, [gottenAmplitude, 0, 0]])
        raysAtScreenList = np.dstack((raysAtScreenList, rayAtScreen))
        # print(raysAtScreenList.shape)
    raysAtScreenList = np.moveaxis(raysAtScreenList, -1, 0)
    print(raysAtScreenList.shape)
    return raysAtScreenList


def ray_propogation(rayList, mirrorInterpolator, mirrorBorders):
    reflectedRayList = build_intersections_with_mirror(rayList, mirrorInterpolator, mirrorBorders,
                                                       config["mirrorErrorValue"])

    screenNormal = np.array([[config["xScreenLocation"],
                              config["yScreenLocation"],
                              config["zScreenLocation"]], [1, 0, 0]], dtype=np.float_)

    print(screenNormal.dtype)
    # Ray(Vector(config["xScreenLocation"],
    #                           config["yScreenLocation"],
    #                           config["zScreenLocation"]),
    #                    Vector(1, 0, 0), 1)

    return build_intersections_with_screen(reflectedRayList, screenNormal)

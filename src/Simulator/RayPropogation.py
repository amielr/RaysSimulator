import json
import numpy as np
from numba import njit, jit, prange, vectorize
from numba.typed import List



from src.Simulator.MirrorIntersectionFunctions import *
from src.Simulator.PlotFunctions import *
from src.Simulator.Ray import *

with open('./config.json') as config_file:
    config = json.load(config_file)


@njit()
def line_plane_collision(planeNormal, ray, epsilon=1e-6):
    # print("we are in line plane collision")
    # print(planeNormal.dtype)

    planeNormalDirection = planeNormal[1]  # getDirection(planeNormal)
    rayDirection = ray[1]  # getDirection(ray)
    perpendicularFactor = np.dot(planeNormalDirection, rayDirection)
    if abs(perpendicularFactor) < epsilon:
        return np.array([0, 0, 0], dtype=np.float_)

    w = planeNormal[0] - ray[0]      #getOrigin(planeNormal) - getOrigin(ray)
    t = np.dot(planeNormal[1], w) / perpendicularFactor # np.dot(getDirection(planeNormal), w) / perpendicularFactor
    return ray[0] + ray[1] * t # getOrigin(ray) + getDirection(ray) * t


# @njit(parallel=True)
def build_intersections_with_mirror(rayList, mirrorInterpolator, mirrorBorders, errorValue):
    #reflectedRayList = List()
    reflectedRayList = []
    #for x in range(len(rayList)):
    #    reflectedRayList.append(np.array([[origin], [direction], [amplitude]]))  #this could cause and error
        # reflectedRayListnp
    # print(rayList.size, "our raylist size")
    # print(len(rayList))
    # print(len(rayList[0]))
    for rayIndex in range(len(rayList)):
        ray = rayList[rayIndex]
        mirrorHitPoint = get_ray_mirror_intersection_point(errorValue, mirrorInterpolator, ray)

        if is_ray_in_mirror_bounds(mirrorHitPoint, mirrorBorders):
            reflectedRayDirection = get_reflected_ray_from_mirror(mirrorHitPoint, mirrorInterpolator, ray)
            Amplitude = np.array([getAmplitude(ray), 0, 0])
            reflectedRay = np.array([mirrorHitPoint, reflectedRayDirection, Amplitude])
            reflectedRayList.append(reflectedRay)
            # reflectedRayList[rayIndex] = reflectedRay
    # reflectedRayListnp = np.array(reflectedRayList)
    # plot_scatter(reflectedRayList)

    print("end mirror intersection")
    print(len(reflectedRayList))

    return reflectedRayList


# @vectorize(['Ray(Ray)'], target='cuda')
def intersect_with_mirror_parrallelly(ray, mirrorInterpolator, mirrorBorders, errorValue):
    mirrorHitPoint = get_ray_mirror_intersection_point(errorValue, mirrorInterpolator, ray)

    if is_ray_in_mirror_bounds(mirrorHitPoint, mirrorBorders):
        reflectedRayDirection = get_reflected_ray_from_mirror(mirrorHitPoint, mirrorInterpolator, ray)
        return np.array([mirrorHitPoint], [reflectedRayDirection], getAmplitude(ray))

    return None


#@njit(nogil=True)
def build_intersections_with_screen(rayList, screenNormal):
    raysAtScreenListHolder = np.zeros((3, 3), dtype=np.float_)
    # for x in range(len(rayList)):
    #    raysAtScreenList.append(np.array([origin], [direction], [amplitude]))
    rayAtScreen = np.empty((3, 3), dtype=np.float_)
    raysAtScreenList = np.dstack((raysAtScreenListHolder, rayAtScreen))

    for rayIndex in range(len(rayList)):
        ray = rayList[rayIndex]
        screenPoints = line_plane_collision(screenNormal, ray)
        gottenDirection = ray[0] # getDirection(ray)
        gottenAmplitude = ray[2][0] #getAmplitude(ray)
        rayAtScreen[0], rayAtScreen[1], rayAtScreen[2] = screenPoints, gottenDirection, [gottenAmplitude, 0, 0]
        expandedRayAtScreen = np.expand_dims(rayAtScreen, 2)
        #print("our expandedRayAtScreen at screen are of shape", expandedRayAtScreen.shape, expandedRayAtScreen.dtype)
        #print("our rays at screen are of shape", rayAtScreen.shape, rayAtScreen.dtype)

        raysAtScreenList = np.dstack((raysAtScreenList, expandedRayAtScreen))
        #print(raysAtScreenList.shape)
    print(raysAtScreenList.shape)
    return raysAtScreenList


def ray_propogation(rayList, mirrorInterpolator, mirrorBorders):
    reflectedRayList = build_intersections_with_mirror(rayList, mirrorInterpolator, mirrorBorders,
                                                       config["mirrorErrorValue"])
    #plot_scatter(reflectedRayList)

    reflectedRayListnp = np.array(reflectedRayList, dtype=np.float_)

    screenNormal = np.array([[config["xScreenLocation"],
                              config["yScreenLocation"],
                              config["zScreenLocation"]], [1, 0, 0]], dtype=np.float_)

    print(screenNormal.shape, screenNormal.dtype)
    # Ray(Vector(config["xScreenLocation"],
    #                           config["yScreenLocation"],
    #                           config["zScreenLocation"]),
    #                    Vector(1, 0, 0), 1)

    raysArrivedAtScreen = build_intersections_with_screen(reflectedRayList, screenNormal)

    return np.moveaxis(raysArrivedAtScreen, -1, 0)


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


def remove_rays_of_zero_amplitude(nonZeroIndices, array):
    newArray = array[nonZeroIndices]
    return newArray


def create_rays_from_wigner_transform(rayObjectList, lineWignerTransform, functionValues, lineTranslatedWignerTransform,
                                      isrow):
    zDistance = 750
    repetitions = len(np.ravel(lineWignerTransform[0, 1]))

    for location in range(len(lineWignerTransform)):

        # for location in range(len(wignerrow)):
        #     print(wignerrow[location])

        Amplitudes = np.ravel(lineWignerTransform[location, 0])
        nonzeroindices = np.nonzero(Amplitudes)
        Amplitudes = remove_rays_of_zero_amplitude(nonzeroindices, Amplitudes)

        rowXOriginalLocations = np.ravel(lineWignerTransform[location, 1])
        rowXOriginalLocations = remove_rays_of_zero_amplitude(nonzeroindices, rowXOriginalLocations)

        z0 = np.zeros(repetitions)
        z0 = remove_rays_of_zero_amplitude(nonzeroindices, z0)

        rowXTranslatedLocations = np.ravel(lineTranslatedWignerTransform[location, 1])
        rowXTranslatedLocations = remove_rays_of_zero_amplitude(nonzeroindices, rowXTranslatedLocations)

        zd = np.repeat(zDistance, repetitions)
        zd = remove_rays_of_zero_amplitude(nonzeroindices, zd)

        if isrow:
            yValue = functionValues[location, 1]
            rowYLocations = np.repeat(yValue, repetitions)
            rowYLocations = remove_rays_of_zero_amplitude(nonzeroindices, rowYLocations)

            rayPacket = np.stack(
                (Amplitudes, rowXOriginalLocations, rowYLocations, z0, rowXTranslatedLocations, rowYLocations, zd))
        else:
            yValue = functionValues[1, location]
            rowYLocations = np.repeat(yValue, repetitions)
            rowYLocations = remove_rays_of_zero_amplitude(nonzeroindices, rowYLocations)
            rayPacket = np.stack(
                (Amplitudes, rowYLocations, rowXOriginalLocations, z0, rowYLocations, rowXTranslatedLocations, zd))

        if len(rayPacket) != 0:
            rayObjectList.append(rayPacket)
    return rayObjectList


def prepare_wigner_data_for_deconstruction_to_rays(wignerTransform, translatedWignerTransform, lightSourceCoordinates):
    return wignerTransform, translatedWignerTransform, lightSourceCoordinates


def convert_rays_to_objects(stackedRays):
    rayList = np.empty(len(stackedRays) * len(stackedRays[0]))
    for row in range(len(stackedRays)):
        for column in range(len(stackedRays[row])):
            singleRow = stackedRays[row]
            rayNumber = singleRow[column]
            origin, direction = return_vector_properties(singleRow, rayNumber)
            rayExample = Ray(origin, direction, singleRow[0, rayNumber])
            rayList[row * len(stackedRays[row]) + column] = rayExample
    return rayList


def build_ray_object_list_from_wigner(wignerTransform, translatedWignerTransform, lightSource):
    rowsWignerTransform, rowsTranslatedWignerTransform, lightSourceXCoordinates = \
        prepare_wigner_data_for_deconstruction_to_rays(wignerTransform[0], translatedWignerTransform[0], lightSource[0])
    columnsWignerTransform, columnsTranslatedWignerTransform, lightSourceYCoordinates = \
        prepare_wigner_data_for_deconstruction_to_rays(wignerTransform[1], translatedWignerTransform[1], lightSource[1])
    isRow = True

    stacked_ray_list = []

    stacked_ray_list = create_rays_from_wigner_transform(stacked_ray_list, rowsWignerTransform, lightSourceYCoordinates,
                                                         rowsTranslatedWignerTransform, isRow)
    stacked_ray_list = create_rays_from_wigner_transform(stacked_ray_list, columnsWignerTransform,
                                                         lightSourceXCoordinates, columnsTranslatedWignerTransform,
                                                         not isRow)

    return convert_rays_to_objects(stacked_ray_list)


@jit(parallel=True)
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

    return reflectedRayList


@vectorize(['Ray(Ray)'], target='cuda')
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

from src.Simulator.MirrorIntersectionFunctions import *
from src.Simulator.Ray import Ray
import numpy as np
import numpy.ma as ma

import json

with open('config.json') as config_file:
    config = json.load(config_file)


def line_plane_collision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        a = np.empty((3,))
        a[:] = np.nan
        return a

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    psi = w + si * rayDirection + planePoint
    return psi


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
            print("finished row")
        else:
            yValue = functionValues[1, location]
            rowYLocations = np.repeat(yValue, repetitions)
            rowYLocations = remove_rays_of_zero_amplitude(nonzeroindices, rowYLocations)
            rayPacket = np.stack(
                (Amplitudes, rowYLocations, rowXOriginalLocations, z0, rowYLocations, rowXTranslatedLocations, zd))
            print("finished columns")

        if len(rayPacket) != 0:
            rayObjectList.append(rayPacket)
    return rayObjectList


def prepare_wigner_data_for_deconstruction_to_rays(wignerTransform, translatedWignerTransform, lightSourceCoordinates):
    return wignerTransform, translatedWignerTransform, lightSourceCoordinates


def convertRaysToObjects(stackedRays):
    print("length of stacked rays" + str(len(stackedRays)))
    ListOfRayObjects = []
    for singleRow in stackedRays:
        for rayNumber in range(len(singleRow[0])):
            origin, direction = return_vector_properties2(singleRow, rayNumber)
            # return_vector_properties(singleRow, rayNumber)
            # dx = singleRow[4, rayNumber] - singleRow[1, rayNumber]
            # dy = singleRow[5, rayNumber] - singleRow[2, rayNumber]
            # dz = singleRow[6, rayNumber] - singleRow[3, rayNumber]
            rayexample = Ray(origin, direction, singleRow[0, rayNumber])
            # print(rayexample.amplitude)
            ListOfRayObjects.append(rayexample)
    # print("lenght of list of ray objects")
    # print(len(ListOfRayObjects))
    # print(ListOfRayObjects[45].getAmplitude())
    # print(Ray.getNumberOfRays)
    # print(Ray.number_of_rays)
    return ListOfRayObjects


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

    print("number of rays are:")
    print(len(stacked_ray_list))
    ray_list = convertRaysToObjects(stacked_ray_list)

    return ray_list


def build_intersections_with_mirror(rayList, mirrorInterpolator, mirrorBorders):
    reflectedRayList = []
    errorValue = config["mirrorErrorValue"]

    for ray in rayList:
        mirrorHitPoint = get_ray_mirror_intersection_point(errorValue, mirrorInterpolator, ray)

        if is_ray_in_mirror_bounds(mirrorHitPoint, mirrorBorders):
            reflectedRayDirection = get_reflected_ray_from_mirror(mirrorHitPoint, mirrorInterpolator, ray)

            reflectedRay = Ray(mirrorHitPoint, reflectedRayDirection, ray.getAmplitude())
            reflectedRayList.append(reflectedRay)

    return reflectedRayList


def build_intersections_with_screen(raysobject):
    rayobjectreturn = []

    for rayholder in raysobject:  # for each function row of rays
        rayholder = np.stack(
            (rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5], rayholder[6],
             # 1* Amplitude     # 3* spatial origins of ray, 3* intersection point on mirror
             rayholder[7], rayholder[8], rayholder[9], rayholder[10]
             # 3* normal of the surface  1* angle between
             , rayholder[11], rayholder[12], rayholder[13],  # 3* reflected rays
             rayholder[11], rayholder[12], rayholder[13],))  # 3* intersection points with screen
        for i in range(len(rayholder[0])):  # run through all elements along array
            mx = rayholder[11][i]  # - rayholder[4][i]
            ny = rayholder[12][i]  # - rayholder[5][i]
            oz = rayholder[13][i]  # - rayholder[6][i]

            raydirection = np.array((mx, ny, oz))
            raypoint = np.array((rayholder[4][i], rayholder[5][i], rayholder[6][i]))

            planenormal = np.array([1, 0, 0])
            planepoint = np.array([config["xScreenLocation"], config["yScreenLocation"], config["zScreenLocation"]])

            intersection = line_plane_collision(planenormal, planepoint, raydirection, raypoint)

            rayholder[14][i] = intersection[0]
            rayholder[15][i] = intersection[1]
            rayholder[16][i] = intersection[2]

        rayobjectreturn.append(rayholder)

    return rayobjectreturn


def ray_propogation(zwignerobject, zwignerobjecttrans, lightsource, mirrorInterpolator, mirrorBorders):
    rayObjectList = build_ray_object_list_from_wigner(zwignerobject, zwignerobjecttrans, lightsource)

    rayBundle = build_intersections_with_mirror(rayObjectList, mirrorInterpolator, mirrorBorders)

    rayBundle = build_intersections_with_screen(rayBundle)
    return rayBundle

import json
import numpy as np

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
    rayList = []
    for singleRow in stackedRays:
        for rayNumber in range(len(singleRow[0])):
            origin, direction = return_vector_properties(singleRow, rayNumber)
            # return_vector_properties(singleRow, rayNumber)
            # dx = singleRow[4, rayNumber] - singleRow[1, rayNumber]
            # dy = singleRow[5, rayNumber] - singleRow[2, rayNumber]
            # dz = singleRow[6, rayNumber] - singleRow[3, rayNumber]
            rayexample = Ray(origin, direction, singleRow[0, rayNumber])
            # print(rayexample.amplitude)
            rayList.append(rayexample)
    # print("lenght of list of ray objects")
    # print(len(rayList))
    # print(rayList[45].getAmplitude())
    # print(Ray.getNumberOfRays)
    # print(Ray.number_of_rays)
    print("number of rays are:")
    print(len(rayList))
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


def build_intersections_with_mirror(rayList, mirrorInterpolator, mirrorBorders):
    reflectedRayList = []
    errorValue = config["mirrorErrorValue"]

    for ray in rayList:
        mirrorHitPoint = get_ray_mirror_intersection_point(errorValue, mirrorInterpolator, ray)

        if is_ray_in_mirror_bounds(mirrorHitPoint, mirrorBorders):
            reflectedRayDirection = get_reflected_ray_from_mirror(mirrorHitPoint, mirrorInterpolator, ray)

            reflectedRay = Ray(mirrorHitPoint, reflectedRayDirection, ray.getAmplitude())
            reflectedRayList.append(reflectedRay)

    print("end mirror intersection")

    return reflectedRayList


def build_intersections_with_screen(rayList):
    screenNormal = Ray(Vector(config["xScreenLocation"],
                              config["yScreenLocation"],
                              config["zScreenLocation"]),
                       Vector(1, 0, 0), 1)

    raysAtScreenList = []

    for ray in rayList:

        screenPoints = line_plane_collision(screenNormal, ray)
        rayAtScreen = Ray(screenPoints, ray.getDirection(), ray.getAmplitude())
        raysAtScreenList.append(rayAtScreen)
    return raysAtScreenList


def ray_propogation(zwignerobject, zwignerobjecttrans, lightsource, mirrorInterpolator, mirrorBorders):
    rayList = build_ray_object_list_from_wigner(zwignerobject, zwignerobjecttrans, lightsource)

    reflectedRayList = build_intersections_with_mirror(rayList, mirrorInterpolator, mirrorBorders)

    return build_intersections_with_screen(reflectedRayList)

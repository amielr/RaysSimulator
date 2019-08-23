from src.Simulator.MirrorIntersectionFunctions import *
import numpy as np
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


def build_ray_matrices_from_wigner(wignertransform, translatedwignertransform, functionobject):
    zDist = 750
    rayobject = []

    rowsWignerTransform = wignertransform[0]
    columnsWignerTransform = wignertransform[1]

    rowsTranslatedWignerTransform = translatedwignertransform[0]
    columnsTranslatedWignerTransform = translatedwignertransform[1]

    functionXValues = functionobject[0]
    functionYValues = functionobject[1]

    repetitions = len(np.ravel(rowsWignerTransform[0, 1]))

    for i in range(len(rowsWignerTransform)):
        rowAmplitudes = np.ravel(rowsWignerTransform[i, 0])
        rowXOriginalLocations = np.ravel(rowsWignerTransform[i, 1])
        yValue = functionYValues[i, 1]
        rowYLocations = np.repeat(yValue, repetitions)
        z0 = np.zeros(repetitions)
        rowXTranslatedLocations = np.ravel(rowsTranslatedWignerTransform[i, 1])
        zd = np.repeat(zDist, repetitions)

        raypacket = np.stack((rowAmplitudes, rowXOriginalLocations, rowYLocations, z0, rowXTranslatedLocations, rowYLocations, zd))
        print("finished row")
        rayobject.append(raypacket)

    for i in range(len(columnsWignerTransform)):
        print(i)
        colAmplitudes = np.ravel(columnsWignerTransform[i, 0])
        colYOriginalLocations = np.ravel(columnsWignerTransform[i, 1])
        xValue = functionXValues[1, i]
        colXLocations = np.repeat(xValue, repetitions)
        z0 = np.zeros(repetitions)
        colYTranslatedLocations = np.ravel(columnsTranslatedWignerTransform[i, 1])
        zd = np.repeat(zDist, repetitions)

        raypacket = np.stack(
            (colAmplitudes, colXLocations, colYOriginalLocations, z0, colXLocations, colYTranslatedLocations, zd))
        print("finished column")
        rayobject.append(raypacket)
    return rayobject


def build_intersections_with_mirror(rayBundle, mirrorInterpolator, mirrorFunction):
    rayObjectReturn = []
    errorValue = config["mirrorErrorValue"]

    mirrorBorders = max_min(mirrorFunction)

    for allRaysFromLine in rayBundle:  # for each function row of rays
        removeelements = []

        allRaysFromLine = np.stack((allRaysFromLine[config["rayAmplitudeValue"]], allRaysFromLine[1], allRaysFromLine[2]
                                    , allRaysFromLine[3], allRaysFromLine[4], allRaysFromLine[5],
                                    allRaysFromLine[6], allRaysFromLine[4], allRaysFromLine[5], allRaysFromLine[5],
                                    allRaysFromLine[4], allRaysFromLine[5], allRaysFromLine[5], allRaysFromLine[5]))

        for rayNumber in range(len(allRaysFromLine[0])):  # run through all rays along array

            xPoint, yPoint, zPoint, xDirection, yDirection, zDirection = return_vector_properties(allRaysFromLine, rayNumber)

            checkPoint = iterate_till_error_reached(errorValue, mirrorInterpolator, [xPoint, yPoint, zPoint, xDirection, yDirection, zDirection])

            allRaysFromLine[4][rayNumber], allRaysFromLine[5][rayNumber], allRaysFromLine[6][rayNumber] = \
                mirror_intersection_points(allRaysFromLine, rayNumber, checkPoint)

            allRaysFromLine[7][rayNumber], allRaysFromLine[8][rayNumber], allRaysFromLine[9][rayNumber] = \
                planes_of_mirror_intersections(allRaysFromLine, rayNumber, mirrorInterpolator)

            allRaysFromLine[10][rayNumber] = angle_of_ray_with_mirror(allRaysFromLine, rayNumber)

            allRaysFromLine[11][rayNumber], allRaysFromLine[12][rayNumber], allRaysFromLine[13][rayNumber] = \
                reflected_vector_from_mirror(allRaysFromLine, rayNumber)

            if is_ray_in_mirror_bounds(allRaysFromLine, rayNumber, mirrorBorders):
                continue
            else:
                removeelements.append(rayNumber)
                continue

        rayholderbuild = np.delete(allRaysFromLine[:], np.s_[removeelements], 1)
        rayObjectReturn.append(rayholderbuild)

    return rayObjectReturn


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
            planepoint = np.array([-200, 0, 450])

            intersection = line_plane_collision(planenormal, planepoint, raydirection, raypoint)

            rayholder[14][i] = intersection[0]
            rayholder[15][i] = intersection[1]
            rayholder[16][i] = intersection[2]

        rayobjectreturn.append(rayholder)

    return rayobjectreturn


def ray_propogation(zwignerobject, zwignerobjecttrans, lightsource, mirrorInterpolator, mirrorobject):
    rayBundle = build_ray_matrices_from_wigner(zwignerobject, zwignerobjecttrans, lightsource)

    rayBundle = build_intersections_with_mirror(rayBundle, mirrorInterpolator, mirrorobject)

    rayBundle = build_intersections_with_screen(rayBundle)
    return rayBundle

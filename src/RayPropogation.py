from src.MirrorIntersectionFunctions import *
import numpy as np
import time


def line_plane_collision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        # raise RuntimeError("no intersection or line is within plane")
        a = np.empty((3,))
        a[:] = np.nan
        return a

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    psi = w + si * rayDirection + planePoint
    return psi


def build_ray_matrices_from_wigner(wignertransform, translatedwignertransform, functionobject):
    # b = Line3D(Point3D(1, 3, 4), Point3D(2, 2, 2))
    zDist = 750
    # transformedwignerobject = raytransforms(wignerobject,zdist)
    rayobject = []

    print("wignerobject shapes are the sames?")
    print(wignertransform.shape)

    rowsWignerTransform = wignertransform[0]
    columnsWignerTransform = wignertransform[1]

    rowsTranslatedWignerTransform = translatedwignertransform[0]
    columnsTranslatedWignerTransform = translatedwignertransform[1]

    functionXValues = functionobject[0]
    functionYValues = functionobject[1]

    print(translatedwignertransform.shape)
    print("rowswignertransform", np.shape(rowsWignerTransform[:, 1]))
    print("function shape is ", np.shape(functionXValues))
    # print(functionYValues)
    print("function object shape is: ", np.shape(functionobject))
    repetitions = len(np.ravel(rowsWignerTransform[0, 1]))
    print("The number of repetitions ", repetitions)

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
    errorValue = 0.1
    print("raybundle shape")
    print(np.shape(rayBundle))
    mirrorBorders = max_min(mirrorFunction)

    tb = time.time()
    ta = time.time()
    for allRaysFromLine in rayBundle:  # for each function row of rays
        removeelements = []

        print("rayholder before")
        print(ta - tb)
        print(np.shape(allRaysFromLine))
        tb = time.time()
        twb = time.time()

        allRaysFromLine = np.stack((allRaysFromLine[0], allRaysFromLine[1], allRaysFromLine[2], allRaysFromLine[3],
                                    allRaysFromLine[4], allRaysFromLine[5],
                                    allRaysFromLine[6], allRaysFromLine[4], allRaysFromLine[5], allRaysFromLine[5],
                                    allRaysFromLine[4], allRaysFromLine[5], allRaysFromLine[5], allRaysFromLine[5]))

        for rayNumber in range(len(allRaysFromLine[0])):  # run through all rays along array

            xPoint, yPoint, zPoint, xDirection, yDirection, zDirection = return_vector_properties(allRaysFromLine, rayNumber)

            checkPoint = iterate_till_error_reached(errorValue, mirrorInterpolator, [xPoint, yPoint, zPoint, xDirection, yDirection, zDirection])

            allRaysFromLine[4][rayNumber], allRaysFromLine[5][rayNumber], allRaysFromLine[6][rayNumber] = \
                mirror_intersection_points(allRaysFromLine, rayNumber, checkPoint)

            # print(rayholder[4][i], maxmin[0, 0])

            allRaysFromLine[7][rayNumber], allRaysFromLine[8][rayNumber], allRaysFromLine[9][rayNumber] = \
                planes_of_mirror_intersections(allRaysFromLine, rayNumber, mirrorInterpolator)

            allRaysFromLine[10][rayNumber] = angle_of_ray_with_mirror(allRaysFromLine, rayNumber)

            allRaysFromLine[11][rayNumber], allRaysFromLine[12][rayNumber], allRaysFromLine[13][rayNumber] = \
                reflected_vector_from_mirror(allRaysFromLine, rayNumber)

            if is_ray_in_mirror_bounds(allRaysFromLine, rayNumber, mirrorBorders):
                continue
            else:
                # allRaysFromLine[:][i] = 0
                removeelements.append(rayNumber)
                continue

        tsa = time.time()
        print("x", tsa - twb)

        ta = time.time()
        rayholderbuild = np.delete(allRaysFromLine[:], np.s_[removeelements], 1)
        rayObjectReturn.append(rayholderbuild)

    return rayObjectReturn


def build_intersections_with_screen(raysobject):
    print("rayholder before")
    # print(raysobject)
    rayobjectreturn = []

    for rayholder in raysobject:  # for each function row of rays
        # print("rayholder before")
        # print(rayholder)
        # print("new line")
        rayholder = np.stack(
            (rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5], rayholder[6],
             # 1* Amplitude     # 3* spatial origins of ray, 3* intersection point on mirror
             rayholder[7], rayholder[8], rayholder[9], rayholder[10]
             # 3* normal of the surface  1* angle between
             , rayholder[11], rayholder[12], rayholder[13]  # 3* reflected rays
             , rayholder[11], rayholder[12], rayholder[13],))  # 3* intersection points with screen
        for i in range(len(rayholder[0])):  # run through all elements along array
            # print("iteration number",i)
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
    print("rayholder after")
    # print(rayobjectreturn)
    return rayobjectreturn


def ray_propogation(zwignerobject, zwignerobjecttrans, lightsource, mirrorInterpolator, mirrorobject, screen_function):

    rayBundle = build_ray_matrices_from_wigner(zwignerobject, zwignerobjecttrans, lightsource)

    print("in the main after matrix built")
    print(np.shape(rayBundle))

    rayBundle = build_intersections_with_mirror(rayBundle, mirrorInterpolator, mirrorobject)

    print("in the main, the rayobject size is")
    print(rayBundle[0].shape)

    screenobject, screeninterp = screen_function()

    rayBundle = build_intersections_with_screen(rayBundle)
    return rayBundle

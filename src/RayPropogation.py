import numpy as np
import time
from numpy import linalg as LA
import math


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


def dot_product(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))


def length(v):
    return math.sqrt(dot_product(v, v))


def build_ray_matrices_from_wigner(wignerobject, transformedwignerobject, functionobject):
    # b = Line3D(Point3D(1, 3, 4), Point3D(2, 2, 2))
    zdist = 750
    # transformedwignerobject = raytransforms(wignerobject,zdist)
    rayobject = []
    # rayobject = np.array
    updatedravelamp = ()
    ravelspace = np.array
    ravelamplitude = np.array

    print("wignerobject shaspes are the sames?")
    print(wignerobject.shape)
    print(transformedwignerobject.shape)
    for x in range(0, len(wignerobject)):  # for rows and then columns\
        if x == 0:  # if rows then we need y values from function and x values from wigner
            direction = 1
        else:
            direction = 0  # if columns we need x values from function and y values from wigner

        for y in range(0, len(wignerobject[x])):  # for each row or column - iterate through all rows and then columns
            ravelspace = np.ravel(wignerobject[x, y, 1])  # ravel space matrix of wigner transform to become 1D
            print(ravelspace.shape)
            ravelamplitude = np.ravel(wignerobject[x, y, 0])
            # print(ravelspace)

            print("New iteration of updateravelamp")
            # print(updatedravelamp)
            ravelspacetrans = np.ravel(transformedwignerobject[x, y, 1])
            # print("translated x values")
            # print(ravelspacetrans)
            # print("original ravelspace")
            # print(ravelspace)
            # ravelspacetrans = transformedwignerobject[x,y,1]
            # print(ravelspacetrans)
            # print(ravelspace.shape)
            # print(ravelspacetrans.shape)

            z0 = np.zeros(ravelspace.shape)
            zd = np.repeat(zdist, len(ravelspace))
            if direction == 1:  # if rows then we need y values from function and x values from wigner
                perpvalues = np.repeat(functionobject[direction, y, 0],
                                       len(ravelspace))  # get the perpendicular location of space matrix

                raypacket = np.stack((ravelamplitude, ravelspace, perpvalues, z0, ravelspacetrans, perpvalues, zd))
                # function is built like: X then Y then Amplitude
                perpvaluestrans = 0
                # print("our perp axis values are")
                # print(perpvalues)

            else:  # if columns we need x values from function and y values from wigner
                perpvalues = np.repeat(functionobject[direction, 0, y],
                                       len(ravelspace))  # get the perpendicular location of space matrix
                raypacket = np.stack((ravelamplitude, perpvalues, ravelspace, z0, perpvalues, ravelspacetrans, zd))
                # raypacket = np.stack((perpvalues, ravelspace, z0, ravelspacetrans, perpvalues, zd))

                perpvaluestrans = 0
                # print("our perp axis values are")
                # print(perpvalues)
            # print(ravelspacetrans.shape,ravelspace.shape,perpvalues.shape,z0.shape,zd.shape)
            # raypacket = np.stack((ravelspace,perpvalues,z0,ravelspacetrans,perpvalues,zd))
            # print("the raypackets' shape is")
            # print(raypacket.shape)
            # rayarray = np.array([raypacket])
            # np.append(rayobject, raypacket)

            rayobject.append(raypacket)

            # functionspace= np.ravel(functionobject[1][])
            # z = mirrorinterpolator(perpspace,ravelspace)
            # print("our ray objects shape is")
            # print(len(rayobject))
            # print(rayobject.shape)
            # print("rayobject length is")
            # print(len(rayobject))

    return rayobject


def build_intersections_with_mirror(raysobject, mirrorinterp, mirrorobject):
    print(np.max(mirrorobject[0]))
    maxmin = np.array(
        ([np.max(mirrorobject[0]), np.min(mirrorobject[0])], [np.max(mirrorobject[1]), np.min(mirrorobject[1])]))
    print(maxmin)
    print("rayholder before mirror")
    print(np.shape(raysobject))
    rayobjectreturn = []
    print(np.shape(raysobject[0])[0])

    tb = time.time()
    ta = time.time()
    for rayholder in raysobject:  # for each function row of rays
        removeelements = []

        print("rayholder before")
        print(ta - tb)
        tb = time.time()

        # print(rayholder)
        # print("new line")
        # rayholder = rayobject[j]
        # print(rayholder)
        rayholderbuild = np.array
        # rayholderbuild = np.stack(
        #    (rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5], rayholder[6]))

        twb = time.time()
        for i in range(len(rayholder[0])):  # run through all elements along array
            # print("iteration number",i)
            # print("new iteration through rayholder", i)

            mx = rayholder[4][i] - rayholder[1][i]
            ny = rayholder[5][i] - rayholder[2][i]
            oz = rayholder[6][i] - rayholder[3][i]
            zint = 10
            zray = 1
            top = 1
            bottom = 0
            delta = (top - bottom) / 2
            error = 10000
            checkpoint = delta
            # zray = rayholder[2][i] + oz*checkpoint
            # print(zray)
            zint = mirrorinterp((rayholder[1][i] + mx * checkpoint), (rayholder[2][i] + ny * checkpoint))

            while error > 0.1:

                tsb = time.time()
                zray = rayholder[3][i] + oz * checkpoint

                # print(zint,zray,error)
                if zray < zint:
                    # print("we are in if")
                    bottom = bottom + delta
                    delta = (top - bottom) / 2
                    checkpoint = bottom + delta
                    # print("checkpoint, top, bottom, delta, error")
                    # print(checkpoint,top,bottom,delta,error)
                else:
                    # print("we are in else")
                    top = top - delta
                    delta = (top - bottom) / 2
                    checkpoint = top - delta
                    # print("checkpoint, top, bottom, delta, error")
                    # print(checkpoint,top,bottom,delta,error)

                # print("the checkpoint value is")
                # print(checkpoint)
                xloc = rayholder[1][i] + mx * checkpoint
                yloc = rayholder[2][i] + ny * checkpoint
                zint = mirrorinterp((rayholder[1][i] + mx * checkpoint), (rayholder[2][i] + ny * checkpoint))
                zray = rayholder[3][i] + oz * checkpoint
                error = abs(zray - zint)

            # print("found the spot")

            # print("checkpoint, x and y location:",checkpoint,xloc,yloc)
            # print("our interp z:",zint,"our ray location is:",zray,"our measure:",error)
            # print("before")
            # print(rayholder[5][i])
            # np.append(rayholder[4], rayholder[1][i] + mx * checkpoint)
            # np.append(rayholder[5], rayholder[2][i] + mx * checkpoint)
            # np.append(rayholder[6], rayholder[3][i] + mx * checkpoint)
            # print("size of rayholder :")
            # print(np.shape(rayholder[4]))

            rayholder[4][i] = (rayholder[1][i] + mx * checkpoint)
            rayholder[5][i] = rayholder[2][i] + ny * checkpoint
            rayholder[6][i] = rayholder[3][i] + oz * checkpoint

            # print(rayholder[4][i], maxmin[0, 0])

            if rayholder[4][i] < maxmin[0, 0] and rayholder[4][i] > maxmin[0, 1] and rayholder[5][i] < maxmin[1, 0] and \
                    rayholder[5][i] > maxmin[1, 1]:
                x = 0
                # np.delete(rayholder[:], i)
                # print("we are in bounds")
                # print(rayholderbuild.shape)
            else:
                # print("we are out of bounds")
                removeelements.append(i)
                # print(removeelements)
                # print(len(rayholder[0]))
                continue
            # print("just after remove elements")
            # print(removeelements)
            # print(rayholderbuild.shape)

        tsa = time.time()
        print("x", tsa - twb)
        # print("remove elements")
        # print(removeelements)

        ta = time.time()
        rayholderbuild = np.delete(rayholder[:], np.s_[removeelements], 1)
        # print(len(rayholderbuild[0]))
        rayobjectreturn.append(rayholderbuild)
        # print("end of iteration for rayholderbuild")
        # print(len(rayobjectreturn))
        # print(np.shape(rayobjectreturn))

        # print(rayholder[4][i])
        # print(np.shape(rayholder[4]))

        # print("after")
        # print(rayholder[5][i])
        # print("saved value")
        # print("the intersection point is ")
        # print(rayholder[0][i]+mx*checkpoint,rayholder[1][i]+ny*checkpoint,rayholder[2][i]+oz*checkpoint)
    print(len(rayobjectreturn))
    print("rayholder after")
    # print(rayobjectreturn)
    return rayobjectreturn


def build_plane_of_mirror_intersections(raysobject, mirrorinterpolator):  # get the normalized normal of the plane
    rayobjectreturn = []
    for rayholder in raysobject:  # for each function row of rays
        # print("rayholder before")
        # print(rayholder)
        # print("new line")
        rayholder = np.stack((rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5],
                              rayholder[6], rayholder[4], rayholder[5], rayholder[5]))
        print("rayholder shape is:")
        print(rayholder.shape)
        for i in range(len(rayholder[0])):  # run through all elements along array
            # print("iteration number",i)
            x = rayholder[4][i]  # get the xyz coordinates of intersection with mirror
            y = rayholder[5][i]
            z = rayholder[6][i]
            dx = 0.2
            dy = 0.2

            p1x = x  # create triangulation points
            p1y = y + dy * np.sqrt(2)  # In order to be able to calculate reflection normal
            p2x = x + dx
            p2y = y - dy
            p3x = x - dx
            p3y = y - dy

            # z = mirrorinterpolator(x, y)
            p1z = mirrorinterpolator(p1x, p1y)  # get equivelant z points of interpelation points
            p2z = mirrorinterpolator(p2x, p2y)
            p3z = mirrorinterpolator(p3x, p3y)

            p1 = np.array([p1x, p1y, float(p1z)])
            p2 = np.array([p2x, p2y, float(p2z)])
            p3 = np.array([p3x, p3y, float(p3z)])
            # print(p1,p2,p3)
            # These two vectors are in the plane
            v1 = p3 - p1
            v2 = p2 - p1
            # the cross product is a vector normal to the plane
            cp = np.cross(v2, v1)

            # print("the normal vector and its normalized version")
            # print(cp)
            cp = cp / LA.norm(cp)
            # print("our normal vector is")
            # print(cp)

            a, b, c = cp
            # print(a,b,c)
            rayholder[7][i] = a
            rayholder[8][i] = b
            rayholder[9][i] = c
            # cp = np.reshape(cp,(-1,1))
            # X_normalized = preprocessing.normalize(cp , axis=1,norm='l1')
            # print(cp)
            norm = [float(i) / sum(cp, 0) for i in cp]
        rayobjectreturn.append(rayholder)

        # print("above us is cp, below normalized")
        # print(norm)
        # This evaluates a * x3 + b * y3 + c * z3 which equals d
        # d = np.dot(cp, p3)
        # print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))

        # normal = a.normal_vector
    print("our plane is defined as")
    # print(a)
    print("Shape of the object")
    print(rayobjectreturn[0].shape)
    # print(normal)
    return rayobjectreturn


def angle(v1, v2):
    return math.acos(dot_product(v1, v2) / (length(v1) * length(v2)))


def get_angles_between_ray_and_normal(raysobject):
    rayobjectreturn = []
    for rayholder in raysobject:  # for each function row of rays
        # print("rayholder before")
        # print(rayholder)
        # print("new line")
        rayholder = np.stack((rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5],
                              rayholder[6], rayholder[7], rayholder[8], rayholder[9], rayholder[9]))
        # print("rayholder shape is:")
        # print(rayholder.shape)
        for i in range(len(rayholder[0])):  # run through all elements along array
            # print("iteration number",i)
            x = rayholder[4][i]
            y = rayholder[5][i]
            z = rayholder[6][i]

            a = rayholder[7][i]
            b = rayholder[8][i]
            c = rayholder[9][i]
            v1 = np.array([x, y, z])
            v2 = np.array([a, b, c])
            rayholder[10][i] = angle(v1, v2)
            # print(rayholder[9][i])
        rayobjectreturn.append(rayholder)
    return rayobjectreturn


def get_reflected_vector(raysobject):
    rayobjectreturn = []
    for rayholder in raysobject:  # for each function row of rays
        # print("rayholder before")
        # print(rayholder)
        # print("new line")
        rayholder = np.stack(
            (rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5], rayholder[6],
             # 1* Amplitude     # 3* spatial origins of ray, 3* intersection point on surface
             rayholder[7], rayholder[8], rayholder[9], rayholder[10]
             # 3* normal of the surface  1* angle between
             , rayholder[9], rayholder[9], rayholder[
                 9]))  # 3* reflected rays                                                                                         #

        # print("rayholder shape is:")
        # print(rayholder.shape)
        for i in range(len(rayholder[0])):  # run through all elements along array
            # print("iteration number",i)
            normal = np.array([rayholder[7][i], rayholder[8][i], rayholder[9][i]])

            x = rayholder[4][i] - rayholder[1][i]
            y = rayholder[5][i] - rayholder[2][i]
            z = rayholder[6][i] - rayholder[3][i]
            d = np.array([x, y, z])
            n = np.array([rayholder[7][i], rayholder[8][i], rayholder[9][i]])

            ndot = dot_product(d, n)
            # print(ndot)
            r = d - 2 * (ndot * n)
            # print(r)

            # print("the reflected vector is")
            # print(r)
            r = r / LA.norm(r)
            r = r * 100
            # print(r)

            rayholder[11][i] = r[0]
            rayholder[12][i] = r[1]
            rayholder[13][i] = r[2]

        rayobjectreturn.append(rayholder)
    print("rayholder shape is:")
    print(rayholder.shape)
    print(len(rayobjectreturn))
    return rayobjectreturn


def build_intersections_with_screen(raysobject, screeninterp):
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
            # if np.absolute(intersection[2])>10000 :
            #    print("the ray has intersected out of bounds")
            #    continue
            # intersection = [0,0,0]
            # print(intersection)
            rayholder[14][i] = intersection[0]
            rayholder[15][i] = intersection[1]
            rayholder[16][i] = intersection[2]

            # print(rayholder[11][i], rayholder[12][i], rayholder[13][i])

            # print("intersection point")
            # print(rayholder[14][i],rayholder[15][i],rayholder[16][i])     #intersection points with screen
            # print("after")
            # print(rayholder[5][i])
            # print("saved value")
            # print("the intersection point is ")
            # print(rayholder[0][i]+mx*checkpoint,rayholder[1][i]+ny*checkpoint,rayholder[2][i]+oz*checkpoint)
        rayobjectreturn.append(rayholder)
    print("rayholder after")
    # print(rayobjectreturn)
    return rayobjectreturn


def ray_propogation(zwignerobject, zwignerobjecttrans, lightsource, mirrorinterpolator, mirrorobject, screen_function):

    rayobject = build_ray_matrices_from_wigner(zwignerobject, zwignerobjecttrans, lightsource)

    print("in the main after matrix built")
    print(np.shape(rayobject))

    # viewvectors(rayobject)

    # plotscatter(rayobject)

    # print(mirrorinterpolator(150.254355400696866, -8.0))

    rayobjects = build_intersections_with_mirror(rayobject, mirrorinterpolator, mirrorobject)

    print("in the main, the rayobject size is")
    print(rayobjects[0].shape)

    # buildplaneofmirrorintersections(rayobject,mirrorinterpolator)

    # viewvectors(rayobject)


    # plotscatter(rayobject)


    rayobject = build_plane_of_mirror_intersections(rayobjects, mirrorinterpolator)

    rayobject = get_angles_between_ray_and_normal(rayobject)

    rayobject = get_reflected_vector(rayobject)

    # print("the length of ray object is")
    # print(len(rayobject))
    # print(len(rayobject[3]))
    #
    # print("our mirror intropolator for values out of bounds are: ")
    # print(mirrorinterpolator(151, 0))
    # print(mirrorinterpolator(155, 0))
    # print(mirrorinterpolator(200, 0))

    screenobject, screeninterp = screen_function()

    rayobject = build_intersections_with_screen(rayobject, screeninterp)
    return rayobject

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np


def plot_3d_to_2d(X, Y, Z):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.set_title('')

    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    # xv, yv = np.meshgrid(X, Y)
    # plt.contour(xv, yv, Z, 100, cmap='viridis')
    plt.show()
    return


def plot_quiver_3dobject(rayobject, x, y, z, u, v, w):
    X, Y, Z, U, V, W = [], [], [], [], [], []  # np.array,np.array,np.array,np.array,np.array,np.array

    for i in range(len(rayobject)):
        # X = X + (rayobject[i][x],)
        # Y = Y + (rayobject[i][y],)
        # Z = Z + (rayobject[i][z],)
        # U = U + (rayobject[i][u],)
        # V = V + (rayobject[i][v],)
        # W = W + (rayobject[i][w],)
        X = np.append(X, rayobject[i][x])
        Y = np.append(Y, rayobject[i][y])
        Z = np.append(Z, rayobject[i][z])
        U = np.append(U, rayobject[i][u])
        V = np.append(V, rayobject[i][v])
        W = np.append(W, rayobject[i][w])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.8))

    # Make the direction data for the arrows
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
         np.sin(np.pi * z))

    print("our coordinate values are: ")
    print(X, Y, Z)

    ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, linewidths=0.5)
    # ax.set_xlim([-150, 150])
    # ax.set_ylim([-150, 150])
    # ax.set_zlim([0, 550])
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    plt.show()

    return


def plot_quiver3d(X, Y, Z, U, V, W):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.8))

    # Make the direction data for the arrows
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
         np.sin(np.pi * z))

    ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, linewidths=0.1)
    ax.set_xlim([-150, 150])
    ax.set_ylim([-150, 150])
    ax.set_zlim([0, 550])
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    plt.show()

    return


def plot_scatter(raysobject, a, b, c):
    X = []
    Y = []
    Z = []
    for rayholder in raysobject:  # for each function row of rays

        print("newrow")
        for i in range(len(rayholder[0])):
            mx = rayholder[a][i]  # - rayholder[0][i]
            ny = rayholder[b][i]  # - rayholder[1][i]
            oz = rayholder[c][i]  # - rayholder[2][i]
            X.append(mx)
            Y.append(ny)
            Z.append(oz)
            vector = np.array([mx, ny, oz])
            # print(vector)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.scatter(X, Y, Z, c='r', marker='o', s=0.1)

    ax.set_title('surface')

    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    # xv, yv = np.meshgrid(X, Y)
    # plt.contour(xv, yv, Z, 100, cmap='viridis')
    plt.show()
    return


def plot_gridata(functiondata):
    # functiondata[0]
    # functiondata[1]
    # functiondata[2]

    points = np.asarray([functiondata[0], functiondata[1]])  # np.random.rand(1000, 2)
    # points = [functiondata[0],functiondata[1]]
    # points.tolist()
    # test = float(functiondata[0][0])
    values = np.asarray(functiondata[2])  # func(points[:, 0], points[:, 1])
    # values.tolist()

    # print("checkout our functiondata")
    # print(functiondata[0])
    # print(test)
    # print(points)
    # print(values)
    # print(len(values))
    # print(values.shape)
    values = np.expand_dims(values, axis=1)
    print(values.shape)

    # xmin = np.amin(functiondata[0])
    # min = np.amin(points[0][0])
    # xmax = int(np.max(functiondata[0]))
    #
    # ymin = int(np.min(functiondata[1]))
    # ymax = int(np.max(functiondata[1]))

    grid_x, grid_y = np.meshgrid(np.linspace(np.amin(functiondata[0]), np.amax(functiondata[0]), 30),
                                 np.linspace(np.amin(functiondata[1]), np.amax(functiondata[1]), 30))

    # grid_z0 = griddata(points.T, values, (grid_x, grid_y), method='nearest')
    grid_z1 = griddata(points.T, values, ((grid_x, grid_y)), method='linear')
    # grid_z2 = griddata(points.T, values, (grid_x, grid_y), method='cubic')

    print("griddata is:")
    # print(len(grid_z0))
    # print(grid_z0.shape)
    # grid_z0 = np.squeeze(grid_z0)
    grid_z1 = np.squeeze(grid_z1)
    # grid_z2 = np.squeeze(grid_z2)

    # plt.subplot(221)
    # plt.imshow(values.T, extent=(0, 1, 0, 1), origin='lower')
    # plt.plot(points[:, 0], points[:, 1], 'k.', ms=1)
    # plt.title('Original')
    # plt.subplot(222)
    # plt.imshow(grid_z0.T)
    # plt.title('Nearest')
    # plt.subplot(223)
    # plt.imshow(grid_z1.T,  origin='lower')
    # plt.title('Linear')
    # plt.subplot(224)
    # #plt.imshow(grid_z2.T, extent=(0, 1, 0, 1), origin='lower')
    # plt.imshow(grid_z2.T, origin='lower')
    # plt.title('Cubic')
    # plt.gcf().set_size_inches(6, 6)
    # plt.show()

    plt.title('Linear')
    # plt.subplot(224)
    plt.imshow(grid_z1.T, extent=(0, 1, 0, 1), origin='lower')

    return


def plot_contour(X, Y, Z):
    plt.contourf(X, Y, Z, 1, cmap='viridis')
    plt.colorbar()
    plt.show()
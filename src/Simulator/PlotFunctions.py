import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.interpolate import griddata
import numpy as np


def plot_3d_to_2d(X, Y, Z):
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.set_title('')

    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    plt.show()


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

    ax = plt.axes(projection='3d')

    ax.scatter(X, Y, Z, c='r', marker='o', s=0.1)

    ax.set_title('surface')

    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    plt.show()


def plot_gridata(functiondata):
    points = np.asarray([functiondata[0], functiondata[1]])
    values = np.asarray(functiondata[2])
    values = np.expand_dims(values, axis=1)
    print(values.shape)
    print("we are in plot gridata")
    grid_x, grid_y = np.meshgrid(np.linspace(np.amin(functiondata[0]), np.amax(functiondata[0]), 30),
                                 np.linspace(np.amin(functiondata[1]), np.amax(functiondata[1]), 30))

    # grid_z0 = griddata(points.T, values, (grid_x, grid_y), method='nearest')
    grid_z1 = griddata(points.T, values, (grid_x, grid_y), method='linear')
    # grid_z2 = griddata(points.T, values, (grid_x, grid_y), method='cubic')
    grid_z1 = np.squeeze(grid_z1)

    plt.title('Linear')
    plt.imshow(grid_z1.T, extent=(0, 1, 0, 1), origin='lower')
    plt.show()

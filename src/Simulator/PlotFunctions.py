import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import griddata
import numpy as np
import json
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from src.Simulator.Ray import *
from src.Simulator.Vector import *
from mpl_toolkits.mplot3d import Axes3D




with open('../config.json') as config_file:
    config = json.load(config_file)

matplotlib.use('TkAgg')


def plot_3d_to_2d(X, Y, Z):
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.set_title('')

    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    plt.show()


def plot_mirror(mirrorBorders, mirrorInterpolatedBuilder):
    x_axis = np.arange(mirrorBorders[0], mirrorBorders[2], 1)
    y_axis = np.arange(mirrorBorders[1], mirrorBorders[3], 1)

    X, Y = np.meshgrid(x_axis, y_axis)

    Z = mirrorInterpolatedBuilder(X, Y)

    plt.figure(1)
    plt.clf()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', linewidth=0)

    ax.set_title('')

    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    plt.show(block=False)
    plt.pause(1)


def plot_scatter(rays):

    raysY = []
    raysX = []
    for ray in rays:
        raysY.append(getY(getOrigin(ray)))
        raysX.append(getX(getOrigin(ray)))

    padding = 5

    plt.figure(2)
    plt.clf()
    plt.axis([min(raysY) - padding, max(raysY) + padding, min(raysX) - padding, max(raysX) + padding])
    plt.scatter(raysY, raysX, c='r', marker='o', s=[getAmplitude(ray) for ray in rays])
    plt.show(block=False)
    #plt.pause(1)

def plot_heatmap(rays):
    raysY = np.array([ray.getOrigin().getY() for ray in rays])
    raysZ = np.array([ray.getOrigin().getZ() for ray in rays])

    padding = 5

    heatmap, xedges, yedges = np.histogram2d(raysY, raysZ, bins=50, weights=[ray.getAmplitude() for ray in rays])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.figure(2)
    plt.clf()
    plt.axis([raysY.min() - padding, raysY.max() + padding, raysZ.min() - padding, raysZ.max() + padding])

    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show(block=False)
    plt.pause(1)


def plot_wigner(row, rays):


    [ray for ray in rays]


    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    #ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

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

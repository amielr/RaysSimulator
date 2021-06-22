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




with open('./config.json') as config_file:
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
    x_axis = np.arange(mirrorBorders[0], mirrorBorders[2], 10)
    y_axis = np.arange(mirrorBorders[1], mirrorBorders[3], 10)

    X, Y = np.meshgrid(x_axis, y_axis)

    Z = mirrorInterpolatedBuilder(x_axis, y_axis)

    plt.figure(1)
    plt.clf()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none', linewidth=0)

    ax.set_title('')

    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")

    plt.show(block=False)
    #plt.pause(1)


def plot_scatter(rays):
    raysY = []
    raysX = []
    raysZ = []

    for ray in rays:
        raysY.append(getY(getOrigin(ray)))
        raysX.append(getX(getOrigin(ray)))
        raysZ.append(getZ(getOrigin(ray)))

    padding = 5

    #fig = plt.figure()

    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')

    #ax.clf()
    ax.axis([min(raysY) - padding, max(raysY) + padding, min(raysX) - padding, max(raysX) + padding])
    #ax.scatter(raysY, raysX, raysZ, c='r', marker='o', s=[getAmplitude(ray) for ray in rays])
    ax.scatter(raysY, raysX, raysZ, c='r', marker='.')

    plt.show()


def plot_3dheatmap(rays):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Make data.
    raysYY = []
    raysXX = []
    raysZZ = []
    amp = []
    # raysY = np.array([ray.getOrigin().getY() for ray in rays])
    # raysY = np.array([getY(getOrigin(ray)) for ray in rays])

    # raysZ = np.array([ray.getOrigin().getZ() for ray in rays])
    # raysZ = np.array([getZ(getOrigin(ray)) for ray in rays])
    for ray in rays:
        raysYY.append(getY(getOrigin(ray)))
        raysXX.append(getX(getOrigin(ray)))
        raysZZ.append(getZ(getOrigin(ray)))
        amp.append(getAmplitude(ray))


    raysX = np.array(raysXX)
    raysY = np.array(raysYY)
    raysZ = np.array(raysZZ)
    npAmp = np.array(amp)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    # Plot the surface.
    surf = ax.plot_trisurf(raysX, raysY, npAmp, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_heatmap(rays, direction):
    raysYY = []
    raysXX = []
    raysZZ = []

    #raysY = np.array([ray.getOrigin().getY() for ray in rays])
    #raysY = np.array([getY(getOrigin(ray)) for ray in rays])

    #raysZ = np.array([ray.getOrigin().getZ() for ray in rays])
    #raysZ = np.array([getZ(getOrigin(ray)) for ray in rays])
    for ray in rays:
        raysYY.append(getY(getOrigin(ray)))
        raysXX.append(getX(getOrigin(ray)))
        raysZZ.append(getZ(getOrigin(ray)))

    raysX = np.array(raysXX)
    raysY = np.array(raysYY)
    raysZ = np.array(raysZZ)
    padding = 1.3

    rngY = (raysY.max() - raysY.min()) * padding
    cY = (raysY.max() + raysY.min()) / 2
    minY, maxY = cY - rngY / 2, cY + rngY / 2

    rngZ = (raysZ.max() - raysZ.min()) * padding
    cZ = (raysZ.max() + raysZ.min()) / 2
    minZ, maxZ = cZ - rngZ / 2, cZ + rngZ / 2

    rngX = (raysX.max() - raysX.min()) * padding
    cX = (raysX.max() + raysX.min()) / 2
    minX, maxX = cX - rngX / 2, cX + rngX / 2



    if direction == 'z':
        heatmap, xedges, yedges = np.histogram2d(raysX, raysY, bins=50, weights=[getAmplitude(ray) for ray in rays])
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        plt.axis([minX, maxX, minY, maxY])
    elif direction =='x':
        heatmap, zedges, yedges = np.histogram2d(raysZ, raysY, bins=50, weights=[getAmplitude(ray) for ray in rays])
        extent = [zedges[0], zedges[-1], yedges[0], yedges[-1]]
        plt.axis([minZ, maxZ, minY, maxY])
    else:
        heatmap, zedges, xedges = np.histogram2d(raysZ, raysX, bins=50, weights=[getAmplitude(ray) for ray in rays])
        extent = [zedges[0], zedges[-1], xedges[0], xedges[-1]]
        plt.axis([minZ, maxZ, minX, maxX])

    #plt.figure(2)
    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()


def plot_wigner(row, rays):
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


def plot_error_over_time(errors):
    y = errors
    x = range(len(errors))

    plt.plot(x, y)
    plt.yscale("log")
    plt.show()


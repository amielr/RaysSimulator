import numpy as np
from time import time
import matplotlib.pyplot as plt
from scipy.io import loadmat  # this is the SciPy module that loads mat-files

def time_function(func):
    def wrapper():
        start = time()
        func()
        print(f'{func.__name__} run time: {time() - start}')
    return wrapper

def plot_3d_to_2d(X, Y, Z, name='Plot'):
    # print("X=", X)
    # print("Y=", Y)
    # print("X \n", X)
    X, Y = np.meshgrid(X, Y)
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.set_title(name)

    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    # xv, yv = np.meshgrid(X, Y)
    # plt.contour(xv, yv, Z, 100, cmap='viridis')
    plt.show()
    # plt.close(fig)
    return




def Wigner_func_for_element(temp_data, name="name"):
    Nx, Ny = temp_data.shape
    dx, dy = 1, 1  # step size in the x- and y- direction
    dkx = 2 * np.pi / (2 * Nx) / dx
    dky = 2 * np.pi / (2 * Ny) / dy
    E_shift = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
    W = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
    Wig = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
    A = np.zeros((2 * Nx, 2 * Ny))
    B = np.zeros((2 * Nx, 2 * Ny), dtype=complex)
    Nxs = np.linspace(-Nx, Nx, dx)
    Nys = np.linspace(-Ny, Ny, dy)
    # Nxs = np.linspace(-0, 2*Nx, dx)
    # Nys = np.linspace(-0, 2*Ny, dy)
    WigF = []
    Etot = np.zeros((Nx, Ny, Nx, Ny), dtype=complex)
    for i in range(Nx):
        for j in range(Ny):
            for m in range(-Nx, Nx, dx):
                for n in range(-Ny, Ny, dy):
                    if (i - m) >= 0 and (j - n) >= 0 and (i + m) < Nx and (j + n) < Ny and (i - m) < Nx and (
                            j - n) < Ny and (i + m) >= 0 and (j + n) >= 0:
                        # A[m + Nx, n + Ny] = E[i + m, j + n] * np.conj(E[i - m, j - n]) # modified coorr matrix
                        # print("A[m+Nx,n+Ny]=",A[m+Nx,n+Ny], "i=", i, "j=", j, "m=", m, "n=", n,  m + Nx,  n + Ny)
                        E_shift[i, j, m + Nx, n + Ny] = E[i + m, j + n] * np.conj(E[i - m, j - n])
                        # print("E_shift[i,j,m+Nx,n+Ny]=", E_shift[i, j, m + Nx, n + Ny], "                  i=", i, "j=", j, "m=", m, "n=", n)
            # print("E_shift[i,j]=\n", E_shift[i, j])
            Exij = E_shift
            # Exij[i,j,:,:] = E_shift
            # print("Exij[i, j] \n", Exij[i, j])

            fft_Et = np.fft.fftshift(np.fft.fft2(Exij[i, j]))
            Wig[i, j, :, :] = fft_Et
            # print("Exij[i, j] SHAPE \n", fft_Et[i, j].shape)
            # Efft.append(fft_Et)
            Kx = np.linspace(-Nx // 2, Nx // 2, Exij.shape[0])  # * dkx
            Ky = np.linspace(-Ny // 2, Ny // 2, Exij.shape[1])  # * dky
            Kx = np.linspace(-Nx, Nx, Exij[i, j].shape[0]) * dkx
            Ky = np.linspace(-Ny, Ny, Exij[i, j].shape[1]) * dky
            # print("Exij.shape[0] :", Exij[i, j].shape[0])
            fft_z = np.fft.fftshift(np.fft.fft2(Exij))

    return Wig


def Raybuilder(WignerMatrix):
    # rays = np.array((3, 3), complex)
    rayList = []

    xRange, yRange, resolution = 7.5, 5, 32
    dx = xRange / resolution
    dy = yRange / resolution
    dKx = 2 * np.pi / (2 * Nx) / dx
    dKy = 2 * np.pi / (2 * Nx) / dx

    for posX in range(Nx):
        for posY in range(Ny):
            for Kx in range(-Nx, Nx):
                for Ky in range(-Ny, Ny):
                    if WignerMatrix[posX, posY, Kx, Ky] != 0:
                        origin = np.array(
                            [-(0.5 * xRange) + posX * (xRange / resolution), -(0.5 * yRange) + posY * (yRange / resolution),
                             0], dtype=np.float_)
                        direction = np.array([Kx * dKx, Ky * dKy, 1], dtype=np.float_)
                        amplitude = np.array([WignerMatrix[posX, posY, Kx, Ky], 0, 0])
                        ray = np.array([origin, direction, amplitude])
                        rayList.append(ray)
    return rayList


def michaelMain():
    twodsquarewave1j = [[1, -1, 2, -1j, 1j],
                        [3, -3, 4, -3j, 3j],
                        [2, -2, 5, -2j, 1j],
                        [1, 2, 3, 4j, 5j],
                        [1, 2, 3, 4j, 5j]]

    ThzFieldData = loadmat('EqxtTHZField.mat')
    ThzFieldData = ThzFieldData['Eqxt']
    E = ThzFieldData
    Nx, Ny = E.shape
    nx, ny = np.shape(E)

    WignerField = Wigner_func_for_element(ThzFieldData, "Eqxt")

    TwoDRayPackage = Raybuilder(WignerField)

    return TwoDRayPackage

########################################################################################################################
####################################### Main Program ###################################################################
########################################################################################################################




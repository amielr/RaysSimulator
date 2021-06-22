import numpy as np
from numpy import genfromtxt
from time import time
import matplotlib.pyplot as plt
from scipy.io import loadmat  # this is the SciPy module that loads mat-files
import os
from Simulator.PlotFunctions import plot_scatter, plot_3d_to_2d, plot_heatmap, plot_3dheatmap
from Simulator.RayPropogation import propogate_rays_in_free_space


def time_function(func):
    def wrapper():
        start = time()
        func()
        print(f'{func.__name__} run time: {time() - start}')
    return wrapper


def rotate_180(array, M, N, out):
    M = array.shape[0]
    N = array.shape[1]
    for i in range(M):
        for j in range(N):
            out[i, N-1-j] = array[M-1-i, j]


def Wigner_func_for_element(temp_data, name="name"):
    Nx, Ny = temp_data.shape
    E = temp_data
    dx, dy = 1, 1  # step size in the x- and y- direction
    # dkx = 2 * np.pi / (2 * Nx) / dx
    # dky = 2 * np.pi / (2 * Ny) / dy
    E_shift = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
    # W = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
    Wig = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny), dtype=complex)
    # A = np.zeros((2 * Nx, 2 * Ny))
    # B = np.zeros((2 * Nx, 2 * Ny), dtype=complex)
    # Nxs = np.linspace(-Nx, Nx, dx)
    # Nys = np.linspace(-Ny, Ny, dy)
    # Nxs = np.linspace(-0, 2*Nx, dx)
    # Nys = np.linspace(-0, 2*Ny, dy)
    # WigF = []
    # Etot = np.zeros((Nx, Ny, Nx, Ny), dtype=complex)
    for i in range(Nx):
        for j in range(Ny):
            for m in range(-Nx, Nx, dx):
                for n in range(-Ny, Ny, dy):
                    if (i - m) >= 0 and (j - n) >= 0 and (i + m) < Nx and (j + n) < Ny and (i - m) < Nx and (
                            j - n) < Ny and (i + m) >= 0 and (j + n) >= 0:
                        # A[m + Nx, n + Ny] = E[i + m, j + n] * np.conj(E[i - m, j - n]) # modified coorr matrix
                        # print("A[m+Nx,n+Ny]=",A[m+Nx,n+Ny], "i=", i, "j=", j, "m=", m, "n=", n,  m + Nx,  n + Ny)
                        E_shift[i, j, m + Nx, n + Ny] = E[i + m, j + n] * np.conj(E[i - m, j - n])
                        #ERotated = rotate_180(E_shift)
                        # E_shift = E_shift
                        # print("E_shift[i,j,m+Nx,n+Ny]=", E_shift[i, j, m + Nx, n + Ny], "                  i=", i, "j=", j, "m=", m, "n=", n)
            # print("E_shift[i,j]=\n", E_shift[i, j])
            Exij = E_shift
            # Exij[i,j,:,:] = E_shift
            # print("Exij[i, j] \n", Exij[i, j])

            #fft_Et = np.fft.fftshift(np.fft.fft2(Exij[i, j]))
            fft_Et = (np.fft.fft2(Exij[i, j]))

            #fft_Et[::2] = -fft_Et[::2]
            #fft_Et = abs(fft_Et)
            Wig[i, j, :, :] = abs(fft_Et)
            # print("Exij[i, j] SHAPE \n", fft_Et[i, j].shape)
            # Efft.append(fft_Et)
            # Kx = np.linspace(-Nx // 2, Nx // 2, Exij.shape[0])  # * dkx
            # Ky = np.linspace(-Ny // 2, Ny // 2, Exij.shape[1])  # * dky
            # Kx = np.linspace(-Nx, Nx, Exij[i, j].shape[0]) * dkx
            # Ky = np.linspace(-Ny, Ny, Exij[i, j].shape[1]) * dky
            # print("Exij.shape[0] :", Exij[i, j].shape[0])
            # fft_z = np.fft.fftshift(np.fft.fft2(Exij))

            # if (i == int(Nx/2) and j == int(Ny/2)):
            #     X, Y = np.meshgrid(np.linspace(-Nx, Nx, Exij[i, j].shape[0]), np.linspace(-Ny, Ny, Exij[i, j].shape[1]))
            #     fft_Et[1::2] = -np.conj(fft_Et[1::2])
            #     fft_Et = np.fft.fftshift(fft_Et)
            #     Z = abs(fft_Et)
            #     plot_3d_to_2d(X, Y, Z)

    return Wig


def Raybuilder(WignerMatrix):
    # rays = np.array((3, 3), complex)
    WignerMatrix = WignerMatrix.real
    shape = WignerMatrix.shape
    Nx = shape[0]
    Ny = shape[1]
    xResolution = shape[0]
    yResolution = shape[1]

    rayList = []

    xRange, yRange, resolution = 7.5, 5, 32   # resolution is not predefined
    dx = xRange / Nx
    dy = yRange / Ny
    # dKx = 2 * np.pi / (2 * Nx) / dx
    # dKy = 2 * np.pi / (2 * Ny) / dy
    # dKx = 0.5 / (2 * Nx)
    # dKy = 0.5 / (2 * Ny)
    rng = 2*Nx
    dKx = 0.033/rng
    #dKy = 0.046
    dKy = 0.033/rng
    for posX in range(Nx):
        for posY in range(Ny):
            for Kx in range(-Nx, Nx):
                for Ky in range(-Ny, Ny):
                    if WignerMatrix[posX, posY, Kx, Ky] != 0:
                        origin = np.array(
                            #[posX * (xRange / resolution), posY * (yRange / resolution),0 ], dtype=np.float_)
                            [-(0.5 * xRange) + posX * (xRange / xResolution), (0.5 * yRange) - posY * (yRange / yResolution),
                             0], dtype=np.float_)
                        #direction = np.array([Kx, Ky , 1], dtype=np.float_)
                        direction = np.array([Kx * dKx, Ky * dKy, 1], dtype=np.float_)
                        # direction = np.array([np.cos(Kx * dKx), np.sin(Ky * dKy), 1], dtype=np.float_)

                        #direction = np.array([Kx * dx, Ky * Ky, 1], dtype=np.float_)
                        amplitude = np.array([WignerMatrix[posX, posY, Kx, Ky], 0, 0])
                        ray = np.array([origin, direction, amplitude])
                        rayList.append(ray)

    # plot_scatter(rayList)

    return rayList


def michaelMain():
    twodsquarewaveComplex = np.array([[1, -1, 2, -1j, 1j],
                        [3, -3, 4, -3j, 3j],
                        [2, -2, 5, -2j, 1j],
                        [1, 2, 3, 4j, 5j],
                        [1, 2, 3, 4j, 5j]])

    twodwavereal = np.array([[1, -1, 2, -1, 1],
                                   [3, -3, 4, -3, 3],
                                   [2, -2, 5, -2, 1],
                                   [1, 2, 3, 4, 5],
                                   [1, 2, 3, 4, 5]])

    twodsquarewavereal = np.array([[0, 0, 0, 0, 0],
                                   [0, 1, 1, 1, 0],
                                   [0, 1, 1, 1, 0],
                                   [0, 1, 1, 1, 0],
                                   [0, 0, 0, 0, 0]])

    twodsquarewavereal = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])



    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    my_file = os.path.join(THIS_FOLDER, 'EqxtTHZField.mat')
    my_CSV_file = os.path.join(THIS_FOLDER, 'Eqxtcsv.csv')


    ThzFieldData = loadmat(my_file)
    # ThzFieldData = ThzFieldData['Eqxt']

    my_CSV_data = genfromtxt(my_CSV_file, dtype=complex, delimiter=',')

    E = my_CSV_data
    Nx, Ny = E.shape
    nx, ny = np.shape(E)

    SampleData = twodsquarewavereal

    WignerField = Wigner_func_for_element(SampleData, "Eqxt")

    TwoDRayPackage = Raybuilder(WignerField)
    # plot_scatter(TwoDRayPackage)
    # plot_3dheatmap(TwoDRayPackage)
    plot_heatmap(TwoDRayPackage, 'z')

    # TwoDRayPackage = propogate_rays_in_free_space(TwoDRayPackage, 450)
    # plot_heatmap(TwoDRayPackage)

    # for i in range(3):
    #     TwoDRayPackage = propogate_rays_in_free_space(TwoDRayPackage, 400)
    #     plot_heatmap(TwoDRayPackage, 'z')
    #     plot_3dheatmap(TwoDRayPackage)
    #     plot_scatter(TwoDRayPackage)


    return TwoDRayPackage

########################################################################################################################
####################################### Main Program ###################################################################
########################################################################################################################




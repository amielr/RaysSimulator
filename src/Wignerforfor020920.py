import numpy as np
from time import  sleep, time
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits import mplot3d
import csv
from scipy.linalg import circulant
from scipy.fftpack import fft
import pandas as pd
from scipy.io import loadmat  # this is the SciPy module that loads mat-files

def time_function(func):
    def wrapper():
        start = time()
        func()
        print(f'{func.__name__} run time: {time() - start}')
    return wrapper

# f = time_function(f)

def time_function1(func):
    start = time()
    func()
    print(f'{func.__name__} run time: {time() - start}')

# time_function1(f)

def light_source_function():
    n=1
    # # x= np.linspace(-10, 10, 11)    print(x)
    # # y = np.linspace(-10, 10, 11)   print(y)
    # x, y = np.meshgrid(np.linspace(-n*10, n*10, n*11), np.linspace(-n*10, n*10, n*11))     # function space and parameters
    # # print(abs(x) <= 5, 1, 0)
    # # print(y)
    # twodsquarewave = np.where(abs(x) <= n*4, 1, 0) & np.where(abs(y) <= n*3, 1, 0)
    # #######################################################################################################
    # x, y = np.meshgrid(np.linspace(-n * 1, n * 1, n * 3),
    #                    np.linspace(-n * 1, n * 1, n * 3))  # function space and parameters
    # x, y = np.meshgrid(np.linspace(-n * 6, n * 6, n * 7),
    #                    np.linspace(-n * 4, n * 4, n * 5))  # function space and parameters
    # twodsquarewaveE = np.where(abs(x) <= n * 2, 1, 0) & np.where(abs(y) <= n * 2, 1, 0)
    # print("twodsquarewaveE \n", twodsquarewaveE)
    # twodsquarewave = [[0, 1, 2],
    #                   [0, 1, 0],
    #                   [3, 0, 4]]
    # twodsquarewave = [[0, 0, 0],
    #                   [0, 1, 0],
    #                   [0, 0, 0]]
    # twodsquarewave = [[1, 2, 3],
    #                   [4, 5, 6],
    #                   [7, 8, 9]]
    # print("twodsquarewave: ", twodsquarewave)
    # reshapedtwodsquarewave = np.reshape(twodsquarewave, (3, 3))
    # twodsquarewave = reshapedtwodsquarewave
    #########################################################################################################
    data111 = loadmat('Eqxtmat.mat')
    # data111 = loadmat('C:\Users\michaelge\PycharmProjects\WIGNERCHIRP\Eqxtmat.mat')
    data1111 = data111['Eqxt']
    # print(data1111.shape)
    x, y = np.meshgrid(np.linspace(-n * 7.5, n * 7.5, n * 32), np.linspace(-n * 5, n * 5, n * 32))
    absdata1111 = np.abs(data1111)
    twodsquarewave = absdata1111
    ##########################################################################################################

    # print("x:", x)
    # print("x function dimension is")
    # print(x.shape)
    # print("y function dimension is")
    # print(y.shape)
    #
    # print(np.linspace(-10, 10, 11), np.linspace(-10, 10, 11))
    # print("x", x)
    # print("x,y",  x, y)
    # print("y", y)
    # print("x[0, :], y[:, 0]", x[5, :], y[:, 5])

    raveledx, raveledy, raveledfunc = np.ravel(x), np.ravel(y), np.ravel(twodsquarewave)

    # print(twodsquarewave.shape[0])
    # print(twodsquarewave)
    # print("raveledx", raveledx)

    xdimension, ydimension = twodsquarewave.shape[0], twodsquarewave.shape[1]
    # print("xdimension, ydimension", xdimension, ydimension)

    functiondata = np.stack((raveledx, raveledy, raveledfunc))
    # print("functiondata ", functiondata)

    reshapedfunctiondata = np.reshape(functiondata, (3, xdimension, ydimension))
    # print("reshapedfunctiondata /",  reshapedfunctiondata)

    return reshapedfunctiondata  # x, z , amplitude



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


def WDF(data):
    sout_fft = np.zeros((len(data), len(data)), dtype=complex)
    sout_fftTotserial=sout_fft[0, :]
    soutTotParalel= np.zeros((6, 6), dtype=complex)
    # soutTot = np.zeros(1, len(sout_fft[1]), dtype=complex)
    soutTot = sout_fft[0, :]

    # print("soutTot \n", soutTot)
    # sout_fft = np.zeros((tlngth * 2, flngth), dtype=complex)
    # print("sout_fft \n", sout_fft)
    # print("tlngth=", tlngth)
    # print("flngth=", flngth)
    # # print(sout_fft.shape)
    # print("data:", data)
    sp = np.copy(data)
    # sp = np.roll(sp, len(data) // 4)
    sm = np.copy(data)
    sm = np.conj(sm)
    sm = np.roll(sm, len(data)//2)
    # print('sp : ', sp, '\nsm : ', sm)


    for i in range(len(data)):
        # if (i % 10) == 0:
        #     print("i = ", i)
        # sm = np.roll(sm, -1)
        # print('i= : ', i)
        # print('sp : ', sp)
        # print('sm : ', sm)
        # sm = np.roll(sm,-1)
        # sm = np.roll(sp, -1)
        sout = sp * sm
        soutTotSerial = soutTot+sout
        # print('soutTot i=', i, 'is', soutTotSerial)
        sout[1::2] = (-1)*sout[1::2]
        sout_fft[i] = np.fft.fft(sout, len(data))
        # sout_fft[i] = np.fft.fft(sout, flngth)
        # print('sout:', sout) #, "      flngth=", flngth)
        # print('sout_fft of i=', i, 'is', sout_fft[i])
        sout_fftTotserial=sout_fftTotserial+sout_fft[i]
        soutPar = np.reshape(sout_fft[i], (6, 6))
        soutTotParalel=soutTotParalel+soutPar
        sm = np.roll(sm, -1)
        # sp = np.roll(sp, 1)
    sout_fft[0::, 1::2] = (-1)*sout_fft[0::, 1::2]
    return sout_fft, soutTotSerial, soutTotParalel #, sout_fftTotserial

def Wigner_func_for_element(temp_data, name="name"):
    Nx, Ny = temp_data.shape
    dx, dy = 1, 1  # step size in the x- and y- direction
    dkx = 2 * np.pi / (2 * Nx) / dx
    dky = 2 * np.pi / (2 * Ny) / dy
    E_shift = np.zeros((Nx, Ny, 2 * Nx, 2 * Ny))
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
                        A[m + Nx, n + Ny] = E[i + m, j + n] * np.conj(E[i - m, j - n]) # modified coorr matrix
                        # print("A[m+Nx,n+Ny]=",A[m+Nx,n+Ny], "i=", i, "j=", j, "m=", m, "n=", n,  m + Nx,  n + Ny)
                        E_shift[i, j, m + Nx, n + Ny] = E[i + m, j + n] * np.conj(E[i - m, j - n])
                        # print("E_shift[i,j,m+Nx,n+Ny]=", E_shift[i, j, m + Nx, n + Ny], "                  i=", i, "j=", j, "m=", m, "n=", n)
            # print("E_shift[i,j]=\n", E_shift[i, j])
            Exij = E_shift
            # Exij[i,j,:,:] = E_shift
            # print("Exij[i, j] \n", Exij[i, j])

            fft_Et = np.fft.fftshift(np.fft.fft2(Exij[i, j]))
            # print("Exij[i, j] SHAPE \n", fft_Et[i, j].shape)
            # Efft.append(fft_Et)
            Kx = np.linspace(-Nx // 2, Nx // 2, Exij.shape[0])  # * dkx
            Ky = np.linspace(-Ny // 2, Ny // 2, Exij.shape[1])  # * dky
            Kx = np.linspace(-Nx, Nx, Exij[i, j].shape[0]) * dkx
            Ky = np.linspace(-Ny, Ny, Exij[i, j].shape[1]) * dky
            # print("Exij.shape[0] :", Exij[i, j].shape[0])
            fft_z = np.fft.fftshift(np.fft.fft2(Exij))
            # fft_zT = np.fft.fftshift(np.fft.fft2(Exij.T))
            fx = np.fft.fftshift(np.fft.fftfreq(Kx.shape[0], Kx[1] - Kx[0]))
            fy = np.fft.fftshift(np.fft.fftfreq(Ky.shape[0], Ky[1] - Ky[0]))
            # print("fx,fy \n", fx, fy)
            # print("fft_Et[i, j, :, :]shape \n", fft_Et[i,j].shape)
            # plot_3d_to_2d(fx, fy, np.abs(fft_z), "fft_Et")
            # plot_3d_to_2d(fx, fy, np.abs(fft_zT), "fft_Et Transposed")
            print("fft_z[i, j, :, :] \n", fft_z)
            # print("Exij[i, j] \n", fft_Et.shape)
            plot_3d_to_2d(fx, fy, np.abs(fft_Et), "fft_Et")

            for kx in range(-Nx, Nx, dx):  # index kx
                for ky in range(-Ny, Ny, dy):  # index ky
                    for p in range(-Nx, Nx, dx):
                        for q in range(-Ny, Ny, dy):
                            B[p + Nx, q + Ny] = 4 * A[p + Nx, q + Ny] * np.exp(
                                -1j * 4 * np.pi * kx * p * dx * dkx) * np.exp(
                                -1j * 4 * np.pi * ky * p * dy * dky) * dx * dy
                            # print("B[p+Nx,q+Ny]=", B[p+Nx,q+Ny])
                            # print("B[p+Nx,q+Ny]=",B[p+Nx,q+Ny], 'kx=', kx, 'ky=', ky, 'p=', p, 'q=', q)
                    # print("B[p+Nx,q+Ny]=", B.shape)
                    Kx = np.linspace(-Nx, Nx, B.shape[0]) * dkx
                    Ky = np.linspace(-Ny, Ny, B.shape[1]) * dky
                    # plot_3d_to_2d(Kx, Ky, np.abs(B), "B")
                    # C= sum(B)
                    print("C=", B)

                    # WigF.append(sum(sum(B))
                    W[i, j, kx + Nx, ky + Ny] = sum(sum(B))  # W(x,y,kx,ky)
                    # W[i, j, kx + Nx, ky + Ny] = sum(sum(B))
                    Wig[i, j] = B

            return W, Wig, fft_Et, fft_z

########################################################################################################################
####################################### Main Program ###################################################################
########################################################################################################################
n=1
x, y = np.meshgrid(np.linspace(-n * 6*2, n * 6*2, n * 7*2),
                   np.linspace(-n * 4*2, n * 4*2, n * 5*2))  # function space and parameters
twodsquarewaveE = np.where(abs(x) <= n * 2*2, 1, 0) & np.where(abs(y) <= n * 2*2, 1, 0)
twodsquarewave = [[1, 2, 1],
                  [4, 5, 4],
                  [1, 2, 1]]
twodsquarewave123 = [[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]]
twodsquarewave1 = [[1, 2, 0, 0],
                   [3, 4, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]
twodsquarewave1 = [[0, 0, 0, 0],
                   [0, 1, 2, 0],
                   [0, 3, 4, 0],
                   [0, 0, 0, 0]]
twodsquarewave = [[0, 0, 0],
                  [0, 5, 0],
                  [0, 0, 0]]
twodsquarewave1j = [[1, -1, 2, -1j, 1j],
                    [3, -3, 4, -3j, 3j],
                    [2, -2, 5, -2j, 1j],
                    [1,  2, 3,  4j, 5j],
                    [1,  2, 3, 4j, 5j]]
# print("twodsquarewave123 \n", twodsquarewave123)
origindata = np.copy(twodsquarewave123)
data111 = loadmat('Eqxtmat.mat')
    # data111 = loadmat('C:\Users\michaelge\PycharmProjects\WIGNERCHIRP\Eqxtmat.mat')
data1111 = data111['Eqxt']
# origindata = np.copy(data1111)
E=origindata
Nx, Ny = E.shape
nx,ny= np.shape(E)
# print("(Nx, Ny) \n", Nx, Ny,nx,ny)
# print("(Nx, Ny) \n", type(Nx), Ny,type(nx),ny)

# Wigner_func_for_element = time_function(Wigner_func_for_element)
# # W, Wig, fft_Et, fft_z = Wigner_func_for_element(origindata, "Eqxt")
# Wigner_func_for_element(origindata, "Eqxt")
########################################################################################################################
# time_function(Wigner_func_for_element)
W, Wig, fft_Et, fft_z = Wigner_func_for_element(origindata, "Eqxt")
# time_function(Wigner_func_for_element(origindata, "Eqxt"))
#######################################################################################################################
# print(" fft_z\n",  fft_z)
# print("B[p+Nx,q+Ny]=", B.shape)
    # print("C=", C.shape)
    # print("WigF=", len(WigF))
    # print("WigF=", WigF[0])
    # WigF.append(sum(B))
    # reshapedWigF = np.reshape(WigF, (Nx, Nx, B.shape[0], B.shape[1]))

    # Kx = [-Nx:Nx]*dkx
    # Ky = [-Ny:Ny]*dky

# Kx = np.linspace(-Nx, Nx, dx) * dkx
# Ky = np.linspace(-Ny, Ny, dy) * dky
    # print("Kx\n",  np.linspace(-Nx, Nx, dx))
    #
    # plot_3d_to_2d(Kx, Ky, W[2][1], "W")
    # print("Wshape\n", np.shape(W))
    # print("Wshape\n", np.shape(Wig))
# print("Wig\n", Wig.shape, W.shape)
    # print("W\n", W[1,2].real)
# Kxw = np.linspace(-Nx, Nx, len(W[1, 2][0].real)) * dkx
# Kyw = np.linspace(-Ny, Ny, len(W[1, 2][1].real)) * dky
    # plot_3d_to_2d(Kxw, Kyw, np.abs(W[2,2]), "W")
    # print(" E_shift\n",  E_shift)
    # print("Type(Nx, Ny) \n", type(Nx), Ny,type(Kx),ny,type(dkx))

    # print("E_shift[2,2]=\n", Exij[2, 2])
    # print("W\n", W[2,2])
    # print("D=", Wig)


import numpy as np
from scipy.linalg import circulant
from scipy.fftpack import fft

from Simulator.PlotFunctions import plot_scatter
from src.Simulator.Ray import *
from src.Simulator.Vector import *
from numba.typed import List


def get_fourier_frequencies(wignerFunction):
    frequencies = np.fft.fftfreq(len(wignerFunction))
    return frequencies


def fft_row(array):
    fourierVector = np.real(np.abs(fft(array)))
    return fourierVector


def fourier_each_vector(corrmatrix):
    wvd = np.apply_along_axis(fft_row, axis=1, arr=corrmatrix)  # forier each row
    return wvd.T


def prep_corr_matrix_from_vector(vector):
    zerovector = np.zeros(len(vector))
    doublevector = np.append(vector, zerovector)
    matrixcirculant = circulant(doublevector)
    matrixcirculant[1::2] = -matrixcirculant[1::2]
    conjreversedoublevector = np.conjugate(doublevector)
    iteratedmatrix = np.multiply(matrixcirculant, conjreversedoublevector)
    return iteratedmatrix


def wigner_function(scalarArray, suitedCoordinate, space, frequencies):
    corrMatrix = prep_corr_matrix_from_vector(scalarArray)
    wignerDist = fourier_each_vector(corrMatrix)

    rayList = []

    for wignerRow in range(len(wignerDist)):
        for wignerColumn in range(len(wignerDist[wignerRow])):
            origin = np.array([space[wignerColumn], suitedCoordinate, 0], dtype=np.float_)
            direction = np.array([frequencies[wignerRow], 0, 1], dtype=np.float_)
            amplitude = np.array([wignerDist[wignerRow][wignerColumn], 0, 0])
            ray = np.array([origin, direction, amplitude])
            rayList.append(ray)

    return rayList


def fit_axis(axis):
    return np.linspace(np.amin(axis), np.amax(axis), 2 * len(axis))


def apply_wigner_along_axis(scalarField, axis):
    axisUpSample = fit_axis(axis)

    axisFrequencies = get_fourier_frequencies(axisUpSample)

    rayList = []

    for index in range(len(scalarField)):
        array = scalarField[index]
        rays = wigner_function(array, axisUpSample[index] * 2 - axisUpSample[0], axisUpSample, axisFrequencies)
        rayList.append(rays)

    reArrangedRayList = []
    for sublist in rayList:
        for item in sublist:
            reArrangedRayList.append(item)

    return reArrangedRayList  # [item for sublist in rayList for item in sublist]








def wigner_transform(lightSource, xVec, yVec):
    updatedColumnsWignerTransform = []
    allRaysFromWigner = []

    rowsWignerTransform = apply_wigner_along_axis(lightSource, xVec)
    columnsWignerTransform = apply_wigner_along_axis(lightSource.T, yVec)

    for ray in columnsWignerTransform:
        orig = np.array([getY(getOrigin(ray)), getX(getOrigin(ray)), getZ(getOrigin(ray))], np.float_)
        dire = np.array([getY(getDirection(ray)), getX(getDirection(ray)), getZ(getDirection(ray))], np.float_)
        amp = np.array([getAmplitude(ray), 0, 0], np.float_)

        ray = np.array([orig, dire, amp], dtype=np.float_)
        updatedColumnsWignerTransform.append(ray)

    print("our updated columns length is: ", len(updatedColumnsWignerTransform))
    print("our updated columns length is: ", len(updatedColumnsWignerTransform[0]))

    for counter in range(len(rowsWignerTransform)):
        allRaysFromWigner.append(rowsWignerTransform[counter])

    for counter in range(len(updatedColumnsWignerTransform)):
        allRaysFromWigner.append(updatedColumnsWignerTransform[counter])
    #allRaysFromWigner.append(updatedColumnsWignerTransform)

    nonZeroRays = [ray for ray in allRaysFromWigner if getAmplitude(ray) > 0]

    return nonZeroRays

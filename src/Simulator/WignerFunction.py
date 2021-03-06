import numpy as np
from scipy.linalg import circulant
from scipy.fftpack import fft

from src.Simulator.Ray import Ray
from src.Simulator.Vector import Vector


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
            origin = Vector(space[wignerColumn], suitedCoordinate, 0)
            direction = Vector(frequencies[wignerRow], 0, 1)
            amplitude = wignerDist[wignerRow][wignerColumn]
            ray = Ray(origin, direction, amplitude)
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

    return [item for sublist in rayList for item in sublist]


def wigner_transform(lightSource, xVec, yVec):
    rowsWignerTransform = apply_wigner_along_axis(lightSource, xVec)
    columnsWignerTransform = apply_wigner_along_axis(lightSource.T, yVec)
    columnsWignerTransform = [Ray(Vector(ray.getOrigin().getY(),
                                         ray.getOrigin().getX(),
                                         ray.getOrigin().getZ()),
                                  Vector(ray.getDirection().getY(),
                                         ray.getDirection().getX(),
                                         ray.getDirection().getZ()
                                         ), ray.amplitude) for ray in columnsWignerTransform]
    return rowsWignerTransform + columnsWignerTransform

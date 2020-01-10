import numpy as np
from scipy.linalg import circulant
from scipy.fftpack import fft


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


def wigner_function(scalarField, space, frequencies):
    corrmatrix = prep_corr_matrix_from_vector(scalarField)
    wignerdist = fourier_each_vector(corrmatrix)
    ravelwig = np.ravel(wignerdist)
    ravelspace = np.ravel(space)
    ravelfreq = np.ravel(frequencies)

    wignerblock = np.stack((ravelwig, ravelspace, ravelfreq))
    rows = wignerdist.shape[0]  # space
    cols = wignerdist.shape[1]  # phase
    shape = -1, rows, cols
    wignerblock = np.reshape(wignerblock, shape)

    return wignerblock


def fit_axis(axis):
    return np.linspace(np.amin(axis), np.amax(axis), 2 * len(axis))


def apply_wigner_along_axis(scalarField, axis):
    axisUpSample = fit_axis(axis)

    axisFrequencies = get_fourier_frequencies(axisUpSample)

    axisSpaceGrid, axisFrequenciesGrid = np.meshgrid(axisUpSample, axisFrequencies)

    return np.apply_along_axis(wigner_function, axis=1, arr=scalarField, space=axisSpaceGrid,
                               frequencies=axisFrequenciesGrid)


def wigner_transform(functionobject):
    rowsWignerTransform = apply_wigner_along_axis(functionobject[2], functionobject[0][0, :])
    columnsWignerTransform = apply_wigner_along_axis(functionobject[2].T, functionobject[1][:, 0])

    ravelrowwvd = np.ravel(rowsWignerTransform)
    ravelcolwvd = np.ravel(columnsWignerTransform)

    wignerobject = np.stack((ravelrowwvd, ravelcolwvd))
    a, b, c, d = columnsWignerTransform.shape[0], columnsWignerTransform.shape[1], columnsWignerTransform.shape[2], columnsWignerTransform.shape[3]
    wignerobject = np.reshape(wignerobject, (-1, a, b, c, d))

    return wignerobject

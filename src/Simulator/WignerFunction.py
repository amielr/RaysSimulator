import numpy as np
from scipy.linalg import circulant
from scipy.fftpack import fft


def get_fourier_frequencies(wFunction):
    freqs = np.fft.fftfreq(len(wFunction))
    return freqs


def fft_row(array):
    fouriervector = np.real(np.abs(fft(array)))
    return fouriervector


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


def wigner_function(lightsourcefunction, space, frequencies):
    corrmatrix = prep_corr_matrix_from_vector(lightsourcefunction)
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
    x = np.linspace(np.amin(axis), np.amax(axis), 2 * len(axis))
    return x


def get_meshgrid(x, y):
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def wignerize_each_function(functionobject):
    xinterpaxis = fit_axis(functionobject[0][0, :])  # upsample - calculate x axis that suits wigner - ie rows
    yinterpaxis = fit_axis(functionobject[1][:, 0])  # upsample - calculate y axis that suits wigner - ie columns

    rowfreqs = get_fourier_frequencies(xinterpaxis)  # get the phasespace for rows
    colfreqs = get_fourier_frequencies(yinterpaxis)  # get the phasespace for columns

    rowspace, rowfrequencies = get_meshgrid(xinterpaxis, rowfreqs)
    colspace, colfrequencies = get_meshgrid(yinterpaxis, colfreqs)

    rowwvd = np.apply_along_axis(wigner_function, axis=1, arr=functionobject[2], space=rowspace,
                                 frequencies=rowfrequencies)  # rows
    colwvd = np.apply_along_axis(wigner_function, axis=1, arr=functionobject[2].T, space=colspace,
                                 frequencies=colfrequencies)  # columns

    ravelrowwvd = np.ravel(rowwvd)
    ravelcolwvd = np.ravel(colwvd)

    wignerobject = np.stack((ravelrowwvd, ravelcolwvd))
    a, b, c, d = colwvd.shape[0], colwvd.shape[1], colwvd.shape[2], colwvd.shape[3]
    wignerobject = np.reshape(wignerobject, (-1, a, b, c, d))

    return wignerobject

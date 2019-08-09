import numpy as np
from scipy.linalg import circulant
from scipy.fftpack import fft


def get_fourier_frequencies(wFunction):
    freqs = np.fft.fftfreq(len(wFunction))
    return freqs


def fft_row(array):
    # return fftshift(fft(array))
    # array[0::2] = -array[0::2]
    fouriervector = np.real(np.abs(fft(array)))
    # print(fouriervector)
    # freqs = np.fft.fftfreq(len(fouriervector))
    # print(freqs)
    return fouriervector


def fourier_each_vector(corrmatrix):
    wvd = np.apply_along_axis(fft_row, axis=1, arr=corrmatrix)  # forier each row
    wvd = wvd.T
    # freqs = np.fft.fftfreq(len(wvd))
    # print("the fourier dimensions are")
    # print(wvd.shape)
    return wvd


def prep_corr_matrix_from_vector(vector):
    zerovector = np.zeros(len(vector))
    doublevector = np.append(vector, zerovector)
    matrixcirculant = circulant(doublevector)
    # print(matrixcirculant)
    # matrixcirculant = matrixcirculant.T
    matrixcirculant[1::2] = -matrixcirculant[1::2]
    # reversedoublevector = np.append(zerovector, vector)
    # reversedoublevector[1::2] = -reversedoublevector[1::2]
    conjreversedoublevector = np.conjugate(doublevector)
    iteratedmatrix = np.multiply(matrixcirculant, conjreversedoublevector)
    # print(matrixcirculant)
    return iteratedmatrix


def wigner_function(lightsourcefunction, space, frequencies):
    # print("iteration")
    # print(function)
    corrmatrix = prep_corr_matrix_from_vector(lightsourcefunction)
    wignerdist = fourier_each_vector(corrmatrix)
    # wignerdist[1::2] = -wignerdist[1::2]

    # print("the shape of the wigner is:")
    # print(wignerdist.shape)
    # print(space.shape)
    # print(frequencies.shape)

    # wignerdist = np.abs(wignerdist)
    ravelwig = np.ravel(wignerdist)
    ravelspace = np.ravel(space)
    ravelfreq = np.ravel(frequencies)

    wignerblock = np.stack((ravelwig, ravelspace, ravelfreq))
    rows = wignerdist.shape[0]  # space
    cols = wignerdist.shape[1]  # phase
    shape = -1, rows, cols
    wignerblock = np.reshape(wignerblock, shape)
    # print("the shape of the wigner block is:")
    # print(wignerblock.shape)

    return wignerblock


def fit_axis(axis):
    x = np.linspace(np.amin(axis), np.amax(axis), 2 * len(axis))
    print("axis min and axis max")
    print(np.amin(axis))
    print(np.amax(axis))
    # interpaxis = np.interp(x,axis,axis)
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
    # build axis dimensions correlating space and phase for rows of function
    colspace, colfrequencies = get_meshgrid(yinterpaxis, colfreqs)
    # build axis dimensions correlating space and phase for columns of function

    rowwvd = np.apply_along_axis(wigner_function, axis=1, arr=functionobject[2], space=rowspace,
                                 frequencies=rowfrequencies)  # rows
    print("finished with the rows")
    colwvd = np.apply_along_axis(wigner_function, axis=1, arr=functionobject[2].T, space=colspace,
                                 frequencies=colfrequencies)  # columns
    # first dimension is the function, second is the correlation matrix, 3rd is the fft

    print("our row and column shape is:")
    print(rowwvd.shape)
    print(colwvd.shape)

    ravelrowwvd = np.ravel(rowwvd)
    ravelcolwvd = np.ravel(colwvd)

    print("shape of ravel is")
    print(ravelcolwvd.shape)
    print(ravelrowwvd.shape)

    ravelx = np.ravel(functionobject[0])  # X values
    ravely = np.ravel(functionobject[1])  # Y values
    ravel2d = np.ravel(functionobject[2])  # Amplitude values

    wFunction = np.stack((ravel2d, ravelx, ravely))
    # function = np.reshape(function,(3,a,b))
    print("our function's shape is")
    print(wFunction.shape)

    wignerobject = np.stack((ravelrowwvd, ravelcolwvd))
    a, b, c, d = colwvd.shape[0], colwvd.shape[1], colwvd.shape[2], colwvd.shape[3]
    wignerobject = np.reshape(wignerobject, (-1, a, b, c, d))

    print("wignerobject shape is:")
    print(wignerobject.shape)

    return wignerobject

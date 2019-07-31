import numpy as np

from numpy import sin, cos, pi
from numpy import linalg as LA
import matplotlib.pyplot as plt
import tensorflow as tf
import scipy as sp
import time
from scipy import constants,interpolate
from scipy.fftpack import fft
from scipy.fftpack import fftshift
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import average_precision_score
import math
from scipy.linalg import circulant
from scipy.interpolate import griddata
#from sympy.physics.optics import RayTransferMatrix, ThinLens
#from sympy import Plane, Point3D,Line3D
#from sympy.abc import x
from itertools import chain
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
from keras.models import Sequential
from keras.layers import Dense,Activation


#global variables

def miscellaneous():
    pi = np.pi
    ThzFreq = 10**12
    lamda = constants.c/ThzFreq
    print(str(lamda)+' meters')
    frequency = constants.c / lamda
    print (str(frequency)+' cycles per second')
    K = 2*pi/lamda
    Tperiod = 1/frequency
    Omega = 2*pi * frequency
    permiabilityFS = constants.epsilon_0
    print(permiabilityFS)

def InputGaussElectricField():
    x, y = np.meshgrid(np.linspace(-2,2,1000), np.linspace(-2,2,1000))
    d = np.sqrt(x*x+y*y)
    sigma = 0.3
    mu =0.0
    g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    print("2D Gaussian-like array:")
    print(g)

    #N*np.exp(-((x**2/2a**2)+(y**2/2b**2)) np.exp(j*kx* x *ky* y)

    ind = np.unravel_index(np.argmax(g, axis=None), g.shape)  # returns a tuple
    print(ind)
    print(g.shape)

    return g

def inputfunction():
    # function space and parameters
    x, y = np.meshgrid(np.linspace(-10, 10, 11), np.linspace(-10, 10, 11))


    print("x function dimension is")
    print(x.shape)
    #print(x)
    print("y function dimension is")
    print(y.shape)
    #print(y)
    #print(y)
    #print(x)
    #print(y)

    a = 1
    b = 1
    N = 1
    kx = 1
    ky = 1
    omega = 5
    # Data for plotting
    t = np.linspace(-5, 5, 100)
    t2 = np.linspace(-5,5,200)
    #function Defintions
    np.where(abs(x) <= 0.5, 1, 0)
    s = 1 + np.sin(2 * np.pi * t)
    s2d = np.where(abs(x)<=4,1,0) & np.where(abs(y)<=3,1,0)
    #s2d = np.real(N*np.exp(-((x**2/(2*a**2))+(y**2/(2*b**2)))*np.exp(np.imag*kx*x*ky*y)))
    sdelta = np.where(abs(t) >0.5,1,0) & np.where(abs(t) <0.7,1,0)
    data = np.zeros((len(x), len(y)))
    data[2:5, 9:14] = 1


    np.where(abs(y) <= 0.25, 1, 0)

    s1 = np.where(abs(t) <= 1, 1, 0)
    st = ((a/np.pi)**0.25) * np.exp(((-a*t**2)/2)+1j*b*t**2/2 + 1j*omega*t)

    rx,ry,rfunc = np.ravel(x),np.ravel(y),np.ravel(s2d)
    sx,sy = s2d.shape[0],s2d.shape[1]
    functionobject = np.stack((rx,ry,rfunc))

    functionobject = np.reshape(functionobject,(3,sx,sy))

    return functionobject    #x, z , amplitude

def integrateintensitywig(wignerobject, functionobject):
    print("we are in integrate function our x shape is")
    print(functionobject[0].shape)
    #print(x)
    wvdr = np.ravel(wignerobject)
    xr = np.ravel(functionobject[0])
    yr = np.ravel(functionobject[1])
    xr = np.ravel(wignerobject[0][2][1])


    uniquex,uniquexloc = np.unique(xr,return_inverse = 'true')
    uniquey,uniqueyloc = np.unique(yr,return_inverse = 'true')

    #print("xr")
    #print(xr)
    #print("uniquex")
    #print(uniquex)
    #print("uniquexloc")
    #print(uniquexloc)
    #print("wignerobjectspace")
    #print(wignerobject[0][2][1])

    #print("length of x")
    # print(len(xr))
    # print("length of wvd")
    # print(len(wignerobject[0]))


    #sumvec = []
    mapping = {}
    functionreturn = []


    for x in range(0,len(wignerobject)):           #for rows and columns

        if x == 0:  # if rows then we need y values from function and x values from wigner
            direction = 1
        else:
            direction = 0  # if columns we need x values from function and y values from wigner

        for y in range(0,len(wignerobject[x])):    #iterate through all rows/columns
            ravelamplitude = np.ravel(wignerobject[x][y][0])
            ravelspace = np.ravel(wignerobject[x][y][1]) #ravel space matrix to become 1D
            ravelphase = np.ravel(wignerobject[x][y][2])  #ravel the phase matrix to become 1D

            uniquizedspace = np.unique(ravelspace)

            sumvec = []
            for i in np.unique(ravelspace):
                # mapping[i] = np.where(x == i)[0]
                sum = ravelamplitude[np.where(ravelspace == i)].sum()
                # print(np.where(xr==i))
                # print("where the values correlate")
                sumvec.append(sum)

            if direction == 1:  # if rows then we need y values from function and x values from wigner
                perpvalues = np.repeat(functionobject[direction, y, 0],
                                       len(uniquizedspace))  # get the perpendicular location of space matrix
                functionreconstruct = np.stack((uniquizedspace,perpvalues,sumvec))

            else:  # if columns we need x values from function and y values from wigner
                perpvalues = np.repeat(functionobject[direction, 0, y],
                                       len(uniquizedspace))  # get the perpendicular location of space matrix
                functionreconstruct = np.stack((perpvalues,uniquizedspace,sumvec))

            functionreturn.append(functionreconstruct)

            print("the length of sumvec is: ")
            print(len(ravelamplitude))
            print(len(sumvec))

            # XY = np.stack((ravelspace, ravelphase))  # stack the two 1D matrices and create a 2xN matrix space and then phase
            # # print("raveled matrix shape")
            # # print(XY.shape)
            # transfravelxy = np.matmul(shearmatrix, XY)  # multiply matrix 2x2 raymatrix by 2xN xy matrix
            # xtransfered = transfravelxy[0]  # unstack into X and Y
            # ytransfered = transfravelxy[1]
            #
            # transformedwigner[x][y][0] = np.reshape(ravelamplitude,(wignerobject[x][y][0].shape))
            # transformedwigner[x][y][1] = np.reshape(xtransfered, (wignerobject[x][y][1].shape))  # reshape to original state and update object
            # transformedwigner[x][y][2] = np.reshape(ytransfered, (wignerobject[x][y][2].shape))



            #print(wignerobject[x][y])

    print("ourfunction dimensions are:")
    print(len(functionreturn))
    print(functionreturn[0].shape)

    print(functionreturn[15])
    # print(transformedwigner.shape)

    X,Y,Z = np.empty(0,float),np.empty(0,float),np.empty(0,float)
    print("our X is:")
    print(len(X))
    for i in range(len(functionreturn)):
        X = np.append(X,functionreturn[i][0])
        Y = np.append(Y, functionreturn[i][1])
        Z = np.append(Z, functionreturn[i][2])
        X.tolist()
        Y.tolist()
        Z.tolist()
    #functionobjectreturn = np.stack((X, Y, Z))
    functionobjectreturn = np.stack((X,Y,Z))
    #np.reshape(functionobjectreturn,(3,,-1))
    print("functionobjectreturn")
    print(functionobjectreturn.shape)
    # print(functionobjectreturn.shape)
    # hello = np.ravel(functionreturn)
    # print("hello")
    # print(len(hello))
    # print(hello)

    return functionobjectreturn

def screenfunction():
    z,y  = np.meshgrid(np.linspace(-250, -600, 81), np.linspace(-150, 150, 81))
    x = z*0 + y*0 - 450
    f = interpolate.interp2d(z[0, :], y[:, 0], x, kind='cubic')
    screenobject = np.stack((x, y, z))
    return screenobject, f

def mirrorfunction(type,x,y,zmin):
    a = -600
    b = -600
    if type == 'standard':
        #x, y = np.meshgrid(np.linspace(-150, 150, 50), np.linspace(-150, 150, 50))
        #z = (x**2)/a + (y**2)/b + 450

        #zplay[5][:] = -20
        #zplay = np.zeros(z.shape)
        #z = z+zplay

        x, y = np.meshgrid(np.linspace(-150, 150, 50), np.linspace(-150, 150, 50))
        z = (x ** 2) / a + (y ** 2) / b
        #z = 0*x + 0*y + 200
        # zplay[5][:] = -20
        #zplay = np.zeros(z.shape)
        zplay = 0*x
        #print(zplay)
        z = z + zplay
        f = interpolate.interp2d(x[0,:], y[:,0], z, kind='cubic')

        #zint = f(x[0,:],y[:,0])
        zinterp = f(x[0,:],y[:,0])

        x,y,zrotate = rotate(x,y,zinterp,15,'y')
        zrotate = zrotate + 450
        f = interpolate.interp2d(x[0,:], y[:,0], zrotate, kind='cubic')
        print("shape of zrotate is:" + str(zrotate.shape))

        #print("interpolated value is: ")
        #print(zinterp)
        z0 = np.zeros(zinterp.shape)
        zdist = zrotate-z0
        #zout = zdist+z
        #print("the value of the difference")
        #print(zdist)
        #print(zdist.shape)
        rx = np.ravel(x)
        ry = np.ravel(y)
        #z = np.ravel(z)
        zinterp = np.ravel(zinterp)
        zrotate = np.ravel(zrotate)
        zdist = np.ravel(zdist)
        mirrorobject = np.stack((rx,ry,zrotate,zdist))
        mirrorobject = np.reshape(mirrorobject,(4,len(x),len(y)))

        return mirrorobject, f
    else:
        z =  (x**2)/a + (y**2)/b + 450
        zplay = np.zeros(z.shape)
        z0 = np.zeros(z.shape)
        zdist = z + zplay - zmin
        zout = zdist + z
        print("the value of the difference")
        print(zdist)
        print(zdist.shape)
        x = np.ravel(x)
        #y = np.ravel(y)
        z = np.ravel(z)
        zdist = np.ravel(zdist)

        mirrorobject = np.stack((x, z, zdist))
        return mirrorobject

def plot2d(s2d, axis):
    H = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])  # added some commas and array creation code

    fig = plt.figure(figsize=(6, 3.2))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    plt.imshow(s2d)
    ax.set_aspect('equal')

    cax = fig.add_axes([0.12, 0.1, 0.78, 0.8])
    cax.get_xaxis().set_visible(False)
    cax.get_yaxis().set_visible(False)
    cax.patch.set_alpha(0)
    cax.set_frame_on(False)
    plt.colorbar(orientation='vertical')
    plt.show()

def plotinputs(st,axis):
    fig, ax = plt.subplots()
    #ax.plot(t, s1)
    ax.set(xlabel='x', ylabel='y',
       title='simple plot')
    ax.grid()
    ax.plot(axis,st)

    #conjst = np.conjugate(st)
    #corr = np.correlate(st,conjst,'full')
    #corrmatrix = np.tile(corr, ((len(corr)), 1))
    #np.savetxt('test.out', corrmatrix, delimiter=',')  # X is an array
    #print(corrmatrix)
    #tricorrmatrix = createtrimatrix(corrmatrix)
    #diagonalcorrmatrix = np.diag(np.diag(corrmatrix))
    #np.savetxt('test.out', tricorrmatrix, delimiter=',')  # X is an array
    #autocorrmatrix = np.matmul(trionesmatrix, corrmatrix)
    #np.savetxt('test.out', autocorrmatrix, delimiter=',')  # X is an array
    #wvd = np.apply_along_axis(fftrow, axis=1, arr=tricorrmatrix)
    #np.savetxt('test.out', wvd, delimiter=',')  # X is an array
    #wvd = abs(wvd)
    #realwvd = wvd.real
    #abswvd = abs(wvd)
    #scalewvd = np.interp(realwvd, (realwvd.min(), realwvd.max()), (-1, +1))

    #plt.matshow(st)
    #plt.colorbar()
    #plt.pcolor( vmin=-1, vmax=0.8)
    #plt.show()


    #print(corr[1000])

   # print("our correlation values are")
   # print(corr)

    #fcorr = fft(corr)

    #x = np.arange(1000)
    #fcorr = fftshift(fcorr)
    #plt.matshow(scalewvd)
    #plt.show(corr)
#fig.savefig("test.png")
    plt.show()

def plotMatrix(st):
    fig, ax = plt.subplots()
    # ax.plot(t, s1)
    ax.set(xlabel='space', ylabel='phase',
           title='Wigner Distribution')
    ax.grid()
    ax.plot(st)

    plt.matshow(st,origin='lower')
    #xpos = np.arange(len(xaxisline))
    #ypos = np.arange(len(freqs))
    plt.locator_params(nbins=10)
    #plt.xticks(xaxisline)
    #plt.yticks(freqs)
    plt.colorbar()
    plt.show()

def plot3dto2d(X,Y,Z):
    fig = plt.figure()
    ax = plt.axes(projection = '3d')

    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')

    ax.set_title('')


    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    #xv, yv = np.meshgrid(X, Y)
    #plt.contour(xv, yv, Z, 100, cmap='viridis')
    plt.show()
    return

def plotquiver3dobject(rayobject,x,y,z,u,v,w):

    X,Y,Z,U,V,W =[],[],[],[],[],[] #np.array,np.array,np.array,np.array,np.array,np.array


    for i in range(len(rayobject)):
        # X = X + (rayobject[i][x],)
        # Y = Y + (rayobject[i][y],)
        # Z = Z + (rayobject[i][z],)
        # U = U + (rayobject[i][u],)
        # V = V + (rayobject[i][v],)
        # W = W + (rayobject[i][w],)
        X = np.append(X,rayobject[i][x])
        Y = np.append(Y,rayobject[i][y])
        Z = np.append(Z,rayobject[i][z])
        U = np.append(U,rayobject[i][u])
        V = np.append(V,rayobject[i][v])
        W = np.append(W,rayobject[i][w])

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.8))

    # Make the direction data for the arrows
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
         np.sin(np.pi * z))

    print("our coordinate values are: ")
    print(X,Y,Z)

    ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, linewidths=0.5)
    #ax.set_xlim([-150, 150])
    #ax.set_ylim([-150, 150])
    #ax.set_zlim([0, 550])
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    plt.show()


    return

def plotquiver3d(X,Y,Z,U,V,W):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make the grid
    x, y, z = np.meshgrid(np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.2),
                          np.arange(-0.8, 1, 0.8))

    # Make the direction data for the arrows
    u = np.sin(np.pi * x) * np.cos(np.pi * y) * np.cos(np.pi * z)
    v = -np.cos(np.pi * x) * np.sin(np.pi * y) * np.cos(np.pi * z)
    w = (np.sqrt(2.0 / 3.0) * np.cos(np.pi * x) * np.cos(np.pi * y) *
         np.sin(np.pi * z))

    ax.quiver(X, Y, Z, U, V, W, length=1, normalize=False, linewidths = 0.1)
    ax.set_xlim([-150, 150])
    ax.set_ylim([-150, 150])
    ax.set_zlim([0, 550])
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    plt.show()

    return

def plotscatter(raysobject,a,b,c):
    X = []
    Y = []
    Z = []
    for rayholder in raysobject:  # for each function row of rays

        print("newrow")
        for i in range(len(rayholder[0])):
            mx = rayholder[a][i] #- rayholder[0][i]
            ny = rayholder[b][i] #- rayholder[1][i]
            oz = rayholder[c][i] #- rayholder[2][i]
            X.append(mx)
            Y.append(ny)
            Z.append(oz)
            vector = np.array([mx,ny,oz])
            #print(vector)

      # print("Hello plot points")
    # print(X)
    # print(Y)
    # print(Z)

    fig = plt.figure()
    ax = plt.axes(projection='3d')

    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.scatter(X, Y, Z, c = 'r', marker='o',s=0.1)

    ax.set_title('surface')

    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.clabel("z axis")
    # xv, yv = np.meshgrid(X, Y)
    # plt.contour(xv, yv, Z, 100, cmap='viridis')
    plt.show()
    return

def plotgridata(functiondata):


    # functiondata[0]
    # functiondata[1]
    # functiondata[2]

    points = np.asarray([functiondata[0],functiondata[1]]) # np.random.rand(1000, 2)
    #points = [functiondata[0],functiondata[1]]
    #points.tolist()
    #test = float(functiondata[0][0])
    values = np.asarray(functiondata[2]) #func(points[:, 0], points[:, 1])
    #values.tolist()

    print("checkout our functiondata")
    print(functiondata[0])
    #print(test)
    print(points)
    print(values)
    print(len(values))
    print(values.shape)
    values = np.expand_dims(values,axis=1)
    print(values.shape)

    #xmin = np.amin(functiondata[0])
    # min = np.amin(points[0][0])
    # xmax = int(np.max(functiondata[0]))
    #
    # ymin = int(np.min(functiondata[1]))
    # ymax = int(np.max(functiondata[1]))


    grid_x, grid_y = np.meshgrid(np.linspace(np.amin(functiondata[0]), np.amax(functiondata[0]), 30),
                                 np.linspace(np.amin(functiondata[1]), np.amax(functiondata[1]), 30))

    #grid_z0 = griddata(points.T, values, (grid_x, grid_y), method='nearest')
    grid_z1 = griddata(points.T, values, ((grid_x, grid_y)), method='linear')
    #grid_z2 = griddata(points.T, values, (grid_x, grid_y), method='cubic')

    print("griddata is:")
    #print(len(grid_z0))
    #print(grid_z0.shape)
    #grid_z0 = np.squeeze(grid_z0)
    grid_z1 = np.squeeze(grid_z1)
    #grid_z2 = np.squeeze(grid_z2)



    # plt.subplot(221)
    # plt.imshow(values.T, extent=(0, 1, 0, 1), origin='lower')
    # plt.plot(points[:, 0], points[:, 1], 'k.', ms=1)
    # plt.title('Original')
    # plt.subplot(222)
    # plt.imshow(grid_z0.T)
    # plt.title('Nearest')
    # plt.subplot(223)
    # plt.imshow(grid_z1.T,  origin='lower')
    # plt.title('Linear')
    # plt.subplot(224)
    # #plt.imshow(grid_z2.T, extent=(0, 1, 0, 1), origin='lower')
    # plt.imshow(grid_z2.T, origin='lower')
    # plt.title('Cubic')
    # plt.gcf().set_size_inches(6, 6)
    # plt.show()

    plt.title('Linear')
    #plt.subplot(224)
    plt.imshow(grid_z1.T, extent=(0, 1, 0, 1), origin='lower')







    return

def contourplot(X,Y,Z):
    plt.contourf(X, Y, Z, 1, cmap='viridis')
    plt.colorbar()
    plt.show()

def getfourierfrequencies(function):
    freqs = np.fft.fftfreq(len(function))
    return freqs

def fftrow(array):
    #return fftshift(fft(array))
    #array[0::2] = -array[0::2]
    fouriervector = np.real(np.abs(fft(array)))
    #print(fouriervector)
    freqs = np.fft.fftfreq(len(fouriervector))
    #print(freqs)
    return fouriervector

def fouriereachvector(corrmatrix):
    wvd = np.apply_along_axis(fftrow, axis=1, arr=corrmatrix)  #forier each row
    wvd = wvd.T
    freqs = np.fft.fftfreq(len(wvd))
    #print("the fourier dimensions are")
    #print(wvd.shape)
    return wvd

def prepcorrmatrixfromvector(vector):
    zerovector = np.zeros(len(vector))
    doublevector = np.append(vector,zerovector)
    matrixcirculant = circulant(doublevector)
    #print(matrixcirculant)
    #matrixcirculant = matrixcirculant.T
    matrixcirculant[1::2] = -matrixcirculant[1::2]
    reversedoublevector = np.append(zerovector,vector)
    #reversedoublevector[1::2] = -reversedoublevector[1::2]
    conjreversedoublevector = np.conjugate(doublevector)
    iteratedmatrix = np.multiply(matrixcirculant,conjreversedoublevector)
    #print(matrixcirculant)
    return iteratedmatrix

def wignerfunction(function,space,frequencies):

    # print("iteration")
    # print(function)
    corrmatrix = prepcorrmatrixfromvector(function)
    wignerdist = fouriereachvector(corrmatrix)
    #wignerdist[1::2] = -wignerdist[1::2]

    # print("the shape of the wigner is:")
    # print(wignerdist.shape)
    # print(space.shape)
    # print(frequencies.shape)

    #wignerdist = np.abs(wignerdist)
    ravelwig = np.ravel(wignerdist)
    ravelspace = np.ravel(space)
    ravelfreq = np.ravel(frequencies)

    wignerblock = np.stack((ravelwig,ravelspace,ravelfreq))
    rows = wignerdist.shape[0]  #space
    cols = wignerdist.shape[1]  #phase
    shape = -1,rows,cols
    wignerblock = np.reshape(wignerblock,shape)
    #print("the shape of the wigner block is:")
    #print(wignerblock.shape)


    return wignerblock

def fitaxis(axis):
    x = np.linspace(np.amin(axis), np.amax(axis), 2*len(axis))
    print("axis min and axis max")
    print(np.amin(axis))
    print(np.amax(axis))
    #interpaxis = np.interp(x,axis,axis)
    return x

def getmeshgrid(x,y):
    xx, yy = np.meshgrid(x, y)
    return xx,yy

def wignerizeeachfunction(functionobject):

    xinterpaxis = fitaxis(functionobject[0][0,:])  # upsample - calculate x axis that suits wigner - ie rows
    yinterpaxis = fitaxis(functionobject[1][:, 0])  # upsample - calculate y axis that suits wigner - ie columns

    #print("the x and y upsampled region is")
    #print(xinterpaxis)
    #print(yinterpaxis)

    rowfreqs = getfourierfrequencies(xinterpaxis)  # get the phasespace for rows
    colfreqs = getfourierfrequencies(yinterpaxis)  # get the phasespace for columns

    rowspace, rowfrequencies = getmeshgrid(xinterpaxis,rowfreqs)  # build axis dimensions correlating space and phase for rows of function
    colspace, colfrequencies = getmeshgrid(yinterpaxis,colfreqs)  # build axis dimensions correlating space and phase for columns of function

    # x,y,xp,yp = np.meshgrid(xinterpaxis,yinterpaxis,rowfrequencies,colfrequencies)
    # print("our 4 dimensional grid is")
    # print(x,y,xp,yp)


    #print(twodinput)
    #print(twodinput.shape)
    rowwvd = np.apply_along_axis(wignerfunction ,axis = 1, arr = functionobject[2],space = rowspace, frequencies = rowfrequencies)  #rows
    print("finished with the rows")
    colwvd = np.apply_along_axis(wignerfunction, axis = 1, arr = functionobject[2].T,space = colspace, frequencies = colfrequencies) #columns
    # first dimension is the function, second is the correlation matrix, 3rd is the fft



    #print(rowwvd)

    #print(xaxisline)
    #print(yaxisline)

    print("our row and column shape is:")
    print(rowwvd.shape)
    print(colwvd.shape)

    ravelrowwvd = np.ravel(rowwvd)
    ravelcolwvd = np.ravel(colwvd)

    print("shape of ravel is")
    print(ravelcolwvd.shape)
    print(ravelrowwvd.shape)

    ravelx = np.ravel(functionobject[0])  # X values
    ravely = np.ravel(functionobject[1])    #Y values
    ravel2d = np.ravel(functionobject[2])   #Amplitude values

    function = np.stack((ravel2d,ravelx,ravely))
    #function = np.reshape(function,(3,a,b))
    print("our function's shape is")
    print(function.shape)


    wignerobject = np.stack((ravelrowwvd,ravelcolwvd))
    a,b,c,d = colwvd.shape[0],colwvd.shape[1],colwvd.shape[2],colwvd.shape[3]
    wignerobject = np.reshape(wignerobject,(-1,a,b,c,d))

    print("wignerobject shape is:")
    print(wignerobject.shape)
    #rowwvd = np.reshape(rowwvd,(20,1600))

    #np.savetxt('test.out', rowwvd, delimiter=',')   # X is an array

    #WVD = np.append(rowwvd,colwvd)

    return wignerobject

def rotationmatrices(theta,direction):
    Rx = [[1,0,0],                             [0, np.cos(theta), -np.sin(theta)],      [0, np.sin(theta),np.cos(theta)]]
    Ry = [[cos(theta), 0 , sin(theta)],  [0,1,0],                                 [-sin(theta),0, cos(theta)]]
    Rz = [[cos(theta),-sin(theta),0],    [sin(theta),cos(theta),0],         [0,0,1]]

    if direction =='x':
        Rreturn = Rx
    elif direction =='y':
        Rreturn = Ry
    else:
        Rreturn = Rz

    return Rreturn

def projection(source, mirrorloc, mirrorfunction):

    return source

def rotate(x,y,z,angle,direction):
    ravelx = np.ravel(x)  # ravel the X matrix to become 1D
    ravely = np.ravel(y)  # ravel the Y matrix to become 1D
    ravelz = np.ravel(z)
    XYZ = np.stack((ravelx, ravely,ravelz))  # stack the two 1D matrices and create a 2xN matrix X and then Y
    # print(XY.shape)

    radian = math.radians(angle)
    transfravelxyz = np.matmul(rotationmatrices(radian,direction), XYZ)  # multiply matrix 2x2 raymatrix by 2xN xy matrix
    # print(transfravelxy.shape)

    xtransfered = transfravelxyz[0]  # unstack into X and Y
    ytransfered = transfravelxyz[1]
    ztransfered = transfravelxyz[2]

    xtransfered = np.reshape(xtransfered, (x.shape))  # reshape to original state
    ytransfered = np.reshape(ytransfered, (y.shape))
    ztransfered = np.reshape(ztransfered, (z.shape))

    return (xtransfered,ytransfered,ztransfered)

def getthresholdlocations(function):            #create matrix where the value of function in each row is above the threshold
    rows = np.where(function.any(axis=1))     #rows
    columns = np.where(function.any(axis=0))      #columns
    rowscolumns = [rows,columns]
    print(rowscolumns)
    return rowscolumns

def raytransforms(wignerobject,dist):
    #nrows, ncolumns = wvd.shape

    refl = -2/1000
    shearmatrix = np.array([[1, dist], [0, 1]])
    #shearmatrix = np.array([[1,0],[refl,1]])
    #transformedwigner = []
    transformedwigner = np.empty(wignerobject.shape)

    for x in range(0,len(wignerobject)):           #for rows and columns
        for y in range(0,len(wignerobject[x])):    #iterate through all rows/columns
            ravelamplitude = np.ravel(wignerobject[x][y][0])
            ravelspace = np.ravel(wignerobject[x][y][1]) #ravel space matrix to become 1D
            ravelphase = np.ravel(wignerobject[x][y][2])  #ravel the phase matrix to become 1D
            XY = np.stack((ravelspace, ravelphase))  # stack the two 1D matrices and create a 2xN matrix space and then phase
            # print("raveled matrix shape")
            # print(XY.shape)
            transfravelxy = np.matmul(shearmatrix, XY)  # multiply matrix 2x2 raymatrix by 2xN xy matrix
            xtransfered = transfravelxy[0]  # unstack into X and Y
            ytransfered = transfravelxy[1]

            transformedwigner[x][y][0] = np.reshape(ravelamplitude,(wignerobject[x][y][0].shape))
            transformedwigner[x][y][1] = np.reshape(xtransfered, (wignerobject[x][y][1].shape))  # reshape to original state and update object
            transformedwigner[x][y][2] = np.reshape(ytransfered, (wignerobject[x][y][2].shape))



            #print(wignerobject[x][y])

    print("our new shape is")
    print(transformedwigner.shape)
    return transformedwigner

def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step

def buildraymatrices(wignerobject, transformedwignerobject, functionobject):
    # b = Line3D(Point3D(1, 3, 4), Point3D(2, 2, 2))
    zdist = 750
    # transformedwignerobject = raytransforms(wignerobject,zdist)
    rayobject = []
    #rayobject = np.array
    updatedravelamp = ()
    ravelspace = np.array
    ravelamplitude = np.array

    print("wignerobject shaspes are the sames?")
    print(wignerobject.shape)
    print(transformedwignerobject.shape)
    for x in range(0, len(wignerobject)):  # for rows and then columns\
        if x == 0:  # if rows then we need y values from function and x values from wigner
            direction = 1
        else:
            direction = 0  # if columns we need x values from function and y values from wigner

        for y in range(0, len(wignerobject[x])):  # for each row or column - iterate through all rows and then columns
            ravelspace =  np.ravel(wignerobject[x, y, 1])  # ravel space matrix of wigner transform to become 1D
            print(ravelspace.shape)
            ravelamplitude = np.ravel(wignerobject[x, y, 0])
            # print(ravelspace)

            print("New iteration of updateravelamp")
            # print(updatedravelamp)
            ravelspacetrans = np.ravel(transformedwignerobject[x, y, 1])
            # print("translated x values")
            # print(ravelspacetrans)
            # print("original ravelspace")
            # print(ravelspace)
            # ravelspacetrans = transformedwignerobject[x,y,1]
            # print(ravelspacetrans)
            # print(ravelspace.shape)
            # print(ravelspacetrans.shape)

            z0 = np.zeros(ravelspace.shape)
            zd = np.repeat(zdist, len(ravelspace))
            if direction == 1:  # if rows then we need y values from function and x values from wigner
                perpvalues = np.repeat(functionobject[direction, y, 0],
                                       len(ravelspace))  # get the perpendicular location of space matrix

                raypacket = np.stack((ravelamplitude,ravelspace, perpvalues, z0, ravelspacetrans, perpvalues, zd))
                # function is built like: X then Y then Amplitude
                perpvaluestrans = 0
                # print("our perp axis values are")
                # print(perpvalues)

            else:  # if columns we need x values from function and y values from wigner
                perpvalues = np.repeat(functionobject[direction, 0, y],
                                       len(ravelspace))  # get the perpendicular location of space matrix
                raypacket = np.stack((ravelamplitude,perpvalues,ravelspace, z0, perpvalues, ravelspacetrans, zd))
                # raypacket = np.stack((perpvalues, ravelspace, z0, ravelspacetrans, perpvalues, zd))

                perpvaluestrans = 0
                # print("our perp axis values are")
                # print(perpvalues)
            # print(ravelspacetrans.shape,ravelspace.shape,perpvalues.shape,z0.shape,zd.shape)
            # raypacket = np.stack((ravelspace,perpvalues,z0,ravelspacetrans,perpvalues,zd))
            # print("the raypackets' shape is")
            # print(raypacket.shape)
            # rayarray = np.array([raypacket])
            #np.append(rayobject, raypacket)

            rayobject.append(raypacket)

            # functionspace= np.ravel(functionobject[1][])
            # z = mirrorinterpolator(perpspace,ravelspace)
            # print("our ray objects shape is")
            # print(len(rayobject))
            # print(rayobject.shape)
            # print("rayobject length is")
            # print(len(rayobject))

    return rayobject

def viewvectors(raysobject):
    for rayholder in raysobject:  # for each function row of rays
        print("newrow")
        for i in range(len(rayholder[0])):
            mx = rayholder[3][i] - rayholder[0][i]
            ny = rayholder[4][i] - rayholder[1][i]
            oz = rayholder[5][i] - rayholder[2][i]

            vector = np.array([mx,ny,oz])
            print(vector)

    return

def buildintersectionswithmirror(raysobject, mirrorinterp, mirrorobject):

    print(np.max(mirrorobject[0]))
    maxmin = np.array(([np.max(mirrorobject[0]),np.min(mirrorobject[0])],[np.max(mirrorobject[1]),np.min(mirrorobject[1])]))
    print(maxmin)
    print("rayholder before mirror")
    print(np.shape(raysobject))
    rayobjectreturn = []
    print(np.shape(raysobject[0])[0])

    tb = time.time()
    ta = time.time()
    for rayholder in raysobject:                    #for each function row of rays
        removeelements = []

        print("rayholder before")
        print(ta-tb)
        tb = time.time()

        #print(rayholder)
        #print("new line")
        #rayholder = rayobject[j]
        #print(rayholder)
        rayholderbuild = np.array
        #rayholderbuild = np.stack(
        #    (rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5], rayholder[6]))

        twb = time.time()
        for i in range(len(rayholder[0])):          # run through all elements along array
            #print("iteration number",i)
            #print("new iteration through rayholder", i)

            mx = rayholder[4][i]-rayholder[1][i]
            ny = rayholder[5][i]-rayholder[2][i]
            oz = rayholder[6][i]-rayholder[3][i]
            zint = 10
            zray = 1
            top = 1
            bottom = 0
            delta = (top-bottom)/2
            error = 10000
            checkpoint = delta
            #zray = rayholder[2][i] + oz*checkpoint
            #print(zray)
            zint = mirrorinterp((rayholder[1][i] + mx * checkpoint), (rayholder[2][i] + ny * checkpoint))

            while error>0.1:

                tsb = time.time()
                zray = rayholder[3][i] + oz * checkpoint

                #print(zint,zray,error)
                if zray < zint:
                    #print("we are in if")
                    bottom = bottom + delta
                    delta = (top - bottom) / 2
                    checkpoint = bottom+delta
                    #print("checkpoint, top, bottom, delta, error")
                    #print(checkpoint,top,bottom,delta,error)
                else:
                    #print("we are in else")
                    top = top - delta
                    delta = (top - bottom) / 2
                    checkpoint = top - delta
                    #print("checkpoint, top, bottom, delta, error")
                    #print(checkpoint,top,bottom,delta,error)


                # print("the checkpoint value is")
                # print(checkpoint)
                xloc = rayholder[1][i] + mx * checkpoint
                yloc = rayholder[2][i] + ny * checkpoint
                zint = mirrorinterp((rayholder[1][i] + mx * checkpoint), (rayholder[2][i] + ny * checkpoint))
                zray = rayholder[3][i] + oz * checkpoint
                error = abs(zray - zint)



            #print("found the spot")

                # print("checkpoint, x and y location:",checkpoint,xloc,yloc)
                # print("our interp z:",zint,"our ray location is:",zray,"our measure:",error)
            #print("before")
            #print(rayholder[5][i])
            #np.append(rayholder[4], rayholder[1][i] + mx * checkpoint)
            #np.append(rayholder[5], rayholder[2][i] + mx * checkpoint)
            #np.append(rayholder[6], rayholder[3][i] + mx * checkpoint)
            #print("size of rayholder :")
            #print(np.shape(rayholder[4]))

            rayholder[4][i] = (rayholder[1][i] + mx * checkpoint)
            rayholder[5][i] = rayholder[2][i] + ny * checkpoint
            rayholder[6][i] = rayholder[3][i] + oz * checkpoint

            #print(rayholder[4][i], maxmin[0, 0])

            if rayholder[4][i] < maxmin[0, 0] and rayholder[4][i] > maxmin[0, 1] and rayholder[5][i]<maxmin[1,0] and rayholder[5][i]>maxmin[1,1]:
                x = 0
                #np.delete(rayholder[:], i)
                #print("we are in bounds")
                #print(rayholderbuild.shape)
            else:
                #print("we are out of bounds")
                removeelements.append(i)
                #print(removeelements)
                #print(len(rayholder[0]))
                continue
            #print("just after remove elements")
            #print(removeelements)
            #print(rayholderbuild.shape)

        tsa = time.time()
        print("x",tsa - twb)
        #print("remove elements")
        #print(removeelements)

        ta = time.time()
        rayholderbuild = np.delete(rayholder[:], np.s_[removeelements], 1)
        #print(len(rayholderbuild[0]))
        rayobjectreturn.append(rayholderbuild)
        #print("end of iteration for rayholderbuild")
        #print(len(rayobjectreturn))
        #print(np.shape(rayobjectreturn))

            #print(rayholder[4][i])
            #print(np.shape(rayholder[4]))

            #print("after")
            #print(rayholder[5][i])
            # print("saved value")
            # print("the intersection point is ")
            # print(rayholder[0][i]+mx*checkpoint,rayholder[1][i]+ny*checkpoint,rayholder[2][i]+oz*checkpoint)
    print(len(rayobjectreturn))
    print("rayholder after")
    #print(rayobjectreturn)
    return rayobjectreturn

def buildplaneofmirrorintersections(raysobject,mirrorinterpolator):         #get the normalized normal of the plane
    rayobjectreturn =[]
    for rayholder in raysobject:  # for each function row of rays
        # print("rayholder before")
        # print(rayholder)
        # print("new line")
        rayholder = np.stack((rayholder[0],rayholder[1],rayholder[2],rayholder[3],rayholder[4],rayholder[5],rayholder[6],rayholder[4],rayholder[5],rayholder[5]))
        print("rayholder shape is:")
        print(rayholder.shape)
        for i in range(len(rayholder[0])):  # run through all elements along array
            # print("iteration number",i)
            x = rayholder[4][i]    # get the xyz coordinates of intersection with mirror
            y = rayholder[5][i]
            z = rayholder[6][i]
            dx = 0.2
            dy = 0.2

            p1x = x                             #create triangulation points
            p1y = y + dy*np.sqrt(2)              #In order to be able to calculate reflection normal
            p2x = x +dx
            p2y = y - dy
            p3x = x - dx
            p3y = y -dy

            #z = mirrorinterpolator(x, y)
            p1z = mirrorinterpolator(p1x, p1y)                 #get equivelant z points of interpelation points
            p2z = mirrorinterpolator(p2x, p2y)
            p3z = mirrorinterpolator(p3x,p3y)

            p1 = np.array([p1x, p1y, float(p1z)])
            p2 = np.array([p2x, p2y, float(p2z)])
            p3 = np.array([p3x, p3y, float(p3z)])
            #print(p1,p2,p3)
            # These two vectors are in the plane
            v1 = p3 - p1
            v2 = p2 - p1
            # the cross product is a vector normal to the plane
            cp = np.cross( v2,v1)

            # print("the normal vector and its normalized version")
            # print(cp)
            cp = cp/LA.norm(cp)
            #print("our normal vector is")
            #print(cp)

            a, b, c = cp
            #print(a,b,c)
            rayholder[7][i] = a
            rayholder[8][i] = b
            rayholder[9][i] = c
            #cp = np.reshape(cp,(-1,1))
            #X_normalized = preprocessing.normalize(cp , axis=1,norm='l1')
            #print(cp)
            norm = [float(i) / sum(cp) for i in cp]
        rayobjectreturn.append(rayholder)



            #print("above us is cp, below normalized")
            #print(norm)
            # This evaluates a * x3 + b * y3 + c * z3 which equals d
            #d = np.dot(cp, p3)
            #print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))

            #normal = a.normal_vector
    print("our plane is defined as")
    #print(a)
    print("Shape of the object")
    print(rayobjectreturn[0].shape)
    #print(normal)
    return rayobjectreturn

def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        #raise RuntimeError("no intersection or line is within plane")
        a = np.empty((3,))
        a[:] = np.nan
        return a

    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    psi = w + si * rayDirection + planePoint
    return psi

def getrayfocalpoints(raysobject):
    return

def getvector(raysobject):
    rayobjectreturn = []
    for rayholder in raysobject:  # for each function row of rays
        rayholder = np.stack((rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5],rayholder[6],
                              # 1* amplitude # 3* spatial origins of ray, 3* intersection point on surface
                               rayholder[7], rayholder[8], rayholder[9],
                              # 3* normal of the surface  1* angle between
                              rayholder[9], rayholder[9], rayholder[9]))  # 3* reflected rays                                                                                         #

        print("rayholder shape is:")
        print(rayholder.shape)
        for i in range(len(rayholder[0])):  # run through all elements along array
            # print("iteration number",i)
            normal = np.array([rayholder[6][i], rayholder[7][i], rayholder[8][i]])

            # r = d−2(d⋅n)n

            # rayholder[10][i] =
            # rayholder[11][i] =
            # rayholder[12][i] =
    return rayobjectreturn

def dotproduct(v1, v2):
  return sum((a*b) for a, b in zip(v1, v2))

def length(v):
  return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
  return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def getanglesbetweenrayandnormal(raysobject):
    rayobjectreturn = []
    for rayholder in raysobject:  # for each function row of rays
        # print("rayholder before")
        # print(rayholder)
        # print("new line")
        rayholder = np.stack((rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5],rayholder[6], rayholder[7], rayholder[8],rayholder[9],rayholder[9]))
        # print("rayholder shape is:")
        # print(rayholder.shape)
        for i in range(len(rayholder[0])):  # run through all elements along array
            # print("iteration number",i)
            x = rayholder[4][i]
            y = rayholder[5][i]
            z = rayholder[6][i]

            a = rayholder[7][i]
            b = rayholder[8][i]
            c = rayholder[9][i]
            v1 = np.array([x, y, z])
            v2 = np.array([a, b, c])
            rayholder[10][i] = angle(v1, v2)
            #print(rayholder[9][i])
        rayobjectreturn.append(rayholder)
    return rayobjectreturn

def getreflectedvector(raysobject):
    rayobjectreturn = []
    for rayholder in raysobject:  # for each function row of rays
        # print("rayholder before")
        # print(rayholder)
        # print("new line")
        rayholder = np.stack((rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5], rayholder[6],
                              # 1* Amplitude     # 3* spatial origins of ray, 3* intersection point on surface
                               rayholder[7], rayholder[8], rayholder[9],rayholder[10]
                              # 3* normal of the surface  1* angle between
                              ,rayholder[9],rayholder[9],rayholder[9]))     # 3* reflected rays                                                                                         #

        # print("rayholder shape is:")
        # print(rayholder.shape)
        for i in range(len(rayholder[0])):  # run through all elements along array
            # print("iteration number",i)
            normal = np.array([ rayholder[7][i], rayholder[8][i], rayholder[9][i]])


            x = rayholder[4][i]-rayholder[1][i]
            y = rayholder[5][i]-rayholder[2][i]
            z = rayholder[6][i]-rayholder[3][i]
            d = np.array([x, y, z])
            n= np.array([rayholder[7][i],rayholder[8][i],rayholder[9][i]])

            ndot = dotproduct(d,n)
            #print(ndot)
            r = d- 2*(ndot*n)
            #print(r)

            # print("the reflected vector is")
            # print(r)
            r = r/LA.norm(r)
            r = r*100
            # print(r)



            rayholder[11][i] = r[0]
            rayholder[12][i] = r[1]
            rayholder[13][i] = r[2]

        rayobjectreturn.append(rayholder)
    print("rayholder shape is:")
    print(rayholder.shape)
    print(len(rayobjectreturn))
    return rayobjectreturn

def buildintersectionswithscreen(raysobject,screeninterp):
    print("rayholder before")
    # print(raysobject)
    rayobjectreturn = []

    for rayholder in raysobject:  # for each function row of rays
        # print("rayholder before")
        # print(rayholder)
        # print("new line")
        rayholder = np.stack(
            (rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5], rayholder[6],
             # 1* Amplitude     # 3* spatial origins of ray, 3* intersection point on mirror
             rayholder[7], rayholder[8], rayholder[9], rayholder[10]
             # 3* normal of the surface  1* angle between
             , rayholder[11], rayholder[12], rayholder[13]     # 3* reflected rays
             , rayholder[11], rayholder[12], rayholder[13],))   # 3* intersection points with screen
        for i in range(len(rayholder[0])):  # run through all elements along array
            # print("iteration number",i)
            mx =  rayholder[11][i] #- rayholder[4][i]
            ny =  rayholder[12][i] #- rayholder[5][i]
            oz =  rayholder[13][i] #- rayholder[6][i]

            raydirection = np.array((mx,ny,oz))
            raypoint = np.array((rayholder[4][i],rayholder[5][i],rayholder[6][i]))

            planenormal = np.array([1,0,0])
            planepoint = np.array([-200,0,450])

            intersection = LinePlaneCollision(planenormal, planepoint, raydirection, raypoint)
            #if np.absolute(intersection[2])>10000 :
            #    print("the ray has intersected out of bounds")
            #    continue
                #intersection = [0,0,0]
            #print(intersection)
            rayholder[14][i] = intersection[0]
            rayholder[15][i] = intersection[1]
            rayholder[16][i] = intersection[2]

            #print(rayholder[11][i], rayholder[12][i], rayholder[13][i])

            #print("intersection point")
            #print(rayholder[14][i],rayholder[15][i],rayholder[16][i])     #intersection points with screen
            # print("after")
            # print(rayholder[5][i])
            # print("saved value")
            # print("the intersection point is ")
            # print(rayholder[0][i]+mx*checkpoint,rayholder[1][i]+ny*checkpoint,rayholder[2][i]+oz*checkpoint)
        rayobjectreturn.append(rayholder)
    print("rayholder after")
    # print(rayobjectreturn)
    return rayobjectreturn

def errorvaluecalc(raysobject, actualoutputfunction):
    accuracy = 0
    precision = 0

    idealx,idealy,idealz = -200, 0,130
    print("our raysobject shape is in the error")
    print(len(raysobject))
    averagedist = []

    for rayholder in raysobject:                    #for each function row of rays
        ytrue = np.zeros(len(rayholder[15]))
        #accuracy = accuracy_score(ytrue, rayholder[15])
        #print(accuracy)
        true = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.4, 0.35, 0.8])
        print(ytrue.shape)
        print(rayholder[15].shape)
        y_val_true, val_pred = ytrue.reshape((-1)), rayholder[15,:].reshape((-1))

        rayholder = np.stack(
            (rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5], rayholder[6],
             # 1* Amplitude     # 3* spatial origins of ray, 3* intersection point on mirror
             rayholder[7], rayholder[8], rayholder[9], rayholder[10]
             # 3* normal of the surface  1* angle between
             , rayholder[11], rayholder[12], rayholder[13]  # 3* reflected rays
             , rayholder[14], rayholder[15], rayholder[16], # 3* intersection points with screen
             rayholder[16],))  #distance from desired focal point
        total = 0
        for i in range(len(rayholder[0])):  # run through all elements along array

            rayholder[17][i] = math.sqrt((rayholder[14][i]- idealx)**2 +(rayholder[15][i]- idealy)**2 +(rayholder[16][i]- idealz)**2)*abs(rayholder[0][i])
        averagedist.append(np.mean(rayholder[17]))
        print('our average distance is: ' + str(averagedist))
        #tot += ((((data[i + 1:] - data[i]) ** 2).sum(1)) ** .5).sum()
        #avg = tot / ((data.shape[0] - 1) * (data.shape[0]) / 2.)

    averagedistance = np.mean(averagedist)
    print("the average precision: "+ str(averagedistance))

#        print(average_precision_score(ytrue.flatten(), rayholder[15].flatten()))
        #     for i in range(len(rayholder[0])):          # run through all elements along array
    #         #print("iteration number",i)
    #         #print("new iteration through rayholder", i)
    #         mx = idealx-rayholder[14][i]
    #         ny = idealy-rayholder[15][i]
    #         oz = idealz-rayholder[16][i]
    #
    # print("rayholder after")
    # #print(rayobjectreturn)
    return 0

def derayandrebuildxwigner(raysobject,wignerobject):

    restoredwignerobject = []

    for rayholder in raysobject:  # for each function row of rays
        # print("rayholder before")
        # print(rayholder)
        # print("new line")
        rayholder = np.stack(
            (rayholder[0], rayholder[1], rayholder[2], rayholder[3], rayholder[4], rayholder[5], rayholder[6],
             # 1* Amplitude     # 3* spatial origins of ray, 3* intersection point on mirror
             rayholder[7], rayholder[8], rayholder[9], rayholder[10]
             # 3* normal of the surface  1* angle between
             , rayholder[11], rayholder[12], rayholder[13]  # 3* reflected rays
             , rayholder[14], rayholder[15], rayholder[16],))  # 3* intersection points with screen                                                                                   #

        # print("rayholder shape is:")
        # print(rayholder.shape)
        for i in range(len(rayholder[0])):  # run through all elements along array
            # print("iteration number",i)

            location = np.array([rayholder[14][i], rayholder[15][i], rayholder[16][i]])
            xy = rayholder[12][i]/rayholder[11][i]
            xz = rayholder[13][i]/rayholder[11][i]
            zplusy = rayholder[12]+rayholder[13]
            xyamp =rayholder[12]/zplusy
            xzamp = rayholder[13]/zplusy

            wignerblock = np.array([[rayholder[0]*xzamp],[rayholder[16]],[xz]])
            wignerblock = np.array([[rayholder[0]*xyamp],[rayholder[15]],[xy]])


            # rayholder[11][i] = r[0]
            # rayholder[12][i] = r[1]
            # rayholder[13][i] = r[2]

        restoredwignerobject.append(wignerblock)
    print("rayholder shape is:")



    restoredwignerobject = wignerobject


    return

def errortest():
    return 0

def neuralnet(mirrordimensions,inputvalues, outputvalues):
    #model = Sequential()
    #model.add(Dense(units=64, activation='relu', input_dim=100))
    #model.add(Dense(units=10, activation='softmax'))

    print(len(mirrordimensions[0]))

    model = Sequential()
    model.add(Dense(7, input_shape=(7,), activation='relu'))
    # model.add(Dense(40, activation='relu'))
    model.add(Dense(7*len(mirrordimensions[0])**2, activation='relu'))
    model.add(Dense(len(mirrordimensions[0])**2, activation='relu'))
    model.add(Dense(1))
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    return


##################################################################################################################################
######################################### PROGRAM RUNS FROM HERE #################################################################
##################################################################################################################################




mirrorobject, mirrorinterpolator = mirrorfunction('standard',0,0,0)
#print("the mirror distance function is")
#print(mirrorobject[2])

#plot3dto2d(mirrorobject[0],mirrorobject[1],mirrorobject[2])

functionobject = inputfunction()   #returns x , y and the Function value/amplitude

# print(xaxisline.shape)
# print(yaxisline.shape)

#plot3dto2d(functionobject[0],functionobject[1],functionobject[2])   #xyz of function input

zwignerobject = wignerizeeachfunction(functionobject)    #wigner amp, space, frequency


#wignerobject = abs(wignerobject)

plot3dto2d(zwignerobject[0][5][1],zwignerobject[0][5][2],zwignerobject[0][5][0])   #pass in space, phase and wigner amplitude


zwignerobjecttrans = raytransforms(zwignerobject,50)


#plot3dto2d(wignerobjecttrans[0][5][1],wignerobjecttrans[0][5][2],wignerobjecttrans[0][5][0])   #pass in space, phase and wigner
reversefunction = integrateintensitywig(zwignerobject,functionobject)
plotgridata(reversefunction)


#contourplot(reversefunction[0],reversefunction[1],reversefunction[2])


############## raysobject incorporates the following information
##############   0,1,2,
##############

rayobject = buildraymatrices(zwignerobject,zwignerobjecttrans,functionobject)


print("in the main after matrix built")
print(np.shape(rayobject))

#viewvectors(rayobject)

#plotscatter(rayobject)

#print(mirrorinterpolator(150.254355400696866, -8.0))

rayobjects = buildintersectionswithmirror(rayobject,mirrorinterpolator,mirrorobject)


print("in the main, the rayobject size is")
print(rayobjects[0].shape)


#buildplaneofmirrorintersections(rayobject,mirrorinterpolator)

#viewvectors(rayobject)


#plotscatter(rayobject)


rayobject = buildplaneofmirrorintersections(rayobjects,mirrorinterpolator)

rayobject = getanglesbetweenrayandnormal(rayobject)

rayobject = getreflectedvector(rayobject)


print("the length of ray object is")
print(len(rayobject))
print(len(rayobject[3]))


print("our mirror intropolator for values out of bounds are: ")
print(mirrorinterpolator(151,0))
print(mirrorinterpolator(155,0))
print(mirrorinterpolator(200,0))

screenobject,screeninterp = screenfunction()

rayobject = buildintersectionswithscreen(rayobject,screeninterp)

#print(len(rayobject))
#print(rayobject[0].shape)


errorvaluecalc(rayobject, 0)



print("the length of ray object is")
print(len(rayobject))
print(len(rayobject[3]))

#plot3dto2d(screenobject[0],screenobject[1],screenobject[2])
#plot3dto2d(mirrorobject[0],mirrorobject[1],mirrorobject[2])

plotscatter(rayobject,14,15,16)   #plot the locations where the reflected rays hit the screen
#plotscatter(rayobject,4,5,6)   #plot the locations where the reflected rays hit the screen

#plotquiver3dobject(rayobject,14,15,16,11,12,13) #plot the vectors at the locations wehre the reflected rays hit the screen

#plotquiver3d(rayobject[6][1], rayobject[6][2], rayobject[6][3], rayobject[6][4], rayobject[6][5], rayobject[6][6])   #where rays meet mirror
#plotquiver3d(rayobject[6][4], rayobject[6][5], rayobject[6][6], rayobject[6][11], rayobject[6][12], rayobject[6][13])    #reflected rays
#plotquiver3d(rayobject[6][4], rayobject[6][5], rayobject[6][6], rayobject[6][14], rayobject[6][15], rayobject[6][16]) #where rays meet screen  ??????

errorvaluecalc(rayobject, 0)

neuralnet(mirrorobject,0,0)


#print incident rays
#plotquiver3d(rayobject[8][4],rayobject[8][5],rayobject[8][6],rayobject[8][7],rayobject[8][8],rayobject[8][9])  #print normal
#plotquiver3d(rayobject[6][4],rayobject[6][5],rayobject[6][6],rayobject[6][11],rayobject[6][12],rayobject[6][13])  #print reflected rays

#plot3dto2d(wignerobject[1][10][1],wignerobject[1][10][2],wignerobject[1][10][0])   #pass in space, phase and wigner


#buildrayincidentlocations(functionobject,mirrorobject,wignerobject,mirrorinterpolator)



#colwvd = wignerobject[1]




#plot3dto2d(wignerobject[0][11][1],wignerobject[0][11][2],wignerobject[0][10][0])




#colwvd = np.add(colwvd[10],colwvd[10].T)
#colwvd = np.rot90(colwvd[10])

#contourplot(xaxis,yaxis,colwvd[10])       #this is a plot that gives you insight to the workings of the system
#plotMatrix(colwvd)
#plot3dto2d(xaxisline,freqs,abswvd)

#np.savetxt('test.out', wvd, delimiter=',')   # X is an array

#wvd = wignerfunction(function)
#print("the shape of our wigner dist function is")
#print(wvd.shape)
#print(wvd)
#realwvd = wvd.real

#newx,newy = raytransforms(xaxis,yaxis)



#print("we have subtracted and we get...")
#subx = np.subtract(newx,xaxis)
#contourplot(newx,newy,colwvd)

#plotMatrix(transfwvd)

#columnsrows = wvdT.shape
#print("the shape of the array is")
#print(columnsrows)

#intense,xaxis = integrateintensitywig(wignerobject,functionobject)
#plotinputs(intense,xaxis)
#getthresholdlocations(function)

#plotMatrix(wvdT)
#plotinputs(intense)
#mirrorfunction()
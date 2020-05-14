import numpy as np
from scipy import interpolate, special, constants
import json
import math
from src.Simulator import PlotFunctions
import numpy.polynomial.hermite as Hermite



from src.Simulator.ScalarField import ScalarField

with open('config.json') as config_file:
    config = json.load(config_file)


def gauss_beam(x, y, z, E0, w0, wz, zr, Rz, k):

    beam = E0*(w0/wz)*math.exp(-(x**2+y**2)/wz**2)*math.exp(1j*(k*z)-np.arctan(z/zr)+k*((x**2+y**2)/(2*Rz)))

    return beam

def generate_light_source():
<<<<<<< HEAD
    xGrid, yGrid = np.meshgrid(np.linspace(0, 7.5, config["lightSourceDensity"]),
                               np.linspace(0, 5, config["lightSourceDensity"]))

    pulse2d = np.where(abs(xGrid) <= 4, 1, 0) & np.where(abs(yGrid) <= 3, 1, 0)

    nMode, mMode = 2, 2
    aXDimension, bYDimension = 7.5, 5
    z = 0
    frequency = 2.9*pow(10, 12)

    permiabilitymue = constants.mu_0
    permittivity = constants.epsilon_0
    omega = 2*np.pi*frequency

    kc = np.sqrt(np.square(nMode*np.pi/aXDimension)+np.square(mMode*np.pi/bYDimension))

    Beta = np.sqrt(omega*permiabilitymue*permittivity-kc)

    TM_ElectricZ_Field = np.square(np.sin(mMode*np.pi*xGrid/aXDimension)*np.sin(nMode*np.pi*yGrid/bYDimension))#*np.square(math.exp(-1j*Beta*z))



    PlotFunctions.plot_3d_to_2d(xGrid, yGrid, TM_ElectricZ_Field)

    return np.stack((xGrid, yGrid, TM_ElectricZ_Field))
=======
    xVec = np.linspace(-10, 10, config["lightSourceDensity"])
    yVec = np.linspace(-10, 10, config["lightSourceDensity"])
    xGrid, yGrid = np.meshgrid(xVec, yVec)

    pulse2d = np.where(abs(xGrid) <= 4, 1, 0) & np.where(abs(yGrid) <= 3, 1, 0)

    return xVec, yVec, pulse2d
>>>>>>> CR_Fix


def create_interpolated_mirror(mirrorCorrections):
    xMirrorScale = config["xMirrorScale"]
    yMirrorScale = config["yMirrorScale"]
    mirrorGridDensity = config["mirrorGridDensity"]
    mirrorDimensions = config["mirrorDimensions"]
    mirrorOffsetFromSource = config["mirrorOffsetFromSource"]
    angle = config["mirrorRotationAngle"]
    direction = config["mirrorRotationDirection"]

    axis = np.linspace(-mirrorDimensions, mirrorDimensions, mirrorGridDensity)

    xGrid, yGrid = np.meshgrid(axis, axis)
    # mirrorBaseShape = (xGrid ** 2) / xMirrorScale + (yGrid ** 2) / yMirrorScale
    mirrorBaseShape = np.ones(mirrorGridDensity * mirrorGridDensity).reshape((mirrorGridDensity, mirrorGridDensity))
    mirrorShape = mirrorBaseShape + mirrorCorrections

    field = ScalarField(xGrid, yGrid, mirrorShape)
    field.apply_rotation(angle, direction)
    field.add_offset(mirrorOffsetFromSource)

    interpolatedMirrorBuilder = interpolate.interp2d(axis, axis, field.zScalarField, kind='cubic')
    mirrorBorders = np.array(([field.xGrid.max(), field.xGrid.min()], [field.yGrid.max(), field.yGrid.min()]))

    return mirrorBorders, interpolatedMirrorBuilder

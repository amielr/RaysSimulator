import math

from src.Simulator.FunctionSources import generate_light_source, create_interpolated_mirror
from src.Simulator.WignerFunction import wigner_transform
from src.Simulator.Wigner2DElement import *
from src.Simulator.RayPropogation import *
from src.Simulator.PlotFunctions import *
from numpy import linalg as LA
import json

with open('./config.json') as config_file:
    config = json.load(config_file)


xVec, yVec, lightSource = generate_light_source()

rays = michaelMain()
# rays = wigner_transform(lightSource, xVec, yVec)

# plot_scatter(rays)

def getStochasticRaysList(rays):
    stochasticRaysSelection = [ray for ray in rays if np.random.random() < 100 / len(rays)]
    npArrayStochasticRays = np.array(stochasticRaysSelection)
    return npArrayStochasticRays


def getHighestAmplitudeList(rays):
    highestRayList = np.array(rays)
    sortedList = np.argsort(highestRayList)[-100:][::-1]
    return sortedList

#sortedList = getHighestAmplitudeList(rays)

stochasticRayList = getStochasticRaysList(rays)


def getFullRayList(rays):
    fullRayList = np.array(rays)
    return fullRayList

# print("Our raylist size is: ", rayList.size)
# print("Our raylist length is: ", len(rayList))
#plot_scatter(rayList)
#x = 2




def error_value_calc(screenRays):
    idealPoint = np.array([config["xScreenLocation"],
                        config["yScreenLocation"],
                        config["zScreenLocation"]])

    meanDistance = 0
    for ray in screenRays:
        meanDistance += getAmplitude(ray) * LA.norm(getOrigin(ray) - idealPoint)

    meanDistance = meanDistance / len(screenRays)

    variance = 0
    for ray in screenRays:
        variance += (getAmplitude(ray) * LA.norm(getOrigin(ray) - idealPoint)) ** 2

    variance = math.sqrt(variance) / len(screenRays)

    return variance + meanDistance


def simulate_mirror(mirrorCorrections, plot, stochasticFlag):
    # mirrorGridDensity = config["mirrorGridDensity"]
    # mirrorDimensions = config["mirrorDimensions"]
    # mirrorOffsetFromSource = config["mirrorOffsetFromSource"]
    # angle = config["mirrorRotationAngle"]
    # direction = config["mirrorRotationDirection"]

    mirrorBorders, mirrorInterpolatedBuilder = create_interpolated_mirror(mirrorCorrections, config["mirrorGridDensity"],
                                                                          config["mirrorDimensions"], config["mirrorOffsetFromSource"],
                                                                          config["mirrorRotationAngle"], config["mirrorRotationDirection"])

    if stochasticFlag:
        global stochasticRayList
        stochasticRayList = getStochasticRaysList(rays)
        print("the stochastic raylist length is: ", str(len(stochasticRayList)))

    counter = 0
    if plot:
        # plot_mirror(mirrorBorders, mirrorInterpolatedBuilder)
        mirrorBorders, mirrorInterpolatedBuilderZeroed = create_interpolated_mirror(mirrorCorrections,
                                                                              config["mirrorGridDensity"],
                                                                              config["mirrorDimensions"],
                                                                              0, 0,
                                                                              config["mirrorRotationDirection"])
        plot_mirror(mirrorBorders, mirrorInterpolatedBuilderZeroed)

        #rayList = np.array(rays)
        fullRayList = getFullRayList(rays)
        print("the full raylist length is: ", str(len(fullRayList)))
        #screenRays = ray_propogation(fullRayList,
        #                             mirrorInterpolatedBuilder,
        #                             mirrorBorders)
        screenRays = ray_propogation(fullRayList,
                                     mirrorInterpolatedBuilder,
                                     mirrorBorders)
        x=0
    else:
        screenRays = ray_propogation(stochasticRayList, mirrorInterpolatedBuilder, mirrorBorders)


    if plot:
        plot_heatmap(screenRays, 'x')
        #plot_3dheatmap(screenRays)

#    if plot:
#        plot_wigner()

    error = error_value_calc(screenRays)

    return error

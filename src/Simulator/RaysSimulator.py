import math

from src.Simulator.FunctionSources import generate_light_source, create_interpolated_mirror
from src.Simulator.WignerFunction import wigner_transform
from src.Simulator.RayPropogation import *
from src.Simulator.PlotFunctions import *
import json

with open('../config.json') as config_file:
    config = json.load(config_file)


def error_value_calc(screenRays):
    idealPoint = np.array([config["xScreenLocation"],
                        config["yScreenLocation"],
                        config["zScreenLocation"]])

    meanDistance = 0
    for ray in screenRays:
        meanDistance += getAmplitude(ray) * (getOrigin(ray) - idealPoint).length()

    meanDistance = meanDistance / len(screenRays)

    variance = 0
    for ray in screenRays:
        variance += (getAmplitude(ray) * (getOrigin(ray) - idealPoint).length()) ** 2

    variance = math.sqrt(variance) / len(screenRays)

    return variance + meanDistance


xVec, yVec, lightSource = generate_light_source()
rays = wigner_transform(lightSource, xVec, yVec)

plot_scatter(rays)

rayList = np.array(rays)
print("Our raylist size is: ", rayList.size)
#plot_scatter(rayList)
# x = 2


def simulate_mirror(mirrorCorrections, plot):
    mirrorBorders, mirrorInterpolatedBuilder = create_interpolated_mirror(mirrorCorrections)

    #if plot:
    #    plot_mirror(mirrorBorders, mirrorInterpolatedBuilder)

    screenRays = ray_propogation(rayList,
                                 mirrorInterpolatedBuilder,
                                 mirrorBorders)

    #if plot:
    #    plot_heatmap(screenRays)

#    if plot:
#        plot_wigner()

    error = error_value_calc(screenRays)

    return error

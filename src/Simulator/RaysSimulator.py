import math

from src.Simulator.FunctionSources import generate_light_source, create_interpolated_mirror
from src.Simulator.WignerFunction import wigner_transform
from src.Simulator.RayPropogation import *
from src.Simulator.PlotFunctions import *
import json

with open('config.json') as config_file:
    config = json.load(config_file)


def error_value_calc(screenRays):
    idealPoint = Vector(config["xScreenLocation"],
                        config["yScreenLocation"],
                        config["zScreenLocation"])

    meanDistance = 0
    for ray in screenRays:
        meanDistance += ray.getAmplitude() * (ray.getOrigin() - idealPoint).length()

    meanDistance = meanDistance / len(screenRays)

    variance = 0
    for ray in screenRays:
        variance += (ray.getAmplitude() * (ray.getOrigin() - idealPoint).length()) ** 2

    variance = math.sqrt(variance) / len(screenRays)

    return variance + meanDistance


xVec, yVec, lightSource = generate_light_source()
rays = wigner_transform(lightSource, xVec, yVec)

# plot_wigner(rays)

rayList = np.array(rays)

# plot_scatter(rayList)
# x = 2


def simulateMirror(mirrorCorrections, plot):
    mirrorBorders, mirrorInterpolatedBuilder = create_interpolated_mirror(mirrorCorrections)

    if plot:
        plot_mirror(mirrorBorders, mirrorInterpolatedBuilder)

    screenRays = ray_propogation(rayList,
                                 mirrorInterpolatedBuilder,
                                 mirrorBorders)

    if plot:
        plot_heatmap(screenRays)

    error = error_value_calc(screenRays)

    return error

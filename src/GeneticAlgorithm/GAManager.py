import numpy as np
import json

from Simulator.PlotFunctions import plot_error_over_time
from src.GeneticAlgorithm.MirrorCreature import MirrorCreature
from src.GeneticAlgorithm.MirrorPopulation import MirrorPopulation

errors = []

def startSimulation():
    with open('config.json') as config_file:
        config = json.load(config_file)

    mirrorGridDensity = config["mirrorGridDensity"]
    mirrors = MirrorPopulation()
    index = 1

    best = MirrorCreature([0 for _ in np.zeros([mirrorGridDensity**2])])
    errors.append(best.simulate(plot=True))
    print("Generation number: 0")
    print(best.get_fitness())
    print()

    while True:
        mirrors.simulate(index)
        new_population = mirrors.next_generation()
        mirrors.set_population(new_population)

        print("Generation number: " + str(index))
        print(mirrors.get_best().get_fitness())
        # if (index % 10) == 0:
        errors.append(mirrors.get_best().simulate(plot=True))
        plot_error_over_time(errors)
        print()
        index += 1


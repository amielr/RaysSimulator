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
    errors.append(best.simulate_single_mirror(plot=True))
    print("Generation number: 0")
    print(best.get_fitness())
    print()
    maxMutationRate = config["max_mutation_rate"]
    minMutationRate = config["min_mutation_rate"]
    generationMutationRelation = config["generation_mutation_relation"]

    while True:
        # simulate all of the mirrors
        mirrors.simulate_generation(index)

        # regenerate the mirrors
        if index < generationMutationRelation:
            rate = maxMutationRate - index * (maxMutationRate - minMutationRate) / generationMutationRelation
        else:
            rate = minMutationRate
        new_population = mirrors.next_generation(rate)
        mirrors.set_population(new_population)

        # summarize the result of this generation
        print("Generation number: " + str(index))
        print(mirrors.get_best().get_fitness())
        err = mirrors.get_best().simulate_single_mirror(plot=(index % 50) == 0)
        errors.append(err)
        if index % 50 == 0:
            plot_error_over_time(errors)
        print()
        index += 1


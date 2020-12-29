import numpy as np
import json

from src.GeneticAlgorithm.MirrorCreature import *
from src.GeneticAlgorithm.MirrorPopulation import *
import platform
import struct

def is_os_64bit():
    return platform.machine().endswith('64')


def is_python_64bit():
    return (struct.calcsize("P") == 8)

def startSimulation():

    is_os_64bit()
    is_python_64bit()

    with open('./config.json') as config_file:
        config = json.load(config_file)

    mirrorGridDensity = config["mirrorGridDensity"]
    mirrors = initiate_mirror_population()
    index = 1

    bestDNA = initiate_mirror_creature([0 for _ in np.zeros([mirrorGridDensity**2])])

    print("Generation number: 0")
    print(simulate_mirror_creature_return_fitness(bestDNA, plot=True))  #get from this fitness/error)
    print()

    while True:
        mirrorsfitnesses = simulate_mirror_population(mirrors)
        new_population = next_generation(mirrors, mirrorsfitnesses)
        # mirrors.set_population(new_population)
        mirrors = new_population
        print("Generation number: " + str(index))
        # print(mirrors.get_best().get_fitness())
        # if (index % 10) == 0:
        #simulate_mirror_population(mirrors, plot=True)
        #mirrors.get_best().simulate(plot=True)
        print()
        index += 1


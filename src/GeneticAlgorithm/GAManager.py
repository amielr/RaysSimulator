import numpy as np
import json

from src.GeneticAlgorithm.MirrorCreature import MirrorCreature
from src.GeneticAlgorithm.MirrorPopulation import MirrorPopulation

with open('../config.json') as config_file:
    config = json.load(config_file)

mirrorGridDensity = config["mirrorGridDensity"]

mirrors = MirrorPopulation()
index = 1

best = MirrorCreature([0 for _ in np.zeros([mirrorGridDensity**2])])
best.simulate(plot=True)

while True:
    mirrors.simulate()
    new_population = mirrors.next_generation()
    mirrors.set_population(new_population)

    print("Generation number: " + str(index))
    print(mirrors.get_best().get_fitness())
    mirrors.get_best().simulate(plot=True)
    print()
    index += 1

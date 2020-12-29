import random
import json
import numpy as np

from src.Simulator.RaysSimulator import simulate_mirror

with open('./config.json') as config_file:
    config = json.load(config_file)

mirrorGridDensity = config["mirrorGridDensity"]
mutationRate = config["mutation_rate"]
_dna = []
_picked_probability = 0
_fitness = 0


def random_integer():
    return random.random() * 0.6 - 0.3


def initiate_mirror_creature(dna=None):
    global _dna
    _dna = dna
    if not _dna:
        _dna = [random_integer() for _ in np.zeros(mirrorGridDensity ** 2)]
    return _dna


def get_fitness():
    return _fitness


def get_dna():
    return _dna


def change_gene(dna, index):
    dna[index] += random_integer()
    return dna


def mutate(dna):
    for index, gene in enumerate(dna):
        if random.random() < mutationRate:
            dna = change_gene(dna, index)
    return dna


def get_picked_probability():
    return _picked_probability


def set_picked_probability_of_mirror(probability):
    global _picked_probability
    _picked_probability = probability


def calculate_fitness(error):
    return -error


def simulate_mirror_creature_return_fitness(MirrorDNA, plot=False):
    mirrorGrid = np.array(MirrorDNA).reshape((mirrorGridDensity, mirrorGridDensity))
    error = simulate_mirror(mirrorGrid, plot)
    return calculate_fitness(error)







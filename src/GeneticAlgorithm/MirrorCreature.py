import random
import json
import numpy as np

from src.Simulator.RaysSimulator import simulateMirror

with open('../config.json') as config_file:
    config = json.load(config_file)

mirrorGridDensity = config["mirrorGridDensity"]


def random_integer():
    return random.random() * 2 - 1


class MirrorCreature:
    _dna = []
    _fitness = 0
    _picked_probability = 0

    def __init__(self, dna=None):
        self._dna = dna
        if not self._dna:
            self._dna = [random_integer() for _ in np.zeros(config["mirrorGridDensity"] ** 2)]

    def get_picked_probability(self):
        return self._picked_probability

    def get_fitness(self):
        return self._fitness

    def get_dna(self):
        return self._dna

    def mutate(self):
        for index, gene in enumerate(self._dna):
            if random.random() < config["mutation_rate"]:
                self._dna[index] += random_integer()

    def set_picked_probability(self, probability):
        self._picked_probability = probability

    def calculate_fitness(self, error):
        self._fitness = -error

    def simulate(self):
        error = simulateMirror(np.array(self._dna).reshape((config["mirrorGridDensity"], config["mirrorGridDensity"])))
        print(error)
        self.calculate_fitness(error)

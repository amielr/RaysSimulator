import random
import json
import numpy as np

from src.Simulator.RaysSimulator import simulate_mirror

with open('config.json') as config_file:
    config = json.load(config_file)

mirrorGridDensity = config["mirrorGridDensity"]


def random_integer():
    return random.random() * 2 - 1


class MirrorCreature:
    _dna = []
    _fitness = 0
    _initialMirrorAngle = 0
    _picked_probability = 0

    def __init__(self, dna=None, initialMirrorAngle=None):
        self._dna = dna
        if not self._dna:
            self._dna = [random_integer() for _ in np.zeros(mirrorGridDensity ** 2)]
        self._initialMirrorAngle = initialMirrorAngle
        if not self._initialMirrorAngle:
            self._initialMirrorAngle = [-45 + random_integer()]

    def get_picked_probability(self):
        return self._picked_probability

    def get_fitness(self):
        return self._fitness

    def get_dna(self):
        return self._dna

    def change_gene(self, index):
        self._dna[index] += random_integer()

    def mutate(self,rate):
        for index, gene in enumerate(self._dna):
            if random.random() < rate:
                self.change_gene(index)

    def set_picked_probability(self, probability):
        self._picked_probability = probability

    def calculate_fitness(self, error):
        self._fitness = -error

    def simulate_single_mirror(self, plot=False, stochasticFlag=False):
        mirrorGrid = np.array(self._dna).reshape((mirrorGridDensity, mirrorGridDensity))
        error = simulate_mirror(mirrorGrid, plot, stochasticFlag)
        self.calculate_fitness(error)
        return error

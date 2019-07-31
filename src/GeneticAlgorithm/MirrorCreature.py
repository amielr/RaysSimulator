import random
import json

with open('config.json') as config_file:
    config = json.load(config_file)


def random_integer():
    return random.random() * 2 - 1


class MirrorCreature:
    _dna = []
    _fitness = 0
    _picked_probability = 0

    def __init__(self, dna=None):
        self._dna = dna
        if not self._dna:
            self._dna = [random_integer() for x in range(30)]

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

    def calculate_fitness(self):
        self._fitness = random.random()

    def simulate(self):
        # evaluate mirror performance
        self.calculate_fitness()

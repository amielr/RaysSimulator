import random
import numpy as np
from src.GeneticAlgorithm.MirrorCreature import MirrorCreature
import json

with open('config.json') as config_file:
    config = json.load(config_file)


def crossover(dna1, dna2):
    dna = dna1
    dna = [(dna[i] + x)/2 for i, x in enumerate(dna2)]
    return dna


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def non_zero_random():
    return 1 - random.random()


class MirrorPopulation:
    _population = []
    _best = MirrorCreature()

    def __init__(self):
        self._population = [MirrorCreature() for x in range(config["population_size"])]

    def simulate(self):
        for index, mirror in enumerate(self._population):
            # print("Mirror #"+str(index))
            mirror.simulate()

    def get_best(self):
        return self._best

    def natural_select(self):
        selected = non_zero_random()
        selected_index = -1
        while selected > 0:
            selected_index += 1
            selected -= self._population[selected_index].get_picked_probability()

        return self._population[selected_index]

    def create_child(self):
        parent1 = self.natural_select()
        parent2 = self.natural_select()

        dna = crossover(parent1.get_dna(), parent2.get_dna())
        child = MirrorCreature(dna)
        child.mutate()

        return child

    def set_picked_probability(self):
        best_count = int(len(self._population) * config["best_percent"])

        self._population.sort(key=lambda m: m.get_fitness(), reverse=True)
        best_population = self._population[0:best_count]
        all_best_population_score = [mirror.get_fitness() for mirror in best_population]
        all_best_population_probability = softmax(all_best_population_score)

        [mirror.set_picked_probability(all_best_population_probability[index])
         for index, mirror in enumerate(best_population)]

        self._population = best_population
        self._best = best_population[0]

    def next_generation(self):
        self.set_picked_probability()

        new_population = [self.create_child() for x in range(config["population_size"])]

        return new_population

    def set_population(self, new_population):
        self._population = new_population

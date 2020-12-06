import random
import numpy as np
from src.GeneticAlgorithm.MirrorCreature import *
import json

with open('../config.json') as config_file:
    config = json.load(config_file)

_population = []
_bestPopulationProbability = []
_fitness = []
_best = initiate_mirror_creature()


def crossover(dna1, dna2):
    dna = dna1
    dna = [(dna[i] + x)/2 for i, x in enumerate(dna2)]
    return dna


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def non_zero_random():
    return 1 - random.random()



def initiate_mirror_population():
    _population = [initiate_mirror_creature() for x in range(config["population_size"])]
    return _population



def simulate_mirror_population(mirrors):
    for index, mirror in enumerate(mirrors):
        # print("Mirror #"+str(index))
        _fitness.append(simulate_mirror_creature_return_fitness(mirror))
        print(str(index) + " : " + str(_fitness))
    return _fitness


def get_best():
    return _best


def next_generation(mirrors, mirrorsFitness):
    mirrorPopulation, bestMirrorPopulationProbability, bestMirror =  set_picked_probability_from_population(mirrors)

    new_population = [create_child(mirrorPopulation, bestMirrorPopulationProbability) for x in range(config["population_size"])]

    return new_population

def set_picked_probability_from_population(mirrorsPopulation):
    best_count = int(len(mirrorsPopulation) * config["best_percent"])

    mirrorsPopulation.sort(key=lambda m: simulate_mirror_creature_return_fitness(m), reverse=True)
    best_population = mirrorsPopulation[0:best_count]
    all_best_population_score = [simulate_mirror_creature_return_fitness(mirror) for mirror in best_population]
    all_best_population_probability = softmax(all_best_population_score)   #I dont understand what purpose this serves

    [mirror.set_picked_probability_of_mirror(all_best_population_probability[index])
     for index, mirror in enumerate(best_population)]   #need to think about this how to create matching index

    _bestPopulationProbability = all_best_population_probability  #maybe this is the way
    _population = best_population
    _best = best_population[0]

    return _population, _bestPopulationProbability, _best


def create_child(mirrors, bestProbabilities):
    parentMirror1 = natural_select(mirrors, bestProbabilities)
    parentMirror2 = natural_select(mirrors, bestProbabilities)

    dna = crossover(parentMirror1, parentMirror2)
    child = initiate_mirror_creature(dna)
    child = mutate(child)

    return child


def natural_select(mirrorsPopulation, bestPopulationProbability ):
    selected = non_zero_random()
    selected_index = -1
    while selected > 0:
        selected_index += 1
        selected -= bestPopulationProbability(selected_index)

    return mirrorsPopulation[selected_index]



# class MirrorPopulation:
#
#
#
#     def natural_select(self):
#         selected = non_zero_random()
#         selected_index = -1
#         while selected > 0:
#             selected_index += 1
#             selected -= self._population[selected_index].get_picked_probability()
#
#         return self._population[selected_index]
#
#     def create_child(self):
#         parent1 = self.natural_select()
#         parent2 = self.natural_select()
#
#         dna = crossover(parent1.get_dna(), parent2.get_dna())
#         child = initiate_mirror_creature(dna)
#         child.mutate()
#
#         return child
#
#     def set_picked_probability(self):
#         best_count = int(len(self._population) * config["best_percent"])
#
#         self._population.sort(key=lambda m: m.get_fitness(), reverse=True)
#         best_population = self._population[0:best_count]
#         all_best_population_score = [mirror.get_fitness() for mirror in best_population]
#         all_best_population_probability = softmax(all_best_population_score)
#
#         [mirror.set_picked_probability(all_best_population_probability[index])
#          for index, mirror in enumerate(best_population)]
#
#         self._population = best_population
#         self._best = best_population[0]
#
#
#
#     def set_population(self, new_population):
#         self._population = new_population

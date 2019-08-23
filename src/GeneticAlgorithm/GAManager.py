from src.GeneticAlgorithm.MirrorPopulation import MirrorPopulation

mirrors = MirrorPopulation()
index = 1

while True:
    mirrors.simulate()
    new_population = mirrors.next_generation()
    mirrors.set_population(new_population)

    print("Generation number: " + str(index))
    print(''.join(mirrors.get_best().get_dna()))
    print()
    index += 1

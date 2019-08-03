from MirrorPopulation import MirrorPopulation

mirrors = MirrorPopulation()
index = 1

while True:
    mirrors.simulate()
    new_population = mirrors.next_generation()
    mirrors.set_population(new_population)

    best_str = ''.join([chr(x) for x in mirrors.get_best().get_dna()])
    print("Generation number: " + str(index))
    print(best_str)
    print()
    index += 1

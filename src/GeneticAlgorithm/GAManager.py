from MirrorPopulation import MirrorPopulation


mirrors = MirrorPopulation()

while True:
    mirrors.simulate()
    new_population = mirrors.next_generation()
    mirrors.set_population(new_population)

from ariel.ec.a004 import Individual, Population

from ea.config import config

def initial_pop() -> Population:
    """Create initial population with random genotypes."""
    population = []
        
    for _ in range(config.POPULATION_SIZE):
        ind = Individual()
        ind.genotype = [
            config.RNG.uniform(-config.SCALE, config.SCALE, config.GENOTYPE_SIZE).tolist(),
            config.RNG.uniform(-config.SCALE, config.SCALE, config.GENOTYPE_SIZE).tolist(),
            config.RNG.uniform(-config.SCALE, config.SCALE, config.GENOTYPE_SIZE).tolist(),
        ]
        ind.fitness = 0.0
        ind.requires_eval = True
        # ind.tags['current'] = True
        
        population.append(ind)

    return population

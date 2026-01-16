from ariel.ec.a004 import Population

from ea.config import config


def survivor_selection(population: Population) -> Population:
    """
    Standard (Mu + Lambda) Selection:
    Deterministically keeps the top N individuals.
    """
    target_size = config.EA_SETTINGS.target_population_size
    
    valid_pop = [ind for ind in population if ind.alive]
    
    valid_pop.sort(key=lambda ind: ind.fitness, reverse=config.EA_SETTINGS.is_maximisation)
    

    for i in range(len(valid_pop)):
        if i < target_size:
            valid_pop[i].alive = True
        else:
            valid_pop[i].alive = False

    return population

# def survivor_selection(population: Population) -> Population:
#     """
#     K-tournament survivor selection with configurable tournament size.
#     Higher tournament size = higher selection pressure.
#     Repeatedly runs tournaments to eliminate worst individuals.
#     """
#     current_pop_size = len([ind for ind in population if ind.alive])

#     while current_pop_size > config.EA_SETTINGS.target_population_size:
#         alive = [ind for ind in population if ind.alive]

#         if len(alive) <= config.EA_SETTINGS.target_population_size:
#             break

#         tournament_size = min(config.SURVIVOR_TOURNAMENT_SIZE, len(alive))
        
#         competitor_indices = config.RNG.choice(len(alive), size=tournament_size, replace=False)

#         competitors = [alive[i] for i in competitor_indices]

#         if config.EA_SETTINGS.is_maximisation:
#             loser = min(competitors, key=lambda ind: ind.fitness)
#         else:
#             loser = max(competitors, key=lambda ind: ind.fitness)

#         loser.alive = False
#         current_pop_size -= 1

#     return population

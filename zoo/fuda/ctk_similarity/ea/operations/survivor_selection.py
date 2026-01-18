from ariel.ec.a004 import Population

from ea.config import config
import canonical_toolkit as ctk

def survivor_selection(population: Population) -> Population:
    """
    Standard (Mu + Lambda) Selection:
    Deterministically keeps the top N individuals.
    """
    target_size = config.EA_SETTINGS.target_population_size
    valid_pop = [ind for ind in population if ind.alive]
    
    valid_pop.sort(key=lambda ind: ind.fitness, reverse=config.EA_SETTINGS.is_maximisation)

    for i in range(len(valid_pop)):
        if i >= target_size:
            valid_pop[i].alive = False
    
    # TODO could add series metadata for what is saved when
    new_archived = [ctk.node_from_nde_genotype(ind.genotype, config.NDE) for ind in population if ind.tags.get('archived') and not ind.alive]
    if new_archived:
        new_series = ctk.series_from_node_population(new_archived, space_config=config.SIM_CONFIG)
        try:
            archive_series = ctk.SimilaritySeries.load(config.OUTPUT_FOLDER / 'archive')
            archive_series = new_series | archive_series
        except Exception:
            archive_series = new_series
        archive_series.save(config.OUTPUT_FOLDER, series_name='archive', overwrite=True)   

    return population

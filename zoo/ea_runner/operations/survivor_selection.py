from ariel.ec.a004 import Population

from ea.config import config
import canonical_toolkit as ctk

def survivor_selection(population: Population) -> Population:
    if config.TOURNAMENT:
        population = _tournament_survival(population)
    else:
        population = _deterministic_survival(population)
    
    _fix_archive(population)
    return population

def _tournament_survival(population: Population) -> Population:
    """
    K-tournament survivor selection with configurable tournament size.
    Higher tournament size = higher selection pressure.
    Repeatedly runs tournaments to eliminate worst individuals.
    """
    current_pop_size = len([ind for ind in population if ind.alive])

    while current_pop_size > config.EA_SETTINGS.target_population_size:
        alive = [ind for ind in population if ind.alive]

        if len(alive) <= config.EA_SETTINGS.target_population_size:
            break

        tournament_size = min(config.K_TOURNAMENT, len(alive))
        
        competitor_indices = config.RNG.choice(len(alive), size=tournament_size, replace=False)

        competitors = [alive[i] for i in competitor_indices]

        if config.IS_MAXIMISATION:
            loser = min(competitors, key=lambda ind: ind.fitness)
        else:
            loser = max(competitors, key=lambda ind: ind.fitness)

        loser.alive = False
        current_pop_size -= 1

    return population

def _deterministic_survival(population: Population) -> Population:
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
    
    return population

def _fix_archive(population: Population):
    match config.NOVELTY_METHOD:
        case "ctk":
            _save_ctk_archive(population)
        case "fuda":
            _save_fuda_archive(population)
        case _:
            pass
    return  

def _save_ctk_archive(population: Population):
    new_archived = [
        ctk.node_from_string(ind.tags['ctk_string'])
        for ind in population
        if ind.tags.get('archived') and not ind.alive
    ]
    if not new_archived:
        return

    # Save archive per space (all STORE_SPACES, not just NOVELTY_SPACES)
    for sim_config in config.SIM_CONFIGS:
        space_name = sim_config.space.name
        archive_path = config.OUTPUT_FOLDER / 'archive' / space_name

        new_series = ctk.series_from_node_population(new_archived, space_config=sim_config)
        try:
            existing = ctk.SimilaritySeries.load(archive_path)
            combined = new_series | existing
        except Exception:
            combined = new_series

        combined.save(config.OUTPUT_FOLDER / 'archive', series_name=space_name, overwrite=True)   

def _save_fuda_archive(population: Population):
    new_archived = [
        ctk.node_from_string(ind.tags['ctk_string'])
        for ind in population
        if ind.tags.get('archived') and not ind.alive
    ]
    if not new_archived:
        return

    from ariel_experiments.characterize.individual import (
        analyze_branching,
        analyze_coverage,
        analyze_joints,
        analyze_number_of_limbs,
        analyze_proportion_literature,
        analyze_symmetry,
    )
    from ariel_experiments.characterize.population import (
        get_raw_population_properties,
    )
    import numpy as np
    
    morphological_analysers = [
        analyze_branching,
        analyze_number_of_limbs,
        analyze_coverage,
        analyze_joints,
        analyze_proportion_literature,
        analyze_symmetry,
    ]
    
    graph_list = [ind.to_graph() for ind in new_archived]

    documents = get_raw_population_properties(
        graph_list, morphological_analysers,
        n_jobs=1,
        hide_tracker=True
    )
    archive = np.array(list(documents.values())).T
    
    archive_path = config.OUTPUT_FOLDER / 'archive' / 'fuda'
    matrix = ctk.SimilarityMatrix(archive, 'fuda_archive', -1)
    try:
        existing = ctk.SimilarityMatrix.load(archive_path)
        combined = matrix | existing
    except Exception:
        combined = matrix

    combined.save(config.OUTPUT_FOLDER / 'archive' / 'fuda', overwrite=True)   

from functools import partial

from ariel_experiments.characterize.canonical.core.toolkit import (
    CanonicalToolKit as ctk,
)
from ariel_experiments.characterize.individual import analyze_neighbourhood
from ariel_experiments.characterize.population import (
    get_full_analyzed_population,
)

type Individual = object # ofc should be the ariel EC EA thing


def tfidf_fitness(
    population: list[Individual],
    config: ctk.SimilarityConfig,
    tfidf_archive_prev: dict[int, ctk.RadiusData],
    tfidf_archive_now: dict[int, ctk.RadiusData],
) -> None:
    """Because the tfidf_archives are pointers in memory, make sure to not fully replace them but update everything in place here."""
    analyzed_population = get_full_analyzed_population(
        population,
        analyzers=[
            partial(analyze_neighbourhood, config=config),
        ],
        derivers=[],
        n_jobs=-1,
        hide_tracker=True,
    )
    treehash_list = analyzed_population.raw["neighbourhood"]

    # first update the 'empty' current population list
    ctk.update_tfidf_dictionary(treehash_list, tfidf_archive_now)
    # now update the previous generation archive
    ctk.update_tfidf_dictionary(treehash_list, tfidf_archive_prev)

    # now give the scores for each individual
    for idx, treehash_dict in enumerate(treehash_list):
        individual = population[idx]
        individual.tfidf_diversity = ctk.calculate_tfidf(
            treehash_dict,
            tfidf_archive_now,
            config=tfidf_config,
        )
        individual.tfidf_novelty = ctk.calculate_tfidf(
            treehash_dict,
            tfidf_archive_prev,
            config=tfidf_config,
        )

    # Save current archive to previous archive for next iteration
    tfidf_archive_prev.clear()
    tfidf_archive_prev |= tfidf_archive_now
    # Clear current archive for next iteration
    tfidf_archive_now.clear()



# USAGE:
tfidf_config = (
    ctk.create_tfidf_config()
)  # just to this to get our chosen default values
tfidf_archive_now = {}  # initialise the 'global' archive
tfidf_archive_prev = {}

# use this in the EA OPS
tfidf_fitness_eval_fn = partial(
    tfidf_fitness,
    config=tfidf_config,
    tfidf_archive_prev=tfidf_archive_prev,
    tfidf_archive_now=tfidf_archive_now,
)

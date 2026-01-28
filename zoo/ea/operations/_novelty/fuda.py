import numpy as np
from ea.config import config

from typing import Any
import canonical_toolkit as ctk
from ariel.ec.a004 import Population

__all__ = [
    "fuda_novelty",
    "save_fuda_archive"
]



def fuda_novelty(population: Population) -> list[Any]:  # type: ignore
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
    from sklearn.metrics import pairwise_distances

    morphological_analysers = [
        analyze_branching,
        analyze_number_of_limbs,
        analyze_coverage,
        analyze_joints,
        analyze_proportion_literature,
        analyze_symmetry,
    ]

    K = config.K_NOVELTY
    graph_list = [ctk.node_from_string(ind.tags['ctk_string']).to_graph() for ind in population]

    documents = get_raw_population_properties(
        graph_list, morphological_analysers,
        n_jobs=1,
        hide_tracker=True
    )
    pop_matrix = np.array(list(documents.values())).T

    try:
        archive_matrix = ctk.SimilarityMatrix.load(
            config.OUTPUT_FOLDER / "archive" / "fuda"
        )
        archive_matrix = archive_matrix.matrix.toarray()

        combined = np.vstack((pop_matrix,archive_matrix))
    except Exception:
        combined = pop_matrix

    distance_matrix = pairwise_distances(combined, metric="euclidean")
    novelty_list = [0] * len(population)

    for i in range(len(population)):
        distances_sorted = np.sort(distance_matrix[i])
        novelty_list[i] = distances_sorted[1 : K + 1].sum() / K

    return novelty_list




def save_fuda_archive(population: Population):
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

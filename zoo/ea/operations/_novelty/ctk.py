import numpy as np
from ea.config import config

import canonical_toolkit as ctk
from ariel.ec.a004 import Population

__all__ = [
    "ctk_novelty",
    "save_ctk_archive"
]


def ctk_novelty(population: Population) -> np.ndarray:
    """Requres pre-saved features sorted by ctk_string order."""
    try:
        folder_names = sorted(
            (config.OUTPUT_FOLDER / "feature_frames").glob("gen_*"),
        )
        last_path = folder_names[-1]

        pop_frame = ctk.SimilarityFrame.load(last_path)
    except:
        msg = "cant find reature frame to do novelty analysis"
        raise FileNotFoundError(
            msg,
        )

    novelty_spaces = config.NOVELTY_SPACES
    assert novelty_spaces, "noveltyspaces must not be none"
    novelty_series_list = []
    for space in novelty_spaces:
        space_name = space.name
        pop_series = pop_frame[space_name]

        try:
            archive_series = ctk.SimilaritySeries.load(
                config.OUTPUT_FOLDER / "archive" / space_name,
            )
            combined = pop_series | archive_series
        except Exception:
            combined = pop_series

        novelty_series_list.append(combined)

    novelty_frame = ctk.SimilarityFrame(series=novelty_series_list)
    novelty_frame.map("cosine_similarity")
    novelty_frame.to_cumulative()

    assert config.MAX_HOP_RADIUS
    frame_slice = novelty_frame[config.MAX_HOP_RADIUS]

    frame_slice.map("normalize_by_radius")

    matrix = frame_slice.aggregate().aggregate()
    matrix /= len(config.NOVELTY_SPACES)

    sorted_similarity_array = matrix.sum_to_rows(
        zero_diagonal=True, k=config.K_NOVELTY, largest=True, normalise=True,
    )[: len(population)]
    sorted_novelty_array = 1 - sorted_similarity_array

    sorted_indices = sorted(
        range(len(population)),
        key=lambda i: str(population[i].tags["ctk_string"]),
    )
    inverse_indices = np.argsort(sorted_indices)
    novelty_array = sorted_novelty_array[inverse_indices]

    return np.clip(novelty_array, 0, 1.0)



def save_ctk_archive(population: Population):
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

from ariel.ec.a004 import Population
import canonical_toolkit as ctk
from ..config import config

def save_features(population: Population):
    """
    saves features sorted by alphabetical ctk_string value
    """ 
    sorted_indices = sorted(range(len(population)), key=lambda i: population[i].tags["ctk_string"])
    sorted_pop = [population[i] for i in sorted_indices]
    nodes = [ctk.node_from_string(ind.tags["ctk_string"]) for ind in sorted_pop]

    series_list = [
        ctk.series_from_node_population(nodes, space_config=sim_config)
        for sim_config in config.SIM_CONFIGS
    ]
    pop_frame = ctk.SimilarityFrame(series=series_list)
    pop_frame.save(config.OUTPUT_FOLDER / 'feature_frames', 'gen')
    return population

from ariel.ec.a004 import Population, Individual

from ea.config import config

from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
import numpy as np
import canonical_toolkit as ctk

def evaluate_pop(population: Population):
    
    for ind in population:
        if ind.requires_eval:
            matrixes = config.NDE.forward(np.array(ind.genotype))
            hpd = HighProbabilityDecoder(num_modules=config.NUM_MODULES)
            ind_graph = hpd.probability_matrices_to_graph(
                matrixes[0], matrixes[1], matrixes[2],
            )
            ind.tags["ctk_string"] = ctk.node_from_graph(ind_graph).to_string()
            
            ind.fitness = float(len(ind.tags["ctk_string"]))
            ind.requires_eval = False
    
    return population


def _evaluate_speed(individual: Individual) -> type[float]:
    return float

def _evaluate_novelty(population: Population, k_novelty: int = 1):
    return 

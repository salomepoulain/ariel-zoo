from __future__ import annotations

from networkx import DiGraph
from ariel.ec.a004 import Population, Individual

from ea.config import config

from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
import numpy as np
import canonical_toolkit as ctk
import torch

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from networkx import DiGraph

def _ind_to_graph(individual: Individual) -> DiGraph[Any]:
    with torch.no_grad():
        matrixes = config.NDE.forward(np.array(individual.genotype))
        hpd = HighProbabilityDecoder(num_modules=config.NUM_MODULES)
        ind_graph = hpd.probability_matrices_to_graph(
            matrixes[0], matrixes[1], matrixes[2],
        )
        return ind_graph
    
def _store_ctk_string(individual: Individual, graph: DiGraph[Any]):
    ctk_string = ctk.node_from_graph(graph).to_string()
    individual.tags["ctk_string"] = ctk_string
    
def evaluate_pop(population: Population):
    novelty_scores = evaluate_novelty(population)
    
    for idx, ind in enumerate(population):
        if ind.requires_eval:
            ind_graph = _ind_to_graph(ind)
            _store_ctk_string(ind, ind_graph) if config.STORE_STRING else None

            novelty = novelty_scores[idx]
            assert 0 < novelty <= 1
            
            speed = _evaluate_speed(ind)
            ind.fitness = novelty * speed

            ind.requires_eval = False
    return population


def evaluate_novelty(population: Population) -> np.ndarray:
    raise NotImplementedError
    
    current_graph_population = [
        _ind_to_graph(ind) 
        for ind in population 
        if not ind.tags.get('archive', False)
    ]
    archive_graph_population = [
        _ind_to_graph(ind)
        for ind in population 
        if not ind.tags.get('archive', True)
    ]
    whole_population = current_graph_population + archive_graph_population
    
    series = ctk.graph_population_to_series(whole_population, space_config=config.SIM_CONFIG)
    # series.save(config.OUTPUT_FOLDER) #TODO: buggy!!!!
    
    series.cosine_similarity()
    matrix = series.aggregate()
    matrix.normalize_by_radius()
    
    similarity_array = matrix.sum_to_rows(zero_diagonal=True, k=config.K_NOVELTY, largest=True)
    return similarity_array - 1
    


def _evaluate_speed(individual: Individual) -> type[float]:
    return 1.0

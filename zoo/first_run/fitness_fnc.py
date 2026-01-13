"""
This file contains the fitness function from the Fuda paper

Use: import the fitness function and place it in the EA operators 
"""


import matplotlib.pyplot as plt

# from examples.z_ec_course.A3_template import NUM_OF_MODULES
import numpy as np

# from ariel_experiments.characterize.individual import analyze_neighbourhood
# from ariel_experiments.characterize.population import (
# get_full_analyzed_population,
# matrix_derive_neighbourhood,
# matrix_derive_neighbourhood_cross_pop,
# )
import torch
import umap
from rich.console import Console
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.a004 import (
    EA,
    EASettings,
    EAStep,
    Individual,
    Population,
)
from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding
from ariel_experiments.characterize.canonical.core.toolkit import (
    CanonicalToolKit as ctk,
)

# some global vars that will be removed when the function is complete, only used vars will be kept

console = Console()
SEED = 42

# Seed everything for determinism
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

# Seed PyTorch for deterministic behavior
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)

# Global var
global_fitness_history = []
global_best_fitness_history = []
diversity_archive = []  # NEW: Archive of past tree hashes for novelty search

# EA settings
EA_CONFIG = EASettings(
    is_maximisation=True,
    num_of_generations=50,
    target_population_size=100,
)

# evaluation settings
SIM_CONFIG = ctk.SimilarityConfig()
SIM_CONFIG.max_tree_radius = 10
SIM_CONFIG.radius_strategy = ctk.CollectionStrategy.SUBTREES


NUM_OF_MODULES = 20
GENOTYPE_SIZE = 64
SCALE = 8192  # ADDED: Following A3_template pattern

MUTATION_RATE = 0.05  # FIXED: 5% mutation rate (was 0)

# Selection pressure settings
PARENT_TOURNAMENT_SIZE = 4  # Reduced for more diversity
SURVIVOR_TOURNAMENT_SIZE = 4  # Reduced for more diversity

# Archive settings
MAX_ARCHIVE_SIZE = 500_000  # Limit archive growth (keeps last 500)

GLOBAL_N_NEIGHBORS = EA_CONFIG.target_population_size - 1

K_NEIGHBORS = 5

# ADDED: Global NDE for deterministic genotype-to-phenotype mapping (following A3_template pattern)
GLOBAL_NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)




def decode_genotype_to_string(genotype: list) -> str:
    """Helper: decode genotype to phenotype string using GLOBAL_NDE."""
    matrixes = GLOBAL_NDE.forward(np.array(genotype))
    hpd = HighProbabilityDecoder(num_modules=NUM_OF_MODULES)
    graph = hpd.probability_matrices_to_graph(matrixes[0], matrixes[1], matrixes[2])
    tree = ctk.from_graph(graph)
    return ctk.to_string(tree)

def fitness(population:Population)->Population:

    K = K_NEIGHBORS  # Use the global K
    
    for ind in population:
        if ind.requires_eval:
            ind.tags['ctk_string'] = decode_genotype_to_string(ind.genotype)

    n_pop = len(population)
    novelty_list = [0]*n_pop

    simliarity_matrix = np.zeros((n_pop, n_pop))
    for ind in population:
        # TODO fill simliarity matrix
        break 

    # TODO take a row take the average of the K lowest score in row, then put in a list
    speed_list = []
    for index, ind in enumerate(population):
        speed = speed_test(ind)
        speed_list.append(speed)
        ind.fitness = speed*novelty_list[index]



    return population



import numpy as np

# Assuming K_NEIGHBORS is defined globally, e.g.,


def speed_test(ind:Individual) -> float:
    # TODO calculate the speed of the robot
    return 0



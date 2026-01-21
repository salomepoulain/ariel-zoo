

# from examples.z_ec_course.A3_template import NUM_OF_MODULES

import numpy as np

# from ariel_experiments.characterize.individual import analyze_neighbourhood
# from ariel_experiments.characterize.population import (
# get_full_analyzed_population,
# matrix_derive_neighbourhood,
# matrix_derive_neighbourhood_cross_pop,
# )
import torch


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
import canonical_toolkit as ctk

from fitness_fnc import fitness


#from examples.z_ec_course.A3_template import NUM_OF_MODULES

import networkx as nx
from ariel_experiments.gui_vis.view_mujoco import view
import matplotlib.pyplot as plt


NUM_OF_MODULES = 20
SEED = 42
RNG = np.random.default_rng(SEED)

# Global var
global_fitness_history = []
global_best_fitness_history = []
diversity_archive = []  # NEW: Archive of past tree hashes for novelty search

# EA settings
EA_CONFIG = EASettings(
    is_maximisation=True,
    num_of_generations=100,
    target_population_size=500,  
)





GENOTYPE_SIZE = 64
SCALE = 8192  # ADDED: Following A3_template pattern

MUTATION_RATE = 0.05  # FIXED: 5% mutation rate (was 0)

# Selection pressure settings
PARENT_TOURNAMENT_SIZE = 2  # Reduced for more diversity
SURVIVOR_TOURNAMENT_SIZE = 2  # Reduced for more diversity

# Archive settings
MAX_ARCHIVE_SIZE = 500_000  # Limit archive growth (keeps last 500)

GLOBAL_N_NEIGHBORS = EA_CONFIG.target_population_size - 1

K_NEIGHBORS = 9

# ADDED: Global NDE for deterministic genotype-to-phenotype mapping (following A3_template pattern)
GLOBAL_NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)

def _ind_to_graph(individual: Individual):
    with torch.no_grad():
        matrixes = GLOBAL_NDE.forward(np.array(individual.genotype))
        hpd = HighProbabilityDecoder(num_modules=NUM_OF_MODULES)
        ind_graph = hpd.probability_matrices_to_graph(
            matrixes[0], matrixes[1], matrixes[2],
        )
        return ind_graph

def _store_ctk_string(individual: Individual):
    ctk_string = ctk.node_from_graph(_ind_to_graph(individual)).to_string()
    individual.tags["ctk_string"] = ctk_string 




def float_creep(
    individual: list[list[float]] | list[list[list[float]]],
    mutation_probability: float,
    mutation_step_size: float = 100.0,  # Small step relative to SCALE
) -> list[list[float]]:
    """
    Creep mutation: adds small random perturbations to genes.
    
    Args:
        individual: Genotype to mutate
        mutation_probability: Probability each gene mutates (0-1)
        mutation_step_size: Maximum step size for mutation (default 100)
    """
    ind_arr = np.array(individual)
    shape = ind_arr.shape

    # Generate SMALL mutation values (not full range!)
    mutator = RNG.uniform(-mutation_step_size, mutation_step_size, size=shape)

    # Determine which positions to mutate
    do_mask = RNG.choice(
        [1, 0],
        size=shape,
        p=[mutation_probability, 1 - mutation_probability],
    )
    
    mutation_mask = mutator * do_mask
    new_genotype = ind_arr + mutation_mask
    return new_genotype.tolist()


def make_random_robot(genotype_size: int = GENOTYPE_SIZE) -> Individual:
    """
    Produces a robot with only its genotype.
    Following A3_template.py pattern: uses uniform distribution from -SCALE to +SCALE.
    """
    ind = Individual()
    ind.genotype = [
        RNG.uniform(-SCALE, SCALE, genotype_size).tolist(),
        RNG.uniform(-SCALE, SCALE, genotype_size).tolist(),
        RNG.uniform(-SCALE, SCALE, genotype_size).tolist(),
    ]
    ind.fitness = 0.0
    ind.requires_eval = True
    _store_ctk_string(ind)
    return ind


def parent_selection_tournament(population: Population) -> Population:
    """
    K-tournament parent selection with configurable tournament size.
    Higher tournament size = higher selection pressure.
    """
    tournament_size = PARENT_TOURNAMENT_SIZE

    # Run tournaments to select parents
    num_parents = len(population) // 2  # Select ~50% as parents
    selected_parents = []

    for _ in range(num_parents):
        # Randomly select tournament_size individuals
        competitors = RNG.choice(population, size=min(tournament_size, len(population)), replace=False)

        # Find the best in the tournament
        if EA_CONFIG.is_maximisation:
            winner = max(competitors, key=lambda ind: ind.fitness)
        else:
            winner = min(competitors, key=lambda ind: ind.fitness)

        selected_parents.append(winner)

    # Clear all ps tags first
    for ind in population:
        ind.tags["ps"] = False

    # Tag selected parents
    for parent in selected_parents:
        parent.tags["ps"] = True

    return population


def crossover(population: Population) -> Population:
    """
    Generational crossover with 100% reproduction rate.
    Creates offspring equal to target population size by selecting parents with replacement.
    """
    children = []

    # Get selected parents (those with ps=True from parent selection)
    parents = [ind for ind in population if ind.tags.get("ps", False)]

    # If no parents selected (shouldn't happen), use all population
    if len(parents) == 0:
        parents = population

    # Create target_population_size offspring
    for _ in range(EA_CONFIG.target_population_size):
        child = Individual()

        # Select two parents WITH replacement (can select same parent twice)
        parent1 = RNG.choice(parents)
        parent2 = RNG.choice(parents)

        # Generate a NEW random mask for every child
        shape = np.array(parent1.genotype).shape
        mask = RNG.random(size=shape) < 0.5

        # Perform uniform crossover
        child.genotype = np.where(
            mask,
            np.array(parent1.genotype),
            np.array(parent2.genotype),
        ).tolist()

        # Tags
        child.requires_eval = True
        child.tags["mut"] = True
        children.append(child)

    population.extend(children)
    return population


def mutation(population: Population) -> Population:
    """Mutates offspring by adding small random noise to genotype values."""
    mutation_rate = MUTATION_RATE
    
    # Debug: only print first 3 individuals to avoid spam
    # debug_count = 0
    # max_debug = 3
    
    for ind in population:
        if ind.tags.get("mut", False):


            # Apply mutation
            genes = ind.genotype
            mutated = [
                float_creep(individual=genes[0], mutation_probability=mutation_rate),
                float_creep(individual=genes[1], mutation_probability=mutation_rate),
                float_creep(individual=genes[2], mutation_probability=mutation_rate),
            ]
            ind.genotype = mutated

  
                            
            ind.tags["mut"] = False
            ind.requires_eval = True

    return population





def survivor_selection_tournament(population: Population) -> Population:
    """
    K-tournament survivor selection with configurable tournament size.
    Higher tournament size = higher selection pressure.
    Repeatedly runs tournaments to eliminate worst individuals.
    """
    tournament_size = SURVIVOR_TOURNAMENT_SIZE
    current_pop_size = len([ind for ind in population if ind.alive])

    while current_pop_size > EA_CONFIG.target_population_size:
        # Get all alive individuals
        alive = [ind for ind in population if ind.alive]

        # Safety check
        if len(alive) <= EA_CONFIG.target_population_size:
            break

        # Randomly select tournament_size individuals for tournament
        competitors = RNG.choice(alive, size=min(tournament_size, len(alive)), replace=False)

        # Find the WORST individual in the tournament (to eliminate)
        if EA_CONFIG.is_maximisation:
            # In maximization, lowest fitness is worst
            loser = min(competitors, key=lambda ind: ind.fitness)
        else:
            # In minimization, highest fitness is worst
            loser = max(competitors, key=lambda ind: ind.fitness)

        # Eliminate the loser
        loser.alive = False
        current_pop_size -= 1

    return population










if __name__ == "__main__":
    population_list = [
    make_random_robot(GENOTYPE_SIZE) for _ in range(EA_CONFIG.target_population_size)
    ]

    ops = [
    EAStep("parent_selection", parent_selection_tournament),  # K-tournament (size=2)
    EAStep("crossover", crossover),
    EAStep("mutation", mutation),


    EAStep("evaluation", fitness),


    EAStep("survivor_selection", survivor_selection_tournament),  # K-tournament (size=2)

    ]


    ea = EA(
    population_list,
    operations=ops,
    num_of_generations=EA_CONFIG.num_of_generations,
    )
    try:
        print("Starting EA")
        ea.run()
    except KeyboardInterrupt:
        print("Got ctrl+c, stopping ea")






# FUDA STYLE POST EA NOVELTY SELECTION
from sklearn.metrics import pairwise_distances
from fitness_fnc import _ind_to_graph
from ariel_experiments.characterize.individual import analyze_branching, analyze_coverage, analyze_joints, analyze_number_of_limbs, analyze_proportion_literature, analyze_symmetry
from ariel_experiments.characterize.population import get_raw_population_properties
import numpy as np

morphological_analysers = [analyze_branching,analyze_number_of_limbs,analyze_coverage,analyze_joints,analyze_proportion_literature,analyze_symmetry]

# Selecting size of test suite
N_SELECT = 20

def fuda_selection(
    population: Population,
    morphological_analysers,
    n_select: int = 20,
    seed: int = 42,
):

    rng = np.random.default_rng(seed)

    # Decoding all individuals to graphs
    graphs = []
    valid_inds = []

    for ind in population:
        try:
            graph = _ind_to_graph(ind)
            graphs.append(graph)
            valid_inds.append(ind)
        except Exception:
            continue

    # Extracting morphological features
    documents = get_raw_population_properties(graphs, morphological_analysers)
    X = np.array(list(documents.values())).T

    # Computing distance matrix
    D = pairwise_distances(X, metric="euclidean")

    n = len(valid_inds)
    assert n >= n_select, "Population smaller than needed"

    # Greedy novelty selection
    selected_idx = []

    # Starting from a random individual
    first = rng.integers(0, n)
    selected_idx.append(first)
    remaining = set(range(n))
    remaining.remove(first)

    while len(selected_idx) < n_select:
        best_candidate = None
        best_score = -np.inf
        for i in remaining:
            # Computing the distance to the closest selected individual
            d_min = min(D[i, j] for j in selected_idx)
            if d_min > best_score:
                best_score = d_min
                best_candidate = i
        selected_idx.append(best_candidate)
        remaining.remove(best_candidate)
    selected_individuals = [valid_inds[i] for i in selected_idx]

    return selected_individuals


final_population = ea.population

test_suite = fuda_selection(
    population=final_population,
    morphological_analysers=morphological_analysers,
    n_select=20,
)

print(f"Selected {len(test_suite)} morphologically novel robots using Fuda's method")

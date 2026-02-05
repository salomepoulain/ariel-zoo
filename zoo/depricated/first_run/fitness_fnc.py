"""
This file contains the fitness function from the Fuda paper

Use: import the fitness function and place it in the EA operators 
"""


import time
from typing import Any
import matplotlib.pyplot as plt

# from examples.z_ec_course.A3_template import NUM_OF_MODULES
import mujoco
import networkx as nx
import numpy as np

# from ariel_experiments.characterize.individual import analyze_neighbourhood
# from ariel_experiments.characterize.population import (
# get_full_analyzed_population,
# matrix_derive_neighbourhood,
# matrix_derive_neighbourhood_cross_pop,
# )
import torch

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import NaCPG, create_fully_connected_adjacency
from ariel.simulation.environments._simple_flat import SimpleFlatWorld

from ariel.utils.tracker import Tracker

from rich.console import Console

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
# from ariel_experiments.characterize.canonical.core.toolkit import (
#     CanonicalToolKit as ctk,
# )
import canonical_toolkit as ctk

from ariel_experiments.characterize.individual import analyze_branching, analyze_coverage, analyze_joints, analyze_number_of_limbs, analyze_proportion_literature, analyze_symmetry
from ariel_experiments.characterize.population import get_raw_population_properties

from sklearn.metrics import pairwise_distances


# some global vars that will be removed when the function is complete, only used vars will be kept

console = Console()
SEED = 42

# Seed everything for determinism
np.random.seed(SEED)
RNG = np.random.default_rng(SEED)

# Seed PyTorch for deterministic behavior
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True, warn_only=True)



NUM_OF_MODULES = 20
GENOTYPE_SIZE = 64
SCALE = 8192  # ADDED: Following A3_template pattern

MUTATION_RATE = 0.05  # FIXED: 5% mutation rate (was 0)

# Selection pressure settings
PARENT_TOURNAMENT_SIZE = 2  # Reduced for more diversity
SURVIVOR_TOURNAMENT_SIZE = 2  # Reduced for more diversity

K_NEIGHBORS = 9

# ADDED: Global NDE for deterministic genotype-to-phenotype mapping (following A3_template pattern)
GLOBAL_NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)

morphological_analysers = [analyze_branching,analyze_number_of_limbs,analyze_coverage,analyze_joints,analyze_proportion_literature,analyze_symmetry]

def _ind_to_graph(individual: Individual):
    with torch.no_grad():
        matrixes = GLOBAL_NDE.forward(np.array(individual.genotype))
        hpd = HighProbabilityDecoder(num_modules=NUM_OF_MODULES)
        ind_graph = hpd.probability_matrices_to_graph(
            matrixes[0], matrixes[1], matrixes[2],
        )
        return ind_graph

def _store_ctk_string(individual: Individual):
    graph = _ind_to_graph(individual)
    ctk_string = ctk.node_from_graph(graph).to_string()
    individual.tags["ctk_string"] = ctk_string 
    return graph


def fitness(population:Population)->Population:

    K = K_NEIGHBORS  # Use the global K
    graph_list = []
    # Decoding genotypes
    for ind in population:
        if ind.requires_eval:
            graph = _store_ctk_string(ind)
            graph_list.append(graph)
            

    # n_pop = len(population)

    #similarity_matrix = np.zeros((n_pop, n_pop))
    documents = get_raw_population_properties(graph_list, morphological_analysers)# [list(ind.tags['subtrees']) for ind in population]
    #hasher = FeatureHasher(n_features=7)
    #X = hasher.transform(documents)
    #X = normalize(X, norm="l2")
    x = np.array(list(documents.values())).T

    #for i in range(n_pop):
    #    for j in range(n_pop):
    #        similarity_matrix[i, j] = X[i] @ X[j].T

    distance_matrix = pairwise_distances(x, metric="euclidean")

    novelty_list = [0]*len(distance_matrix)

    for i in range(len(distance_matrix)):
        distances_sorted = np.sort(distance_matrix[i])
        novelty_list[i] = distances_sorted[1:K+1].sum()/K


    index = 0
    for ind in population:
        if ind.requires_eval:
            speed = speed_test(graph_list[index])
            ind.fitness = speed*novelty_list[index]
            index += 1 


    return population




def speed_test(graph , nr_of_tests:int=1) -> float:
    best = 0
    

    for _ in range(nr_of_tests):
        robot = construct_mjspec_from_graph(graph) 
        score = run_simulation(robot)
        if score > best:
            best = score

    return best


def run_simulation(robot:CoreModule, time:int=60): 
    """Entry function to run the simulation with random movements, with added speed measuring"""
    # Initialise controller to controller to None, always in the beginning.

    mujoco.set_mjcb_control(None)  # DO NOT REMOVE
    steps = time *500  # TODO check how many steps are in a second
    # Initialise world
    # Import environments from ariel.simulation.environments
    world = SimpleFlatWorld()



    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(robot.spec)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Initialise data tracking
    # to_track is automatically updated every time step
    # You do not need to touch it.
    mujoco_type_to_find = mujoco.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    # Setup the NaCPG controller
    adj_dict = create_fully_connected_adjacency(len(data.ctrl.copy()))
    na_cpg_mat = NaCPG(adj_dict)

   

    # Simulate the robot
    ctrl = Controller(
        controller_callback_function=lambda _, d: na_cpg_mat.forward(d.time),
        tracker=tracker,
    )

    # Set the control callback function
    # This is called every time step to get the next action.
    # Pass the model and data to the tracker
    ctrl.tracker.setup(world.spec, data)

    # Set the control callback function
    mujoco.set_mjcb_control(
        ctrl.set_control,
    )

    # Reset state and time of simulation
    # mujoco.mj_resetData(model, data)



    # Move simulation forward one iteration/step nsteps times
    mujoco.mj_step(model, data, nstep=steps)

    pos_data = np.array(tracker.history["xpos"][0])
    speed = np.sqrt((pos_data[-1, 0] - pos_data[0, 0])**2 + (pos_data[-1, 1] - pos_data[0, 1])**2)/60 # change for 60 second
    # print("data time:", data.time)
    return float(speed)


# if __name__ == "__main__":
## just some testing code feel free to change

    # for _ in tqdm(range(500)):
        # print(format(run_simulation('C[f(H6H2B[t(B2[l(B2[t(BH4BB6[r(B6)]H6)])])]HH4)]', time=60),'.8f'))
        # run_simulation('C[f(H6H2B[t(B2[l(B2[t(BH4BB6[r(B6)]H6)])])]HH4)]', time=60)



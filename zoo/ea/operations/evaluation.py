from __future__ import annotations

import time
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any, Iterable

import mujoco
from mujoco import viewer
import numpy as np
import torch
import canonical_toolkit as ctk
from networkx import DiGraph
import os
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.a004 import Population, Individual
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import (
    NaCPG,
    create_fully_connected_adjacency,
)
from ariel.simulation.environments._simple_flat import SimpleFlatWorld
from ariel.simulation.tasks.gait_learning import xy_displacement
from ariel.utils.tracker import Tracker

from ._timer import time_func
from ea.config import config, logger

NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))


@time_func
def evaluate_pop_novelty(population: Population) -> Iterable[float]:
    """
    returns array of size len(population)
    with population being the full entire population.
    and the value representing the individual novelty scores
    """
    match config.NOVELTY_METHOD:
        case "ctk":
            novelty = _ctk_novelty(population)
        case "fuda":
            novelty = _fuda_novelty(population)
        case _:
            novelty = [1] * len(population)
            logger.warn('found novelty method thats not been implemented')
            
    return novelty


@time_func
def evaluate_pop(population: Population) -> Population:
    if config.STORE_NOVELTY:
        novelty_scores = evaluate_pop_novelty(population)
    else:
        novelty_scores = None

    eval_indices = [
        idx for idx, ind in enumerate(population) if ind.requires_eval
    ]

    # --- Parallel MuJoCo speed evaluation ---
    speed_map = {}
    if config.STORE_SPEED and eval_indices:
        tasks = [(idx, population[idx].genotype) for idx in eval_indices]
        with Pool(processes=NUM_WORKERS) as pool:
            results = pool.map(_evaluate_speed_from_genotype, tasks)
        speed_map = dict(results)

    # --- Sequential: novelty assignment, penalty, fitness ---
    for idx in eval_indices:
        ind = population[idx]

        fitness_novelty = 1.0
        if config.STORE_NOVELTY:
            novelty = novelty_scores[idx] if config.STORE_NOVELTY else 1.0  # type: ignore
            ind.tags["novelty"] = novelty
            if novelty < 0 or novelty > 1:
                logger.warn(f"found novelty {novelty} for index {idx} {novelty_scores}")
            if config.FITNESS_NOVELTY:
                fitness_novelty = novelty

        fitness_speed = 1.0
        if config.STORE_SPEED:
            speed = speed_map[idx]
            ind.tags["speed"] = speed
            if config.FITNESS_SPEED:
                fitness_speed = speed

        fitness_penalty = 1.0
        if config.PENALTY:
            fitness_penalty = _calc_penalty(ind)

        ind.fitness = fitness_novelty * fitness_speed * fitness_penalty

        if config.RNG.random() < config.ARCHIVE_CHANCE:
            ind.tags["archived"] = True

        ind.requires_eval = False

    return population


def _ctk_novelty(population: Population) -> np.ndarray:
    """
    Requres pre-saved features sorted by ctk_string order
    """
    try:
        folder_names = sorted(
            (config.OUTPUT_FOLDER / "feature_frames").glob("gen_*")
        )#[-1]
        last_path = folder_names[-1]
        print(last_path)
        # print(folder_name)
        pop_frame = ctk.SimilarityFrame.load(last_path)
    except:
        raise FileNotFoundError(
            "cant find reature frame to do novelty analysis"
        )

    novelty_spaces = config.NOVELTY_SPACES
    assert novelty_spaces, 'noveltyspaces must not be none'
    novelty_series_list = []
    for space in novelty_spaces:
        space_name = space.name
        pop_series = pop_frame[space_name]

        try:
            archive_series = ctk.SimilaritySeries.load(
                config.OUTPUT_FOLDER / "archive" / space_name
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
    
    frame_slice.map('normalize_by_radius')
    
    matrix = frame_slice.aggregate().aggregate()
    
    matrix = matrix / len(config.NOVELTY_SPACES)

    sorted_similarity_array = matrix.sum_to_rows(
        zero_diagonal=True, k=config.K_NOVELTY, largest=True, normalise=True
    )[: len(population)]
    sorted_novelty_array = 1 - sorted_similarity_array

    print(sorted_novelty_array)

    sorted_indices = sorted(
        range(len(population)),
        key=lambda i: str(population[i].tags["ctk_string"]),
    )
    inverse_indices = np.argsort(sorted_indices)
    novelty_array = sorted_novelty_array[inverse_indices]

    return np.clip(novelty_array, 0, 1.0)


def _evaluate_speed_from_genotype(args: tuple) -> tuple[int, float]:
    """Standalone function for multiprocessing - takes (idx, genotype) and returns (idx, speed)."""
    idx, genotype = args
    from ea.config import (
        config,
    )  # Import inside worker to avoid pickling issues

    # with torch.no_grad():
    matrixes = config.NDE.forward(np.array(genotype))
    hpd = HighProbabilityDecoder(num_modules=config.NUM_MODULES)
    ind_graph = hpd.probability_matrices_to_graph(
        matrixes[0], matrixes[1], matrixes[2]
    )

    robot = construct_mjspec_from_graph(ind_graph)
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    world.spawn(robot.spec)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM, name_to_bind="core"
    )
    adj_dict = create_fully_connected_adjacency(len(data.ctrl))
    na_cpg = NaCPG(adj_dict)
    ctrl = Controller(
        controller_callback_function=lambda _, d: na_cpg.forward(d.time),
        tracker=tracker,
    )
    ctrl.tracker.setup(world.spec, data)
    mujoco.set_mjcb_control(ctrl.set_control)

    duration, warmup = config.TOTAL_SIM_DURATION, config.SIM_WARMUP
    mujoco.mj_resetData(model, data)
    while data.time < duration:
        mujoco.mj_step(model, data, nstep=100)

    pos_data = tracker.history["xpos"][0]
    warmup_index = int((warmup / duration) * len(pos_data))
    start_xy = (pos_data[warmup_index][0], pos_data[warmup_index][1])
    end_xy = (pos_data[-1][0], pos_data[-1][1])
    speed = xy_displacement(start_xy, end_xy) / (duration - warmup)

    return (idx, speed)


def _fuda_novelty(population: Population) -> list[Any]:  # type: ignore
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


def _calc_limb_length(node: ctk.Node) -> float:
    """Calculate the Length of Limbs descriptor (E) for a robot morphology.

    E measures how extended/elongated the limbs are versus how branched:
    - High E (approaching 1.0): Long chain-like limbs (snake-like)
    - Low E (approaching 0.0): Short limbs with lots of branching (tree-like)

    Formula: E = e / emax if m >= 3, else 0
    Where:
        m = total modules
        e = modules with exactly 2 faces attached (excluding core)
        emax = m - 2
    """
    m = 0
    e = 0

    def _count_module(n: ctk.Node) -> None:
        nonlocal m, e
        m += 1
        if n.is_root:
            return
        num_children = sum(1 for _ in n.children)
        num_connected_faces = 1 + num_children
        if num_connected_faces == 2:
            e += 1

    node.traverse_depth_first(_count_module)
    if m < 3:
        return 0.0

    emax = m - 2
    return e / emax if emax > 0 else 0.0


def _calc_penalty(individual: Individual) -> float:
    node = ctk.node_from_nde_genotype(individual.genotype, config.NDE)
    E = _calc_limb_length(node)

    return max(0.1, 1.0 - E)

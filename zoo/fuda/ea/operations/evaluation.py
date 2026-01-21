from __future__ import annotations

import time
from multiprocessing import Pool
from typing import TYPE_CHECKING, Any

import mujoco
from mujoco import viewer
import numpy as np
import torch
import canonical_toolkit as ctk
from networkx import DiGraph
import os
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.a004 import Population, Individual
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import NaCPG, create_fully_connected_adjacency
from ariel.simulation.environments._simple_flat import SimpleFlatWorld
from ariel.simulation.tasks.gait_learning import xy_displacement
from ariel.utils.tracker import Tracker

from ea.config import config, logger

NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))

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
    

def evaluate_pop(population: Population):
    t_start = time.perf_counter()

    t_novelty_total = 0.0  #~~~~~~~~~~~~ time
    if config.STORE_NOVELTY:
        t0 = time.perf_counter()  #~~~~~~~~~~~~ time
        novelty_scores = evaluate_novelty(population)
        t_novelty_total = time.perf_counter() - t0  #~~~~~~~~~~~~ time
    else:
        novelty_scores = None
    
    eval_indices = [idx for idx, ind in enumerate(population) if ind.requires_eval]
    n_evaluated = len(eval_indices)

    # --- Parallel MuJoCo speed evaluation ---
    speed_map = {}
    t_speed_total = 0.0  #~~~~~~~~~~~~ time
    if config.STORE_SPEED and eval_indices:
        t0 = time.perf_counter()  #~~~~~~~~~~~~ time
        tasks = [(idx, population[idx].genotype) for idx in eval_indices]
        with Pool(processes=NUM_WORKERS) as pool:
            results = pool.map(_evaluate_speed_from_genotype, tasks)
        speed_map = dict(results)
        t_speed_total = time.perf_counter() - t0  #~~~~~~~~~~~~ time

    # --- Sequential: novelty assignment, penalty, fitness ---
    t_penalty_total = 0.0 #~~~~~~~~~~~~ time

    for idx in eval_indices:
        ind = population[idx]

        fitness_novelty = 1.0
        if config.STORE_NOVELTY:
            novelty = novelty_scores[idx] if config.STORE_NOVELTY else 1.0  # type: ignore
            ind.tags['novelty'] = novelty
            assert novelty <= 1, f'found novelty {novelty} for index {idx} {novelty_scores}'
            if config.FITNESS_NOVELTY:
                fitness_novelty = novelty

        fitness_speed = 1.0
        if config.STORE_SPEED:
            speed = speed_map[idx]
            ind.tags['speed'] = speed
            if config.FITNESS_SPEED:
                fitness_speed = speed

        fitness_penalty = 1.0
        if config.PENALTY:
            t0 = time.perf_counter() #~~~~~~~~~~~~ time
            fitness_penalty = _calc_penalty(ind)
            t_penalty_total += time.perf_counter() - t0 #~~~~~~~~~~~~ time

        ind.fitness = fitness_novelty * fitness_speed * fitness_penalty

        if config.RNG.random() < config.ARCHIVE_CHANCE:
            ind.tags['archived'] = True
            
        ind.requires_eval = False

    logger.info(f"[TIMING] Evaluated {n_evaluated} individuals:")
    if config.STORE_NOVELTY:
        logger.info(f"         Novelty: {t_novelty_total:.2f}s ({t_novelty_total/max(n_evaluated,1)*1000:.1f}ms/ind)")
    if config.STORE_SPEED:
        logger.info(f"         Speed (MuJoCo parallel): {t_speed_total:.2f}s ({t_speed_total/max(n_evaluated,1)*1000:.1f}ms/ind)")
    if config.PENALTY:
        logger.info(f"         Penalty: {t_penalty_total:.2f}s ({t_penalty_total/max(n_evaluated,1)*1000:.1f}ms/ind)")
    logger.info(f"[TIMING] Total evaluate_pop: {time.perf_counter() - t_start:.2f}s")

    return population


def evaluate_novelty(population: Population) -> np.ndarray:
    # Sort by ctk_string for consistent feature storage order
    sorted_indices = sorted(range(len(population)), key=lambda i: population[i].tags["ctk_string"])
    sorted_pop = [population[i] for i in sorted_indices]
    nodes = [ctk.node_from_string(ind.tags["ctk_string"]) for ind in sorted_pop]

    # Build frame with ALL STORE_SPACES
    series_list = [
        ctk.series_from_node_population(nodes, space_config=sim_config)
        for sim_config in config.SIM_CONFIGS
    ]
    pop_frame = ctk.SimilarityFrame(series=series_list)
    pop_frame.save(config.OUTPUT_FOLDER / 'feature_frames', 'gen')

    # For novelty: select only NOVELTY_SPACES and combine with archive
    novelty_spaces = config.NOVELTY_SPACES or config.STORE_SPACES
    novelty_series_list = []
    for space in novelty_spaces:
        space_name = space.name
        pop_series = pop_frame[space_name]

        # Try to load and combine with archive
        try:
            archive_series = ctk.SimilaritySeries.load(config.OUTPUT_FOLDER / 'archive' / space_name)
            combined = pop_series | archive_series
        except Exception:
            combined = pop_series

        novelty_series_list.append(combined)

    # Aggregate across spaces and calculate novelty
    novelty_frame = ctk.SimilarityFrame(series=novelty_series_list)
    aggregated_series = novelty_frame.aggregate()
    aggregated_series.map('cosine_similarity')
    matrix = aggregated_series.aggregate()
    matrix.normalize_by_radius()

    # Create reverse mapping to return novelty in original population order
    reverse_indices = [0] * len(population)
    for new_idx, orig_idx in enumerate(sorted_indices):
        reverse_indices[orig_idx] = new_idx

    pop_size = len(population)
    similarity_array = matrix.sum_to_rows(zero_diagonal=True, k=config.K_NOVELTY, largest=True, normalise=True)[:pop_size]
    novelty_sorted = 1 - similarity_array

    # Reorder novelty back to original population order
    novelty_array = np.array([novelty_sorted[reverse_indices[i]] for i in range(pop_size)])
    return np.clip(novelty_array, 0, 1.0)
    
def _evaluate_speed_from_genotype(args: tuple) -> tuple[int, float]:
    """Standalone function for multiprocessing - takes (idx, genotype) and returns (idx, speed)."""
    idx, genotype = args
    from ea.config import config  # Import inside worker to avoid pickling issues

    with torch.no_grad():
        matrixes = config.NDE.forward(np.array(genotype))
        hpd = HighProbabilityDecoder(num_modules=config.NUM_MODULES)
        ind_graph = hpd.probability_matrices_to_graph(matrixes[0], matrixes[1], matrixes[2])

    robot = construct_mjspec_from_graph(ind_graph)
    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    world.spawn(robot.spec)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    tracker = Tracker(mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    adj_dict = create_fully_connected_adjacency(len(data.ctrl))
    na_cpg = NaCPG(adj_dict)
    ctrl = Controller(controller_callback_function=lambda _, d: na_cpg.forward(d.time), tracker=tracker)
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


# TODO. is displacement in meter
def _evaluate_speed(
    individual: Individual,
    duration: float = 60.0,
    warmup: float = 30.0,
    show_viewer: bool = False,
) -> float:
    """Simulate robot and return speed (displacement / time).

    Speed is measured only after the warmup period to allow the robot's
    gait to stabilize before measuring locomotion performance.
    """
    robot = construct_mjspec_from_graph(_ind_to_graph(individual))

    mujoco.set_mjcb_control(None)
    world = SimpleFlatWorld()
    world.spawn(robot.spec)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    tracker = Tracker(
        mujoco_obj_to_find=mujoco.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
    )

    adj_dict = create_fully_connected_adjacency(len(data.ctrl))
    na_cpg = NaCPG(adj_dict)

    ctrl = Controller(
        controller_callback_function=lambda _, d: na_cpg.forward(d.time),
        tracker=tracker,
    )
    ctrl.tracker.setup(world.spec, data)
    mujoco.set_mjcb_control(ctrl.set_control)

    mujoco.mj_resetData(model, data)
    while data.time < duration:
        mujoco.mj_step(model, data, nstep=100)

    pos_data = tracker.history["xpos"][0]
    total_entries = len(pos_data)
    warmup_index = int((warmup / duration) * total_entries)

    start_xy = (pos_data[warmup_index][0], pos_data[warmup_index][1])
    end_xy = (pos_data[-1][0], pos_data[-1][1])

    displacement = xy_displacement(start_xy, end_xy)
    measurement_time = duration - warmup
    speed = displacement / measurement_time

    if show_viewer:
        mujoco.mj_resetData(model, data)
        ctrl.tracker.reset()
        viewer.launch(model, data)

    return speed



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

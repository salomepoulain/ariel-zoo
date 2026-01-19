from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mujoco
from mujoco import viewer
import numpy as np
import torch
import canonical_toolkit as ctk
from networkx import DiGraph

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.a004 import Population, Individual
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.controllers.na_cpg import NaCPG, create_fully_connected_adjacency
from ariel.simulation.environments._simple_flat import SimpleFlatWorld
from ariel.simulation.tasks.gait_learning import xy_displacement
from ariel.utils.tracker import Tracker

from ea.config import config

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
    novelty_scores = evaluate_novelty(population) if config.STORE_NOVELTY else None
    
    for idx, ind in enumerate(population):
        if ind.requires_eval:
            ind_graph = _ind_to_graph(ind)
            _store_ctk_string(ind, ind_graph) if config.STORE_STRING else None
            
            fitness_novelty = 1.0
            if config.STORE_NOVELTY:
                novelty = novelty_scores[idx] if config.STORE_NOVELTY else 1.0 # type: ignore
                ind.tags['novelty'] = novelty
                assert novelty <= 1, f'found novelty {novelty} for index {idx} {novelty_scores}'
                if config.FITNESS_NOVELTY:
                    fitness_novelty = novelty
            
            fitness_speed = 1.0
            if config.STORE_SPEED:
                speed = _evaluate_speed(ind) if config.STORE_SPEED else 1.0
                ind.tags['speed'] = speed
                if config.FITNESS_SPEED:
                    fitness_speed = speed
                    
            fitness_penalty = 1.0
            if config.PENALTY:
                fitness_penalty = _calc_penalty(ind)
            
            ind.fitness = fitness_novelty * fitness_speed * fitness_penalty
            
            if config.RNG.random() < config.ARCHIVE_CHANCE:
                ind.tags['archived'] = True
            ind.requires_eval = False
    
    return population


def evaluate_novelty(population: Population) -> np.ndarray:
    pop_series = ctk.series_from_graph_population([_ind_to_graph(ind) for ind in population], space_config=config.SIM_CONFIG)
    pop_series.save(config.OUTPUT_FOLDER / 'features', 'generation')

    try:                                   
        archive_series = ctk.SimilaritySeries.load(config.OUTPUT_FOLDER / 'archive')                                                                                             
        series = pop_series | archive_series                                                                                          
    except Exception:                                                                                                                                             
        series = pop_series  

    series.cosine_similarity()
    matrix = series.aggregate()
    matrix.normalize_by_radius()
    pop_size = len(population)
    similarity_array = matrix.sum_to_rows(zero_diagonal=True, k=config.K_NOVELTY, largest=True)[:pop_size]
    novelty_array = 1 - similarity_array
    return np.clip(novelty_array, 0, 1.0)
    
#TODO setup multiprocess?

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

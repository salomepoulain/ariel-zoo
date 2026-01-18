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
    novelty_scores = evaluate_novelty(population)
    
    for idx, ind in enumerate(population):
        if ind.requires_eval:
            ind_graph = _ind_to_graph(ind)
            _store_ctk_string(ind, ind_graph) if config.STORE_STRING else None

            novelty = novelty_scores[idx]
            ind.tags['novelty'] = novelty
            assert novelty <= 1, f'found novelty {novelty} for index {idx} {novelty_scores}'
            
            speed = _evaluate_speed(ind) if config.SPEED_EVAL else 1.0
            ind.tags['speed'] = speed
            ind.fitness = novelty * speed
            
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
    

# TODO. is displacement in meter
def _evaluate_speed(
    individual: Individual,
    duration: float = 60.0,
    show_viewer: bool = False,
) -> float:
    """Simulate robot and return speed (displacement / time)."""
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
    start_xy = (pos_data[0][0], pos_data[0][1])
    end_xy = (pos_data[-1][0], pos_data[-1][1])

    displacement = xy_displacement(start_xy, end_xy)
    speed = displacement / data.time

    if show_viewer:
        mujoco.mj_resetData(model, data)
        ctrl.tracker.reset()
        viewer.launch(model, data)

    return speed

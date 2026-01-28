from __future__ import annotations

from typing import  TYPE_CHECKING

import canonical_toolkit as ctk

from ea.config import config

if TYPE_CHECKING:
    from ariel.ec.a004 import Population

def store_string(population: Population):
    for individual in population:
        if individual.requires_eval:
            node = ctk.node_from_nde_genotype(individual.genotype, num_modules=config.NUM_MODULES)
            ctk_string = node.to_string()
            individual.tags["ctk_string"] = ctk_string

    return population

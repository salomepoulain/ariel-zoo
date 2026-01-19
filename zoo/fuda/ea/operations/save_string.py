from __future__ import annotations

from typing import  TYPE_CHECKING

import canonical_toolkit as ctk

from ..config import config

if TYPE_CHECKING:
    from ariel.ec.a004 import Population

def save_string(population: Population):
    for individual in population:
        if individual.requires_eval:
            node = ctk.node_from_nde_genotype(individual.genotype, NDE=config.NDE)
            ctk_string = node.to_string()
            individual.tags["ctk_string"] = ctk_string
    
    return population

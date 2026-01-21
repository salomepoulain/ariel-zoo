
import numpy as np
from ..config import config
from ariel.ec.a004 import Population, Individual


def parent_select_genotypes(population: Population) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    """
    Selects 3 distinct parents uniformly at random using the config RNG.
    Returns their genotypes (List of Arrays).
    """
    indices = config.RNG.choice((len(population)), 3, replace=False)
    
    p1: list[list[float]] = population[indices[0]].genotype
    p2: list[list[float]] = population[indices[1]].genotype
    p3: list[list[float]] = population[indices[2]].genotype
    
    return p1, p2, p3


def revde(population: Population) -> Population:
    """
    Applies Reversible Differential Evolution to create a new generation.
    Handles the 'List of Arrays' genotype structure by looping over parts.
    """
    # valid_population = [ind for ind in population if ind.tags.get('current') == True]
    offspring_list: list[Individual] = []        
    num_triplets = np.ceil(len(population) // 3 + 1)

    for _ in range(num_triplets):
        
        g_parent_a, g_parent_b, g_parent_c = parent_select_genotypes(population)
        
        c1_parts = []
        c2_parts = []
        c3_parts = []

        for i in range(len(g_parent_a)):
            
            part_a = np.array(g_parent_a[i])
            part_b = np.array(g_parent_b[i])
            part_c = np.array(g_parent_c[i])

            mutated_parts = config.REVDE.mutate(part_a, part_b, part_c)
            
            c1_parts.append(list(mutated_parts[0]))
            c2_parts.append(list(mutated_parts[1]))
            c3_parts.append(list(mutated_parts[2]))

        child_1 = Individual()
        child_1.genotype = c1_parts
        child_1.requires_eval = True
        
        child_2 = Individual()
        child_2.genotype = c2_parts
        child_2.requires_eval = True
        
        child_3 = Individual()
        child_3.genotype = c3_parts
        child_3.requires_eval = True

        offspring_list.extend([child_1, child_2, child_3])

    return population + offspring_list

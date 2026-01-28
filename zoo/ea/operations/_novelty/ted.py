from __future__ import annotations

import numpy as np
import zss
from ariel.ec.a004 import Population
import canonical_toolkit as ctk
from ea.config import config
import pandas as pd

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from networkx import DiGraph


__all__ = [
    "ted_novelty",
    "save_ted_archive"
]


def _nx_to_zss_node(graph: DiGraph[Any], node_idx) -> zss.Node:
    """
    Recursively converts a NetworkX node and its children to a zss.Node.
    """
    label = graph.nodes[node_idx].get("module_type", "None")
    z_node = zss.Node(label)

    # We sort children to ensure deterministic order if the graph structure is ambiguous
    children = sorted(list(graph.successors(node_idx)))
    for child_idx in children:
        z_node.addkid(_nx_to_zss_node(graph, child_idx))
    return z_node


def _calculate_similarity_ted(
    individual: DiGraph[Any], target_graph: DiGraph[Any]
) -> float:
    """
    Calculates Tree Edit Distance normalized by the size of the target graph.
    """
    # 0. Handle empty graphs
    len_a = len(individual)
    len_b = len(target_graph)

    if len_b == 0:
        if len_a == 0:
            return 0.0  # Both empty -> Identical
        else:
            return float(
                "inf"
            )  # Infinite distance if target is empty but source is not

    # 1. Find roots
    try:
        root_a = next(n for n, d in individual.in_degree() if d == 0)
        root_b = next(n for n, d in target_graph.in_degree() if d == 0)
    except StopIteration:
        # Handle cases where graphs might be cyclic or have no clear root
        return float("inf")

    # 2. Convert NetworkX graphs to ZSS trees
    tree_a = _nx_to_zss_node(individual, root_a)
    tree_b = _nx_to_zss_node(target_graph, root_b)

    # 3. Calculate Raw Distance
    raw_distance = zss.simple_distance(tree_a, tree_b)

    # 4. Normalize
    # This tells you: "How much error is there per node in the target?"
    normalized_score = raw_distance / len_b

    return normalized_score


def ted_novelty(population: Population) -> list[float]:
    # 1. Prepare Population Graphs
    pop_graphs = [
        ctk.node_from_string(ind.tags['ctk_string']).to_graph()
        for ind in population
    ]
    n_pop = len(pop_graphs)
    K = config.K_NOVELTY

    # 2. Load and Prepare Archive Graphs
    archive_file = config.OUTPUT_FOLDER / 'archive' / 'ted_archive.csv'
    archive_graphs = []
    if archive_file.exists():
        archive_df = pd.read_csv(archive_file)
        archive_graphs = [
            ctk.node_from_string(s).to_graph()
            for s in archive_df['ctk_string'].tolist()
        ]
    n_arch = len(archive_graphs)

    # 3. Build Distance Matrix (Population vs Combined Pool)
    # Each row is a population individual; columns are (population + archive)
    total_pool_size = n_pop + n_arch
    combined_dist = np.zeros((n_pop, total_pool_size))

    # A. Intra-population distances (Symmetric)
    for i in range(n_pop):
        for j in range(i + 1, n_pop):
            dist = _calculate_similarity_ted(pop_graphs[i], pop_graphs[j])
            combined_dist[i, j] = dist
            combined_dist[j, i] = dist

    # B. Population vs Archive distances
    if n_arch > 0:
        for i in range(n_pop):
            for j in range(n_arch):
                dist = _calculate_similarity_ted(pop_graphs[i], archive_graphs[j])
                # Store in the archive-dedicated columns (after n_pop)
                combined_dist[i, n_pop + j] = dist

    # 4. Calculate Novelty (Mean distance to K-nearest neighbors)
    novelty_list = [0.0] * n_pop
    for i in range(n_pop):
        # Sort distances for individual i against everyone (self, others, archive)
        distances_sorted = np.sort(combined_dist[i])

        # index 0 is always 0.0 (self), so we take 1 to K+1
        # If the pool is smaller than K, we average over what we have
        actual_k = min(K, total_pool_size - 1)
        if actual_k > 0:
            novelty_list[i] = distances_sorted[1 : actual_k + 1].sum() / actual_k

    return novelty_list


def save_ted_archive(population: Population):
    new_strings = [
        ind.tags['ctk_string']
        for ind in population
        if ind.tags.get('archived') and not ind.alive
    ]

    if not new_strings:
        return

    archive_dir = config.OUTPUT_FOLDER / 'archive'
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_file = config.OUTPUT_FOLDER / 'archive' / 'ted_archive.csv'

    new_df = pd.DataFrame({'ctk_string': new_strings})

    # Header only exists if the file is new
    header = not archive_file.exists()
    new_df.to_csv(archive_file, mode='a', index=False, header=header)

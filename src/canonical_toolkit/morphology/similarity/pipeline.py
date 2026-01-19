"""
Similarity module - Similarity analysis and fingerprinting.

This module provides high-level functions for calculating similarity between
trees (node) using neighborhood fingerprints and feature hashing, and providing a pipeline
with the analysis tools form the *MATRIX* and *VISUAL* package
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from sklearn.feature_extraction import FeatureHasher

from .sim_matrix import SimilarityMatrix, SimilaritySeries

from .options import (
    MatrixDomain,
    Space,
)

from ..node import (
    deriver,
    node_from_graph
)

from .options import (
    SimilaritySpaceConfig,
    OutputType
)

from .sim_matrix import SimilarityMatrix, SimilaritySeries, SimilarityFrame

if TYPE_CHECKING:
    from collections.abc import Callable

    from .sim_types import (
        HashFingerprint,
        PopulationFingerprint,
        FeatureHasherProtocol,
    )

    from ..node import (
        Node,
    )
    
    from networkx import DiGraph


__all__ = [
    "collect_subtrees",
    "collect_hash_fingerprint",
    "collect_population_fingerprint",
    "series_from_graph_population", # Questionable? Might be too abstract. If users should do it themselves, get better understanding of what they are doing?
    "series_from_node_population", # Idem.
    "series_from_population_fingerprint",
    "frame_from_graph_population"
]


def collect_subtrees(node: Node, output_type: OutputType):
    serializer_fn = output_type.value  # Get the callable
    return deriver.collect_subtrees(node, serializer_fn)

def collect_hash_fingerprint(
    node: Node, 
    *, 
    config: SimilaritySpaceConfig | None
) -> HashFingerprint:
    if not config:
        config = SimilaritySpaceConfig()
    
    if not config.space.value == '':
        node = node.copy()
        
        # detatch radial children, if you want axial hashes
        if config.space.value == 'A_#__':
            for radial_child in node.radial_children:
                radial_child.detatch_from_parent()
                
        # detatch axial children, if you want radial hashes
        elif config.space.value == 'R_#__':
            for axial_child in node.axial_children:
                axial_child.detatch_from_parent()
        
        # just get hashes from 1 limb   
        else:
            node = node.get(config.space.name)
            if not node:
                return {-1: []}
            node.detatch_from_parent()
                
    return deriver.collect_neighbourhoods(
        starting_node=node,
        serializer_fn=OutputType.STRING,
        use_node_max_radius=config.radius_strategy.value,
        tree_max_radius=config.max_hop_radius,
        canonicalized=True,
        do_radius_prefix=True,
        hash_prefix=config.space.value,
    )
    
def collect_population_fingerprint(
    node_population: list[Node], 
    *, 
    config: SimilaritySpaceConfig | None = None
) -> PopulationFingerprint:
    if not config:
        config = SimilaritySpaceConfig()
    return [collect_hash_fingerprint(node, config=config) for node in node_population]

def series_from_population_fingerprint(
    population_fingerprint: PopulationFingerprint,
    space: Space | str = Space.WHOLE,
    *,
    max_radius: int | None = None,
    hasher: FeatureHasherProtocol | None = None,
    n_features: int = 2 ** 24,
) -> SimilaritySeries:
    """Create SimilaritySeries from population neighbourhood dictionaries.

    Converts neighbourhood strings to sparse feature matrices using
    feature hashing, creating one MatrixInstance per radius.

    Args:
        population_neighbourhoods: List of neighbourhood dicts, one per individual.
            Each dict maps radius -> list of neighbourhood strings.
            Example: [{0: ['r0__B', 'r0__H'], 1: ['r1__BB']}, ...]
        space: VectorSpace or custom space name for the morphology
        n_features: Size of the hash space (number of features)
        hasher: Optional hasher implementing FeatureHasherProtocol.
            Must have .transform(list[list[str]]) -> sparse matrix.
            Defaults to FeatureHasher(n_features, input_type='string').
        max_radius: Maximum radius to process. Auto-detected if None.

    Returns
    -------
        SimilaritySeries with one MatrixInstance per radius, indexed by radius.
        Each matrix has shape (n_individuals, n_features) and is sparse.

    Example:
        >>> neighbourhoods = [
        ...     {0: ["r0__B"], 1: ["r1__BB", "r1__BH"]},  # Individual 0
        ...     {0: ["r0__H"], 1: ["r1__HB"]},  # Individual 1
        ... ]
        >>> series = MatrixSeries.from_neighbourhood_dicts(
        ...     neighbourhoods, space=VectorSpace.FRONT_LIMB, n_features=1000
        ... )
        >>> series[1].shape  # Access radius 1 matrix
        (2, 1000)  # 2 individuals, 1000 features
        >>> series[1][0, 5]  # Individual 0, feature 5
    """
    # Create hasher if not provided
    if hasher is None:
        hasher = FeatureHasher(n_features=n_features, input_type="string")

    # Auto-detect max radius
    if max_radius is None:
        max_radius = max(max(d.keys()) for d in population_fingerprint)

    # Build instances for each radius
    instances_list = []
    for radius in range(max_radius + 1):
        # Collect features at this radius from all individuals
        radius_features = [
            pop_neigh.get(radius, []) for pop_neigh in population_fingerprint
        ]

        # Transform to sparse matrix: (n_individuals, n_features)
        sparse_matrix = hasher.transform(radius_features)

        # Wrap in MatrixInstance
        instance = SimilarityMatrix(
            matrix=sparse_matrix,
            space=space,
            radius=radius,
            domain=MatrixDomain.FEATURES,
        )
        instances_list.append(instance)

    return SimilaritySeries(
        instances_list=instances_list
    )
    
# Region nx based pipeline helpers

def series_from_graph_population(
    population: list[DiGraph[Any]], 
    space_config: SimilaritySpaceConfig
) -> SimilaritySeries:
    node_population = [node_from_graph(graph) for graph in population]
        
    population_fingerprint = collect_population_fingerprint(
        node_population=node_population, 
        config=space_config
    )

    series = series_from_population_fingerprint(
        population_fingerprint=population_fingerprint, 
        space=space_config.space, 
        max_radius=space_config.max_hop_radius, 
        hasher=space_config.hasher
    )
        
    return series

def series_from_node_population(
    population: list[Node], 
    space_config: SimilaritySpaceConfig
) -> SimilaritySeries:  
    population_fingerprint = collect_population_fingerprint(
        node_population=population, 
        config=space_config
    )

    series = series_from_population_fingerprint(
        population_fingerprint=population_fingerprint, 
        space=space_config.space, 
        max_radius=space_config.max_hop_radius, 
        hasher=space_config.hasher
    )
        
    return series

def frame_from_graph_population(
    population: list[DiGraph[Any]], 
    space_configs: list[SimilaritySpaceConfig]
) -> SimilarityFrame:
    all_series = [series_from_graph_population(population, config) for config in space_configs]    
    frame = SimilarityFrame(all_series)
    return frame

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

MAX_ARB_RADIUS = 20

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
    config: SimilaritySpaceConfig | None = None
) -> HashFingerprint:
    config = config or SimilaritySpaceConfig()
    node = node.copy()

    if config.space != Space.WHOLE:

        if config.space.value == 'A_#__':
            for radial_child in node.radial_children:
                radial_child.detatch_from_parent()

        elif config.space.value == 'R_#__':
            for axial_child in node.axial_children:
                axial_child.detatch_from_parent()
        else:
            target_node = node.get(config.space.name)

            # print(config.space)
            if not target_node:
                return {}
                data = {}
                for r in range(min(config.max_hop_radius, MAX_ARB_RADIUS)):
                    data[r] = []
                # for r in range(config.max_hop_radius):
                #     data[r] = [config.space.value + config.emtpy_value if config.emtpy_value else config.space.value]
                return data
            node = target_node
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
    skip_empty: bool = False
) -> SimilaritySeries:
    """
    # TODO: turn into config args
    Transforms tree fingerprints into a typed SimilaritySeries.
    """
    if hasher is None:
        hasher = FeatureHasher(n_features=n_features, input_type="string")

    if max_radius is None:
        max_radius = max((max(d.keys()) for d in population_fingerprint if d), default=0)

    instances = []
    for radius in range(max_radius + 1):
        radius_features = [fp.get(radius) for fp in population_fingerprint]

        if skip_empty and not any(radius_features):
            continue

        fill = [f"{space.value}"] if not skip_empty else []
        processed_features = [feat if feat else fill for feat in radius_features]
        sparse_matrix = hasher.transform(processed_features)

        tags = {
            'domain' : MatrixDomain.FEATURES,
            'radius' : radius, #TODO. might not make it required? could make diy function easier?
        }

        instance = SimilarityMatrix(
            matrix=sparse_matrix,
            label=space.name,
            tags=tags
        )
        instances.append(instance)

    return SimilaritySeries(matrices=instances)

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
        hasher=space_config.hasher,
        skip_empty=space_config.skip_empty
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
        hasher=space_config.hasher,
        skip_empty=space_config.skip_empty
    )

    return series

def frame_from_graph_population(
    population: list[DiGraph[Any]],
    space_configs: list[SimilaritySpaceConfig]
) -> SimilarityFrame:
    """
    Highest level of the pipeline: DiGraph List -> SimilarityFrame.
    """
    node_population = [node_from_graph(graph) for graph in population]

    all_series = []
    for config in space_configs:
        fingerprint = [collect_hash_fingerprint(n, config=config) for n in node_population]

        series = series_from_population_fingerprint(
            population_fingerprint=fingerprint,
            space=config.space,
            max_radius=config.max_hop_radius,
            hasher=config.hasher,
            skip_empty=space_config.skip_empty
        )
        all_series.append(series)

    return SimilarityFrame(series=all_series)

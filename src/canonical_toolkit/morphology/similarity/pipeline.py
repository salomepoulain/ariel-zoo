"""
Similarity module - Similarity analysis and fingerprinting.

This module provides high-level functions for calculating similarity between
trees (node) using neighborhood fingerprints and feature hashing, and providing a pipeline
with the analysis tools form the *MATRIX* and *VISUAL* package
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics.pairwise import cosine_similarity

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

    from ..visual.grid_config import EmbeddingGridConfig, HeatmapGridConfig
    
    from networkx import DiGraph


__all__ = [
    "collect_subtrees",
    "collect_hash_fingerprint",
    "collect_population_fingerprint",
    "series_from_population_fingerprint",
    "frame_to_embedding_grid",
    "frame_to_heatmap_grid",
    "series_to_cumulative_similarity",
    "series_from_graph_population",
    "series_from_node_population",
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


# create your own ea pipeline with actions to be applied to a population of graphs




# --- Grid Conversion Functions ---





















def frame_to_embedding_grid(
    frame: MatrixFrame,
    population_size: int,
    row_order: list | None = None,
    col_order: list | None = None,
    hover_data_fn: Callable[[str, int, int], str] | None = None,
    title_fn: Callable[[str, int], str] | None = None,
    default_dot_size: int = 8,
    highlight_dot_size: int = 8,
) -> list[list[EmbeddingGridConfig]]:
    """
    Convert MatrixFrame with embedding data to grid of EmbeddingGridConfig.

    Frame must contain 2D embeddings (shape: n_robots x 2).
    Typically used after UMAP/PCA dimensionality reduction.

    Args:
        frame: MatrixFrame where each instance.matrix is (n_robots, 2) embeddings
        population_size: Number of robots in population
        row_order: Order of rows (defaults to sorted indices)
        col_order: Order of columns (defaults to series labels)
        hover_data_fn: Optional function (label, index, robot_id) -> str for hover text
        title_fn: Optional function (label, index) -> str for subplot titles
        default_dot_size: Size for regular dots
        highlight_dot_size: Size for highlighted dots

    Returns:
        2D list of EmbeddingGridConfig ready for plotting
    """
    from ..visual.grid_config import EmbeddingGridConfig

    # Determine orders
    if row_order is None:
        all_indices = set()
        for series in frame.series:
            all_indices.update(series.indices)
        row_order = sorted(all_indices)

    if col_order is None:
        col_order = list(frame.keys())

    # Default title function
    if title_fn is None:
        title_fn = lambda label, idx: f"{label} [r={idx}]"

    # Build grid
    grid = []
    for row_idx in row_order:
        row_configs = []
        for col_label in col_order:
            series = frame[col_label]
            inst = series.instances.get(row_idx)

            if inst is None:
                # Empty placeholder
                config = EmbeddingGridConfig(
                    title=title_fn(col_label, row_idx) + " (empty)",
                    embeddings=np.array([]),  # Empty
                    idxs=[],
                    hover_data=None,
                    default_dot_size=default_dot_size,
                    highlight_dot_size=highlight_dot_size,
                )
            else:
                # Extract embeddings
                emb = inst.matrix.toarray() if sp.issparse(inst.matrix) else inst.matrix

                # Validate shape
                if emb.ndim != 2 or emb.shape[1] != 2:
                    msg = f"Expected (n, 2) embeddings at {col_label}[{row_idx}], got {emb.shape}"
                    raise ValueError(msg)

                # Generate hover data if function provided
                hover_data = None
                if hover_data_fn:
                    hover_data = [
                        hover_data_fn(col_label, row_idx, robot_id)
                        for robot_id in range(population_size)
                    ]

                config = EmbeddingGridConfig(
                    title=title_fn(col_label, row_idx),
                    embeddings=emb,
                    idxs=list(range(population_size)),
                    hover_data=hover_data,
                    default_dot_size=default_dot_size,
                    highlight_dot_size=highlight_dot_size,
                )

            row_configs.append(config)
        grid.append(row_configs)

    return grid


def frame_to_heatmap_grid(
    frame: MatrixFrame,
    row_order: list | None = None,
    col_order: list | None = None,
    title_fn: Callable[[str, int], str] | None = None,
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
) -> list[list[HeatmapGridConfig]]:
    """
    Convert MatrixFrame with similarity/distance data to HeatmapGridConfig.

    Frame must contain square matrices (n_robots x n_robots).

    Args:
        frame: MatrixFrame where each instance.matrix is (n x n) similarity/distance
        row_order: Order of rows (defaults to sorted indices)
        col_order: Order of columns (defaults to series labels)
        title_fn: Optional function (label, index) -> str for subplot titles
        colormap: Matplotlib colormap name
        vmin/vmax: Value range for colormap

    Returns:
        2D list of HeatmapGridConfig
    """
    from ..visual.grid_config import HeatmapGridConfig

    # Determine orders
    if row_order is None:
        all_indices = set()
        for series in frame.series:
            all_indices.update(series.indices)
        row_order = sorted(all_indices)

    if col_order is None:
        col_order = list(frame.keys())

    # Default title function
    if title_fn is None:
        title_fn = lambda label, idx: f"{label} [r={idx}]"

    # Build grid
    grid = []
    for row_idx in row_order:
        row_configs = []
        for col_label in col_order:
            series = frame[col_label]
            inst = series.instances.get(row_idx)

            if inst is None:
                # Empty placeholder
                config = HeatmapGridConfig(
                    title=title_fn(col_label, row_idx) + " (empty)",
                    heatmap_data=np.array([[]]),  # Empty 2D array
                    colormap=colormap,
                    vmin=vmin,
                    vmax=vmax,
                )
            else:
                # Extract heatmap data
                heatmap = inst.matrix.toarray() if sp.issparse(inst.matrix) else inst.matrix

                # Validate square
                if heatmap.ndim != 2 or heatmap.shape[0] != heatmap.shape[1]:
                    msg = f"Expected square matrix at {col_label}[{row_idx}], got {heatmap.shape}"
                    raise ValueError(msg)

                config = HeatmapGridConfig(
                    title=title_fn(col_label, row_idx),
                    heatmap_data=heatmap,
                    colormap=colormap,
                    vmin=vmin,
                    vmax=vmax,
                )

            row_configs.append(config)
        grid.append(row_configs)

    return grid


def series_to_cumulative_similarity(
    series: MatrixSeries,
    *,
    inplace: bool = False,
) -> MatrixSeries:
    """
    Convert feature series to cumulative similarity series.

    Takes a series with feature matrices and converts to similarity matrices
    by computing cosine similarity at each radius, then cumulative sum across radii.

    Args:
        series: MatrixSeries with feature matrices
        inplace: If True, modify series in place. If False, return new series.

    Returns:
        MatrixSeries with cumulative similarity matrices
    """
    cumsum = None
    new_instances = []

    for idx in series.indices:
        inst = series[idx]

        # Compute similarity
        sim_matrix = cosine_similarity(inst.matrix)

        # Accumulate
        if cumsum is None:
            cumsum = sim_matrix
        else:
            cumsum = cumsum + sim_matrix

        # Create new instance with cumulative similarity
        new_tags = inst.tags.copy()
        new_tags.update({"domain": "similarity", "cumulative": True})

        new_inst = inst.replace(
            matrix=cumsum.copy(),
            tags=new_tags,
        )
        new_instances.append(new_inst)

    if inplace:
        series._instances = {inst.index: inst for inst in new_instances}
        return series

    return MatrixSeries(instances_list=new_instances)

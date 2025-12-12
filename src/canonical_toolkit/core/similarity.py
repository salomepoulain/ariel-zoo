"""
Similarity module - Similarity analysis and fingerprinting.

This module provides high-level functions for calculating similarity between
trees (node) using neighborhood fingerprints and feature hashing, and providing a pipeline
with the analysis tools form the *MATRIX* and *VISUAL* package
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable

from sklearn.feature_extraction import FeatureHasher

from canonical_toolkit.core.matrix.m_enums import (
    MatrixDomain,
    VectorSpace,
)
from canonical_toolkit.core.matrix.m_series import (
    MatrixSeries,
)
from canonical_toolkit.core.matrix.matrix import (
    MatrixInstance,
)

from canonical_toolkit.core.node.tools import (
    deriver,
    serializer,
)

if TYPE_CHECKING:
    from canonical_toolkit.core.matrix.m_types import (
        FeatureHasherProtocol,
    )


if TYPE_CHECKING:
    from networkx import DiGraph

    from canonical_toolkit.core.node.n_types import (
        hash_fingerprint,
        population_fingerprints
    )
    from canonical_toolkit.core.node.node import (
        Node,
    )


class OutputType(Enum):
    STRING = serializer.to_string
    GRAPH = serializer.to_graph
    NODE = None


class RadiusStrategy(Enum):
    """Strategy for determining neighborhood radius in similarity calculations."""

    NODE_LOCAL = True
    TREE_GLOBAL = False


class HVectorSpace(Enum):
    """hash vectorspace."""

    DEFAULT = ""
    FRONT = "R_f__"
    BACK = "R_b__"
    LEFT = "R_l__"
    RIGHT = "R_r__"
    TOP = "A_t__"
    BOTTOM = "A_b__"


@dataclass(slots=True)
class SimilarityConfig:
    """Configuration for calculating neighborhood similarity."""

    vector_space: HVectorSpace = HVectorSpace.DEFAULT
    radius_strategy: RadiusStrategy = RadiusStrategy.NODE_LOCAL
    max_hop_radius: int | None = None


def collect_subtrees(node: Node, output_type: OutputType):
    serializer_fn = output_type.value  # Get the callable
    return deriver.collect_subtrees(node, serializer_fn)


def collect_hash_fingerprint(
    node: Node, config: SimilarityConfig
) -> hash_fingerprint:
    return deriver.collect_neighbourhoods(
        starting_node=node,
        serializer_fn=OutputType.STRING,
        use_node_max_radius=config.radius_strategy.value,
        tree_max_radius=config.max_hop_radius,
        canonicalized=True,
        do_radius_prefix=True,
        hash_prefix=config.vector_space.value,
    )


def series_from_population_fingerprint(
    population_fingerprints: population_fingerprints,
    space: VectorSpace | str,
    *,
    max_radius: int | None = None,
    hasher: FeatureHasherProtocol | None = None,
    n_features: int = 2 * 24,
) -> MatrixSeries:
    """Create MatrixSeries from population neighbourhood dictionaries.

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
        MatrixSeries with one MatrixInstance per radius, indexed by radius.
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
        max_radius = max(max(d.keys()) for d in population_fingerprints)

    # Build instances for each radius
    instances_list = []
    for radius in range(max_radius + 1):
        # Collect features at this radius from all individuals
        radius_features = [
            pop_neigh.get(radius, []) for pop_neigh in population_fingerprints
        ]

        # Transform to sparse matrix: (n_individuals, n_features)
        sparse_matrix = hasher.transform(radius_features)

        # Wrap in MatrixInstance
        instance = MatrixInstance(
            matrix=sparse_matrix,
            space=space,
            radius=radius,
            domain=MatrixDomain.FEATURES,
        )
        instances_list.append(instance)

    return MatrixSeries(
        instances_list=instances_list,
    )



# TODO ai slop make better
def series_to_grid_configs(
    series: MatrixSeries | list[MatrixSeries],
    *,
    layout: str = "row",
    title_fn: Callable[[MatrixSeries, int], str] | None = None,
    under_title_fn: Callable[[MatrixSeries, int], str] | None = None,
    **subplot_kwargs: Any,
) -> list[list[dict[str, Any]]]:
    """Convert MatrixSeries to 2D grid of subplot configs for visual plotters.

    Creates a 2D array structure suitable for plot_robot_grid or plot_interactive_embed_grid.
    Each cell contains the data and metadata for one subplot.

    Args:
        series: Single MatrixSeries or list of MatrixSeries to visualize
        layout: How to arrange the grid:
            - "row": One series per row, radii as columns (default)
            - "column": One series per column, radii as rows
            - "single_row": All radii in a single row
            - "single_column": All radii in a single column
        title_fn: Optional function(series, radius) -> str for main title.
            Defaults to showing space name.
        under_title_fn: Optional function(series, radius) -> str for subtitle.
            Defaults to showing radius.
        **subplot_kwargs: Additional fields to include in each subplot config
            (e.g., default_dot_size=10, img_under_title_fontsize=12)

    Returns
    -------
        2D list of dictionaries, where each dict contains:
        - title: Main title for the subplot
        - under_title: Subtitle for the subplot
        - matrix: The MatrixInstance at this radius
        - space: The VectorSpace/space name
        - radius: The radius value
        - All additional **subplot_kwargs

    Example:
        >>> # Single series in a row
        >>> configs = series_to_grid_configs(
        ...     front_series,
        ...     layout="single_row",
        ...     default_dot_size=10
        ... )
        >>> # Multiple series for comparison (one per row)
        >>> configs = series_to_grid_configs(
        ...     [front_series, back_series],
        ...     layout="row",
        ...     title_fn=lambda s, r: f"{s.space.value} r{r}"
        ... )
        >>> # Then use with plotter:
        >>> # For embeddings: Convert configs to EmbedSubplot objects
        >>> # For robots: Convert configs to RobotSubplot objects
    """
    # Normalize to list
    if isinstance(series, MatrixSeries):
        series_list = [series]
    else:
        series_list = series

    # Default title functions
    if title_fn is None:
        def title_fn(s: MatrixSeries, r: int) -> str:
            space_name = s.space.value if isinstance(s.space, VectorSpace) else str(s.space)
            return space_name

    if under_title_fn is None:
        def under_title_fn(s: MatrixSeries, r: int) -> str:
            return f"r{r}"

    # Build grid based on layout
    if layout == "row":
        # One series per row, radii as columns
        grid = []
        for s in series_list:
            row = []
            for radius in s.radii:
                config = {
                    "title": title_fn(s, radius),
                    "under_title": under_title_fn(s, radius),
                    "matrix": s[radius],
                    "space": s.space,
                    "radius": radius,
                    **subplot_kwargs,
                }
                row.append(config)
            grid.append(row)

    elif layout == "column":
        # One series per column, radii as rows
        # First, collect all unique radii across all series
        all_radii = sorted(set(r for s in series_list for r in s.radii))
        grid = []
        for radius in all_radii:
            row = []
            for s in series_list:
                if radius in s.radii:
                    config = {
                        "title": title_fn(s, radius),
                        "under_title": under_title_fn(s, radius),
                        "matrix": s[radius],
                        "space": s.space,
                        "radius": radius,
                        **subplot_kwargs,
                    }
                else:
                    # Empty placeholder for missing radius
                    config = {
                        "title": "",
                        "under_title": "",
                        "matrix": None,
                        "space": s.space,
                        "radius": radius,
                        **subplot_kwargs,
                    }
                row.append(config)
            grid.append(row)

    elif layout == "single_row":
        # All series and radii in one row
        row = []
        for s in series_list:
            for radius in s.radii:
                config = {
                    "title": title_fn(s, radius),
                    "under_title": under_title_fn(s, radius),
                    "matrix": s[radius],
                    "space": s.space,
                    "radius": radius,
                    **subplot_kwargs,
                }
                row.append(config)
        grid = [row]

    elif layout == "single_column":
        # All series and radii in one column
        grid = []
        for s in series_list:
            for radius in s.radii:
                config = {
                    "title": title_fn(s, radius),
                    "under_title": under_title_fn(s, radius),
                    "matrix": s[radius],
                    "space": s.space,
                    "radius": radius,
                    **subplot_kwargs,
                }
                grid.append([config])

    else:
        msg = f"Unknown layout: {layout}. Use 'row', 'column', 'single_row', or 'single_column'"
        raise ValueError(msg)

    return grid


# TODO ai slop make better
def embeddings_to_grid(
    embeddings: list[Any],
    idxs: list[int],
    series: MatrixSeries | list[MatrixSeries],
    *,
    hover_data: list[Any] | None = None,
    layout: str = "row",
    title_fn: Callable[[MatrixSeries, int], str] | None = None,
    under_title_fn: Callable[[MatrixSeries, int], str] | None = None,
    **subplot_kwargs: Any,
):
    """Convert embeddings and MatrixSeries to grid of EmbedSubplot objects.

    Convenience wrapper around series_to_grid_configs that creates EmbedSubplot
    objects ready for plot_interactive_embed_grid.

    Args:
        embeddings: List of embedding arrays (one per radius or configuration)
        idxs: List of indices corresponding to the embeddings
        series: MatrixSeries or list of MatrixSeries for metadata
        hover_data: Optional hover tooltip data
        layout: Grid layout ("row", "column", "single_row", "single_column")
        title_fn: Optional title function
        under_title_fn: Optional subtitle function
        **subplot_kwargs: Additional EmbedSubplot parameters

    Returns
    -------
        2D list of EmbedSubplot objects for plot_interactive_embed_grid

    Example:
        >>> # Create embeddings for each radius
        >>> umap_r0 = umap_reducer.fit_transform(series[0].matrix)
        >>> umap_r1 = umap_reducer.fit_transform(series[1].matrix)
        >>> embeddings = [umap_r0, umap_r1]
        >>> idxs = list(range(len(population)))
        >>>
        >>> # Create grid
        >>> grid = embeddings_to_grid(
        ...     embeddings, idxs, series,
        ...     hover_data=[f"Robot {i}" for i in idxs],
        ...     default_dot_size=10
        ... )
        >>> plot_interactive_embed_grid(grid, thumbnails)
    """
    from canonical_toolkit.core.visual import EmbedSubplot

    # Get base configs
    configs = series_to_grid_configs(
        series,
        layout=layout,
        title_fn=title_fn,
        under_title_fn=under_title_fn,
        **subplot_kwargs,
    )

    # Convert to EmbedSubplot objects
    grid = []
    embed_idx = 0
    for row_configs in configs:
        row = []
        for config in row_configs:
            if config["matrix"] is not None:
                # Has data - use corresponding embedding
                subplot = EmbedSubplot(
                    title=config["title"],
                    embeddings=embeddings[embed_idx],
                    idxs=idxs,
                    hover_data=hover_data or [""] * len(idxs),
                    **{k: v for k, v in config.items()
                       if k not in ["title", "matrix", "space", "radius", "under_title"]},
                )
                embed_idx += 1
            else:
                # Empty placeholder
                subplot = EmbedSubplot(
                    title="",
                    embeddings=[],
                    idxs=[],
                    hover_data=[],
                )
            row.append(subplot)
        grid.append(row)

    return grid

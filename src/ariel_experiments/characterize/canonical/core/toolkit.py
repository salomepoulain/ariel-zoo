from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

from ariel_experiments.characterize.canonical.core.tools.deriver import (
    TreeDeriver,
)
from ariel_experiments.characterize.canonical.core.tools.evaluator import (
    Evaluator,
    SimilarityAggregator,
    TanimotoCalculator,
    WeightCalculator,
)
from ariel_experiments.characterize.canonical.core.tools.factory import (
    TreeFactory,
)
from ariel_experiments.characterize.canonical.core.tools.serializer import (
    TreeSerializer,
)
from ariel_experiments.characterize.canonical.core.utils.exceptions import (
    ChildNotFoundError,
    FaceNotFoundError,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from networkx import DiGraph

    from ariel_experiments.characterize.canonical.core.node import (
        CanonicalizableNode,
    )


# region config enums -----


class OutputType(Enum):
    """Output format for tree operations."""

    STRING = auto()
    GRAPH = auto()
    NODE = auto()


class RadiusStrategy(Enum):
    """Strategy for determining neighborhood radius in similarity calculations."""

    NODE_LOCAL = auto()
    TREE_GLOBAL = auto()


class TanimotoMode(Enum):
    """Which Tanimoto function to use."""

    SET = auto()
    COUNTS = auto()


class WeightingMode(Enum):
    """Which weighting strategy to apply to the radii."""

    LINEAR = auto()
    EXPONENTIAL = auto()
    SOFTMAX = auto()
    UNIFORM = auto()


class MissingDataMode(Enum):
    """How to handle radii not present in both neighborhoods."""

    SKIP_RADIUS = auto()
    TREAT_AS_ZERO = auto()


class AggregationMode(Enum):
    """How to combine the per-radius similarities into one score."""

    POWER_MEAN = auto()


@dataclass(frozen=True)
class SimilarityConfig:
    """Configuration for calculating neighborhood similarity."""

    radius_strategy: RadiusStrategy = RadiusStrategy.NODE_LOCAL
    tanimoto_mode: TanimotoMode = TanimotoMode.COUNTS
    weighting_mode: WeightingMode = WeightingMode.LINEAR
    missing_data_mode: MissingDataMode = MissingDataMode.SKIP_RADIUS
    aggregation_mode: AggregationMode = AggregationMode.POWER_MEAN

    max_tree_radius: int | None = None
    softmax_beta: float = 1.0
    power_mean_p: float = 1.0

# endregion

# region class toolkit -----

class _FreshInstanceDescriptor:
    """Descriptor that returns a fresh instance each time it's accessed."""

    def __init__(self, factory_func: Callable[[], Any]) -> None:
        self.factory_func = factory_func

    def __get__(self, obj: object | None, objtype: type | None = None) -> Any:
        return self.factory_func()

    def __set_name__(self, owner: type, name: str) -> None:
        self.name = name


class CanonicalToolKit:
    """Unified API for tree operations."""

    # Enum Config Settings
    OutputType = OutputType
    RadiusStrategy = RadiusStrategy
    TanimotoMode = TanimotoMode
    WeightingMode = WeightingMode
    MissingDataMode = MissingDataMode
    AggregationMode = AggregationMode

    # Factory methods
    create_root: Callable[..., CanonicalizableNode] = TreeFactory.create_root
    create_node: Callable[..., CanonicalizableNode] = TreeFactory.node
    create_brick: Callable[..., CanonicalizableNode] = TreeFactory.brick
    create_hinge: Callable[..., CanonicalizableNode] = TreeFactory.hinge

    from_graph: Callable[..., CanonicalizableNode] = TreeFactory.from_graph
    from_string: Callable[..., CanonicalizableNode] = TreeFactory.from_string

    # Serialization methods
    to_graph: Callable[..., DiGraph[Any]] = TreeSerializer.to_graph
    to_string: Callable[..., str] = TreeSerializer.to_string

    # Derivation methods
    canonicalize: Callable[..., CanonicalizableNode] = TreeDeriver.canonicalize

    # Custom properties
    root = _FreshInstanceDescriptor(TreeFactory.create_root)
    brick = _FreshInstanceDescriptor(TreeFactory.brick)
    hinge = _FreshInstanceDescriptor(TreeFactory.hinge)

    # suppress_face_errors()

    @staticmethod
    def suppress_face_errors() -> None:
        """Install custom exception handler to suppress tracebacks for FaceNotFoundError."""
        original_hook = sys.excepthook

        def custom_hook(
            exc_type: type[BaseException],
            exc_value: BaseException,
            exc_tb: TracebackType | None,
        ) -> None:
            if exc_type is ChildNotFoundError or exc_type is FaceNotFoundError:
                exc_value.print_rich()
            else:
                original_hook(exc_type, exc_value, exc_tb)

        sys.excepthook = custom_hook

    @classmethod
    def create_similarity_config(
        cls,
        *,
        radius_strategy: RadiusStrategy = RadiusStrategy.NODE_LOCAL,
        tanimoto_mode: TanimotoMode = TanimotoMode.COUNTS,
        weighting_mode: WeightingMode = WeightingMode.LINEAR,
        missing_data_mode: MissingDataMode = MissingDataMode.SKIP_RADIUS,
        aggregation_mode: AggregationMode = AggregationMode.POWER_MEAN,
        max_tree_radius: int = 10,
        softmax_beta: float = 1.0,
        power_mean_p: float = 1.0,
    ) -> SimilarityConfig:
        """Create a SimilarityConfig instance with specified parameters."""
        return SimilarityConfig(
            radius_strategy=radius_strategy,
            tanimoto_mode=tanimoto_mode,
            weighting_mode=weighting_mode,
            missing_data_mode=missing_data_mode,
            aggregation_mode=aggregation_mode,
            max_tree_radius=max_tree_radius,
            softmax_beta=softmax_beta,
            power_mean_p=power_mean_p,
        )

    @classmethod
    def collect_subtrees(
        cls,
        node: CanonicalizableNode,
        output_type: OutputType = OutputType.STRING,
    ) -> list[str | DiGraph[Any] | CanonicalizableNode]:
        match output_type:
            case OutputType.STRING:
                return TreeDeriver.collect_subtrees(node, cls.to_string)
            case OutputType.NODE:
                return TreeDeriver.collect_subtrees(node, cls.to_graph)
            case OutputType.GRAPH:
                return TreeDeriver.collect_subtrees(node)
            case _:
                msg = f"Unknown output_type: {output_type}"
                raise ValueError(msg)

    @classmethod
    def collect_neighbours(
        cls,
        node: CanonicalizableNode,
        *,
        use_node_max_radius: bool = False,
        tree_max_radius: int | None = None,
        output_type: OutputType = OutputType.STRING,
    ) -> dict[int, list[str | DiGraph[Any] | CanonicalizableNode]]:
        """Collect neighbourhoods around each node in the tree."""
        match output_type:
            case OutputType.STRING:
                return TreeDeriver.collect_neighbourhoods(
                    node,
                    cls.to_string,
                    use_node_max_radius=use_node_max_radius,
                    tree_max_radius=tree_max_radius,
                )
            case OutputType.NODE:
                return TreeDeriver.collect_neighbourhoods(
                    node,
                    cls.to_graph,
                    use_node_max_radius=use_node_max_radius,
                    tree_max_radius=tree_max_radius,
                )
            case OutputType.GRAPH:
                return TreeDeriver.collect_neighbourhoods(
                    node,
                    use_node_max_radius=use_node_max_radius,
                    tree_max_radius=tree_max_radius,
                )
            case _:
                msg = f"Unknown output_type: {output_type}"
                raise ValueError(msg)

    @classmethod
    def to_canonical_string(
        cls,
        node: CanonicalizableNode,
        *,
        zero_root_angle: bool = True,
        child_order: bool = True,
    ) -> str:
        canonical_tree = cls.canonicalize(
            node,
            zero_root_angle=zero_root_angle,
            child_order=child_order,
        )

        return cls.to_string(canonical_tree)

    @classmethod
    def to_canonical_graph(
        cls,
        node: CanonicalizableNode,
        *,
        zero_root_angle: bool = True,
        child_order: bool = True,
    ) -> DiGraph[Any]:
        canonical_tree = cls.canonicalize(
            node,
            zero_root_angle=zero_root_angle,
            child_order=child_order,
        )

        return cls.to_graph(canonical_tree)

    @staticmethod
    def _resolve_radius_params(
        config: SimilarityConfig,
    ) -> tuple[bool, int | None]:
        """Convert radius strategy enum to concrete parameters."""
        match config.radius_strategy:
            case RadiusStrategy.NODE_LOCAL:
                return True, None
            case RadiusStrategy.TREE_GLOBAL:
                return False, config.max_tree_radius
            case _:
                # Fallback to node local if unknown strategy
                return True, None

    @staticmethod
    def _resolve_tanimoto_function(
        config: SimilarityConfig,
    ) -> TanimotoCalculator:
        """Select Tanimoto similarity function based on config."""
        match config.tanimoto_mode:
            case TanimotoMode.SET:
                return Evaluator.tanimoto_strings_set
            case TanimotoMode.COUNTS:
                return Evaluator.tanimoto_strings_with_counts
            case _:
                msg = f"Unknown TanimotoMode: {config.tanimoto_mode}"
                raise ValueError(msg)

    @staticmethod
    def _resolve_weight_function(
        config: SimilarityConfig,
    ) -> WeightCalculator:
        """Select weight function based on config."""
        match config.weighting_mode:
            case WeightingMode.UNIFORM:
                return Evaluator.uniform_weights
            case WeightingMode.LINEAR:
                return Evaluator.linear_weights
            case WeightingMode.EXPONENTIAL:
                return Evaluator.exponential_weights
            case WeightingMode.SOFTMAX:
                return lambda radii: Evaluator.softmax_weights(
                    radii,
                    beta=config.softmax_beta,
                )
            case _:
                msg = f"Unknown WeightingMode: {config.weighting_mode}"
                raise ValueError(msg)

    @staticmethod
    def _resolve_aggregation_function(
        config: SimilarityConfig,
    ) -> SimilarityAggregator:
        """Select aggregation function based on config."""
        match config.aggregation_mode:
            case AggregationMode.POWER_MEAN:
                return lambda sims, weights: (
                    Evaluator.calc_tanimoto_power_mean(
                        sims,
                        weights,
                        p=config.power_mean_p,
                    )
                )
            case _:
                msg = f"Unknown AggregationMode: {config.aggregation_mode}"
                raise ValueError(msg)

    @classmethod
    def calculate_similarity(
        cls,
        node1: CanonicalizableNode,
        node2: CanonicalizableNode,
        config: SimilarityConfig | None = None,
        *,
        decimals: int = 3,
    ) -> float:
        """Calculate similarity between two nodes based on neighbourhoods."""
        if config is None:
            config = SimilarityConfig()

        # Resolve radius parameters from strategy
        use_node_max_radius, tree_max_radius = cls._resolve_radius_params(
            config,
        )

        # Collect neighbourhoods as strings
        nh1_dict = cls.collect_neighbours(
            node1,
            use_node_max_radius=use_node_max_radius,
            tree_max_radius=tree_max_radius,
            output_type=OutputType.STRING,
        )
        nh2_dict = cls.collect_neighbours(
            node2,
            use_node_max_radius=use_node_max_radius,
            tree_max_radius=tree_max_radius,
            output_type=OutputType.STRING,
        )

        # Resolve similarity calculation functions from config
        tanimoto_fn = cls._resolve_tanimoto_function(config)
        weight_fn = cls._resolve_weight_function(config)
        aggregation_fn = cls._resolve_aggregation_function(config)
        skip_missing = config.missing_data_mode == MissingDataMode.SKIP_RADIUS

        # Delegate to base similarity calculator
        value = Evaluator.similarity_calculator(
            nh1_dict,
            nh2_dict,
            tanimoto_fn=tanimoto_fn,
            weight_fn=weight_fn,
            aggregation_fn=aggregation_fn,
            skip_missing_radii=skip_missing,
        )

        return round(value, decimals)

    #TODO: make some of the similarity helper methods public, so these can also be easily used and just pass the config

# endregion

# Auto-install exception handler when toolkit is imported
CanonicalToolKit.suppress_face_errors()

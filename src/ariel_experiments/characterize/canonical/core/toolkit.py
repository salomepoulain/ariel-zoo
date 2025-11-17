from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum, auto
from functools import partial
from typing import TYPE_CHECKING, Any, cast

from ariel_experiments.characterize.canonical.core.node import (
    CanonicalizableNode,
)
from ariel_experiments.characterize.canonical.core.tools.deriver import (
    TreeDeriver,
)
from ariel_experiments.characterize.canonical.core.tools.evaluator import (
    Evaluator,
    RadiusData,
    Scorer,
    SimilarityAggregator,
    TreeHash,
    Vectorizer,
    WeightCalculator,
)
from ariel_experiments.characterize.canonical.core.tools.factory import (
    TreeFactory,
)
from ariel_experiments.characterize.canonical.core.tools.serializer import (
    TreeSerializer,
)
from ariel_experiments.characterize.canonical.core.tools.vector import (
    HashVector,
)
from ariel_experiments.characterize.canonical.core.utils.exceptions import (
    ChildNotFoundError,
    FaceNotFoundError,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import TracebackType

    from networkx import DiGraph


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


class ScoreStrategy(Enum):
    TFIDF = auto()
    TANIMOTO = auto()
    COSINE = auto()


class TFIDFMode(Enum):
    """How to aggregate TF-IDF vector to a single score."""

    ENTROPY = auto()
    SUM = auto()
    MEAN = auto()
    L1_NORM = auto()


class VectorMode(Enum):
    """Which Vector function to use."""

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


class CollectionStrategy(Enum):
    """Which collection method to use for gathering tree structures."""

    NEIGHBOURHOODS = auto()
    SUBTREES = auto()


# @dataclass(frozen=True)
@dataclass(slots=True)
class SimilarityConfig:
    """Configuration for calculating neighborhood similarity."""

    collection_strategy: CollectionStrategy = CollectionStrategy.NEIGHBOURHOODS

    radius_strategy: RadiusStrategy = RadiusStrategy.NODE_LOCAL
    score_strategy: ScoreStrategy = ScoreStrategy.TANIMOTO

    vector_mode: VectorMode = VectorMode.COUNTS
    tfidf_mode: TFIDFMode = TFIDFMode.ENTROPY

    weighting_mode: WeightingMode = WeightingMode.LINEAR
    missing_data_mode: MissingDataMode = MissingDataMode.TREAT_AS_ZERO
    aggregation_mode: AggregationMode = AggregationMode.POWER_MEAN

    max_tree_radius: int | None = None

    tfidf_smooth: bool = True
    entropy_normalised: bool = True

    softmax_beta: float = 1.0
    power_mean_p: float = 1.0


@dataclass(frozen=True, slots=True)
class SimilarityResults:
    similarity_value: float
    tree_hash_dicts: tuple[dict[int, TreeHash], dict[int, TreeHash] | None]
    per_radius_vectors: tuple[HashVector, None]
    per_radius_scores: dict[int, float]
    obtained_weights: list[float]


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

    Node = CanonicalizableNode

    # Enum Config Settings
    OutputType = OutputType

    CollectionStrategy = CollectionStrategy

    RadiusStrategy = RadiusStrategy
    ScoreStrategy = ScoreStrategy
    TanimotoMode = VectorMode
    VectorMode = VectorMode
    WeightingMode = WeightingMode
    MissingDataMode = MissingDataMode
    AggregationMode = AggregationMode
    TFIDFMode = TFIDFMode

    SimilarityConfig = SimilarityConfig
    SimilarityResults = SimilarityResults
    RadiusData = RadiusData

    # Factory methods
    create_root: Callable[..., CanonicalizableNode] = TreeFactory.create_root
    create_node: Callable[..., CanonicalizableNode] = TreeFactory.node
    create_brick: Callable[..., CanonicalizableNode] = TreeFactory.brick
    create_hinge: Callable[..., CanonicalizableNode] = TreeFactory.hinge

    from_graph: Callable[..., CanonicalizableNode] = TreeFactory.from_graph
    from_string: Callable[..., CanonicalizableNode] = TreeFactory.from_string
    from_nde_genotype: Callable[..., CanonicalizableNode] = (
        TreeFactory.from_nde_genotype
    )

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
    def create_similarity_config(
        collection_strategy: CollectionStrategy = CollectionStrategy.NEIGHBOURHOODS,
        radius_strategy: RadiusStrategy = RadiusStrategy.NODE_LOCAL,
        score_strategy: ScoreStrategy = ScoreStrategy.TANIMOTO,
        vector_mode: VectorMode = VectorMode.COUNTS,
        weighting_mode: WeightingMode = WeightingMode.LINEAR,
        missing_data_mode: MissingDataMode = MissingDataMode.TREAT_AS_ZERO,
        max_tree_radius: int | None = 3,
        softmax_beta: float = 1.0,
    ) -> SimilarityConfig:
        assert score_strategy != ScoreStrategy.TFIDF, (
            "similarity config can't have tfidf as score strategy"
        )
        return SimilarityConfig(
            collection_strategy=collection_strategy,
            radius_strategy=radius_strategy,
            score_strategy=score_strategy,
            vector_mode=vector_mode,
            weighting_mode=weighting_mode,
            missing_data_mode=missing_data_mode,
            max_tree_radius=max_tree_radius,
            softmax_beta=softmax_beta,
        )

    @staticmethod
    def create_tfidf_config(
        collection_strategy: CollectionStrategy = CollectionStrategy.NEIGHBOURHOODS,
        radius_strategy: RadiusStrategy = RadiusStrategy.NODE_LOCAL,
        score_strategy: ScoreStrategy = ScoreStrategy.TFIDF,
        tfidf_mode: TFIDFMode = TFIDFMode.ENTROPY,
        vector_mode: VectorMode = VectorMode.SET,
        weighting_mode: WeightingMode = WeightingMode.LINEAR,
        missing_data_mode: MissingDataMode = MissingDataMode.TREAT_AS_ZERO,
        tfidf_smooth: bool = True,
        entropy_normalised: bool = False,
        max_tree_radius: int | None = 3,
        softmax_beta: float = 1.0,
    ) -> SimilarityConfig:
        assert score_strategy == ScoreStrategy.TFIDF, (
            "tfidf config must have tfidf as score strategy"
        )
        return SimilarityConfig(
            collection_strategy=collection_strategy,
            radius_strategy=radius_strategy,
            score_strategy=score_strategy,
            vector_mode=vector_mode,
            weighting_mode=weighting_mode,
            missing_data_mode=missing_data_mode,
            max_tree_radius=max_tree_radius,
            softmax_beta=softmax_beta,
            tfidf_mode=tfidf_mode,
            tfidf_smooth=tfidf_smooth,
            entropy_normalised=entropy_normalised,
        )

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
    def collect_subtrees(
        cls,
        node: CanonicalizableNode,
        output_type: OutputType = OutputType.STRING,
    ) -> dict[int, list[str | DiGraph[Any] | CanonicalizableNode]]:
        match output_type:
            case OutputType.STRING:
                return TreeDeriver.collect_subtrees(node, cls.to_string)
            case OutputType.NODE:
                return TreeDeriver.collect_subtrees(node, cls.to_graph)
            case OutputType.GRAPH:
                return TreeDeriver.collect_subtrees(node)

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

    @classmethod
    def collect_tree_hash_config_mode(
        cls,
        node: CanonicalizableNode,
        *,
        config: SimilarityConfig,
        output_type: OutputType = OutputType.STRING,
    ) -> dict[int, list[TreeHash]]:
        """Collect neighbourhoods or subtrees based on config strategy."""
        if config.collection_strategy == CollectionStrategy.SUBTREES:
            return cast(
                "dict[int, list[TreeHash]]",
                cls.collect_subtrees(node, output_type=output_type),
            )

        use_node_max_radius, tree_max_radius = cls._resolve_radius_params(
            config,
        )
        return cast(
            "dict[int, list[TreeHash]]",
            cls.collect_neighbours(
                node,
                use_node_max_radius=use_node_max_radius,
                tree_max_radius=tree_max_radius,
                output_type=output_type,
            ),
        )

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

    # region Similarity Calculators -----

    @classmethod
    def calculate_similarity_from_dicts(
        cls,
        hash_dict1: dict[int, list[TreeHash]],
        hash_dict2: dict[int, list[TreeHash]] | dict[int, RadiusData],
        config: SimilarityConfig,
        *,
        decimals: int = 3,
        return_all: bool = False,
    ) -> float | SimilarityResults:
        """
        Primarily used for calculating similarity for treehash (bounded between 0 and 1)
        if used for tfidf, value is not bounded.
        """
        is_tfidf = (
            isinstance(next(iter(hash_dict2.values())), RadiusData)
            if hash_dict2
            else False
        )

        per_radius_scores = {}
        per_radius_vectors = {}
        all_radii = hash_dict1.keys() | hash_dict2.keys()
        vectorizer = cls._resolve_vectorizer(config)

        for radius in sorted(all_radii):
            item1 = hash_dict1.get(radius)
            item2 = hash_dict2.get(radius, None)

            if item1 is None or item2 is None:
                if config.missing_data_mode == MissingDataMode.SKIP_RADIUS:
                    continue
                per_radius_scores[radius] = 0
                continue

            vec1 = vectorizer(item1)
            vec2 = item2.df_counts if is_tfidf else vectorizer(item2)

            score_fn = cls._resolve_score_fn(
                config, N=hash_dict2[radius].N if is_tfidf else None,
            )
            per_radius_scores[radius] = score_fn(vec1, vec2)
            per_radius_vectors[radius] = (vec1, None if is_tfidf else vec2)

        radii = list(per_radius_scores.keys())
        similarities = list(per_radius_scores.values())

        weights = cls._resolve_weight_function(config)(radii)
        final_score = cls._resolve_aggregation_function(config)(
            similarities, weights,
        )

        results = SimilarityResults(
            similarity_value=final_score,
            tree_hash_dicts=(hash_dict1, None if is_tfidf else hash_dict2),
            per_radius_vectors=per_radius_vectors,
            per_radius_scores=per_radius_scores,
            obtained_weights=weights,
        )

        if return_all:
            return results

        return round(results.similarity_value, decimals)

    @classmethod
    def calculate_similarity(
        cls,
        node1: CanonicalizableNode,
        node2: CanonicalizableNode,
        config: SimilarityConfig | None = None,
        *,
        decimals: int = 3,
        return_all: bool = False,
    ) -> float | SimilarityResults:
        """Calculate similarity between two nodes based on subtrees."""
        if config is None:
            config = SimilarityConfig()

        node1 = node1.copy()
        node2 = node2.copy()

        # Collect neighbourhoods or subtrees as strings based on config strategy
        hash_dict1 = cls.collect_tree_hash_config_mode(
            node1,
            config=config,
            output_type=OutputType.STRING,
        )
        hash_dict2 = cls.collect_tree_hash_config_mode(
            node2,
            config=config,
            output_type=OutputType.STRING,
        )

        # Delegate to calculate_similarity_from_dicts for the actual calculation
        return cls.calculate_similarity_from_dicts(
            hash_dict1,
            hash_dict2,
            config,
            decimals=decimals,
            return_all=return_all,
        )

    @classmethod
    def calculate_tfidf(
        cls,
        hash_dict: dict[int, list[TreeHash]],
        population_dict: dict[int, RadiusData],
        config: SimilarityConfig,
        *,
        decimals: int = 3,
        return_all: bool = False,
    ) -> float | SimilarityResults:
        """
        Calculate TFIDF-based similarity between a tree and population.

        This is a convenience wrapper around calculate_similarity_from_dicts
        with more semantic parameter names for the TFIDF use case.
        Assumes that the individual is already in the population dictionary.
        """
        return cls.calculate_similarity_from_dicts(
            hash_dict,
            population_dict,
            config,
            decimals=decimals,
            return_all=return_all,
        )

    # endregion

    # region config resolvers

    @staticmethod
    def _resolve_radius_params(
        config: SimilarityConfig,
    ) -> tuple[bool, int | None]:
        use_node_local = config.radius_strategy == RadiusStrategy.NODE_LOCAL
        tree_max = (
            config.max_tree_radius
            if config.max_tree_radius is not None
            else None
        )
        return use_node_local, tree_max

    @staticmethod
    def _resolve_vectorizer(config: SimilarityConfig) -> Vectorizer:
        """Select vectorizer function based on config."""
        if config.vector_mode == VectorMode.SET:
            return Evaluator.compute_binary_vector

        return Evaluator.compute_count_vector

    @staticmethod
    def _resolve_score_fn(
        config: SimilarityConfig,
        N: int | None = None,
    ) -> Scorer:
        """Select Tanimoto similarity function based on config."""
        if config.score_strategy == ScoreStrategy.TFIDF:
            match config.tfidf_mode:
                case TFIDFMode.ENTROPY:
                    return partial(
                        Evaluator.score_tfidf_entropy,
                        normalized=config.entropy_normalised,
                        smooth=config.tfidf_smooth,
                        N=N,
                    )
                case TFIDFMode.MEAN:
                    return partial(
                        Evaluator.score_tfidf_mean,
                        smooth=config.tfidf_smooth,
                        N=N,
                    )
                case TFIDFMode.SUM:
                    return partial(
                        Evaluator.score_tfidf_sum,
                        smooth=config.tfidf_smooth,
                        N=N,
                    )
                case TFIDFMode.L1_NORM:
                    return partial(
                        Evaluator.score_tfidf_l1,
                        smooth=config.tfidf_smooth,
                        N=N,
                    )

        if config.score_strategy == ScoreStrategy.TANIMOTO:
            return Evaluator.score_tanimoto_similarity

        return Evaluator.score_cosine_similarity

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

    @staticmethod
    def _resolve_aggregation_function(
        config: SimilarityConfig,
    ) -> SimilarityAggregator:
        """Select aggregation function based on config."""
        match config.aggregation_mode:
            case AggregationMode.POWER_MEAN:
                return lambda sims, weights: (
                    Evaluator.power_mean_aggregate(
                        sims,
                        weights,
                        p=config.power_mean_p,
                    )
                )

    # endregion

    @staticmethod
    def update_tfidf_dictionary(
        population_trees: list[dict[int, list[TreeHash]]],
        dictionary: dict[int, RadiusData],
    ) -> None:
        """Update TF-IDF dictionary with new population trees."""
        for tree_hashes in population_trees:
            for radius, hash_list in tree_hashes.items():
                if radius not in dictionary:
                    dictionary[radius] = RadiusData(df_counts=HashVector(), N=0)

                dictionary[radius].N += 1
                vector = Evaluator.compute_binary_vector(hash_list)

                dictionary[radius].df_counts += vector


# Auto-install exception handler when toolkit is imported
CanonicalToolKit.suppress_face_errors()

if __name__ == "__main__":
    from rich.console import Console

    from ariel_experiments.characterize.canonical.core.toolkit import (
        CanonicalToolKit as ctk,
    )
    from ariel_experiments.utils.initialize import generate_random_individual

    console = Console()

    population = [
        ctk.from_graph(generate_random_individual()) for _ in range(1000)
    ]

    config = ctk.SimilarityConfig()
    config.radius_strategy = ctk.RadiusStrategy.TREE_GLOBAL
    config.max_tree_radius = 3
    config.score_strategy = ctk.ScoreStrategy.TFIDF
    config.tfidf_mode = ctk.TFIDFMode.ENTROPY
    config.weighting_mode = ctk.WeightingMode.LINEAR
    config.vector_mode = ctk.VectorMode.COUNTS
    config.missing_data_mode = ctk.MissingDataMode.SKIP_RADIUS
    config.tfidf_smooth = True
    config.entropy_normalised = False
    console.print(config)

    subtrees = [
        ctk.collect_tree_hash_config_mode(individual, config=config)
        for individual in population
    ]

    console.print(subtrees[:2])

    pop_dict = {}
    ctk.update_tfidf_dictionary(subtrees, pop_dict)

    console.print(pop_dict)

    tfidf_similarity = ctk.calculate_similarity_from_dicts(
        subtrees[0], pop_dict, config=config, return_all=True,
    )

    console.print(tfidf_similarity)

    for subtree in subtrees:
        console.print(
            ctk.calculate_similarity_from_dicts(
                subtree, pop_dict, config=config,
            ),
        )

    # first calculate the subtrees for every thing in the population and store in list
    # update the tfidf dictionary
    # calculate the scores per individual

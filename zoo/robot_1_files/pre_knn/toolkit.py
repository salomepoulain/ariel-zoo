from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, cast
from scipy.sparse import csr_matrix # typing
import numpy as np

from ariel_experiments.characterize.canonical_toolkit.core.node.node import (
    Node,
)
from ariel_experiments.characterize.canonical_toolkit.core.tools.deriver import (
    TreeDeriver,
)
from ariel_experiments.characterize.canonical_toolkit.tests.old.evaluator import (
    Evaluator,
    NeighbourhoodData,
    TreeHash,
    WeightCalculator,
    MatrixDict
)
from ariel_experiments.characterize.canonical_toolkit.core.tools.factory import (
    TreeFactory,
)
from ariel_experiments.characterize.canonical_toolkit.core.tools.serializer import (
    TreeSerializer,
)

# from ariel_experiments.characterize.canonical.core.tools.vector import (
#     HashVector,
# )
from ariel_experiments.characterize.canonical_toolkit.core.utils.exceptions import (
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


class CollectionStrategy(Enum):
    """Which collection method to use for gathering tree structures."""

    NEIGHBOURHOODS = auto()
    SUBTREES = auto()


class WeightingMode(Enum):
    """Which weighting strategy to apply to the radii."""

    LINEAR = auto()
    EXPONENTIAL = auto()
    SOFTMAX = auto()
    UNIFORM = auto()


# @dataclass(frozen=True)
@dataclass(slots=True)
class SimilarityConfig:
    """Configuration for calculating neighborhood similarity."""

    collection_strategy: CollectionStrategy = CollectionStrategy.NEIGHBOURHOODS
    radius_strategy: RadiusStrategy = RadiusStrategy.NODE_LOCAL

    # weighting_mode: WeightingMode = WeightingMode.UNIFORM

    # TODO min_tree_radius: int | None = None
    max_tree_radius: int | None = None

    n_features: int = 2**20
    is_binary: bool = False


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

    Node = Node

    # Enum Config Settings
    OutputType = OutputType
    CollectionStrategy = CollectionStrategy
    RadiusStrategy = RadiusStrategy
    WeightingMode = WeightingMode

    SimilarityConfig = SimilarityConfig

    # Factory methods
    create_root: Callable[..., Node] = TreeFactory.create_root
    create_node: Callable[..., Node] = TreeFactory.node
    create_brick: Callable[..., Node] = TreeFactory.brick
    create_hinge: Callable[..., Node] = TreeFactory.hinge

    from_graph: Callable[..., Node] = TreeFactory.from_graph
    from_string: Callable[..., Node] = TreeFactory.from_string
    from_nde_genotype: Callable[..., Node] = (
        TreeFactory.from_nde_genotype
    )

    # Serialization methods
    to_graph: Callable[..., DiGraph[Any]] = TreeSerializer.to_graph
    to_string: Callable[..., str] = TreeSerializer.to_string

    # Derivation methods
    canonicalize: Callable[..., Node] = TreeDeriver.canonicalize

    # Evaluator methods
    matrix_dict_applier = Evaluator.matrix_dict_applier

    # Custom properties
    root = _FreshInstanceDescriptor(TreeFactory.create_root)
    brick = _FreshInstanceDescriptor(TreeFactory.brick)
    hinge = _FreshInstanceDescriptor(TreeFactory.hinge)

    # suppress_face_errors()
    @staticmethod
    def create_similarity_config(
        collection_strategy: CollectionStrategy = CollectionStrategy.NEIGHBOURHOODS,
        radius_strategy: RadiusStrategy = RadiusStrategy.NODE_LOCAL,
        weighting_mode: WeightingMode = WeightingMode.LINEAR,
        max_tree_radius: int | None = 3,
        is_binary: bool = False,
        n_features: int = 2**20,
    ) -> SimilarityConfig:
        return SimilarityConfig(
            collection_strategy=collection_strategy,
            radius_strategy=radius_strategy,
            weighting_mode=weighting_mode,
            max_tree_radius=max_tree_radius,
            is_binary=is_binary,
            n_features=n_features,
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
        node: Node,
        output_type: OutputType = OutputType.STRING,
    ) -> dict[int, list[str | DiGraph[Any] | Node]]:
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
        node: Node,
        *,
        use_node_max_radius: bool = True,
        tree_max_radius: int | None = None,
        output_type: OutputType = OutputType.STRING,
        do_radius_prefix: bool = True,
        hash_prefix: str | None = None
    ) -> dict[int, list[str | DiGraph[Any] | Node]]:
        """Collect neighbourhoods around each node in the tree."""
        match output_type:
            case OutputType.STRING:
                return TreeDeriver.collect_neighbourhoods(
                    node,
                    cls.to_string,
                    use_node_max_radius=use_node_max_radius,
                    tree_max_radius=tree_max_radius,
                    do_radius_prefix=do_radius_prefix,
                    hash_prefix=hash_prefix
                )
            case OutputType.NODE:
                return TreeDeriver.collect_neighbourhoods(
                    node,
                    cls.to_graph,
                    serializer_fn=None,
                    use_node_max_radius=use_node_max_radius,
                    do_radius_prefix=do_radius_prefix,
                    tree_max_radius=tree_max_radius,
                )
            case OutputType.GRAPH:
                return TreeDeriver.collect_neighbourhoods(
                    node,
                    serializer_fn=TreeSerializer.to_graph,
                    use_node_max_radius=use_node_max_radius,
                    do_radius_prefix=do_radius_prefix,
                    tree_max_radius=tree_max_radius,
                )

    @classmethod
    def collect_tree_hash_config_mode(
        cls,
        node: Node,
        *,
        config: SimilarityConfig,
        output_type: OutputType = OutputType.STRING,
        hash_prefix: str | None = None
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
                hash_prefix=hash_prefix
            ),
        )



    # region Similarity Calculators -----

    # # TODO what to add in here
    # # TODO: parrellize this? make it also accept ndes? individuals?
    # @classmethod
    # def get_count_matrix_from_nodes(
    #     cls,
    # ) -> None:
    #     pass

    # @classmethod
    # def get_count_matrix(
    #     cls,
    #     data_list: list[NeighbourhoodData],
    #     config: SimilarityConfig,
    #     *,
    #     give_vocab: bool = False,
    #     aggregated: bool = False,
    # ) -> MatrixDict | csr_matrix | tuple[MatrixDict, np.ndarray]:
    #     """
    #     Converts a list of neighbourhood/subtree dictionaries into a Sparse CSR Matrix.
    #     This acts as the bridge between your High-Level Config and the Low-Level Evaluator.
    #     """
    #     # weight_func = cls._resolve_weight_function(config)

    #     match (give_vocab, aggregated):
    #         # Case 1: (True, True) -> Vocab-based AND Aggregated (sum_vocab_vectors)
    #         case (True, True):
    #             return Evaluator.sum_vocab_vectors(
    #                 data_list=data_list,
    #                 is_binary=config.is_binary,
    #                 max_features=config.n_features,
    #                 # return_vectorizer=True
    #             )
    #         # Case 2: (True, False) -> Vocab-based AND Per-Radius (vectorize_with_vocab)
    #         case (True, False):
    #             # print("Dispatch: CountVectorizer (Vocab) - Per-Radius")
    #             return cast(MatrixDict| tuple[MatrixDict, np.ndarray], Evaluator.vectorize_with_vocab(
    #                 data_list=data_list,
    #                 is_binary=config.is_binary,
    #                 max_features=config.n_features,
    #             ))

    #         # Case 3: (False, True) -> Hashing AND Aggregated (sum_hash_vectors)
    #         case (False, True):
    #             # print("Dispatch: HashingVectorizer - Aggregate (Sum)")
    #             return Evaluator.sum_hash_vectors(
    #                 data_list=data_list,
    #                 is_binary=config.is_binary,
    #                 n_features=config.n_features,
    #             )
    #         # Case 4: (False, False) -> Hashing AND Per-Radius (vectorize_with_hashing)
    #         case (False, False):
    #             # print("Dispatch: HashingVectorizer - Per-Radius")
    #             return Evaluator.vectorize_with_hashing(
    #                 data_list=data_list,
    #                 is_binary=config.is_binary,
    #                 n_features=config.n_features,
    #             )

        # if give_vocab:
        #     if aggregated:
        #         return Evaluator.sum_vocab_vectors(
        #             data_list=data_list,
        #             is_binary=config.is_binary,
        #             min_df=config.min_df,
        #             max_features=config.max_features,
        #             return_vectorizer=True
        #         )
        #     return Evaluator.vectorize_with_vocab(
        #         data_list=data_list,
        #         # weight_calculator=weight_func,
        #         min_df=config.min_df,
        #         max_features=config.max_features,
        #         is_binary=config.is_binary,
        #     )


        # if aggregated:
        #     return Evaluator.sum_hash_vectors(
        #         data_list=data_list,
        #         is_binary=config.is_binary,
        #         n_features=config.n_features,
        #     )

        # return Evaluator.vectorize_with_hashing(
        #     data_list=data_list,
        #     # weight_calculator=weight_func,
        #     is_binary=config.is_binary,
        #     n_features=config.n_features,
        # )

    # endregion

    # region config resolvers

    # @staticmethod
    # def _resolve_radius_params(
    #     config: SimilarityConfig,
    # ) -> tuple[bool, int | None]:
    #     use_node_local = config.radius_strategy == RadiusStrategy.NODE_LOCAL
    #     tree_max = (
    #         config.max_tree_radius
    #         if config.max_tree_radius is not None
    #         else None
    #     )
    #     return use_node_local, tree_max

    # @staticmethod
    # def _resolve_weight_function(
    #     config: SimilarityConfig,
    # ) -> WeightCalculator:
    #     """Select weight function based on config."""
    #     match config.weighting_mode:
    #         case WeightingMode.UNIFORM:
    #             return Evaluator.uniform_weight
    #         case WeightingMode.LINEAR:
    #             return Evaluator.linear_weight
    #         case WeightingMode.EXPONENTIAL:
    #             return Evaluator.exponential_weight
    #         case WeightingMode.INVERSE:
    #             return Evaluator.inverse_weight
    #         case _:
    #             return Evaluator.uniform_weight


# Auto-install exception handler when toolkit is imported
CanonicalToolKit.suppress_face_errors()

if __name__ == "__main__":
    from rich.console import Console

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfTransformer

    from ariel_experiments.characterize.canonical_toolkit.tests.old.toolkit import (
        CanonicalToolKit as ctk,
    )
    from ariel_experiments.utils.initialize import generate_random_individual

    console = Console()


    population = [
        ctk.from_graph(generate_random_individual()) for _ in range(1000)
    ]


    config = ctk.SimilarityConfig()
    subtrees = [
        ctk.collect_tree_hash_config_mode(individual, config=config)
        for individual in population
    ]

    print(subtrees[:2])

    csr = ctk.get_count_matrix(subtrees, config)

    similarity_matrix = cosine_similarity()


    tfidf = TfidfTransformer(norm='l2')
    X_tfidf = tfidf.fit_transform(csr)

    similarity_matrix = cosine_similarity(X_tfidf)

    print(similarity_matrix.shape)

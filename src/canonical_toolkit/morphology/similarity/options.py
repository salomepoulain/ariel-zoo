"""Shared types, enums, and protocols for matrix analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any

from sklearn.feature_extraction import FeatureHasher

from ..node.tools import (
    serializer,
)
from .sim_types import FeatureHasherProtocol

__all__ = [
    "MatrixDomain",
    "OutputType",
    "RadiusStrategy",
    "SimilaritySpaceConfig",
    "Space",
    "UmapConfig",
]


class MatrixDomain(Enum):
    """Defines the topology and mathematical meaning of the matrix."""

    FEATURES = auto()  # N x M: Raw Counts or TFIDF (Must be Sparse)
    SIMILARITY = auto()  # N x N: Pairwise relationships (Must be Dense)
    EMBEDDING = auto()  # N x D: Reduced dimensions (Must be Dense)


class OutputType(Enum):
    STRING = serializer.to_string
    GRAPH = serializer.to_graph
    NODE = None


class RadiusStrategy(Enum):
    """Strategy for determining neighborhood radius in similarity calculations."""

    NODE_LOCAL = True
    TREE_GLOBAL = False


class Space(Enum):
    """
    The immutable physical reality of the robot.
    Used for ground-truth keys. Derived/Experimental keys should use strings.

    Note: 'R' stands for Radial. 'A' stands for Axial
    """

    WHOLE = ""
    FRONT = "R_f__"
    BACK = "R_b__"
    LEFT = "R_l__"
    RIGHT = "R_r__"
    TOP = "A_t__"
    BOTTOM = "A_b__"

    RADIAL = "R_#__"
    AXIAL = "A_#__"
    # AGGREGATED = auto()

    # @classmethod
    # # TODO where used????
    # def limb_spaces_only(cls) -> list[Space]:
    #     return [cls.FRONT_LIMB, cls.LEFT_LIMB, cls.BACK_LIMB, cls.RIGHT_LIMB]


@dataclass
class UmapConfig:
    n_neighbors: int = 15
    n_components: int = 2
    metric: str = "cosine"  # "precomputed"
    random_state: int | None = 42
    init = ("random",)
    transform_seed: int | None = 42
    n_jobs: int = 1

    extra_params: dict[str, Any] = field(default_factory=dict)

    def get_kwargs(self) -> dict[str, Any]:
        """
        Flattens the dataclass into a single dictionary
        ready to be passed into UMAP(**config.get_kwargs()).
        """
        params = asdict(self)
        extras = params.pop("extra_params")
        return {**params, **extras}


@dataclass(slots=True)
class SimilaritySpaceConfig:
    """Configuration for calculating neighborhood similarity space."""

    space: Space = Space.WHOLE

    # collection
    radius_strategy: RadiusStrategy = RadiusStrategy.NODE_LOCAL
    min_hop_radius: int = 0  # TODO
    max_hop_radius: int | None = None

    # processing
    n_features: int = 2**24
    hasher: FeatureHasherProtocol | None = None

    def __post_init__(self):
        if self.hasher is None:
            self.hasher = FeatureHasher(
                n_features=self.n_features,
                input_type="string",
            )

        # TODO in here, test that 0 <= min radius <= max radius

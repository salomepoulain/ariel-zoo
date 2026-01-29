"""Shared types, enums, and protocols for matrix analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, TYPE_CHECKING

from polars import Object
from sklearn.feature_extraction import FeatureHasher

from ..node.tools import (
    serializer,
)
from .sim_types import FeatureHasherProtocol

if TYPE_CHECKING:
    from umap import UMAP # type: ignore (stubs missing)

__all__ = [
    "MatrixDomain",
    "OutputType",
    "RadiusStrategy",
    "SimilaritySpaceConfig",
    "Space",
    "UmapConfig",
]


class MatrixDomain(str, Enum):
    """Defines the topology and mathematical meaning of the matrix."""

    FEATURES = "FEATURES"       # N x M: Raw Counts or TFIDF (likely to be Sparse)
    SIMILARITY = "SIMILARITY"   # N x N: Pairwise relationships (Must be Dense)
    EMBEDDING = "EMBEDDING"     # N x D: Reduced dimensions (Must be Dense?)


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

    # TODO: actually try to make it more efficient? idk

    WHOLE = ""
    FRONT = "R_f__"
    BACK = "R_b__"
    LEFT = "R_l__"
    RIGHT = "R_r__"
    TOP = "A_t__"
    BOTTOM = "A_b__"

    RADIAL = "R_#__"
    AXIAL = "A_#__"

    @classmethod
    def limb_spaces_only(cls) -> list[Space]:
        return [cls.LEFT, cls.FRONT, cls.RIGHT, cls.BACK, cls.TOP, cls.BOTTOM]

    @classmethod
    def all_spaces(cls) -> list[Space]:
        return [cls.WHOLE, cls.RADIAL, cls.AXIAL] + cls.limb_spaces_only()


@dataclass
class UmapConfig:
    n_neighbors: int = 15
    n_components: int = 2
    min_dist: float = 0.0
    metric: str = "cosine"  # "precomputed"
    """use 'precomputed' if already cosine was applied"""
    random_state: int | None = None #42
    init: str = "random"
    transform_seed: int | None = None #42
    n_jobs: int = -1

    extra_params: dict[str, Any] = field(default_factory=dict)

    def get_kwargs(self) -> dict[str, Any]:
        """
        Flattens the dataclass into a single dictionary
        ready to be passed into UMAP(**config.get_kwargs()).
        """
        params = asdict(self)
        extras = params.pop("extra_params")
        return {**params, **extras}


    def get_umap(self, **umap_kwargs: Any) -> UMAP:
        try:
            from umap import UMAP  # type: ignore (stub files missing)
        except ImportError as e:
            msg = "pip install umap-learn to use this method."
            raise ImportError(
                msg,
            ) from e

        kwargs = self.get_kwargs() | umap_kwargs

        print(kwargs)
        return UMAP(**kwargs)


@dataclass(slots=True)
class SimilaritySpaceConfig:
    """Configuration for calculating neighborhood similarity space."""

    space: Space = Space.WHOLE

    # collection
    radius_strategy: RadiusStrategy = RadiusStrategy.TREE_GLOBAL
    min_hop_radius: int = 0  # TODO
    max_hop_radius: int | None = 3

    skip_empty: bool = False

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

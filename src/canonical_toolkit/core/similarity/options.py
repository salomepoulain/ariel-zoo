"""Shared types, enums, and protocols for matrix analysis."""

from __future__ import annotations

from enum import Enum, auto


class VectorSpace(Enum):
    """
    The immutable physical reality of the robot.
    Used for ground-truth keys. Derived/Experimental keys should use strings.
    """

    DEFAULT = ""
    FRONT = "R_f__"
    BACK = "R_b__"
    LEFT = "R_l__"
    RIGHT = "R_r__"
    TOP = "A_t__"
    BOTTOM = "A_b__"
    # AGGREGATED = auto()

    @classmethod
    # TODO where used????
    def limb_spaces_only(cls) -> list[VectorSpace]:
        return [cls.FRONT_LIMB, cls.LEFT_LIMB, cls.BACK_LIMB, cls.RIGHT_LIMB]


class HVectorSpace(Enum):
    """hash vectorspace."""

    DEFAULT = ""
    FRONT = "R_f__"
    BACK = "R_b__"
    LEFT = "R_l__"
    RIGHT = "R_r__"
    TOP = "A_t__"
    BOTTOM = "A_b__"


class MatrixDomain(Enum):
    """Defines the topology and mathematical meaning of the matrix."""

    FEATURES = auto()  # N x M: Raw Counts or TFIDF (Must be Sparse)
    SIMILARITY = auto()  # N x N: Pairwise relationships (Must be Dense)
    EMBEDDING = auto()  # N x D: Reduced dimensions (Must be Dense)

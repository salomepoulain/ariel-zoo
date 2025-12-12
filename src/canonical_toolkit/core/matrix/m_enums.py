"""Shared types, enums, and protocols for matrix analysis."""

from __future__ import annotations

from enum import Enum, auto


class VectorSpace(Enum):
    """
    The immutable physical reality of the robot.
    Used for ground-truth keys. Derived/Experimental keys should use strings.
    """

    ENTIRE_ROBOT = "FULL"
    FRONT_LIMB = "FRONT"
    LEFT_LIMB = "LEFT"
    BACK_LIMB = "BACK"
    RIGHT_LIMB = "RIGHT"
    AGGREGATED = "AGGREGATED"

    @classmethod
    def limb_spaces_only(cls) -> list[VectorSpace]:
        return [cls.FRONT_LIMB, cls.LEFT_LIMB, cls.BACK_LIMB, cls.RIGHT_LIMB]


class MatrixDomain(Enum):
    """Defines the topology and mathematical meaning of the matrix."""

    FEATURES = auto()  # N x M: Raw Counts or TFIDF (Must be Sparse)
    SIMILARITY = auto()  # N x N: Pairwise relationships (Must be Dense)
    EMBEDDING = auto()  # N x D: Reduced dimensions (Must be Dense)

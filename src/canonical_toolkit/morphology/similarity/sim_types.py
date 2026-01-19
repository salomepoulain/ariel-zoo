"""Shared types, enums, and protocols for Matrix and SImilarity Analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from scipy.sparse._matrix import spmatrix

    from ...base.matrix import MatrixInstance

from typing import TYPE_CHECKING

TreeHash = str
"""
TreeHash is a formatted string representing a canonicalized subtree modular part of the robot

"""

HashFingerprint = dict[int, list[TreeHash]]
"""
HashFingerprint is a dictionary with radius : list[TreeHash]
TreeHash is a formatted string representing a canonicalized subtree modular part of the robot
"""

PopulationFingerprint = list[HashFingerprint]
"""
Index of list is HashFingerprint from an individual
HashFingerprint is a dictionary with radius : list[TreeHash]
TreeHash is a formatted string representing a canonicalized subtree modular part of the robot
"""


"""
FingerprintSpace == the vectorized PopulationFingerprint
"""


@runtime_checkable
class FeatureHasherProtocol(Protocol):
    """Protocol for hashers that convert string features to sparse matrices.

    Compatible with sklearn.feature_extraction.FeatureHasher when
    configured with input_type='string'.
    """

    def transform(self, raw_X) -> spmatrix:
        """Transform iterable of feature lists to sparse matrix.

        Args:
            raw_X: Iterable of feature lists (list, iterator, etc.)
                   Each element contains string features for one sample.
                   Example: [['r0__B', 'r0__H'], ['r1__BB']]

        Returns
        -------
            Sparse matrix of shape (n_samples, n_features)
        """
        ...


class InstanceAggregator(Protocol):
    """Callable that aggregates multiple MatrixInstance objects into one."""

    def __call__(
        self,
        instances: list[MatrixInstance],
        **kwargs,
    ) -> MatrixInstance: ...

"""Shared types, enums, and protocols for matrix analysis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import scipy.sparse as sp

if TYPE_CHECKING:
    from canonical_toolkit.core.matrix.matrix import MatrixInstance

# Anchor data path relative to this file location
_CURRENT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = _CURRENT_DIR / "__data__" / "npz"

RadiusKey = int  # Radius is strictly an integer depth


class FeatureHasherProtocol(Protocol):
    """Protocol for hashers that convert string features to sparse matrices.

    Compatible with sklearn.feature_extraction.FeatureHasher when
    configured with input_type='string'.
    """

    def transform(self, raw_X) -> sp.spmatrix:
        """Transform iterable of feature lists to sparse matrix.

        Args:
            raw_X: Iterable of feature lists (list, iterator, etc.)
                   Each element contains string features for one sample.
                   Example: [['r0__B', 'r0__H'], ['r1__BB']]

        Returns:
            Sparse matrix of shape (n_samples, n_features)
        """
        ...


class InstanceAggregator(Protocol):
    """Callable that aggregates multiple MatrixInstance objects into one."""

    def __call__(
        self, instances: list[MatrixInstance], **kwargs
    ) -> MatrixInstance: ...

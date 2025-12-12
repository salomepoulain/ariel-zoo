"""
Matrix package - Analysis and similarity computation.

This package provides classes for analyzing tree structures and computing
similarity metrics across populations.
"""

# ===== Analysis Classes =====
from canonical_toolkit.core.matrix.matrix import MatrixInstance
from canonical_toolkit.core.matrix.m_series import MatrixSeries
from canonical_toolkit.core.matrix.m_frame import MatrixFrame

# ===== Enums =====
from canonical_toolkit.core.matrix.m_enums import (
    MatrixDomain,
    VectorSpace,
)

# ===== Define exports =====
__all__ = [
    # Analysis Classes
    "MatrixInstance",
    "MatrixSeries",
    "MatrixFrame",

    # Enums
    "MatrixDomain",
    "VectorSpace",
]

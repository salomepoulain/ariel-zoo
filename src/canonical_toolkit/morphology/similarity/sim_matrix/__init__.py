"""
Similarity Matrix package - Analysis and similarity computation.

This package provides classes for analyzing tree structures and computing
similarity metrics across populations.
"""

from .matrix import SimilarityMatrix
from .series import SimilaritySeries
from .frame import SimilarityFrame

__all__ = [
    "SimilarityMatrix",
    "SimilaritySeries",
    "SimilarityFrame",
]

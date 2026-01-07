"""
Matrix package - Analysis and similarity computation.

This package provides classes for analyzing tree structures and computing
similarity metrics across populations.
"""

from .matrix import MatrixInstance, DATA_INSTANCES, DATA_FRAMES, DATA_SERIES
from .series import MatrixSeries
from .frame import MatrixFrame

__all__ = [
    "MatrixInstance",
    "MatrixSeries",
    "MatrixFrame",
    
    "DATA_INSTANCES",
    "DATA_FRAMES",
    "DATA_SERIES"
]

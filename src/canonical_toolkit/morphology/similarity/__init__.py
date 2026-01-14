"""
Similarity package - Morphological similarity analysis.

This package provides classes and functions for analyzing tree structures
and computing similarity metrics across robot populations.
"""

# # ===== Similarity-specific Matrix Classes =====
# from .sim_matrix import SimilarityFrame, SimilarityMatrix, SimilaritySeries

# # ===== Enums & Options =====
# from .options import MatrixDomain, OutputType, RadiusStrategy, Space


from .sim_matrix import *
from .options import *
from .pipeline import *




# # ===== Define exports =====
# __all__ = [
#     # Matrix classes
#     "SimilarityFrame",
#     "SimilarityMatrix",
#     "SimilaritySeries",
    
#     # options
#     "MatrixDomain",
#     "OutputType",
#     "RadiusStrategy",
#     "Space"

# ]

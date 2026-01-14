"""
CanonicalToolkit - Morphological analysis toolkit for evolutionary robotics.

This toolkit provides:
- Generic matrix infrastructure (base.matrix)
- Morphological analysis tools (morphology.node, morphology.similarity, morphology.visual)

Quick start:
    # Import quickly [easiest]
    import canonical_toolkit as ctk

    # Import packages
    from canonical_toolkit import base, morphology

    # Or import directly from subpackages
    from canonical_toolkit.base.matrix import MatrixInstance
    from canonical_toolkit.morphology.node import Node
    from canonical_toolkit.morphology.similarity import SimilarityMatrix
"""

# # ===== Main Packages =====
# from . import base
# from . import morphology

from .base import *
from .morphology import *
from .utils import *



# ===== Convenience Imports (Optional - most commonly used) =====
# Users can import directly from subpackages, or use these for convenience

# # Base matrix classes
# from .base.matrix import (
#     MatrixInstance,
#     MatrixSeries,
#     MatrixFrame,
# )

# # Node classes and factories
# from .morphology.node import (
#     Node,
#     create_root_node,
#     create_node,
#     create_brick_node,
#     create_hinge_node,
#     node_from_graph,
#     node_from_string
#     # suppress_face_errors,
# )

# # Similarity classes
# from .morphology.similarity import (
#     SimilarityMatrix,
#     SimilaritySeries,
#     SimilarityFrame,
#     Space,
#     MatrixDomain,
# )

# from .morphology.similarity.pipeline import *

# from .morphology.similarity import *

# from .morphology.visual import(
#     view
# )

# # ===== Exports =====
# __all__ = [
#     # Packages
#     "base",
#     "morphology",

#     # Base matrix
#     "MatrixInstance",
#     "MatrixSeries",
#     "MatrixFrame",

#     # Node
#     "Node",
#     "create_root_node",
#     "create_node",
#     "create_brick_node",
#     "create_hinge_node",
#     "node_from_graph",
#     "node_from_string",

#     # Similarity
#     "SimilarityMatrix",
#     "SimilaritySeries",
#     "SimilarityFrame",
#     # options
#     "Space",
#     "MatrixDomain",
    
#     # Visual functions
#     "view"
# ]

__version__ = "0.1.0"

# Initialize
# suppress_face_errors()

"""
CanonicalToolkit - Tree canonicalization and similarity analysis.

Quick start:
    from canonical_toolkit import create_root_node, Node, MatrixFrame

    root = create_root_node()
    root["front"] = create_brick_node()
"""

# ===== Node Package =====
from canonical_toolkit.core.node import (
    # Core
    Node,

    # Factory
    create_root_node,
    create_node,
    create_brick_node,
    create_hinge_node,
    node_from_graph,
    node_from_string,
    # from_nde_genotype,

    # Types
    hash_fingerprint,
    population_fingerprints,

    # Utilities
    suppress_face_errors,
)

# ===== Node Tool Modules (for advanced use) =====
from canonical_toolkit.core import node

# ===== Matrix Package =====
from canonical_toolkit.core.matrix import (
    # Analysis Classes
    MatrixInstance,
    MatrixSeries,
    MatrixFrame,

    # Enums
    MatrixDomain,
    VectorSpace,
)

# ===== Similarity Analysis =====
from canonical_toolkit.core import similarity
# from canonical_toolkit.core.similarity import (
#     # Config & Enums
#     # OutputType,
#     # RadiusStrategy,
#     # HVectorSpace,
#     SimilarityConfig,

#     # Functions
#     # collect_subtrees,
#     collect_hash_fingerprint,
#     to_canonical_graph,
#     to_canonical_string,
#     series_from_population_fingerprint,
#     series_to_grid_configs,
#     embeddings_to_grid,
# )

# ===== Visual Sub-package =====
from canonical_toolkit.core import visual


suppress_face_errors()

# ===== Define what gets exported with "from canonical_toolkit import *"
__all__ = [
    # Node
    "Node",
    "node",  # Module for advanced use (access to deriver, serializer, factory)

    # Factory
    "create_root_node",
    "create_node",
    "create_brick_node",
    "create_hinge_node",
    "node_from_graph",
    "node_from_string",
    # "from_nde_genotype",

    # Analysis
    "MatrixInstance",
    "MatrixSeries",
    "MatrixFrame",
    "MatrixDomain",
    "VectorSpace",

    # Types
    "hash_fingerprint",
    "population_fingerprints",

    # Similarity Module
    "similarity",

    # Similarity - Config & Enums
    # "OutputType",
    # "RadiusStrategy",
    # "HVectorSpace",
    "SimilarityConfig",

    # Similarity - Functions
    "collect_hash_fingerprint",
    "to_canonical_graph",
    "to_canonical_string",
    "series_from_population_fingerprint",
    "series_to_grid_configs",
    "embeddings_to_grid",

    # Utils
    "suppress_face_errors",

    # Visual sub-package
    "visual",
]

# Optional: Version info
__version__ = "0.1.0"

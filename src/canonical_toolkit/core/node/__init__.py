"""
Node package - Tree structure and operations.

This package provides the core Node class and related functionality for
building, manipulating, and analyzing tree structures.
"""

# ===== Core Node Class =====
from canonical_toolkit.core.node.node import Node

# ===== Tool Modules =====
from canonical_toolkit.core.node.tools import (
    deriver,
    # factory,
    serializer,
)

# ===== Factory Functions =====
from canonical_toolkit.core.node.tools.factory import (
    create_root_node,
    create_node,
    create_brick_node,
    create_hinge_node,
    node_from_graph,
    node_from_string,
    # from_nde_genotype,
)

# # ===== Node Operations =====
# from canonical_toolkit.core.node.tools.deriver import (
#     canonicalize,
#     collect_subtrees,
#     collect_neighbourhoods,
# )

# # ===== Serialization =====
# from canonical_toolkit.core.node.tools.serializer import (
#     to_graph,
#     to_string,
# )

# ===== Types =====
from canonical_toolkit.core.node.n_types import (
    hash_fingerprint,
    population_fingerprints,
)

# ===== Utilities =====
from canonical_toolkit.core.node.exceptions.suppress import suppress_face_errors

# ===== Define exports =====
__all__ = [
    # Core
    "Node",

    # Tool Modules (for advanced use)
    "deriver",
    # "factory",
    "serializer",

    # Factory
    "create_root_node",
    "create_node",
    "create_brick_node",
    "create_hinge_node",
    "node_from_graph",
    "node_from_string",

    # Types
    "hash_fingerprint",
    "population_fingerprints",

    # Utilities
    "suppress_face_errors",
]

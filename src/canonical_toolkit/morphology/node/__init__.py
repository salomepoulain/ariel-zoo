"""
Node package - Tree structure and operations.

This package provides the core Node class and related functionality for
building, manipulating, and analyzing tree structures.
"""

# ===== Core Node Class =====
from .node import Node

# ===== Tool Modules =====
from .tools import (
    deriver,
    # factory,
    serializer,
)

# ===== Factory Functions =====
from .tools.factory import (
    create_root_node,
    create_node,
    create_brick_node,
    create_hinge_node,
    node_from_graph,
    node_from_string,
    # from_nde_genotype,
)

# # ===== Node Operations =====
# from .tools.deriver import (
#     canonicalize,
#     collect_subtrees,
#     collect_neighbourhoods,
# )

# # ===== Serialization =====
# from .tools.serializer import (
#     to_graph,
#     to_string,
# 

# ===== Utilities =====
from .exceptions.suppress import suppress_face_errors

# ===== Define exports =====
__all__ = [
    # Core
    "Node",

    # Tool Modules (for advanced use)
    "deriver",
    "serializer",

    # Factory
    "create_root_node",
    "create_node",
    "create_brick_node",
    "create_hinge_node",
    "node_from_graph",
    "node_from_string",

    # Utilities
    "suppress_face_errors",
]

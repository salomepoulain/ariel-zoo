from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from networkx import topological_sort

from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)

from .._configs.canonical_config import (
    CANONICAL_CONFIGS,
)
from ..node import (
    Node,
)

if TYPE_CHECKING:
    from ariel.ec.genotypes.nde.nde import (
        NeuralDevelopmentalEncoding,
    )
    from networkx import DiGraph

PRE_DEFINED_CONFIGS = CANONICAL_CONFIGS
MODULE_BY_LETTER: dict[str, ModuleType] = {mt.name[0]: mt for mt in ModuleType}

__all__ = [
    "create_root_node",
    "create_node",
    "create_brick_node",
    "create_hinge_node",
    "node_from_graph",
    "node_from_string",
    "node_from_nde_genotype",
]


def _string_to_module_type(s: str) -> ModuleType:
    """Convert string to ModuleType enum."""
    s = s.upper().strip()
    try:
        return ModuleType[s]
    except KeyError as e:
        for module_type in ModuleType:
            if module_type.name.startswith(s):
                return module_type
        valid = ", ".join(mt.name for mt in ModuleType)
        msg = f"Unknown module type '{s}'. Valid: {valid}"
        raise ValueError(msg) from e


def create_root_node(
    module_type_str: str = "CORE",
    rotation: int = 0,
    *,
    auto_ids: bool = True,
) -> Node:
    """
    Create a root CanonicalNode.

    If auto_ids=True, child id's will automatically increment using a shared ID counter.
    """
    module_type = _string_to_module_type(module_type_str)
    root = Node(
        config=PRE_DEFINED_CONFIGS[module_type],
        rotation=ModuleRotationsIdx(rotation).value,
    )

    if auto_ids:
        root.add_id_tags()

    return root


def create_node(
    module_type_str: str,
    rotation: int = 0,
    *,
    node_tags: dict[str, Any] | None = None,
) -> Node:
    """Create a generic node of any module type."""
    module_type = _string_to_module_type(module_type_str)
    if not node_tags:
        node_tags = {}
    node = Node(
        config=PRE_DEFINED_CONFIGS[module_type],
        rotation=ModuleRotationsIdx(rotation).value,
    )
    node.node_tags.update(node_tags)
    return node


def create_brick_node(
    rotation: int = 0,
    *,
    node_tags: dict[str, Any] | None = None,
) -> Node:
    """Create a brick node."""
    if not node_tags:
        node_tags = {}
    brick_node = Node(
        config=PRE_DEFINED_CONFIGS[ModuleType.BRICK],
        rotation=ModuleRotationsIdx(rotation).value,
    )
    brick_node.node_tags.update(node_tags)
    return brick_node


def create_hinge_node(
    rotation: int = 0,
    *,
    node_tags: dict[str, Any] | None = None,
) -> Node:
    """Create a hinge node."""
    if not node_tags:
        node_tags = {}
    hinge_node = Node(
        rotation=ModuleRotationsIdx(rotation).value,
        config=PRE_DEFINED_CONFIGS[ModuleType.HINGE],
    )
    hinge_node.node_tags.update(node_tags)
    return hinge_node


def node_from_graph(
    graph: DiGraph[Any],
    *,
    auto_id: bool = False,
    skip_type: ModuleType = ModuleType.NONE,
) -> Node:
    """Create a node tree from a NetworkX graph."""
    node_map: dict[int, Node] = {}

    # Find and create root node
    root_id = next(n for n in graph.nodes() if graph.in_degree(n) == 0)
    root_attrs = graph.nodes[root_id]

    node_map[root_id] = create_root_node(
        module_type_str=root_attrs["type"],
        rotation=ModuleRotationsIdx[root_attrs["rotation"]].value,
        auto_ids=auto_id,
    )

    if auto_id:
        root = node_map[root_id]
        root.node_tags["id"] = root_id
        root.tree_tags["max_id"] = root_id

    # Build tree structure - process nodes in topological order
    for node_id in topological_sort(graph):
        if node_id not in node_map:
            continue

        parent = node_map[node_id]

        for _, child_id, edge_data in graph.out_edges(node_id, data=True):
            child_type = ModuleType[graph.nodes[child_id]["type"]]
            if child_type == skip_type:
                continue

            if child_id not in node_map:
                attrs = graph.nodes[child_id]
                node_map[child_id] = create_node(
                    module_type_str=attrs["type"],
                    rotation=ModuleRotationsIdx[attrs["rotation"]].value,
                    node_tags={"id": child_id} if auto_id else None,
                )

                if auto_id:
                    parent.tree_tags["max_id"] = max(
                        parent.tree_tags["max_id"],
                        child_id,
                    )

            child_node = node_map[child_id]
            parent[ModuleFaces[edge_data["face"]]] = child_node

    return node_map[root_id]


def node_from_string(s: str) -> Node:
    """
    Parse a string into a CanonicalNode tree.
    See docs/GRAMMAR.ebnf for the complete canonical string grammar specification.
    """
    # Cache common operations
    s_len = len(s)
    isdigit = str.isdigit
    islower = str.islower
    isupper = str.isupper
    ord_ = ord
    min_ = min

    def parse_node(i: int) -> tuple[Node, int]:
        """Parse a single node and return (node, next_index)."""
        # Create node from letter
        c = s[i]
        node = create_root_node(c, auto_ids=False) if i == 0 else create_node(c)
        i += 1

        # Parse rotation number (optimized)
        rotation = 0
        while i < s_len and isdigit(s[i]):
            rotation = rotation * 10 + int(s[i])
            i += 1
        if rotation:
            node.rotate_amt(rotation)

        # Cache face orders
        radial_faces = node.config.radial_face_order
        axial_faces = node.config.axial_face_order

        def parse_children_group(
            i: int,
            parent: Node,
            faces: list[ModuleFaces],
            end_char: str,
        ) -> int:
            """Parse a [...] or <...> group and attach children."""
            while i < s_len and s[i] != end_char:
                # Check for count notation: "4-(X)"
                count = 0
                while i < s_len and isdigit(s[i]):
                    count = count * 10 + (ord_(s[i]) - 48)
                    i += 1

                if count and i < s_len and s[i] == "-":
                    # Count notation: attach same child to first N faces
                    i += 2  # Skip '-('
                    child, i = parse_node(i)
                    i += 1  # Skip ')'

                    limit = min_(count, len(faces))
                    for j in range(limit):
                        parent[faces[j]] = child

                    return i + 1  # Skip ']' or '>'

                # Face letter notation: "nse(X)"
                face_letter_start = i
                while i < s_len and islower(s[i]):
                    i += 1
                face_letters = s[face_letter_start:i]

                i += 1  # Skip '('
                child, i = parse_node(i)
                i += 1  # Skip ')'

                # Attach child to each specified face
                for letter in face_letters:
                    for face in faces:
                        if letter == face.name[0].lower():
                            parent[face.name] = child
                            break  # Move to next letter after first match

            return i + 1  # Skip ']' or '>'

        # Parse radial children: [...]
        if i < s_len and s[i] == "[":
            i = parse_children_group(i + 1, node, radial_faces, "]")

        # Parse axial children: <...>
        if i < s_len and s[i] == "<":
            i = parse_children_group(i + 1, node, axial_faces, ">")

        # Parse single axial child (no brackets)
        if i < s_len and isupper(s[i]):
            child, i = parse_node(i)
            if axial_faces:
                node[axial_faces[0]] = child

        return node, i

    return parse_node(0)[0]


def node_from_nde_genotype(
    genotype: list[list[float]],
    NDE: NeuralDevelopmentalEncoding,
    *,
    num_modules: int = 20,
) -> Node:
    """
    Create a CanonicalizableNode from an NDE genotype.

    Args:
        genotype: NDE genotype (3 matrices: weights for NDE)
        num_modules: Number of modules for NDE decoder
        auto_id: Whether to add ID tags to nodes

    Returns
    -------
        Root CanonicalizableNode of the decoded tree
    """
    try:
        from torch import no_grad
    except ImportError as e:
        raise ImportError("Torch Required") from e

    with no_grad():
        matrixes = NDE.forward(np.array(genotype))

    hpd = HighProbabilityDecoder(num_modules=num_modules)
    ind_graph = hpd.probability_matrices_to_graph(
        matrixes[0],
        matrixes[1],
        matrixes[2],
    )

    return node_from_graph(ind_graph)

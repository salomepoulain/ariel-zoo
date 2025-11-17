from __future__ import annotations

from typing import Any

import networkx as nx

from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel_experiments.characterize.canonical.configs.canonical_config import (
    CANONICAL_CONFIGS,
)
from ariel_experiments.characterize.canonical.core.node import (
    CanonicalizableNode,
)
import numpy as np
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.genotypes.nde.nde import (
    NeuralDevelopmentalEncoding,
)


class TreeFactory:
    """
    Constructs CanonicalNode trees from various formats.
    Handles all ID generation, string parsing, and config lookups.
    """

    PRE_DEFINED_CONFIGS = CANONICAL_CONFIGS
    MODULE_BY_LETTER: dict[str, ModuleType] = {
        mt.name[0]: mt for mt in ModuleType
    }

    @staticmethod
    def _string_to_module_type(s: str) -> ModuleType:
        s = s.upper().strip()
        try:
            return ModuleType[s]
        except KeyError as e:
            for module_type in ModuleType:
                if module_type.name.startswith(s):
                    return module_type
            valid = ", ".join(mt.name for mt in ModuleType)
            msg = f"Unknown module type '{s}'. Valid: {valid}"
            raise ValueError(
                msg,
            ) from e

    @classmethod
    def create_root(
        cls,
        module_type_str: str = "CORE",
        rotation: int = 0,
        *,
        auto_ids: bool = True,
    ) -> CanonicalizableNode:
        """
        Create a root CanonicalNode.

        If auto_ids=True, child id's will automatically increment using a shared ID counter.
        """
        module_type = cls._string_to_module_type(module_type_str)
        root = CanonicalizableNode(
            config=cls.PRE_DEFINED_CONFIGS[module_type],
            rotation=ModuleRotationsIdx(rotation).value,
        )

        if auto_ids:
            root.add_id_tags()

        return root

    @classmethod
    def node(
        cls,
        module_type_str: str,
        rotation: int = 0,
        *,
        node_tags: dict[str, Any] | None = None,
    ) -> CanonicalizableNode:
        module_type = TreeFactory._string_to_module_type(module_type_str)
        if not node_tags:
            node_tags = {}
        node = CanonicalizableNode(
            config=cls.PRE_DEFINED_CONFIGS[module_type],
            rotation=ModuleRotationsIdx(rotation).value,
        )
        node.node_tags.update(node_tags)
        return node

    @classmethod
    def brick(
        cls,
        rotation: int = 0,
        *,
        node_tags: dict[str, Any] | None = None,
    ) -> CanonicalizableNode:
        if not node_tags:
            node_tags = {}
        brick_node = CanonicalizableNode(
            config=cls.PRE_DEFINED_CONFIGS[ModuleType.BRICK],
            rotation=ModuleRotationsIdx(rotation).value,
        )
        brick_node.node_tags.update(node_tags)
        return brick_node

    @classmethod
    def hinge(
        cls,
        rotation: int = 0,
        *,
        node_tags: dict[str, Any] | None = None,
    ) -> CanonicalizableNode:
        if not node_tags:
            node_tags = {}
        hinge_node = CanonicalizableNode(
            rotation=ModuleRotationsIdx(rotation).value,
            config=cls.PRE_DEFINED_CONFIGS[ModuleType.HINGE],
        )
        hinge_node.node_tags.update(node_tags)
        return hinge_node

    @classmethod
    def from_graph(
        cls,
        graph: nx.DiGraph[Any],
        *,
        auto_id: bool = False,
        skip_type: ModuleType = ModuleType.NONE,
    ) -> CanonicalizableNode:
        node_map: dict[int, CanonicalizableNode] = {}

        # Find and create root node
        root_id = next(n for n in graph.nodes() if graph.in_degree(n) == 0)
        root_attrs = graph.nodes[root_id]

        node_map[root_id] = cls.create_root(
            module_type_str=root_attrs["type"],
            rotation=ModuleRotationsIdx[root_attrs["rotation"]].value,
            auto_ids=auto_id,
        )

        if auto_id:
            root = node_map[root_id]
            root.node_tags["id"] = root_id
            root.tree_tags["max_id"] = root_id

        # Build tree structure - process nodes in topological order
        for node_id in nx.topological_sort(graph):
            if node_id not in node_map:
                continue

            parent = node_map[node_id]

            for _, child_id, edge_data in graph.out_edges(node_id, data=True):
                child_type = ModuleType[graph.nodes[child_id]["type"]]
                if child_type == skip_type:
                    continue

                if child_id not in node_map:
                    attrs = graph.nodes[child_id]
                    node_map[child_id] = cls.node(
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

    # TODO: fix the crazy complexity
    # TODO: fix the bug for core b -> should be back/bottom
    @classmethod
    def from_string(cls, s: str) -> CanonicalizableNode:
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

        def parse_node(i: int) -> tuple[CanonicalizableNode, int]:
            """Parse a single node and return (node, next_index)."""
            # Create node from letter
            c = s[i]
            node = cls.create_root(c, auto_ids=False) if i == 0 else cls.node(c)
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
                parent: CanonicalizableNode,
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

    @classmethod
    def from_nde_genotype(
        cls,
        genotype: list[list[float]],
        *,
        num_modules: int = 20,
        auto_id: bool = False,
    ) -> CanonicalizableNode:
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
        nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
        hpd = HighProbabilityDecoder(num_modules=num_modules)

        matrixes = nde.forward(np.array(genotype))
        graph = hpd.probability_matrices_to_graph(
            matrixes[0],
            matrixes[1],
            matrixes[2],
        )

        # Convert graph to canonical node
        return cls.from_graph(graph, auto_id=auto_id)

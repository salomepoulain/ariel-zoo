from __future__ import annotations

from typing import Any
import networkx as nx

from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel_experiments.characterize.canonical.canonical_node import (
    CanonicalNode,
)
from ariel_experiments.characterize.canonical.configs.canonical_config import (
    CANONICAL_CONFIGS,
)


class TreeFactory:
    """
    Constructs CanonicalNode trees from various formats.
    Handles all ID generation, string parsing, and config lookups.
    """

    pre_defined_configs = CANONICAL_CONFIGS
    module_by_letter: dict[str, ModuleType] = {
        mt.name[0]: mt for mt in ModuleType
    }

    @staticmethod
    def id_assigner(node: CanonicalNode) -> None:
        if "max_id" not in node.tree_tags:
            node.tree_tags["max_id"] = 0

        if "id" in node.node_tags:
            node.tree_tags["max_id"] = max(node.tree_tags["max_id"], node.node_tags["id"])
        else:
            node.tree_tags["max_id"] += 1
            node.node_tags["id"] = node.tree_tags["max_id"]

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
    ) -> CanonicalNode:
        """
        Create a root CanonicalNode.

        If auto_ids=True, child id's will automatically increment using a shared ID counter.
        """
        module_type = cls._string_to_module_type(module_type_str)

        kwargs = {
            "module_type": module_type,
            "rotation": ModuleRotationsIdx(rotation).value,
            "config": cls.pre_defined_configs[module_type],
        }

        if auto_ids:
            kwargs["attach_process_fn"] = cls.id_assigner
            kwargs["node_tags"] = {"id": 0}
            kwargs["tree_tags"] = {"max_id": 0}

        return CanonicalNode(**kwargs)

    @classmethod
    def node(
        cls,
        module_type_str: str,
        rotation: int = 0,
        *,
        node_tags: dict[str, Any] | None = None,
    ) -> CanonicalNode:
        module_type = TreeFactory._string_to_module_type(module_type_str)
        if not node_tags:
            node_tags = {}
        return CanonicalNode(
            module_type=module_type,
            rotation=ModuleRotationsIdx(rotation).value,
            config=cls.pre_defined_configs[module_type],
            node_tags=node_tags,
        )

    @classmethod
    def brick(
        cls, rotation: int = 0, *, node_tags: dict[str, Any] | None = None,
    ) -> CanonicalNode:
        if not node_tags:
            node_tags = {}
        return CanonicalNode(
            module_type=ModuleType.BRICK,
            rotation=ModuleRotationsIdx(rotation).value,
            config=cls.pre_defined_configs[ModuleType.BRICK],
            node_tags=node_tags,
        )

    @classmethod
    def hinge(
        cls, rotation: int = 0, *, node_tags: dict[str, Any] | None = None,
    ) -> CanonicalNode:
        if not node_tags:
            node_tags = {}
        return CanonicalNode(
            module_type=ModuleType.HINGE,
            rotation=ModuleRotationsIdx(rotation).value,
            config=cls.pre_defined_configs[ModuleType.HINGE],
            node_tags=node_tags,
        )

    @classmethod
    def from_graph(
        cls,
        graph: nx.DiGraph[Any],
        *,
        auto_id: bool = False,
        skip_type: ModuleType = ModuleType.NONE,
    ) -> CanonicalNode:

        node_map: dict[int, CanonicalNode] = {}

        def _fill_in(parent_id: int) -> None:
            parent = node_map[parent_id]
            for _, child_id, edge_data in graph.out_edges(parent_id, data=True):
                child_type = ModuleType[graph.nodes[child_id]["type"]]
                if child_type == skip_type:
                    continue

                if child_id not in node_map:
                    attrs = graph.nodes[child_id]
                    node_map[child_id] = cls.node(
                        module_type_str=attrs["type"],
                        rotation=ModuleRotationsIdx[attrs["rotation"]].value,
                        node_tags={"id": child_id} if auto_id else {},
                    )

                child_node = node_map[child_id]
                parent[ModuleFaces[edge_data["face"]]] = child_node
                _fill_in(child_id)

        # (just to assume it doesnt always start with id = 0)
        root_id = next(n for n in graph.nodes() if graph.in_degree(n) == 0)
        root_attrs = graph.nodes[root_id]
        node_map[root_id] = cls.create_root(
            module_type_str=root_attrs["type"],
            rotation=ModuleRotationsIdx[root_attrs["rotation"]].value,
            auto_ids=auto_id,
        )

        if auto_id:
            node_map[root_id].node_tags["id"] = root_id

        _fill_in(root_id)

        return node_map[root_id]

    # TODO: fix the crazy complexity
    @classmethod
    def from_string(cls, s: str) -> CanonicalNode:
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

        def parse_node(i: int) -> tuple[CanonicalNode, int]:
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
                node.rotation = rotation

            # Cache face orders
            radial_faces = node.config.radial_face_order
            axial_faces = node.config.axial_face_order

            def parse_children_group(
                i: int, parent: CanonicalNode, faces: list, end_char: str,
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
                        parent[letter] = child

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

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


class TreeExporter:
    """
    Constructs CanonicalNode trees from various formats.
    Handles all ID generation, string parsing, and config lookups.
    """

    @staticmethod
    def to_graph(
        root_node: CanonicalNode, skip_type: ModuleType = ModuleType.NONE
    ) -> nx.DiGraph[Any]:
        """
        Adds id's if there aren't any, then converts to graph.
        """
        # Ensure all nodes have IDs
        if not root_node.node_tags or "id" not in root_node.node_tags:
            id_counter = [0]

            def assign_id(node: CanonicalNode):
                if not node.node_tags or "id" not in node.node_tags:
                    if not node.node_tags:
                        node.node_tags = {}
                    node.node_tags["id"] = id_counter[0]
                    id_counter[0] += 1

            root_node.traverse_depth_first(assign_id, pre_order=True)

        graph = nx.DiGraph()

        def _add_node_and_edge(node: CanonicalNode) -> None:
            """Visitor function executed once per node during traversal."""
            node_attrs = {
                "type": node.module_type.name,
                "rotation": ModuleRotationsIdx(node.rotation).name,
            }
            graph.add_node(node.node_tags["id"], **node_attrs)

            if skip_type == ModuleType.NONE:
                for face, child_node in node.children:
                    graph.add_edge(
                        node.node_tags["id"],
                        child_node.node_tags["id"],
                        face=face.name,
                    )
            else:
                for face, child_node in node.children:
                    if child_node.module_type != skip_type:
                        graph.add_edge(
                            node.node_tags["id"],
                            child_node.node_tags["id"],
                            face=face.name,
                        )

        root_node.traverse_depth_first(_add_node_and_edge, pre_order=True)
        return graph

    @classmethod
    def to_string(cls, node: CanonicalNode) -> str:
        """Generates the canonical string representation."""
        name = node.module_type.name[0]
        if node.rotation != 0:
            name += str(int(node.rotation))

        # Separate axial and radial children
        axial_children = []
        radial_children = []

        for face, child in node.children:
            if face in node.config.axial_face_order:
                axial_children.append((face, child))
            else:
                radial_children.append((face, child))

        result = name

        # Radial children with smart grouping
        if radial_children:
            result += cls._format_children_group(
                node, radial_children, uppercase=False
            )

        # Axial children with smart grouping
        if axial_children:
            needs_labels = len(node.config.axial_face_order) > 1
            if needs_labels:
                result += cls._format_children_group(
                    node, axial_children, uppercase=True
                )
            else:
                # Single axial face: just concatenate
                for _, child in axial_children:
                    result += cls.to_string(child)

        return result

    @classmethod
    def _format_children_group(
        cls,
        node: CanonicalNode,
        children: list[tuple[ModuleFaces, CanonicalNode]],
        uppercase: bool,
    ) -> str:
        """Helper for to_string: Formats children with smart deduplication."""
        if not children:
            return ""

        # Determine total available faces in this context
        if uppercase:
            total_faces = len(node.config.axial_face_order)
        else:
            total_faces = len(node.config.radial_face_order)

        # First pass: compute formulas
        face_formulas = [
            (face, cls.to_string(child)) for face, child in children
        ]

        # Second pass: group by formula
        formula_to_faces = {}
        for face, formula in face_formulas:
            if formula not in formula_to_faces:
                formula_to_faces[formula] = []
            formula_to_faces[formula].append(face)

        # Build output
        parts = []
        for formula, faces in sorted(formula_to_faces.items()):
            faces_sorted = sorted(faces, key=lambda f: f.value)
            face_letters = "".join(f.name[0].lower() for f in faces_sorted)

            # Use count if ALL faces have the same child and there are 2+ faces
            if len(faces) == total_faces and len(faces) > 1:
                parts.append(f"{len(faces)}-({formula})")
            else:
                parts.append(f"{face_letters}({formula})")

        # Radial: [], Axial: <>
        return f"<{''.join(parts)}>" if uppercase else f"[{''.join(parts)}]"

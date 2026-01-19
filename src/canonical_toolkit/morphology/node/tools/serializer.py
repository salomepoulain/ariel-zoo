from __future__ import annotations

from typing import TYPE_CHECKING, Any

from networkx import DiGraph

from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleRotationsIdx,
)

if TYPE_CHECKING:
    from ..node import (
        Node,
    )


__all__ = [
    "to_graph",
    "to_string",
]


def to_graph(root_node: Node) -> DiGraph[Any]:
    """
    Convert a node tree to a NetworkX DiGraph.

    Adds id's if there aren't any, then converts to graph.
    Creates a copy to avoid modifying the original tree.
    """
    root_copy = root_node.copy(deep=True, copy_children=True)
    graph = DiGraph()

    max_attempts = 2
    for attempt in range(max_attempts):
        try:
            for node in root_copy:
                node_id = node.node_tags["id"]

                node_attrs = {
                    "type": node.module_type.name,
                    "rotation": ModuleRotationsIdx(node.internal_rotation).name,
                }
                graph.add_node(node_id, **node_attrs)

                for face, child_node in node.children_items:
                    graph.add_edge(
                        node_id,
                        child_node.node_tags["id"],
                        face=face.name,
                    )
            break
        except KeyError as e:
            if attempt == 0:
                root_copy.add_id_tags()
            else:
                msg = "id not found, even after 2 attempts"
                raise KeyError(msg) from e
    return graph


def to_string(node: Node) -> str:
    """Generate the canonical string representation of a node tree."""
    name = node.module_type.name[0]
    if node.internal_rotation != 0:
        name += str(int(node.internal_rotation))

    # Separate axial and radial children
    axial_children = []
    radial_children = []

    for face, child in node.children_items:
        if face in node.config.axial_face_order:
            axial_children.append((face, child))
        else:
            radial_children.append((face, child))

    result = name

    # Radial children with smart grouping
    if radial_children:
        result += _format_children_group(
            node,
            radial_children,
            uppercase=False,
        )

    # Axial children with smart grouping
    if axial_children:
        needs_labels = len(node.config.axial_face_order) > 1
        if needs_labels:
            result += _format_children_group(
                node,
                axial_children,
                uppercase=True,
            )
        else:
            # Single axial face: just concatenate
            for _, child in axial_children:
                result += to_string(child)

    return result


def _format_children_group(
    node: Node,
    children: list[tuple[ModuleFaces, Node]],
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
    face_formulas = [(face, to_string(child)) for face, child in children]

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

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    from networkx import DiGraph

    from canonical_toolkit.configs.canonical_config import (
        CanonicalConfig,
    )

import sys

from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)


# MARK class ----
class Node:
    """
    A node in a canonical tree structure.

    Core attributes (config, rotation, parent) are private and accessed via properties.
    Module type is derived from the config.
    """

    __slots__ = (
        "__face_to_placement",
        "_axial_list",
        "_config",
        "_full_priority",
        "_internal_rotation",
        "_node_tags",
        "_parent",
        "_radial_list",
        "_tree_tags",
    )

    _STRING_TO_FACE: ClassVar[dict[str, ModuleFaces]] = {
        **{face.name.lower(): face for face in ModuleFaces},
        **{face.name[:2].lower(): face for face in ModuleFaces},
    }
    _MAX_ROTATIONS: ClassVar[int] = 8

    def __init__(
        self,
        config: CanonicalConfig,  # TODO: is this strange?..
        rotation: int,
        *,
        parent: Node | None = None,
    ) -> None:
        """
        Initialize a CanonicalizableNode.

        Args:
            config: Canonical configuration for this node
            internal_rotation: Internal rotation value
            parent: Optional parent node
        """
        self._config = config
        self._internal_rotation = int(ModuleRotationsIdx(rotation).value)
        self._parent = parent

        # Initialize child lists based on config
        self._axial_list: list[Node | None] = [None] * len(
            config.axial_face_order,
        )
        self._radial_list: list[Node | None] = [None] * len(
            config.radial_face_order,
        )

        # Initialize tags
        self._node_tags: dict[str, Any] = {}
        self._tree_tags: dict[str, Any] = {}

        # Set priority from config
        self._full_priority = config.priority

        # Build face to placement mapping
        mapping: dict[
            ModuleFaces, tuple[list[Node | None], int],
        ] = {}
        for i, f in enumerate(config.axial_face_order):
            mapping[f] = (self._axial_list, i)
        for i, f in enumerate(config.radial_face_order):
            mapping[f] = (self._radial_list, i)
        self.__face_to_placement = mapping

    # region dunder methods -----

    def __getitem__(
        self,
        face: ModuleFaces | str,
    ) -> Node:
        if isinstance(face, str):
            face_lower = face.lower()
            try:
                face = self._STRING_TO_FACE[face_lower]
            except KeyError:
                from canonical_toolkit.core.node.exceptions.exceptions import (
                    FaceNotFoundError,
                )

                raise FaceNotFoundError(face, self) from None

        child = self._get_child(face)

        if child is None:
            from canonical_toolkit.core.node.exceptions.exceptions import (
                ChildNotFoundError,
            )

            raise ChildNotFoundError(face, self)
        return child

    def __setitem__(
        self,
        face: ModuleFaces | str,
        child: Node | None,
    ) -> None:
        """
        Set a child at a face, or detach by setting to None.

        Args:
            face: Face to set/detach child at
            child: Node to attach, or None to detach
        """
        if isinstance(face, str):
            face_lower = face.lower()
            try:
                face = self._STRING_TO_FACE[face_lower]
            except KeyError:
                from canonical_toolkit.core.node.exceptions.exceptions import (
                    FaceNotFoundError,
                )

                raise FaceNotFoundError(face, self) from None

        if child is None:
            self._detatch_child(face)
        else:
            self._set_child(face, child)

    def __iter__(self) -> Generator[Node, None, None]:
        """Iterate over all nodes in the tree (depth-first)."""
        yield self
        for child in self.children:
            yield from child

    # def __repr__(self) -> str:
    #     """Detailed attachment structure representation."""
    #     axial_parts = []
    #     for face, child in self.axial_children_items:
    #         module = child.module_type.name[0].upper()
    #         rot = child.internal_rotation
    #         axial_parts.append(f"{face.name.lower()}:{module}{rot}")

    #     radial_parts = []
    #     for face, child in self.radial_children_items:
    #         module = child.module_type.name[0].upper()
    #         rot = child.internal_rotation
    #         radial_parts.append(f"{face.name.lower()}:{module}{rot}")

    #     axial_str = ", ".join(axial_parts) if axial_parts else "empty"
    #     radial_str = ", ".join(radial_parts) if radial_parts else "empty"

    #     if self.parent and self.parent_attachment_face:
    #         prefix_str = f"{self.parent.module_type.name[0].upper()}{self.parent.internal_rotation}:{self.parent_attachment_face.name.lower()} → "
    #     else:
    #         prefix_str = ""

    #     return f"{prefix_str}{self.module_type.name[0].upper()}{self._internal_rotation} → radial[{radial_str}] axial<{axial_str}>"

    def __repr__(self) -> str:
        return self.to_string()

    def __str__(self) -> str:
        """Compact version."""
        return "node-object:" + self.to_string()
        # return f"{self.module_type.name[0].upper()}{self._internal_rotation} (↑{sum(1 for c in self._axial_list if c)} ○{sum(1 for c in self._radial_list if c)})"

    # endregion

    # region internal helpers -----

    def _update_parent_priorites(
        self, delta_priority: int,
    ) -> Node | None:
        def _priority_updater(n: Node) -> None:
            n._full_priority += delta_priority

        self.traverse_through_parents(_priority_updater)

    def _get_child(self, face: ModuleFaces) -> Node | None:
        target_list, index = self.__face_to_placement[face]
        return target_list[index]

    def _set_child_raw(
        self,
        face: ModuleFaces,
        child: Node,
    ) -> None:
        target_list, index = self.__face_to_placement[face]
        target_list[index] = child

    def _set_child(self, face: ModuleFaces, child: Node) -> None:
        child = child if child.parent is None else child.copy()

        self._update_parent_priorites(child.full_priority)
        self._set_child_raw(face, child)
        child.parent = self

        def _propagate(c: Node) -> None:
            self._tree_tags.update(c._tree_tags)
            c._tree_tags = self._tree_tags

        if not child.has_children:
            _propagate(child)
            return

        child.traverse_depth_first(_propagate)

    def _detatch_child(
        self,
        face: ModuleFaces | str,
    ) -> Node | None:
        """Returns the child detatched."""
        if isinstance(face, str):
            face = self._STRING_TO_FACE[face.lower()]

        child = self[face]
        if not child:
            return None

        self._update_parent_priorites(-child.full_priority)

        child.parent = None
        self._set_child_raw(face, None)
        return child

    # endregion

    # region properties ----

    @property
    def config(self) -> CanonicalConfig:
        """Get the node's canonical configuration."""
        return self._config

    @property
    def module_type(self) -> ModuleType:
        """Get the module type (derived from config)."""
        return self._config.module_type

    @property
    def internal_rotation(self) -> int:
        """Get the internal rotation value."""
        return self._internal_rotation

    @property
    def parent(self) -> Node | None:
        """Get the parent node."""
        return self._parent

    @parent.setter
    def parent(self, value: Node | None) -> None:
        """Set the parent node."""
        self._parent = value

    @property
    def full_priority(self) -> int:
        """Get the full priority value."""
        return self._full_priority

    @property
    def parent_attachment_face(self) -> ModuleFaces | None:
        """Get the face on the parent where this node is attached."""
        if self._parent is None:
            return None
        for face, child in self._parent.children_items:
            if child is self:
                return face
        return None

    @property
    def has_children(self):
        return any(
            child is not None for child in self._axial_list + self._radial_list
        )

    @property
    def children(self) -> Generator[Node, None, None]:
        """Yields all non-None children (axial + radial)."""
        for child in self._axial_list:
            if child is not None:
                yield child
        for child in self._radial_list:
            if child is not None:
                yield child

    @property
    def children_items(
        self,
    ) -> Generator[tuple[ModuleFaces, Node], None, None]:
        """Yields (face, child) tuples for all non-None children."""
        for face, (child_list, index) in self.__face_to_placement.items():
            child = child_list[index]
            if child is not None:
                yield face, child

    @property
    def axial_children(self) -> Generator[Node, None, None]:
        """Yields all non-None axial children."""
        for child in self._axial_list:
            if child is not None:
                yield child

    @property
    def axial_children_items(
        self,
    ) -> Generator[tuple[ModuleFaces, Node], None, None]:
        """Yields (face, child) tuples for non-None axial children."""
        for i, face in enumerate(self.config.axial_face_order):
            child = self._axial_list[i]
            if child is not None:
                yield face, child

    @property
    def radial_children(self) -> Generator[Node, None, None]:
        """Yields all non-None radial children."""
        for child in self._radial_list:
            if child is not None:
                yield child

    @property
    def radial_children_items(
        self,
    ) -> Generator[tuple[ModuleFaces, Node], None, None]:
        """Yields (face, child) tuples for non-None radial children."""
        for i, face in enumerate(self.config.radial_face_order):
            child = self._radial_list[i]
            if child is not None:
                yield face, child

    @property
    def priority_face(self) -> ModuleFaces | None:
        """Reveal which face should contain the higest priority."""
        return (
            self.config.radial_face_order[0]
            if self.config.radial_face_order
            else None
        )

    @property
    def is_root(self) -> bool:
        return self.parent is None

    @property
    def node_tags(self) -> dict[str, Any]:
        """Node-specific tags (each node has its own dict)."""
        return self._node_tags

    @property
    def tree_tags(self) -> dict[str, Any]:
        """Tree-wide shared tags (all nodes in tree share same dict)."""
        return self._tree_tags

    # endregion

    # region tags setters -----

    @node_tags.setter
    def node_tags(self, value: Any) -> None:
        """Set node tags (accepts dict only)."""
        if not isinstance(value, dict):
            msg = f"node_tags must be dict, got {type(value).__name__}"
            raise TypeError(
                msg,
            )
        self._node_tags = value

    @tree_tags.setter
    def tree_tags(self, value: Any) -> None:
        """Tree tags is encapsulated and must not be replaced, insatnce is shared across tree."""
        if not isinstance(value, dict):
            msg = f"tree_tags must be dict, got {type(value).__name__}"
            raise TypeError(
                msg,
            )

        self._tree_tags.update(value)

    # endregion

    # region modifyers ----

    def detatch_from_parent(self) -> Node | None:
        """Detach this node from its parent. Returns parent."""
        parent = self.parent
        if not parent:
            return None
        parent[self.parent_attachment_face] = None
        return parent

    def detatch_children(self) -> list[Node]:
        """Detach all children from this node, return the list."""
        detached = list(self.children)

        delta_priority = 0
        for child in detached:
            child.detatch_from_parent()
            delta_priority -= child.full_priority
        self._update_parent_priorites(delta_priority)

        self._radial_list[:] = [None] * len(self._radial_list)
        self._axial_list[:] = [None] * len(self._axial_list)

        return detached

    def rotate_amt(self, delta: int = 1) -> None:
        self._internal_rotation = (
            self._internal_rotation + delta
        ) % Node._MAX_ROTATIONS

    def shift_visual_rotation(self, shift: int) -> None:
        """i.e. visually! rotates the root and everything attached, without losing its shape."""
        rotation_change = shift * self.config.unique_rotation_amt

        for child in self.axial_children:
            child.rotate_amt(-rotation_change)

        if not self._radial_list:
            return

        shift %= len(self._radial_list)
        self._radial_list[:] = (
            self._radial_list[shift:] + self._radial_list[:shift]
        )
        for idx, child in enumerate(self._radial_list):
            if child:
                child.rotate_amt(self.config.radial_adjustments[idx][shift])
        return

    def shift_internal_rotation(self, shift: int) -> None:
        """Shifts the children w.r.t. self, while not chaning the world i.e. changing internal rotation value."""
        rotation_change = shift * self.config.unique_rotation_amt

        self.rotate_amt(-rotation_change)
        for child in self.axial_children:
            child.rotate_amt(rotation_change)

        if not self._radial_list:
            return

        shift %= len(self._radial_list)
        self._radial_list[:] = (
            self._radial_list[shift:] + self._radial_list[:shift]
        )

        for idx, child in enumerate(self._radial_list):
            if child:
                child.rotate_amt(self.config.radial_adjustments[idx][shift])
        return

    # endregion

    # region utility functions -----

    def get(self, face: str, default: Any = None) -> Node | None:
        face_lower = face.lower()
        try:
            face = self._STRING_TO_FACE[face_lower]
        except KeyError:
            from canonical_toolkit.core.utils.exceptions import (
                FaceNotFoundError,
            )
            raise FaceNotFoundError(face, self) from None

        child = self._get_child(face)
        target_list, index = self.__face_to_placement[face]
        child = target_list[index]
        if not child:
            return default
        return child

    def copy(
        self,
        *,
        deep: bool = True,
        copy_children: bool = True,
        empty: bool = False,
    ) -> Node:
        """
        Create a copy of this node.

        Args:
            deep: If True, creates independent tree with new tree_tags.
                If False, shares tree_tags reference (same conceptual tree).
            copy_children: If True, recursively copy all descendants.
        """
        new_node = Node(
            config=self.config,
            rotation=self._internal_rotation,
            parent=None,
        )

        new_node._node_tags = {} if empty else self._node_tags.copy()
        new_node._tree_tags = (
            {}
            if empty
            else (self._tree_tags.copy() if deep else self._tree_tags)
        )

        if copy_children:
            for face, child in self.children_items:
                child_copy = child.copy(
                    deep=deep,
                    copy_children=True,
                    empty=empty,
                )
                new_node[face] = child_copy

        return new_node

    def traverse_depth_first(
        self,
        visit_fn: Callable[[Node], Any]
        | list[Callable[[Node], Any]],
        *,
        pre_order: bool = True,
        post_order: bool = False,
    ) -> None:
        """Will start at node being called."""
        functions = visit_fn if isinstance(visit_fn, list) else [visit_fn]

        if pre_order:
            for fn in functions:
                fn(self)

        for child in self.children:
            child.traverse_depth_first(
                visit_fn,
                pre_order=pre_order,
                post_order=post_order,
            )

        if post_order:
            for fn in functions:
                fn(self)

    def traverse_through_parents(
        self,
        visit_fn: Callable[[Node], Any]
        | list[Callable[[Node], Any]],
        *,
        pre_order: bool = True,
        post_order: bool = False,
    ) -> None:
        """Will start at node being called."""
        functions = visit_fn if isinstance(visit_fn, list) else [visit_fn]

        if pre_order:
            for fn in functions:
                fn(self)

        if self.parent:
            self.parent.traverse_through_parents(
                visit_fn,
                pre_order=pre_order,
                post_order=post_order,
            )

        if post_order:
            for fn in functions:
                fn(self)

    def calc_shift_face_to_target(
        self,
        face_to_move: ModuleFaces,
        target_position_face: ModuleFaces,
    ) -> int:
        if not self._radial_list:
            return 0

        try:
            from_index = self.config.radial_face_order.index(face_to_move)
            to_index = self.config.radial_face_order.index(target_position_face)
        except ValueError:
            return 0

        return (from_index - to_index) % len(self.config.radial_face_order)

    def add_id_tags(self) -> None:
        """Gives each node its own id, using a shared tree_tag tag."""

        def _tag_id_assigner(node: Node) -> None:
            if "max_id" not in node.tree_tags:
                node.tree_tags["max_id"] = -1
            if "id" in node.node_tags:
                node.tree_tags["max_id"] = max(
                    node.tree_tags["max_id"], node.node_tags["id"],
                )
            else:
                node.tree_tags["max_id"] += 1
                node.node_tags["id"] = node.tree_tags["max_id"]

        if self.is_root:
            self.traverse_depth_first(_tag_id_assigner)
            return

        def _find_root_to_start(node: Node) -> None:
            if node.is_root:
                node.add_id_tags()

        self.traverse_through_parents(_find_root_to_start)

    # endregion

    # region tool importations? -----

    def canonicalize(self) -> Node:
        from canonical_toolkit.core.node.tools.deriver import (
            canonicalize,
        )

        return canonicalize(self, return_copy=False)

    def to_graph(self) -> DiGraph[Any]:
        from canonical_toolkit.core.node.tools.serializer import (
            to_graph,
        )

        return to_graph(self)

    def to_string(self) -> str:
        from canonical_toolkit.core.node.tools.serializer import (
            to_string,
        )

        return to_string(self)

    # endregion


if __name__ == "__main__":

    import sys
    sys.excepthook = sys.__excepthook__

    from canonical_toolkit.core.node.exceptions.suppress import (
        suppress_face_errors,
    )

    suppress_face_errors()

    from canonical_toolkit.core.node.tools import factory

    # try:
    root = factory.create_root_node()
    root["front"] = factory.create_brick_node()
    root.rotate_amt(3)
    root["top"]["fr"] = factory.create_brick_node()  # This will fail
    root.canonicalize()
        # ... rest of code

    # except error as e:
    #     # Print ONLY the nice message
    #     e.print_rich()
    #     sys.exit(1)

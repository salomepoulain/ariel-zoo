from __future__ import annotations

# Third-party libraries
from dataclasses import dataclass, field

# Standard library
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

    from ariel_experiments.characterize.canonical.configs.canonical_config import (
        CanonicalConfig,
    )

from typing import ClassVar

# Global constants
from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleType,
)


# MARK class ----
@dataclass(slots=True)
class CanonicalizableNode:
    """make the tree tags not accesible from the outside"""

    """if its in the same tree, it should have the same tree tags"""
    """adding """

    module_type: ModuleType
    internal_rotation: int
    config: CanonicalConfig

    # references to connections
    parent: CanonicalizableNode | None = field(
        kw_only=True,
        default=None,
        repr=False,
    )
    _axial_list: list[CanonicalizableNode | None] = field(init=False)
    _radial_list: list[CanonicalizableNode | None] = field(init=False)

    # to customize stuff
    _node_tags: dict[str, Any] = field(default_factory=dict)
    _tree_tags: dict[str, Any] = field(default_factory=dict)

    # private property so it cant be modified publically
    _full_priority: int = field(init=False)

    # constants
    __face_to_placement: Mapping[
        ModuleFaces, tuple[list[CanonicalizableNode | None], int]
    ] = field(init=False, repr=False)

    _STRING_TO_FACE: ClassVar[dict[str, ModuleFaces]] = {
        **{face.name.lower(): face for face in ModuleFaces},
        **{face.name[:2].lower(): face for face in ModuleFaces},
    }
    _MAX_ROTATIONS: ClassVar[int] = 8

    def __post_init__(self) -> None:
        self._axial_list = [None] * len(self.config.axial_face_order)
        self._radial_list = [None] * len(self.config.radial_face_order)
        self._full_priority = self.config.priority

        mapping = {}
        for i, f in enumerate(self.config.axial_face_order):
            mapping[f] = (self._axial_list, i)
        for i, f in enumerate(self.config.radial_face_order):
            mapping[f] = (self._radial_list, i)
        self.__face_to_placement = mapping

    # region dunder methods -----

    def __getitem__(
        self,
        face: ModuleFaces | str,
    ) -> CanonicalizableNode | None:
        """What the public api should use to get the child at a certain face."""
        if isinstance(face, str):
            face = self._STRING_TO_FACE[face.lower()]
        return self._get_child(face)

    def __setitem__(
        self,
        face: ModuleFaces | str,
        child: CanonicalizableNode,
    ) -> None:
        """What the public api should use to set the child at a certain face."""
        if isinstance(face, str):
            face = self._STRING_TO_FACE[face.lower()]
        self._set_child(face, child)

    def __iter__(self) -> Generator[CanonicalizableNode, None, None]:
        """Iterate over all nodes in the tree (depth-first)."""
        yield self
        for child in self.children:
            yield from child

    def __repr__(self):
        """Detailed attachment structure representation."""
        axial_parts = []
        for face, child in self.axial_children_items:
            module = child.module_type.name[0].upper()
            rot = child.internal_rotation
            axial_parts.append(f"{face.name.lower()}:{module}{rot}")

        radial_parts = []
        for face, child in self.radial_children_items:
            module = child.module_type.name[0].upper()
            rot = child.internal_rotation
            radial_parts.append(f"{face.name.lower()}:{module}{rot}")

        axial_str = ", ".join(axial_parts) if axial_parts else "empty"
        radial_str = ", ".join(radial_parts) if radial_parts else "empty"

        if self.parent and self.parent_attachment_face:
            prefix_str = f"{self.parent.module_type.name[0].upper()}{self.parent.internal_rotation}:{self.parent_attachment_face.name.lower()} → "
        else:
            prefix_str =""

        return f"{prefix_str}{self.module_type.name[0].upper()}{self.internal_rotation} → axial[{axial_str}] radial<{radial_str}>"

    def __str__(self):
        """Compact version."""
        return f"{self.module_type.name[0].upper()}{self.internal_rotation}° (↑{len(self.axial_children)} ○{sum(1 for c in self._radial_list if c)})"

    # endregion

    # region internal helpers -----

    def _update_parent_priorites(
        self, delta_priority: int
    ) -> CanonicalizableNode | None:
        def _priority_updater(n: CanonicalizableNode) -> None:
            n._full_priority += delta_priority

        self.traverse_through_parents(_priority_updater)

    def _get_child(self, face: ModuleFaces) -> CanonicalizableNode | None:
        target_list, index = self.__face_to_placement[face]
        return target_list[index]

    def _set_child_raw(
        self,
        face: ModuleFaces,
        child: CanonicalizableNode,
    ) -> None:
        target_list, index = self.__face_to_placement[face]
        target_list[index] = child

    def _set_child(self, face: ModuleFaces, child: CanonicalizableNode) -> None:
        child = child if child.parent is None else child.copy()

        self._update_parent_priorites(child.full_priority)
        self._set_child_raw(face, child)
        child.parent = self

        def _propagate(c: CanonicalizableNode) -> None:
            self._tree_tags.update(c._tree_tags)
            c._tree_tags = self._tree_tags

        if not child.has_children:
            _propagate(child)
            return

        child.traverse_depth_first(_propagate)

    # endregion

    # region properties ----

    @property
    def full_priority(self) -> int:
        return self._full_priority

    @property
    def parent_attachment_face(self) -> ModuleFaces | None:
        if self.parent is None:
            return None
        for face, child in self.parent.children_items:
            if child is self:
                return face
        return None

    @property
    def has_children(self):
        return any(
            child is not None for child in self._axial_list + self._radial_list
        )

    @property
    def children(self) -> Generator[CanonicalizableNode, None, None]:
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
    ) -> Generator[tuple[ModuleFaces, CanonicalizableNode], None, None]:
        """Yields (face, child) tuples for all non-None children."""
        for face, (child_list, index) in self.__face_to_placement.items():
            child = child_list[index]
            if child is not None:
                yield face, child

    @property
    def axial_children(self) -> Generator[CanonicalizableNode, None, None]:
        """Yields all non-None axial children."""
        for child in self._axial_list:
            if child is not None:
                yield child

    @property
    def axial_children_items(
        self,
    ) -> Generator[tuple[ModuleFaces, CanonicalizableNode], None, None]:
        """Yields (face, child) tuples for non-None axial children."""
        for i, face in enumerate(self.config.axial_face_order):
            child = self._axial_list[i]
            if child is not None:
                yield face, child

    @property
    def radial_children(self) -> Generator[CanonicalizableNode, None, None]:
        """Yields all non-None radial children."""
        for child in self._radial_list:
            if child is not None:
                yield child

    @property
    def radial_children_items(
        self,
    ) -> Generator[tuple[ModuleFaces, CanonicalizableNode], None, None]:
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
        if self.parent is None:
            return True
        return False

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
            raise TypeError(
                f"node_tags must be dict, got {type(value).__name__}"
            )
        self._node_tags = value

    @tree_tags.setter
    def tree_tags(self, value: Any) -> None:
        """tree tags is encapsulated and must not be replaced, insatnce is shared across tree"""
        if not isinstance(value, dict):
            raise TypeError(
                f"tree_tags must be dict, got {type(value).__name__}"
            )

        self._tree_tags.update(value)

    # endregion

    # region modifyers -----

    def attach_parent(
        self,
        face: ModuleFaces | str,
        parent: CanonicalizableNode,
    ) -> None:
        if isinstance(face, str):
            face = self._STRING_TO_FACE[face.lower()]
        parent._set_child(face, self)

    def detatch_parent(self) -> CanonicalizableNode | None:
        """Returns the parent."""
        parent = self.parent
        if not parent:
            return None
        parent.detatch_child(self.parent_attachment_face)
        return parent

    # TODO: add this for the set_item when setting None?
    def detatch_child(
        self,
        face: ModuleFaces | str,
    ) -> CanonicalizableNode | None:
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

    def detatch_children(self) -> None:
        delta_priority = 0
        for child in self.children:
            child.parent = None
            delta_priority -= child.full_priority
        self._update_parent_priorites(delta_priority)

        self._radial_list[:] = [None] * len(self._radial_list)
        self._axial_list[:] = [None] * len(self._axial_list)

    def rotate_amt(self, delta: int = 1) -> None:
        self.internal_rotation = (
            self.internal_rotation + delta
        ) % CanonicalizableNode._MAX_ROTATIONS

    def shift_visual_rotation(self, shift: int) -> None:
        """i.e. visually! rotates the root and everything attached, without losing its shape"""
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

    def copy(
        self,
        *,
        deep: bool = True,
        copy_children: bool = True,
        empty: bool = False,
    ) -> CanonicalizableNode:
        """
        Create a copy of this node.

        Args:
            deep: If True, creates independent tree with new tree_tags.
                If False, shares tree_tags reference (same conceptual tree).
            copy_children: If True, recursively copy all descendants.
        """
        new_node = CanonicalizableNode(
            module_type=self.module_type,
            internal_rotation=self.internal_rotation,
            parent=None,
            config=self.config,
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
        visit_fn: Callable[[CanonicalizableNode], Any]
        | list[Callable[[CanonicalizableNode], Any]],
        *,
        pre_order: bool = True,
        post_order: bool = False,
    ) -> None:
        """will start at node being called"""
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
        visit_fn: Callable[[CanonicalizableNode], Any]
        | list[Callable[[CanonicalizableNode], Any]],
        *,
        pre_order: bool = True,
        post_order: bool = False,
    ) -> None:
        """will start at node being called"""
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

        return (to_index - from_index) % len(self.config.radial_face_order)

    def add_id_tags(self):
        """gives each node its own id, using a shared tree_tag tag"""

        def _tag_id_assigner(node: CanonicalizableNode) -> None:
            if "max_id" not in node.tree_tags:
                node.tree_tags["max_id"] = -1
            if "id" in node.node_tags:
                node.tree_tags["max_id"] = max(
                    node.tree_tags["max_id"], node.node_tags["id"]
                )
            else:
                node.tree_tags["max_id"] += 1
                node.node_tags["id"] = node.tree_tags["max_id"]

        if self.is_root:
            self.traverse_depth_first(_tag_id_assigner)
            return

        def _find_root_to_start(node: CanonicalizableNode):
            if node.is_root:
                node.add_id_tags()

        self.traverse_through_parents(_find_root_to_start)

    # endregion

    # region toolkit importations? -----

    def canonicalize(self, *args: Any, **kwargs: Any) -> CanonicalizableNode:
        from ariel_experiments.characterize.canonical.core.internal.processor import (
            TreeProcessor,
        )

        return TreeProcessor.canonicalize(self, *args, **kwargs)

    # TODO add more??? or not?? what is good design choice?

    # endregion

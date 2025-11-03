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

from typing import ClassVar, Never

# Global constants
from ariel.body_phenotypes.robogen_lite.config import (
    ModuleFaces,
    ModuleType,
)


# MARK class ----
@dataclass(slots=True)
class CanonicalizableNode:
    module_type: ModuleType
    local_rotation_state: int
    config: CanonicalConfig

    # useful flags
    full_priority: int = field(init=False)

    # references to connections
    parent: CanonicalizableNode | None = field(
        kw_only=True, default=None, repr=False,
    )
    _axial_list: list[CanonicalizableNode | None] = field(init=False)
    _radial_list: list[CanonicalizableNode | None] = field(init=False)

    # to customize stuff
    node_tags: dict[str, Any] = field(kw_only=True, default_factory=dict)
    tree_tags: dict[str, Any] = field(kw_only=True, default_factory=dict)
    attach_process_fn: Callable[[CanonicalizableNode], None] | None = field(
        default=None,
        kw_only=True,
    )

    # constants and caches
    _face_to_placement: Mapping[
        ModuleFaces,
        tuple[list[CanonicalizableNode | None], int],
    ] = field(init=False, repr=False, default_factory=dict)

    _STRING_TO_FACE: ClassVar[dict[str, ModuleFaces]] = {
        **{face.name.lower(): face for face in ModuleFaces},
        **{face.name[0].lower(): face for face in ModuleFaces},
    }
    _MAX_ROTATIONS: ClassVar[int] = 8
    _MAX_PRIORITY: ClassVar[int] = 3

    def __post_init__(self) -> None:
        self._axial_list = [None] * len(self.config.axial_face_order)
        self._radial_list = [None] * len(self.config.radial_face_order)
        self.full_priority = self.config.priority

        mapping = {}
        for i, f in enumerate(self.config.axial_face_order):
            mapping[f] = (self._axial_list, i)
        for i, f in enumerate(self.config.radial_face_order):
            mapping[f] = (self._radial_list, i)
        self._face_to_placement = mapping

    # region dunder methods -----

    def __getitem__(
        self, face: ModuleFaces | str,
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

    # TODO just simple, keep chdlren out
    def __str__(self) -> str:
        raise NotImplementedError

    # TODO just simple, keep children out
    def __repr__(self) -> str:
        raise NotImplementedError

    # endregion

    # region internal helpers -----

    def _update_full_priority(self, child: CanonicalizableNode) -> None:
        """Right now goes through all the parents and updates the priprites."""
        parent = self
        while parent is not None:
            parent.full_priority += child.config.priority
            parent = parent.parent

    def _get_child(self, face: ModuleFaces) -> CanonicalizableNode | None:
        target_list, index = self._face_to_placement[face]
        return target_list[index]

    def _set_child_raw(
        self, face: ModuleFaces, child: CanonicalizableNode,
    ) -> None:
        target_list, index = self._face_to_placement[face]
        target_list[index] = child

    def _set_child(self, face: ModuleFaces, child: CanonicalizableNode) -> None:
        child = child if child.parent is None else child.copy()

        self._update_full_priority(child)
        self._set_child_raw(face, child)

        child.parent = self

        if not self.tree_tags and not self.attach_process_fn:
            return

        def _propagate(c: CanonicalizableNode) -> None:
            if self.tree_tags:
                c.tree_tags = self.tree_tags
            if self.attach_process_fn:
                c.attach_process_fn = self.attach_process_fn
                self.attach_process_fn(c)

        if not child.has_children:
            _propagate(child)
            return

        child.traverse_depth_first(_propagate)

    # endregion

    # region properties ----

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
        return any(child is not None for child in self._axial_list + self._radial_list)

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
        for face, (child_list, index) in self._face_to_placement.items():
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
    def global_rotation_state(self) -> int:
        """Number of times the local rotation has cycled through equivalent global orientations."""
        return self.local_rotation_state // self.config.unique_rotation_amt

    @property
    def priority_face(self) -> ModuleFaces | None:
        """Reveal which face should contain the higest priority."""
        return (
            self.config.radial_face_order[0]
            if self.config.radial_face_order
            else None
        )

    @property
    def highest_priority_child_face(self) -> ModuleFaces | None:
        """Returns the face where the child is attached with the higest priority."""
        if not self.has_children or len(self.config.radial_face_order) == 0:
            return None

        max_priority = 0
        winning_face = None
        for face, child in self.radial_children_items:
            if child.full_priority > max_priority:
                max_priority = child.full_priority
                winning_face = face
        return winning_face

    # endregion

    # region modifyers -----

    def attach_parent(
        self, face: ModuleFaces, parent: CanonicalizableNode,
    ) -> None:
        og_tree_tags = self.tree_tags
        parent._set_child(face, self)
        parent.tree_tags = og_tree_tags

    def detatch_parent(self) -> CanonicalizableNode | None:
        """Returns the parent."""
        parent = self.parent
        if not parent:
            return None

        parent.detatch_child(self.parent_attachment_face)
        return parent

    # TODO: add this for the set_item when setting None?
    def detatch_child(
        self, face: ModuleFaces,
    ) -> CanonicalizableNode | None:
        """Returns the child detatched."""
        child = self[face]
        if not child:
            return None

        priority = child.full_priority
        child.parent = None

        def _priority_remover(n: CanonicalizableNode) -> None:
            n.full_priority -= priority

        self._set_child_raw(face, None)
        self.traverse_through_parents(_priority_remover)
        return child

    def detatch_children(self) -> None:
        priority_to_remove = 0
        for child in self.children:
            child.parent = None
            priority_to_remove += child.full_priority

        def _priority_remover(n: CanonicalizableNode) -> None:
            n.full_priority -= priority_to_remove

        self._radial_list[:] = [None] * len(self._radial_list)
        self._axial_list[:] = [None] * len(self._axial_list)
        self.traverse_through_parents(_priority_remover)

    def rotate_amt(self, delta: int = 1) -> None:
        if self.config.priority == self._MAX_PRIORITY:
            self.local_rotation_state = 0
            return
        self.local_rotation_state = (
            self.local_rotation_state + delta
        ) % CanonicalizableNode._MAX_ROTATIONS

    def shift_radial_children_global(self, shift: int) -> None:
        """Shifts the children w.r.t. world reference frame."""
        if not self._radial_list:
            return
        shift %= len(self._radial_list)
        self._radial_list[:] = (
            self._radial_list[shift:] + self._radial_list[:shift]
        )
        return

    def shift_radial_children_local(self, shift: int) -> None:
        """Shifts the children w.r.t. self, while not chaning the world i.e. changing internal rotation value."""
        rotation_change = shift * self.config.unique_rotation_amt
        self.rotate_amt(-rotation_change)

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

    # TODO future: more user friendly way to get the memory adress of certain node?
    def hop_though_tree(self) -> Never:
        """
        Maybe hop_up (gets the parent)
        hop_down (gets the radial_child)
        hop_radial(cycles though the radial children per hop?)
        idk.
        """
        raise NotImplementedError

    def copy(
        self,
        *,
        node_tags: dict[str, Any] | None = None,
        tree_tags: dict[str, Any] | None = None,
        deep: bool = True,
        copy_children: bool = True,
        empty: bool = False,
    ) -> CanonicalizableNode:
        """Create a copy of this node. If copy_children is True, deep copy all descendants.
        If empty is True, do not copy tags or attach_process_fn.
        """
        new_node = CanonicalizableNode(
            module_type=self.module_type,
            local_rotation_state=self.local_rotation_state,
            parent=None,
            config=self.config,
            node_tags=(
                node_tags or ({}
                if empty
                else (self.node_tags.copy() if deep else self.node_tags))
            ),
            tree_tags=(
                tree_tags or ({}
                if empty
                else (self.tree_tags.copy() if deep else self.tree_tags))
            ),
            attach_process_fn=self.attach_process_fn if not empty else None,
        )
        if copy_children:
            for face, child in self.children_items:
                child_copy = child.copy(
                    node_tags=node_tags,
                    tree_tags=tree_tags,
                    deep=deep,
                    copy_children=True,
                    empty=empty,
                )
                new_node[face] = child_copy

        return new_node

    # TODO; change name to make it clear this is for functions to be aplied
    def traverse_depth_first(
        self,
        visit_fn: Callable[[CanonicalizableNode], Any]
        | list[Callable[[CanonicalizableNode], Any]],
        *,
        pre_order: bool = True,
        post_order: bool = False,
    ) -> None:
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

    # endregion

    # region toolkit importations? -----

    def canonicalize(self, *args: Any, **kwargs: Any) -> CanonicalizableNode:
        from ariel_experiments.characterize.canonical.core.internal.processor import (
            TreeProcessor,
        )
        return TreeProcessor.canonicalize(self, *args, **kwargs)

    # TODO add more??? or not?? what is good design choice?

    # endregion

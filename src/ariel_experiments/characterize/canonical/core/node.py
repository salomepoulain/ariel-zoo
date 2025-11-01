from __future__ import annotations

# Third-party libraries
from dataclasses import dataclass, field

# Standard library
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Mapping

    from networkx import DiGraph

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

    module_type: ModuleType
    local_rotation_state: int
    config: CanonicalConfig

    # useful flags
    parent_attachment_face: ModuleFaces | None = field(init=False, default=None)
    has_children: bool = field(init=False, default=False)
    full_priority: int = field(init=False)

    # references to connections
    parent: CanonicalizableNode | None = field(kw_only=True, default=None, repr=False)
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
    _face_to_placement: (
        Mapping[
            ModuleFaces,
            tuple[list[CanonicalizableNode | None], int],
        ]
    ) = field(init=False, repr=False, default_factory=dict)

    _STRING_TO_FACE: ClassVar[dict[str, ModuleFaces]] = {
        **{face.name.lower(): face for face in ModuleFaces},
        **{face.name[0].lower(): face for face in ModuleFaces},
    }
    _MAX_ROTATIONS: ClassVar[int] = 8
    _MAX_PRIORITY: ClassVar[int] = 3

    # _grammar_tracker = GrammarTracker() # TODO: create dynamic string creator

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

    def __getitem__(self, face: ModuleFaces | str) -> CanonicalizableNode | None:
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

    # endregion

    # region helpers -----

    #TODO: to processor?
    def _update_full_priority(self, child: CanonicalizableNode) -> None:
        """Right now goes through all the parents and updates the priprites."""
        parent = self
        # only upward

        while parent is not None:
            parent.full_priority += child.config.priority
            parent = parent.parent

    #TODO: to processor?
    def _get_child(self, face: ModuleFaces):
        target_list, index = self._face_to_placement[face]
        return target_list[index]

    #TODO: to processor?
    def _set_child_raw(self, face: ModuleFaces, child: CanonicalizableNode) -> None:
        target_list, index = self._face_to_placement[face]
        target_list[index] = child

    #TODO: to processor?
    def _set_child(self, face: ModuleFaces, child: CanonicalizableNode) -> None:
        child = child if child.parent is None else child.copy()

        self._update_full_priority(child)
        self.has_children = True
        self._set_child_raw(face, child)

        child.parent = self
        child.parent_attachment_face = face

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
    def children(self) -> Generator[CanonicalizableNode, None, None]:
        """Yields all non-None children (axial + radial)."""
        # Axial children
        for child in self._axial_list:
            if child is not None:
                yield child
        # Radial children
        for child in self._radial_list:
            if child is not None:
                yield child

    @property
    def children_items(self) -> Generator[tuple[ModuleFaces, CanonicalizableNode], None, None]:
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
    def axial_children_items(self) -> Generator[tuple[ModuleFaces, CanonicalizableNode], None, None]:
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
    def radial_children_items(self) -> Generator[tuple[ModuleFaces, CanonicalizableNode], None, None]:
        """Yields (face, child) tuples for non-None radial children."""
        for i, face in enumerate(self.config.radial_face_order):
            child = self._radial_list[i]
            if child is not None:
                yield face, child

    @property
    def global_rotation_state(self) -> int:
        """Number of times the local rotation has cycled through equivalent global orientations."""
        return self.local_rotation_state // self.config.unique_rotation_amt

    # TODO: might change to be somewhere else
    @property
    def priority_face(self) -> ModuleFaces | None:
        """Reveal which face should contain the higest priority."""
        return self.config.radial_face_order[0] if self.config.radial_face_order else None

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

    # region important functions -----

    def copy_node_single(
        self,
        node_tags: dict[str, Any] | None = None,
        tree_tags: dict[str, Any] | None = None,
        deep: bool = True,
    ) -> CanonicalizableNode:
        return CanonicalizableNode(
            module_type=self.module_type,
            local_rotation_state=self.local_rotation_state,
            parent=None,
            config=self.config,
            node_tags=node_tags
            if node_tags is not None
            else (self.node_tags.copy() if deep else self.node_tags),
            tree_tags=tree_tags
            if tree_tags is not None
            else (self.tree_tags.copy() if deep else self.tree_tags),
            attach_process_fn=self.attach_process_fn,
        )

    def copy(
        self,
        node_tags: dict[str, Any] | None = None,
        tree_tags: dict[str, Any] | None = None,
        deep: bool = True,
    ) -> CanonicalizableNode:
        """Create a deep copy of this node and all its descendants."""
        new_node = self.copy_node_single(
            node_tags=node_tags, tree_tags=tree_tags, deep=deep,
        )

        for face, child in self.children_items:
            child_copy = child.copy(
                node_tags=node_tags, tree_tags=tree_tags, deep=deep,
            )
            new_node[face] = child_copy

        return new_node

    def remove_children(self) -> None:
        priority_to_remove = 0
        for child in self.children:
            priority_to_remove += child.full_priority

        def _priority_remover(n: CanonicalizableNode) -> None:
            n.full_priority -= priority_to_remove

        self._radial_list[:] = [None] * len(self._radial_list)
        self._axial_list[:] = [None] * len(self._axial_list)
        self.traverse_through_parents(_priority_remover)

    # NOTE: be careful of the tree tags and make sure parent doesnt  have parents or children
    # not thoroughly tested for the tags
    def attach_parent(self, face: ModuleFaces, parent: CanonicalizableNode) -> None:
        og_tree_tags = self.tree_tags
        parent._set_child(face, self)
        parent.tree_tags = og_tree_tags

    #TODO: to processor?
    def traverse_depth_first(
        self,
        visit_fn: Callable[[CanonicalizableNode], Any]
        | list[Callable[[CanonicalizableNode], Any]],
        *,
        pre_order: bool = True,
        post_order: bool = False,
    ) -> None:
        """Traverses the tree, applying functions in pre/post order."""
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

    #TODO: to processor?
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

    def rotate_amt(self, delta: int = 1) -> None:
        if self.config.priority == self._MAX_PRIORITY:
            self.local_rotation_state = 0
            return
        self.local_rotation_state = (self.local_rotation_state + delta) % CanonicalizableNode._MAX_ROTATIONS

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

        if self.has_children:
            for idx, child in enumerate(self._radial_list):
                if child:
                    child.rotate_amt(self.config.radial_adjustments[idx][shift])
        return

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

    # region immutable functions

    def canonicalize(self) -> CanonicalizableNode:
        from ariel_experiments.characterize.canonical.core.internal.processor import (
            TreeProcessor,
        )
        return TreeProcessor.canonicalize(self)

    def collect_subtrees(self) -> list[str | nx.DiGraph[Any] | CanonicalizableNode]:
        from ariel_experiments.characterize.canonical.core.internal.processor import (
            TreeProcessor,
        )
        return TreeProcessor.collect_subtrees(self)

    def collect_tree_neighbourhoods(self) -> dict[int, list[str | nx.DiGraph[Any] | CanonicalizableNode]]:
        from ariel_experiments.characterize.canonical.core.internal.processor import (
            TreeProcessor,
        )
        return TreeProcessor.collect_tree_neighbourhoods(self)

    # serealize
    def to_string(self) -> str:
        from ariel_experiments.characterize.canonical.core.internal.serializer import (
            TreeSerializer,
        )
        return TreeSerializer.to_string(self)

    def to_graph(self) -> DiGraph[Any]:
        from ariel_experiments.characterize.canonical.core.internal.serializer import (
            TreeSerializer,
        )
        return TreeSerializer.to_graph(self)

    # endregion

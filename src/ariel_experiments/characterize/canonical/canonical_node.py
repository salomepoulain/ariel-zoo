from __future__ import annotations

# Third-party libraries
from dataclasses import dataclass, field
from itertools import chain

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
class CanonicalNode:
    """The main datastructure class that can create a tree."""

    module_type: ModuleType
    rotation: int
    config: CanonicalConfig

    # needed for canonicalization
    full_priority: int = field(init=False)

    # quickacces useful flags
    parent_attachment_face: ModuleFaces | None = field(init=False, default=None)
    has_children: bool = field(init=False, default=False)

    # references to connections
    parent: CanonicalNode | None = field(kw_only=True, default=None, repr=False)
    axial_list: list[CanonicalNode | None] = field(init=False)
    radial_list: list[CanonicalNode | None] = field(init=False)

    # to customize stuff
    node_tags: dict[str, Any] = field(kw_only=True, default_factory=dict)
    tree_tags: dict[str, Any] = field(kw_only=True, default_factory=dict)
    attach_process_fn: Callable[[CanonicalNode], None] | None = field(
        default=None, kw_only=True, repr=False,
    )

    # quick mapping to obtain the right placement for the child
    _face_to_placement: Mapping[
        ModuleFaces, tuple[list[CanonicalNode | None], int],
    ] = field(init=False, repr=False)

    _MAX_ROTATIONS: ClassVar[int] = field(init=False, default=8, repr=False)

    # _grammar_tracker = GrammarTracker() # TODO: create dynamic string creator

    def __post_init__(self) -> None:     
        self.axial_list = [None] * len(self.config.axial_face_order)
        self.radial_list = [None] * len(self.config.radial_face_order)
        self.full_priority = self.config.priority

        self._face_to_placement = {}
        for i, f in enumerate(self.config.axial_face_order):
            self._face_to_placement[f] = (self.axial_list, i)
        for i, f in enumerate(self.config.radial_face_order):
            self._face_to_placement[f] = (self.radial_list, i)

    # region dunder methods ----- 

    def __getitem__(self, face: ModuleFaces | str) -> CanonicalNode | None:
        """Gets a child from the given face."""
        module_face = self._string_to_moduleface(face)
        target_list, index = self._face_to_placement[module_face]
        return target_list[index]

    def __setitem__(
        self, face: ModuleFaces | str, child: CanonicalNode,
    ) -> None:
        """Sets a child at the given face."""
        module_face = self._string_to_moduleface(face)
        self.attach_child(module_face, child)

    def __str__(self, level: int = 0) -> str:
        """Return formatted string representation of the node tree."""
        indent = "    " * level
        attributes = [
            f"{indent}type={self.module_type.name}",
            f"{indent}rotation={self.rotation}",
        ]

        def _format_faces(face_order: list[ModuleFaces]):
            children = []
            for face in face_order:
                child = self[face.name]
                if child is not None:
                    children.append(
                        f"{indent}  {face.name}: {child.__str__(level + 1)}",
                    )
                else:
                    children.append(f"{indent}  {face.name}: None")
            return ",\n".join(children) or f"{indent}  None"

        attributes.extend((
            f"{indent}axial_side=[\n{_format_faces(self.config.axial_face_order)}\n{indent}]",
            f"{indent}radial_side=[\n{_format_faces(self.config.radial_face_order)}\n{indent}]",
        ))

        return (
            f"{indent}CanonicalNode(\n" + "\n".join(attributes) + f"\n{indent})"
        )

    def __iter__(self) -> Generator[CanonicalNode, None, None]:
        """Iterate over all nodes in the tree (depth-first)."""
        yield self
        for _, child in self.children:
            yield from child

    # endregion

    # region helpers -----

    def _update_full_priority(self, child: CanonicalNode) -> None:
        """Right now goes through all the parents and updates the priprites."""
        parent = self
        # only upward
        while parent is not None:
            parent.full_priority += child.config.priority
            parent = parent.parent

    def _inherit_tree_tags(self, child: CanonicalNode) -> None:
        child.tree_tags = self.tree_tags

    def _string_to_moduleface(self, face: ModuleFaces | str) -> ModuleFaces:
        """Accepts ModuleFaces, full name, or single-letter alias."""
        if isinstance(face, ModuleFaces):
            return face
        face_lower = face.lower()
        for f in chain(
            self.config.axial_face_order, self.config.radial_face_order,
        ):
            if face_lower == f.name[0].lower():
                return f
        for f in chain(
            self.config.axial_face_order, self.config.radial_face_order,
        ):
            if face_lower == f.name.lower():
                return f
        msg = f"Face '{face}' is not valid for module type {self.module_type}"
        raise KeyError(
            msg,
        )
        
    # endregion
    
    # region helpers ----

    @property
    def children(
        self,
    ) -> Generator[tuple[ModuleFaces, CanonicalNode], Any, None]:
        """Yields (face, child) tuples for all non-None children."""
        for face, (child_list, index) in self._face_to_placement.items():
            child = child_list[index]
            if child is not None:
                yield (face, child)

    @property
    def axial_children(self) -> Generator[CanonicalNode, Any, None]:
        """Yields all non-None axial children."""
        for child in self.axial_list:
            if child is not None:
                yield child

    @property
    def radial_children(self) -> Generator[CanonicalNode, Any, None]:
        """Yields all non-None radial children."""
        for child in self.radial_list:
            if child is not None:
                yield child
                
    # endregion

    # region important functions -----

    def copy_node(
        self,
        node_tags: dict[str, Any] | None = None,
        tree_tags: dict[str, Any] | None = None,
    ) -> CanonicalNode:
        return CanonicalNode(
            module_type=self.module_type,
            rotation=self.rotation,
            parent=None,
            config=self.config,
            node_tags=node_tags,
            tree_tags=tree_tags,
        )

    def attach_child(self, face: ModuleFaces, child: CanonicalNode) -> None:
        child_node = child if child.parent is None else child.copy_subtree()

        self._update_full_priority(child_node)
        self.has_children = True
        child_node.parent = self
        child_node.parent_attachment_face = face
        target_list, index = self._face_to_placement[face]
        target_list[index] = child_node

        if not self.tree_tags and not self.attach_process_fn:
            return

        tree_tags: dict[str, Any] = self.tree_tags
        attach_fn = self.attach_process_fn
        child_node.tree_tags = tree_tags
        if attach_fn:
            child_node.attach_process_fn = attach_fn
            attach_fn(child_node)

        if not self.has_children:
            return

        if attach_fn:

            def propagate(n: CanonicalNode) -> None:
                n.tree_tags = tree_tags
                n.attach_process_fn = attach_fn
                attach_fn(n)

        else:

            def propagate(n: CanonicalNode) -> None:
                n.tree_tags = tree_tags

        for _, desc in child_node.children:
            desc.traverse_depth_first(propagate, pre_order=True)

    # NOTE: be careful of the tree tags and make sure parent doesnt  have parents or children
    # not thoroughly tested for the tags
    def attach_parent(self, face: ModuleFaces, parent: CanonicalNode):
        og_tree_tags = self.tree_tags
        parent.attach_child(face, self)
        parent.tree_tags = og_tree_tags
        return self

    def copy_subtree(self) -> CanonicalNode:
        """Create a deep copy of this node and all its descendants."""
        new_node = CanonicalNode(
            module_type=self.module_type,
            rotation=self.rotation,
            parent=None,
            config=self.config,
            tree_tags=self.tree_tags,
            attach_process_fn=self.attach_process_fn,
        )
        for face, child in self.children:
            child_copy = child.copy_subtree()
            new_node[face] = child_copy

        return new_node

    def traverse_depth_first(
        self,
        visit_fn: Callable[[CanonicalNode], Any]
        | list[Callable[[CanonicalNode], Any]],
        *,
        pre_order: bool = True,
        post_order: bool = False,
    ) -> None:
        """Traverses the tree, applying functions in pre/post order."""
        functions = visit_fn if isinstance(visit_fn, list) else [visit_fn]

        if pre_order:
            for fn in functions:
                fn(self)

        for _, c in self.children:
            c.traverse_depth_first(
                visit_fn, pre_order=pre_order, post_order=post_order,
            )

        if post_order:
            for fn in functions:
                fn(self)

    # endregion

    # region future ----- stuff that will update the grammar of the string!

    # unused yet
    def shift_radial_children(self, shift: int | None = None) -> None:
        """Rotate radial_list by shift positions to the right."""
        n = len(self.radial_list)
        if n == 0 or shift == 0:
            return

        if not shift:
            shift = self.config.radial_shift

        n = len(self.radial_list)
        shift %= n
        self.radial_list = self.radial_list[shift:] + self.radial_list[:shift]

    # unused yet
    def add_rotation(self, amt: int = 1) -> None:
        self.rotation = (self.rotation + amt) % CanonicalNode._MAX_ROTATIONS

# from __future__ import annotations

# # Standard library
# from pathlib import Path
# from typing import TYPE_CHECKING

# import networkx as nx

# # Third-party libraries
# import numpy as np
# from rich.console import Console

# # Local libraries

# if TYPE_CHECKING:
#     from collections.abc import Generator

# # Global constants
# # SCRIPT_NAME = __file__.split("/")[-1][:-3]
# SCRIPT_NAME = 'canical_test'
# CWD = Path.cwd()
# DATA = Path(CWD / "__data__" / SCRIPT_NAME)
# DATA.mkdir(exist_ok=True)
# SEED = 42

# # Global functions
# console = Console()
# console.is_jupyter = False
# RNG = np.random.default_rng(SEED)


# from builtins import type as create_class
# from enum import Enum
# from typing import Any

# from ariel.body_phenotypes.robogen_lite.config import (
#     ALLOWED_FACES,
#     ModuleFaces,
#     ModuleInstance,
#     ModuleRotationsIdx,
#     ALLOWED_ROTATIONS,
#     ModuleType,
# )

# # * IT IS SUPER VERY IMPORTANT THAT MODULEFACES ORDER NEVER CHANGES<3


# class DIRECTION(Enum):
#     CLOCKWISE = -1
#     COUNTERCLOCKWISE = 1  # RIGHT TOP LEFT BOTTOM | FRONT RIGHT BACK LEFT


# # TODO: COUULD ALSO ADD START IDX????

# ROTATION_DIRECTION = DIRECTION.COUNTERCLOCKWISE.value


# class SymmetryPlane(Enum):
#     """Symmetry plane determines how many unique rotations a module has."""

#     NONE = 1
#     TWO_FOLD = 2
#     FOUR_FOLD = 4


# ATTACHMENT_FACE = {
#     ModuleType.BRICK: ModuleFaces.BACK,
#     ModuleType.HINGE: ModuleFaces.BACK,
#     ModuleType.NONE: ModuleFaces.BACK,
# }

# SYMMETRY_PLANE: dict[ModuleType, SymmetryPlane] = {
#     ModuleType.CORE: SymmetryPlane.NONE,
#     ModuleType.BRICK: SymmetryPlane.FOUR_FOLD,
#     ModuleType.HINGE: SymmetryPlane.TWO_FOLD,
#     ModuleType.NONE: SymmetryPlane.NONE,
# }

# class Priority(Enum):
#     FIRST = 3
#     SECOND = 2
#     THIRD = 1
#     FOURTH = 0


# PRIORITIES: dict[ModuleType, Priority] = {
#     ModuleType.CORE: Priority.FIRST,
#     ModuleType.BRICK: Priority.SECOND,
#     ModuleType.HINGE: Priority.THIRD,
#     ModuleType.NONE: Priority.FOURTH,
# }


# class CanonicalNode:
#     def __init__(
#         self,
#         id: int,
#         type: ModuleType,
#         rotation: int = 0,
#         *,
#         parent: int | None = None,
#         symmetry_plane: int | None = None,
#         attachment_face: ModuleFaces | None = None,
#         allowed_rotations: list[ModuleRotationsIdx] | None = None,
#         priority: int | None = None
#     ) -> None:
#         """
#         This class assumes the rotation axis is the attachment point!
#         Therefore knowing the attachment point is important
#         if there is no attachment point, the rotation axis will default to top and bottom
#         this can be true while rotation being not able to happen.

#         the class makes dynamic properties (methods) based on the configuration,
#         to acces the children quickly:
#         @property
#         def TOP(self):
#             return self.axial_side[0]

#         @TOP.setter
#         def TOP(self, value):
#             self.axial_side[0] = value

#         """
#         # TODO: jacopo question: how would core rotate? which axis?
#         # * HOW DOES TOP GET POSITIONED? as where does it look towards?, what does left right etc mean?

#         # Compute attachment_face if not provided
#         if attachment_face is None:
#             attachment_face = (
#                 ATTACHMENT_FACE.get(type)
#             )

#         # with xor, the oppposite face may get detected
#         axial_pair = (
#             {attachment_face, ModuleFaces(attachment_face.value ^ 1)}
#             if attachment_face
#             else {ModuleFaces.TOP, ModuleFaces.BOTTOM}
#         )
#         reduced_axial = axial_pair & set(ALLOWED_FACES[type])

#         axial_idx = sorted(reduced_axial, key=lambda f: f.value)
#         radial_side = set(ALLOWED_FACES[type]).difference(axial_pair)

#         radial_idx = (lambda s: s[:1] + s[2:3] + s[1:2] + s[3:])(
#             sorted(radial_side, key=lambda f: f.value),
#         )[::ROTATION_DIRECTION]

#         if symmetry_plane is None:
#             symmetry_plane = SYMMETRY_PLANE.get(
#                 type, SymmetryPlane.NONE,
#             ).value

#         if allowed_rotations is None:
#             allowed_rotations = ALLOWED_ROTATIONS.get(
#                 type, [ModuleRotationsIdx.DEG_0]
#             )

#         if priority is None:
#             priority = PRIORITIES.get(
#                 type, Priority.FOURTH
#             ).value

#         # Max normalized values
#         # Could also be switched to cutoff the list values!
#         max_allowed_rotatations = len(allowed_rotations) // symmetry_plane

#         # Store basic attributes
#         self.id: int = id
#         self.type: ModuleType = type  # frozen TODO
#         self.rotation: int = rotation
#         self.parent: CanonicalNode | None = parent

#         # TODO: do i remove the thing entirely if empty list?
#         self.axial_side: list[None | CanonicalNode] = [None] * len(axial_idx)
#         self.radial_side: list[None | CanonicalNode] = [None] * len(radial_idx)

#         # self.axis_face = axis
#         # self._symmetry_plane: int = symmetry_plane

#         self._max_allowed_rotatations = max_allowed_rotatations
#         self._priority = priority
#         self._full_priority = priority

#         # console.print(type, max_allowed_rotatations)

#         # self._allowed_rotate: bool =
#         self._axial_faces: list[ModuleFaces] = axial_idx
#         self._radial_faces: list[ModuleFaces] = radial_idx

#         properties_dict = {}

#         def make_property(index: int, side_list: str):
#             """Factory to create a property for a face (axial or radial)."""

#             def getter(self: CanonicalNode):
#                 return getattr(self, side_list)[index]

#             def setter(self: CanonicalNode, value: CanonicalNode) -> None:
#                 value.parent = self
#                 getattr(self, side_list)[index] = value

#                 def ancestors(node: CanonicalNode):
#                     while node:
#                         yield node
#                         node = node.parent

#                 for ancestor in ancestors(self):
#                     ancestor._full_priority += value._priority


#             return property(getter, setter)

#         # Example usage of the local function
#         properties_dict = {}
#         for idx, face in enumerate(axial_idx):
#             properties_dict[face.name] = make_property(idx, "axial_side")

#         for idx, face in enumerate(radial_idx):
#             properties_dict[face.name] = make_property(idx, "radial_side")

#         # Dynamically create the class
#         NewClass = create_class(
#             f"Canonical{type.name}", (self.__class__,), properties_dict,
#         )
#         self.__class__ = NewClass

#     # TODO: make use of the rich tree print thing
#     def __str__(self, level: int = 0) -> str:
#         """
#         Return a formatted string representation of the CanonicalNode object.

#         Parameters
#         ----------
#         level : int, default 0
#             Indentation level for nested string formatting.

#         Returns
#         -------
#         str
#             Multi-line formatted string showing node attributes and children.

#         Notes
#         -----
#         - Creates hierarchical indentation using 2 spaces per level
#         - Shows axial_side children only for faces in ALLOWED_FACES[self.type]
#         - Shows radial_side children for all faces in ALLOWED_FACES[self.type]
#         - Displays 'None' for missing child nodes at each face position
#         """
#         indent = "    " * level

#         # Start with basic attributes
#         attributes = [
#             f"{indent}id={self.id}",
#             f"{indent}type={self.type.name}",
#             f"{indent}rotation={self.rotation}",
#             # f"{indent}symmetry_plane={self._symmetry_plane}",
#         ]

#         # Add axial children
#         axial_children = ",\n".join(
#             f"{indent}  {face.name}: {getattr(self, face.name).__str__(level + 1)}"
#             if getattr(self, face.name) is not None
#             else f"{indent}  {face.name}: None"
#             for face in ALLOWED_FACES[self.type]
#             if face in self.axial_side
#         )
#         attributes.append(
#             f"{indent}axial_side=[\n{axial_children or f'{indent}  None'}\n{indent}]",
#         )

#         # Add radial children
#         radial_children = ",\n".join(
#             f"{indent}  {face.name}: {getattr(self, face.name).__str__(level + 1)}"
#             if getattr(self, face.name) is not None
#             else f"{indent}  {face.name}: None"
#             for face in ALLOWED_FACES[self.type]
#         )
#         attributes.append(
#             f"{indent}radial_side=[\n{radial_children or f'{indent}  None'}\n{indent}]",
#         )

#         # Format the final string iteratively
#         formatted_attributes = "\n".join(attributes)
#         return f"{indent}CanonicalNode(\n{formatted_attributes}\n{indent})"

#     def __repr__(self) -> str:
#         """
#         Return a string representation of the object.

#         Returns
#         -------
#         str
#             A compact string representation showing type, rotation, and children.

#         Notes
#         -----
#         - Format: "{type_initial}{rotation}[{face_children}]"
#         - Type initial is first character of self.type.name
#         - Rotation only included if non-zero
#         - Face children shown as "{face_initial}:{child_repr}"
#         - Empty brackets omitted if no children exist
#         """
#         # Start with the node's type and conditionally include rotation if it's not 0
#         repr_str = f"{self.type.name[0]}"
#         if self.rotation != 0:
#             repr_str += f"{self.rotation}"
#         repr_str += "{"

#         # Collect children dynamically based on ALLOWED_FACES
#         children_repr = []
#         for face in ALLOWED_FACES[self.type]:
#             child = getattr(self, face.name)
#             if child:
#                 children_repr.append(f"{face.name.lower()[0]}:{child!r}")

#         # Join children without trailing commas
#         repr_str += "".join(children_repr)

#         # Close the representation
#         repr_str += "}" #if children_repr else ""
#         return repr_str

#     @property
#     def child(self):
#         class ChildAccessor:
#             def __init__(self, parent) -> None:
#                 self.parent = parent

#             def __getitem__(self, face: ModuleFaces):
#                 return getattr(self.parent, face.name)

#             def __setitem__(self, face: ModuleFaces, child: CanonicalNode) -> None:
#                 setattr(self.parent, face.name, child)

#         return ChildAccessor(self)

#     #TODO: wrong
#     @property
#     def children(self):
#         for idx, face in enumerate(self._axial_faces):
#             child = self.axial_side[idx]
#             if child is not None:
#                 yield (face, child)

#         for idx, face in enumerate(self._radial_faces):
#             child = self.radial_side[idx]
#             if child is not None:
#                 yield (face, child)

#     @property
#     def axial_children(self):
#         for child in self.axial_side:
#             if child is not None:
#                 yield child

#     @property
#     def radial_children(self):
#         for child in self.radial_side:
#             if child is not None:
#                 yield child

#     @property
#     def children_nodes(self) -> Generator[CanonicalNode, None, None]:
#         yield from (c for c in self.axial_side if c is not None)
#         yield from (c for c in self.radial_side if c is not None)

#     @classmethod
#     def build_tree_from_dict(
#         cls, data: dict[int, ModuleInstance], omit_none: bool = False,
#     ) -> CanonicalNode:
#         root = CanonicalNode(id=0, type=ModuleType.CORE, rotation=0)

#         def fill_in(data: dict[int, ModuleType], parent: CanonicalNode) -> None:
#             for face, id in data[parent.id].links.items():
#                 if omit_none and data[id].type == ModuleType.NONE:
#                     continue
#                 child = CanonicalNode(
#                     id=id,
#                     type=data[id].type,
#                     # parent_id=parent.id,
#                     rotation=data[id].rotation,
#                 )
#                 parent.child[face] = child
#                 fill_in(data, child)

#         fill_in(data, root)
#         return root

#     def to_dict(self, omit_none: bool = False) -> dict[int, ModuleInstance]:
#         """Convert the CanonicalNode tree back into a dictionary of ModuleInstance objects."""
#         result = {}

#         def collect_node(node: CanonicalNode) -> None:
#             if omit_none and node.type == ModuleType.NONE:
#                 return

#             links = {}
#             for face, child in node.children:
#                 if not omit_none or child.type != ModuleType.NONE:
#                     links[face] = child.id

#             result[node.id] = ModuleInstance(
#                 type=node.type,
#                 rotation=ModuleRotationsIdx(node.rotation),
#                 links=links,
#             )

#             for child in node.children_nodes:
#                 collect_node(child)

#         collect_node(self)
#         return result

#     # TODO
#     @classmethod
#     def from_graph(
#         cls, graph: nx.DiGraph, skip_type: ModuleType = ModuleType.NONE,
#     ) -> CanonicalNode:
#         """Build a CanonicalNode tree from a NetworkX graph."""

#         def get_rotation_value(rotation_attr):
#             return (
#                 ModuleRotationsIdx[rotation_attr].value
#                 if isinstance(rotation_attr, str)
#                 else rotation_attr
#             )

#         def create_node(
#             node_id: int,
#         ) -> CanonicalNode:
#             attrs = graph.nodes[node_id]
#             return CanonicalNode(
#                 id=node_id,
#                 type=ModuleType[attrs["type"]],
#                 rotation=get_rotation_value(attrs["rotation"]),
#                 # parent_id=parent.id,
#             )

#         def fill_in(parent: CanonicalNode) -> None:
#             for _, child_id, edge_data in graph.out_edges(parent.id, data=True):
#                 child_type = ModuleType[graph.nodes[child_id]["type"]]

#                 if child_type == skip_type:
#                     continue

#                 child = create_node(child_id)
#                 parent.child[ModuleFaces[edge_data["face"]]] = child
#                 fill_in(child)

#         root_id = next(
#             node for node in graph.nodes() if graph.in_degree(node) == 0
#         )
#         root = create_node(root_id)
#         fill_in(root)

#         return root

#     def to_graph(self, skip_type: ModuleType = ModuleType.NONE) -> nx.DiGraph:
#         """Generate a NetworkX graph from the CanonicalNode tree."""
#         graph = nx.DiGraph()

#         def traverse_node(node: CanonicalNode) -> None:
#             graph.add_node(
#                 node.id,
#                 type=node.type.name,
#                 rotation=ModuleRotationsIdx(node.rotation).name,
#             )

#             for face, child in node.children:
#                 if child.type == skip_type:
#                     print("found skyptype")
#                     continue

#                 graph.add_node(
#                     child.id,
#                     type=child.type.name,
#                     rotation=ModuleRotationsIdx(child.rotation).name,
#                 )
#                 graph.add_edge(
#                     node.id,
#                     child.id,
#                     face=face.name,
#                 )
#                 traverse_node(child)

#         traverse_node(self)
#         return graph

#     # TODO: so wrong. make cumulative1!!!1
#     # def trickle_down_rotations(self, parent_rotation: int = 0) -> None:
#     #     if parent_rotation != 0:
#     #         console.rule()
#     #         console.print("adding rotation ", parent_rotation)
#     #         console.print(self.rotation)

#     #     self.rotation += parent_rotation

#     #     console.print(self.rotation)

#     #     for child in self.radial_side:
#     #         if child is not None:
#     #             child.trickle_down_rotations(parent_rotation=0)

#     #     for child in self.axial_side:
#     #         if child is not None:
#     #             child.trickle_down_rotations(parent_rotation=self.rotation)

#     def re_rotate_down(self, amt: int = 0):
#         self.rotation += amt
#         amt = 0 - self.rotation
#         # self.normalize_single_rotation(amt)

#         for child in self.axial_children:
#             child.re_rotate_down(amt)

#         for child in self.radial_children:
#             child.re_rotate_down(0)

#     def normalize_single_rotation(self, amt: int = 0):
#         self.rotation = (self.rotation + amt) % self._max_allowed_rotatations

#     def add_rotations_down(self, amt: int = 0) -> None:
#         console.print('rotation, amt, id', self.rotation, amt, self.id)


#         self.rotation += amt

#         for child in self.axial_children:
#             child.add_rotations_down(amt)

#         for child in self.radial_children:
#             child.add_rotations_down(0)

#     def invert_symmetry(self) -> None:
#         pass

#     def canonicalize_child_order(self) -> int:
#         """
#         uses the full priority to determine the child order
#         always sets the max at first
#         rotatation always in 1 direction
#         """
#         if len(self.radial_side) == 0:
#             return 0

#         print(len(self.radial_side))
#         priority_list = []
#         for child in self.radial_side:
#             if child is None:
#                 priority_list.append(0)
#             else:
#                 priority_list.append(child._full_priority)

#         max_index = priority_list.index(max(priority_list))
#         self.radial_side = self.radial_side[max_index:] + self.radial_side[:max_index]
#         return max_index

#         # # Compute subtree sizes and sort keys
#         # child_keys = []
#         # for child in node.children:
#         #     size = canonicalize_child_order(child)  # Recurse first for sizes
#         #     priority = {'brick': 1, 'hinge': 2}.get(child.type, 3)
#         #     child_keys.append((priority, -size, child))  # Tuple for sorting

#         # # Sort children: primary type, secondary descending size
#         # child_keys.sort(key=lambda x: (x[0], x[1]))
#         # node.children = [k[2] for k in child_keys]  # Reassign sorted children

#         # return 1 + sum(k[1] for k in child_keys)


#     # def re_map_top(self, amt: int = 0) -> None:
#     #     # console.print(f'trickle looked at {self.id},{self.type},{self.rotation}')
#     #     self.rotation += amt
#     #     current_rotation = self.rotation
#     #     # console.print(f'modified {self.id},{self.type},{self.rotation}')


#     #     for child in self.axial_children:
#     #         # console.print('')
#     #         # child.rotation += current_rotation
#     #         child.trickle_down_rotations(amt)

#     #     for child in self.radial_children:
#     #         child.trickle_down_rotations(0)

#     def normalize_one_rotation(self, amt: int = 0):
#         if self.radial_children:
#             self.radial_side[:] = (
#                 self.radial_side[amt:] + self.radial_side[:amt]
#             )

#     def normalize_rotations(self) -> None:
#         # if should_normalize:
#         before = self.rotation
#         self.rotation = self.rotation % self._max_allowed_rotatations
#         amt = self.rotation - before
#         # console.print(f'normalized looked at {self.id},{self.type},{self.rotation}')

#         if self.radial_children:
#             self.radial_side[:] = (
#                 self.radial_side[amt:] + self.radial_side[:amt]
#             )

#         for _, child in self.children:
#             child.normalize_rotations()

#     # def depth_first_collect(self) -> list[nx.DiGraph[Any]]:
#     #     root = self

#     #     list_graph = []

#     #     # make root node start at 0
#     #     amt = 0 - root.rotation
#     #     self.trickle_down_rotations(amt)
#     #     # account for the change in direction
#     #     self.normalize_rotations()

#     #     list_graph.append(self.to_graph)

#     #     return list_graph

#     def re_add_id(self, start_id: int = 0) -> int:
#         self.id = start_id
#         current_id = start_id + 1
#         for child in self.children_nodes:
#             current_id = child.re_add_id(current_id)
#         return current_id

#     def depth_first_collect(self) -> list[nx.DiGraph[Any]]:
#         list_graph = []

#         # amt = 0 - self.rotation
#         rotation = self.canonicalize_child_order()
#         self.add_rotations_down(rotation)
#         self.rotation = 0
#         # self.normalize_rotations()
#         self.normalize_rotations()


#         console.print(repr(self))

#         self.re_add_id()

#         console.print("id, rotation", self.id, self.rotation, self._priority, self._full_priority)


#         # re-add ids
#         list_graph.append(self.to_graph())


#         # Recurse on children
#         for child in self.children_nodes:
#             list_graph.extend(child.depth_first_collect())

#         return list_graph


#         # for child in self.axial_children:
#         #     child.normalize_rotations()

#         # for child in self.radial_children:
#         #     # self.rotation = self.rotation % self._max_allowed_rotatations
#         #     child.normalize_rotations()


#     # TODO doesnt work
#     # def normalize_rotations(self) -> None:
#     #     """Normalize rotation to valid range based on symmetry plane."""
#     #     before = self.rotation
#     #     self.rotation %= self._symmetry_plane
#     #     change = self.rotation - before
#     #     self._remap_radial_children(change)
#     #     self._apply_rotation_to_axial(change)

#     # def _remap_radial_children(self, amt: int) -> None:
#     #     """Rotate the radial children list by amt positions."""
#     #     if self.radial_side:
#     #         amt %= len(self.radial_side)
#     #         self.radial_side[:] = (
#     #             self.radial_side[amt:] + self.radial_side[:amt]
#     #         )

#     # def _apply_rotation_to_axial(self, amt: int) -> None:
#     #     """Apply rotation change to axial children and recursively normalize."""
#     #     for child in self.axial_side:
#     #         if child is not None:
#     #             child.rotation += amt
#     #             child.normalize_rotations()

#     # def normalize_rotations(self):
#     #     # recursive (kinda already implemented)

#     #     pass

#     # def canonicalize_down(self) -> None:
#     #     # trikle down rotations
#     #     # normalize rotations
#     #     pass

#     # def apply_new_ids(self) -> None:
#     #     pass

#     # def get_normalized_string(self) -> None:
#     #     # might just be the dict?
#     #     pass

#     # def return_canonical_parts(self) -> dict[int, Any] | list[Any]:
#     #     # return dict or list
#     #     pass

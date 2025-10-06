from __future__ import annotations

from typing import Optional, Callable, Any
from enum import Enum
from collections import deque
from builtins import type as create_class

from enum import Enum, IntEnum

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    # ALLOWED_ROTATIONS,
    ModuleType,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleInstance
)

# @dataclass(frozen=True)
# class ModuleAxis():
#     {FRONT, BACK}
#     {RIGHT, LEFT}
#     {TOP, BOTTOM}
    
#     @property
#     def opposite(self) -> 'ModuleFaces':
#         """Get opposite face by XOR with 1 (pairs: 0-1, 2-3, 4-5)"""
#         return ModuleFaces(self.value ^ 1)

# # * IT IS SUPER VERY IMPORTANT THAT THIS ORDER NEVER CHANGES<3
# class ModuleFaces(IntEnum):
#     """Enum for module attachment points."""
    # FRONT = 0
    # BACK = 1
    # RIGHT = 2
    # LEFT = 3
    # TOP = 4
    # BOTTOM = 5
    
#     @property
#     def opposite(self) -> 'ModuleFaces':
#         """Get opposite face by XOR with 1 (pairs: 0-1, 2-3, 4-5)"""
#         return ModuleFaces(self.value ^ 1)



# ADD CONNECTION POINT

class DIRECTION(Enum):
    CLOCKWISE = -1
    COUNTERCLOCKWISE = 1 # RIGHT TOP LEFT BOTTOM | FRONT RIGHT BACK LEFT

#TODO: COUULD ALSO ADD START IDX????

ROTATION_DIRECTION = DIRECTION.COUNTERCLOCKWISE.value

class SymmetryPlane(Enum):
    """Symmetry plane determines how many unique rotations a module has."""
    NONE = 1
    TWO_FOLD = 2
    FOUR_FOLD = 4

ATTACHMENT_FACE = {
    ModuleType.BRICK: ModuleFaces.BACK,
    ModuleType.HINGE: ModuleFaces.BACK,
    ModuleType.NONE: ModuleFaces.BACK,
}

SYMMETRY_PLANE: dict[ModuleType, SymmetryPlane] = {
    ModuleType.CORE: SymmetryPlane.FOUR_FOLD,
    ModuleType.BRICK: SymmetryPlane.FOUR_FOLD,
    ModuleType.HINGE: SymmetryPlane.TWO_FOLD,
    ModuleType.NONE: SymmetryPlane.NONE,
}


class CanonicalNode:
    def __init__(
        self,
        id: int,
        type: ModuleType,
        rotation: int = 0,
        parent_id: int | None = None,
        symmetry_plane: int | None = None,
        attachment_face: ModuleFaces | None = None,
    ):
        """
        This class assumes the rotation axis is the attachment point!
        Therefore knowing the attachment point is important
        if there is no attachment point, the rotation axis will default to top and bottom
        this can be true while rotation being not able to happen
        
        the class makes dynamic properties (methods) based on the configuration,
        to acces the children quickly:
        @property
        def TOP(self):
            return self.axial_side[0]

        @TOP.setter
        def TOP(self, value):
            self.axial_side[0] = value
        
        """
        # TODO: jacopo question: how would core rotate? which axis?

        # Compute attachment_face if not provided
        if attachment_face is None:
            attachment_face = (
                ATTACHMENT_FACE[type]
                if type in ATTACHMENT_FACE
                else None
            )
        
        # Determine the radial faces, and the ones that can attach chidlren
        # with xor, the oppposite face may get detected
        axial_pair = {attachment_face, ModuleFaces(attachment_face.value ^ 1)} if attachment_face else {ModuleFaces.TOP, ModuleFaces.BOTTOM}
        reduced_axial = axial_pair & set(ALLOWED_FACES[type])
        
        # print(set(ALLOWED_FACES[type]))
        
        axial_idx = (sorted(reduced_axial, key=lambda f: f.value))   
        
        # print(axial_pair)
        # print(ALLOWED_FACES[type])
        # print(reduced_axial)
        print("axial idx for: ", type,  axial_idx)
        
        # the radial sides are the allowed faces - axial_pair
        radial_side = set(ALLOWED_FACES[type]).difference(axial_pair)
        
        radial_idx = (lambda s: s[:1] + s[2:3] + s[1:2] + s[3:])(sorted(radial_side, key=lambda f: f.value))[::ROTATION_DIRECTION]   
        
        print("radial idx for: ", type, radial_idx)
        
        # print(radial_idx)
            
        if symmetry_plane is None:
            symmetry_plane = SYMMETRY_PLANE.get(type, SymmetryPlane.NONE).value  # fallback to NONE if not found
        
        # Store basic attributes
        self.id: int = id
        self.type: ModuleType = type # frozen
        self.rotation: int = rotation
        self.parent_id: int = parent_id
        
        
        # todo: do i remove the thing entirely if empty list?
        self.axial_side: list[None | CanonicalNode] = [None] * len(axial_idx) 
        self.radial_side: list[None | CanonicalNode] = [None] * len(radial_idx) 
                
        # self.axis_face = axis
        self.symmetry_plane = symmetry_plane

        # Store mappings for lookups
        # axial_idx  #todo
        # radial_idx  #todo
        
        # Create dynamic properties for all faces
        properties_dict = {}
        
        # TODO: only add the non-attachment point!
        
        # Properties for axial faces (index into axial_children)
        # properties_dict[axis.name] = self._make_axial_property(0)
        # properties_dict[axis.opposite.name] = self._make_axial_property(1)
        
        # Local function to create a property for a face
        def make_property(index: int, side_list: str):
            """Factory to create a property for a face (axial or radial)."""
            def getter(self: 'CanonicalNode'):
                return getattr(self, side_list)[index]
            
            def setter(self: 'CanonicalNode', value: 'CanonicalNode'):
                value.parent_id = self.id
                getattr(self, side_list)[index] = value
              
            
            return property(getter, setter)

        # Example usage of the local function
        properties_dict = {}
        for idx, face in enumerate(axial_idx):
            properties_dict[face.name] = make_property(idx, "axial_side")
        
        for idx, face in enumerate(radial_idx):
            properties_dict[face.name] = make_property(idx, "radial_side")

        # Dynamically create the class
        NewClass = create_class(
            f'Canonical{type.name}',
            (self.__class__,),
            properties_dict
        )
        self.__class__ = NewClass
    
    # TODO: make use of the rich tree print thing            
    def __str__(self, level: int = 0):
        """
        Return a formatted string representation of the CanonicalNode object.
        
        Parameters
        ----------
        level : int, default 0
            Indentation level for nested string formatting.
        
        Returns
        -------
        str
            Multi-line formatted string showing node attributes and children.
        
        Notes
        -----
        - Creates hierarchical indentation using 2 spaces per level
        - Shows axial_side children only for faces in ALLOWED_FACES[self.type]
        - Shows radial_side children for all faces in ALLOWED_FACES[self.type]
        - Displays 'None' for missing child nodes at each face position
        """
        indent = "    " * level

        # Start with basic attributes
        attributes = [
            f"{indent}id={self.id}",
            f"{indent}type={self.type.name}",
            f"{indent}rotation={self.rotation}",
            f"{indent}symmetry_plane={self.symmetry_plane}",
        ]

        # Add axial children
        axial_children = ",\n".join(
            f"{indent}  {face.name}: {getattr(self, face.name).__str__(level + 1)}"
            if getattr(self, face.name) is not None else f"{indent}  {face.name}: None"
            for face in ALLOWED_FACES[self.type]
            if face in self.axial_side
        )
        attributes.append(f"{indent}axial_side=[\n{axial_children if axial_children else f'{indent}  None'}\n{indent}]")

        # Add radial children
        radial_children = ",\n".join(
            f"{indent}  {face.name}: {getattr(self, face.name).__str__(level + 1)}"
            if getattr(self, face.name) is not None else f"{indent}  {face.name}: None"
            for face in ALLOWED_FACES[self.type]
        )
        attributes.append(f"{indent}radial_side=[\n{radial_children if radial_children else f'{indent}  None'}\n{indent}]")

        # Format the final string iteratively
        formatted_attributes = "\n".join(attributes)
        return f"{indent}CanonicalNode(\n{formatted_attributes}\n{indent})"

            
    def __repr__(self):
        """
        Return a string representation of the object.
        
        Returns
        -------
        str
            A compact string representation showing type, rotation, and children.
        
        Notes
        -----
        - Format: "{type_initial}{rotation}[{face_children}]"
        - Type initial is first character of self.type.name
        - Rotation only included if non-zero
        - Face children shown as "{face_initial}:{child_repr}"
        - Empty brackets omitted if no children exist
        """
        # Start with the node's type and conditionally include rotation if it's not 0
        repr_str = f"{self.type.name[0]}"
        if self.rotation != 0:
            repr_str += f"{self.rotation}"
        repr_str += "["

        # Collect children dynamically based on ALLOWED_FACES
        children_repr = []
        for face in ALLOWED_FACES[self.type]:
            child = getattr(self, face.name)
            if child:
                children_repr.append(f"{face.name.lower()[0]}:{repr(child)}")

        # Join children without trailing commas
        repr_str += "".join(children_repr)

        # Close the representation
        repr_str += "]" if children_repr else ""
        return repr_str

    @classmethod
    def from_dict(cls) -> CanonicalNode:
        pass
        
    def to_dict(self):
        pass
    
    #todo
    @classmethod
    def from_graph(cls) -> CanonicalNode:
        pass
    
    #todo
    def to_graph(self):
        pass
    
    def plot_tree_from_node(self):
        # from the current node, create networkx graph, plot tree
        pass
    
    def trickle_down_rotations(self):
        # recursive
        # go through front children, add the rotation
        # for each side child, repeat
        pass
    
    def normalize_rotations(self):
        # recursive (kinda already implemented)
        pass

    def canonicalize_down(self):
        # trikle down rotations
        # normalize rotations
        pass
    
    def apply_new_ids(self):
        pass
    
    def get_normalized_string(self):
        # might just be the dict?
        pass

    def return_canonical_parts(self) -> dict[int, Any] | list[Any]:
        # return dict or list
        pass



    # @staticmethod
    # def _make_axial_property(index: int):
    #     """Factory to create a property for an axial face"""
    #     def getter(self):
    #         return self.axial_side[index]
        
    #     def setter(self, value):
    #         self.axial_side[index] = value
    #         if value is not None:
    #             value.parent_id = self.id
        
    #     return property(getter, setter)
    
    # @staticmethod
    # def _make_side_property(index: int):
    #     """Factory to create a property for a rotating side face"""
    #     def getter(self):
    #         return self.radial_side[index]
        
    #     def setter(self, value):
    #         self.radial_side[index] = value
    #         if value is not None:
    #             value.parent = self
    #             # attachment_face will be set based on current rotation
        
    #     return property(getter, setter)
    
    # def get_child_at_face(self, face: ModuleFaces) -> CanonicalNode | None:
    #     """Get child attached at a specific face."""
    #     if face in self._axial_faces:
    #         idx = self._axial_faces.index(face)
    #         return self.axial_side[idx]
    #     elif face in self._face_to_index:
    #         return self.radial_side[self._face_to_index[face]]
    #     return None
    
    # def set_child_at_face(self, face: ModuleFaces, child: Optional[CanonicalNode]):
    #     """Set child at a specific face."""
    #     if face in self._axial_faces:
    #         idx = self._axial_faces.index(face)
    #         self.axial_side[idx] = child
    #     elif face in self._face_to_index:
    #         self.radial_side[self._face_to_index[face]] = child
    #     else:
    #         raise ValueError(f"Face {face} not allowed for {self.type}")
        
    #     if child:
    #         child.parent = self
    #         child.attachment_face = face
    
    
    
    
    # def normalize_rotation(self):
    #     """
    #     Normalize rotation to valid range based on symmetry plane.
    #     This also rotates the children list to maintain consistency.
    #     """
    #     max_rotations = len(ModuleRotationsIdx) // self.symmetry_plane
    #     rotation_change = -(self.rotation // max_rotations) * max_rotations
        
    #     if rotation_change != 0:
    #         self.rotation = self.rotation % max_rotations
    #         # Rotate children positions
    #         rotation_steps = rotation_change % len(self.children)
    #         if rotation_steps != 0:
    #             self.children = (
    #                 self.children[rotation_steps:] + 
    #                 self.children[:rotation_steps]
    #             )
    
    # def rotate(self, amount: int):
    #     """
    #     Rotate this node by amount steps.
    #     Returns the actual rotation change applied.
    #     """
    #     before = self.rotation
    #     max_rotations = len(ModuleRotationsIdx) // self.symmetry_plane
    #     self.rotation = (self.rotation + amount) % max_rotations
        
    #     actual_change = self.rotation - before
        
    #     # Rotate children list
    #     if len(self.children) > 0:
    #         rotation_steps = actual_change % len(self.children)
    #         if rotation_steps != 0:
    #             self.children = (
    #                 self.children[rotation_steps:] + 
    #                 self.children[:rotation_steps]
    #             )
        
    #     return actual_change
    
    # def depth_first_traversal(self, visit: Callable[[CanonicalNode], None]):
    #     """Depth-first traversal (pre-order)."""
    #     visit(self)
    #     # Visit axial children
    #     for child in self.axial_side:
    #         if child:
    #             child.depth_first_traversal(visit)
    #     # Visit rotating children
    #     for child in self.radial_side:
    #         if child:
    #             child.depth_first_traversal(visit)
    
    # def depth_first_traversal_post_order(self, visit: Callable[[CanonicalNode], None]):
    #     """Depth-first traversal (post-order, from leaves up)."""
    #     # Visit axial children first
    #     for child in self.axial_side:
    #         if child:
    #             child.depth_first_traversal_post_order(visit)
    #     # Visit rotating children
    #     for child in self.radial_side:
    #         if child:
    #             child.depth_first_traversal_post_order(visit)
    #     visit(self)
    
    # def breadth_first_traversal(self, visit: Callable[[CanonicalNode], None]):
    #     """Breadth-first traversal."""
    #     queue = deque([self])
    #     while queue:
    #         node = queue.popleft()
    #         visit(node)
    #         # Add axial children
    #         for child in node.axial_children:
    #             if child:
    #                 queue.append(child)
    #         # Add rotating children
    #         for child in node.sides:
    #             if child:
    #                 queue.append(child)
    
    # @classmethod
    # def from_module_dict(cls, modules: dict[int, ModuleInstance]) -> CanonicalNode:
    #     """
    #     Build canonical tree from a dictionary of ModuleInstances.
    #     Assumes module 0 is always the root.
    #     """
    #     if 0 not in modules:
    #         raise ValueError("Module dictionary must contain root node with id=0")
        
    #     # Create all nodes first
    #     nodes: dict[int, CanonicalNode] = {}
    #     for node_id, module in modules.items():
    #         nodes[node_id] = cls(
    #             id=node_id,
    #             type=module.type,
    #             rotation=module.rotation.value
    #         )
        
    #     # Link nodes based on connections
    #     for node_id, module in modules.items():
    #         parent_node = nodes[node_id]
    #         for face, child_id in module.links.items():
    #             if child_id in nodes:
    #                 parent_node.set_child_at_face(face, nodes[child_id])
        
    #     return nodes[0]
    
    # def to_module_dict(self) -> dict[int, ModuleInstance]:
    #     """Convert canonical tree back to ModuleInstance dictionary."""
    #     result = {}
        
    #     def collect_node(node: CanonicalNode):
    #         links = {}
            
    #         # Add axial children
    #         for face, child in zip(node._axial_faces, node.axial_children):
    #             if child:
    #                 links[face] = child.id
            
    #         # Add rotating children
    #         for face, child in node._face_to_index.items():
    #             if node.sides[child]:
    #                 links[face] = node.sides[child].id
            
    #         result[node.id] = ModuleInstance(
    #             type=node.type,
    #             rotation=ModuleRotationsIdx(node.rotation),
    #             links=links
    #         )
        
    #     self.depth_first_traversal(collect_node)
    #     return result
    
    # def to_networkx(self):
    #     """Convert to NetworkX directed graph."""
    #     import networkx as nx
        
    #     graph = nx.DiGraph()
        
    #     def add_to_graph(node: CanonicalNode):
    #         graph.add_node(
    #             node.id,
    #             type=node.type.name,
    #             rotation=ModuleRotationsIdx(node.rotation).name
    #         )
            
    #         # Add axial children
    #         for face, child in zip(node._axial_faces, node.axial_children):
    #             if child:
    #                 graph.add_node(
    #                     child.id,
    #                     type=child.type.name,
    #                     rotation=ModuleRotationsIdx(child.rotation).name
    #                 )
    #                 graph.add_edge(node.id, child.id, face=face.name)
            
    #         # Add rotating children
    #         for face, idx in node._face_to_index.items():
    #             child = node.sides[idx]
    #             if child:
    #                 graph.add_node(
    #                     child.id,
    #                     type=child.type.name,
    #                     rotation=ModuleRotationsIdx(child.rotation).name
    #                 )
    #                 graph.add_edge(node.id, child.id, face=face.name)
        
    #     self.depth_first_traversal(add_to_graph)
    #     return graph
    
    # def __repr__(self, level=0) -> str:
    #     indent = "  " * level
        
    #     # Format axial children
    #     axial_repr = []
    #     for face, child in zip(self._axial_faces, self.axial_side):
    #         if child:
    #             axial_repr.append(f"{indent}  {face.name}: {child.__repr__(level + 1)}")
    #         else:
    #             axial_repr.append(f"{indent}  {face.name}: None")
        
    #     # Format rotating children
    #     sides_repr = []
    #     for face, idx in self._face_to_index.items():
    #         child = self.radial_side[idx]
    #         if child:
    #             sides_repr.append(f"{indent}  {face.name}: {child.__repr__(level + 1)}")
    #         else:
    #             sides_repr.append(f"{indent}  {face.name}: None")
        
    #     all_children = "\n".join(axial_repr + sides_repr)
        
    #     return (
    #         f"{indent}CanonicalNode(\n"
    #         f"{indent}  id={self.id}, type={self.module_type.name}, "
    #         f"rotation={self.rotation}, symmetry={self.symmetry_plane},\n"
    #         f"{all_children}\n"
    #         f"{indent})"
    #     )


# Example usage
if __name__ == "__main__":
    # Create a simple tree manually using dynamic properties
    root = CanonicalNode(id=0, type=ModuleType.CORE, rotation=0)
    
    # print(root.axial_side)
    # print(root.radial_side)
       
    None1 = CanonicalNode(id=1, type=ModuleType.NONE, rotation=0)
    
    print("---------------------------------------")
    print(None1)
    
    # print(None1.BOTTOM)
       
    brick1 = CanonicalNode(id=1, type=ModuleType.BRICK, rotation=2)
    
    root.TOP = brick1
    
    print("***********************************")
    print(brick1.FRONT)
    print(root)
    print(root.axial_side)
    print(root.radial_side)
    
    print('direct now')
    print(root.BOTTOM)
    print(root.TOP)
    
    # top = root.TOP
    
    # print(brick1.axial_side)
    # print(brick1.radial_side)
    
    root.BOTTOM = brick1
    
    hinge1 = CanonicalNode(id=3, type=ModuleType.HINGE, rotation=0)

    # print(hinge1.axial_side)
    # print(hinge1.radial_side)
    
    brick1.RIGHT = hinge1
    
    # print(brick1)
    
    # print(brick1.axial_side)
    # print(brick1.radial_side)
    
    print(repr(root))
    
    print(root.BOTTOM)
    
    # brick1 = CanonicalNode(id=1, module_type=ModuleType.BRICK, rotation=2)
    # brick2 = CanonicalNode(id=2, module_type=ModuleType.BRICK, rotation=0)
    # hinge1 = CanonicalNode(id=3, module_type=ModuleType.HINGE, rotation=0)
    
    # print("hinge sides", hinge1.sides)
    
    root.LEFT = brick1      # Goes into sides (rotating)
    # root.TOP = brick2       # Goes into axial_children (doesn't rotate)
    brick1.FRONT = hinge1   # FRONT is axial for BRICK
    
    
    # print(root.sides)

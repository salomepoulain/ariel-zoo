from __future__ import annotations
from typing import Optional, Callable
from enum import Enum
from collections import deque

# Assuming these are imported from your config.py
from ariel.body_phenotypes.robogen_lite.config import (
    # ALLOWED_FACES,
    # ALLOWED_ROTATIONS,
    ModuleType,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleInstance
)


from enum import Enum, IntEnum

class ModuleFaces(IntEnum):
    """Enum for module attachment points."""
    FRONT = 0
    BACK = 1
    RIGHT = 2
    LEFT = 3
    TOP = 4
    BOTTOM = 5
    
    @property
    def opposite(self) -> 'ModuleFaces':
        """Get opposite face by XOR with 1 (pairs: 0-1, 2-3, 4-5)"""
        return ModuleFaces(self.value ^ 1)



# Map module types to their rotation axis face
MODULE_AXIS_FACE: dict[ModuleType, ModuleFaces] = {
    ModuleType.CORE: ModuleFaces.TOP,      # Rotates around vertical (TOP-BOTTOM axis)
    ModuleType.BRICK: ModuleFaces.FRONT,   # Rotates around longitudinal (FRONT-BACK axis)
    ModuleType.HINGE: ModuleFaces.FRONT,   # Rotates around longitudinal (FRONT-BACK axis)
    ModuleType.NONE: ModuleFaces.FRONT,    # Doesn't matter for NONE
}

# ADD CONNECTION POINT

class SymmetryPlane(Enum):
    """Symmetry plane determines how many unique rotations a module has."""
    NONE = 1
    TWO_FOLD = 2
    FOUR_FOLD = 4


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
        module_type: ModuleType,
        rotation: int = 0,
        parent_id: int | None = None,
        
        attachment_face: ModuleFaces | None = None,
        axis_face: ModuleFaces | None = None
        # TODO: symmetry
    ):
        # Compute geometry first
        axis = axis_face or MODULE_AXIS_FACE[module_type]
        symmetry = SYMMETRY_PLANE[module_type].value
        
        # TODO: not correct for hinge!!! use allowerd_faces
        all_faces = set(ModuleFaces)
        axis_pair = {axis, axis.opposite}
        rotating = sorted(all_faces - axis_pair, key=lambda f: f.value)
        
        # Store basic attributes
        self.id = id
        self.module_type = module_type
        self.rotation = rotation
        self.attachment_face = attachment_face  # None only for root
        self.parent_id = parent_id
        self.axis_face = axis
        self.symmetry_plane = symmetry
        
        # Store children - separated by rotation behavior
        self.axial_children = [None, None]  # [axis_face child, opposite_face child]
        self.sides = [None] * len(rotating)  # Rotating children
        
        # Store mappings for lookups
        self._face_to_index = {face: idx for idx, face in enumerate(rotating)}
        self._axial_faces = [axis, axis.opposite]
        
        # Create dynamic properties for all faces
        properties_dict = {}
        
        # TODO: only add the non-attachment point!
        # Properties for axial faces (index into axial_children)
        properties_dict[axis.name] = self._make_axial_property(0)
        properties_dict[axis.opposite.name] = self._make_axial_property(1)
        
        # Properties for rotating faces (index into sides)
        for face, idx in self._face_to_index.items():
            properties_dict[face.name] = self._make_side_property(idx)
        
        # Create a new class with these properties and change instance class
        NewClass = type(
            f'CanonicalNode_{module_type.name}',
            (self.__class__,),
            properties_dict
        )
        self.__class__ = NewClass
    
    @staticmethod
    def _make_axial_property(index: int):
        """Factory to create a property for an axial face"""
        def getter(self):
            return self.axial_children[index]
        
        def setter(self, value):
            self.axial_children[index] = value
            if value is not None:
                value.parent = self
                value.attachment_face = self._axial_faces[index]
        
        return property(getter, setter)
    
    @staticmethod
    def _make_side_property(index: int):
        """Factory to create a property for a rotating side face"""
        def getter(self):
            return self.sides[index]
        
        def setter(self, value):
            self.sides[index] = value
            if value is not None:
                value.parent = self
                # attachment_face will be set based on current rotation
        
        return property(getter, setter)
    
    def get_child_at_face(self, face: ModuleFaces) -> CanonicalNode | None:
        """Get child attached at a specific face."""
        if face in self._axial_faces:
            idx = self._axial_faces.index(face)
            return self.axial_children[idx]
        elif face in self._face_to_index:
            return self.sides[self._face_to_index[face]]
        return None
    
    def set_child_at_face(self, face: ModuleFaces, child: Optional[CanonicalNode]):
        """Set child at a specific face."""
        if face in self._axial_faces:
            idx = self._axial_faces.index(face)
            self.axial_children[idx] = child
        elif face in self._face_to_index:
            self.sides[self._face_to_index[face]] = child
        else:
            raise ValueError(f"Face {face} not allowed for {self.type}")
        
        if child:
            child.parent = self
            child.attachment_face = face
    
    
    
    
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
    #     for child in self.axial_children:
    #         if child:
    #             child.depth_first_traversal(visit)
    #     # Visit rotating children
    #     for child in self.sides:
    #         if child:
    #             child.depth_first_traversal(visit)
    
    # def depth_first_traversal_post_order(self, visit: Callable[[CanonicalNode], None]):
    #     """Depth-first traversal (post-order, from leaves up)."""
    #     # Visit axial children first
    #     for child in self.axial_children:
    #         if child:
    #             child.depth_first_traversal_post_order(visit)
    #     # Visit rotating children
    #     for child in self.sides:
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
    #     for face, child in zip(self._axial_faces, self.axial_children):
    #         if child:
    #             axial_repr.append(f"{indent}  {face.name}: {child.__repr__(level + 1)}")
    #         else:
    #             axial_repr.append(f"{indent}  {face.name}: None")
        
    #     # Format rotating children
    #     sides_repr = []
    #     for face, idx in self._face_to_index.items():
    #         child = self.sides[idx]
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
    root = CanonicalNode(id=0, module_type=ModuleType.CORE, rotation=0)
    
    print(root.sides)
    print(root.axial_children)
    
    brick1 = CanonicalNode(id=1, module_type=ModuleType.BRICK, rotation=2)
    brick2 = CanonicalNode(id=2, module_type=ModuleType.BRICK, rotation=0)
    hinge1 = CanonicalNode(id=3, module_type=ModuleType.HINGE, rotation=0)
    
    print("hinge sides", hinge1.sides)
    
    root.LEFT = brick1      # Goes into sides (rotating)
    root.TOP = brick2       # Goes into axial_children (doesn't rotate)
    brick1.FRONT = hinge1   # FRONT is axial for BRICK
    
    
    print(root.sides)
    print(root.axial_children)
    
    print("Tree structure:")
    print(root)
    
    print("\n--- Checking storage ---")
    print(f"root.sides: {root.sides}")
    print(f"root.axial_children: {root.axial_children}")
    print(f"root.LEFT is brick1: {root.LEFT is brick1}")
    print(f"root.TOP is brick2: {root.TOP is brick2}")
    
    print("\n--- Testing rotation ---")
    print(f"Brick1 rotation before: {brick1.rotation}")
    print(f"Brick1 sides before: {brick1.sides}")
    
    brick1.rotate(2)
    
    print(f"Brick1 rotation after: {brick1.rotation}")
    print(f"Brick1 sides after (rotated): {brick1.sides}")
    print(f"FRONT child still accessible: {brick1.FRONT is hinge1}")
    
    print("\n--- Converting to dict ---")
    module_dict = root.to_module_dict()
    for node_id, module in module_dict.items():
        print(f"Node {node_id}: {module}")

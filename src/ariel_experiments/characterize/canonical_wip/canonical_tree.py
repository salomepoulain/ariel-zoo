from __future__ import annotations

# Standard library
import json
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

# Third-party libraries
import numpy as np
from rich.console import Console
import networkx as nx
# Local libraries

if TYPE_CHECKING:
    from networkx import DiGraph

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
SEED = 42

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)


# Type Aliases and TypeVars
T = TypeVar("T")

# Concrete value aliases (don't use TypeVar for concrete aliases)
NumericProperty = int | float
NonNumProperty = str | bool | list | tuple | dict | set
GraphProperty = NumericProperty | NonNumProperty

# Index of a graph in the population
type GraphIndex = int
type IndexMappings = list[GraphIndex]

# Derived property can be a value or list of indexes
DerivedProperty = GraphProperty | IndexMappings

# name aliases
type GraphPropertyName = str
type DerivedPropertyName = str

# Generic mapping when values are homogeneous (use NamedGraphPropertiesT[T])
NamedGraphPropertiesT = dict[GraphPropertyName, T]
# Backwards-compatible mixed container
type NamedGraphProperties = dict[GraphPropertyName, GraphProperty]


# Generic analyzer Protocol: callable that returns dict[str, T]
class PropertyAnalyzer(Protocol[T]):
    def __call__(self, individual: DiGraph) -> NamedGraphPropertiesT[T]: ...


def analyze_canonical_subtrees(individual: DigGraph) -> NamedGraphPropertiesT:
    # from child, find parents,
    # canonciallize subtree
    # move up
    # repeat

    # list = [subtrees]
    pass


from typing import Optional, List, ForwardRef, TypedDict
from pydantic import BaseModel, Field, field_validator, model_validator
from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_FACES,
    ALLOWED_ROTATIONS,
    ModuleType,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleInstance
)
from enum import Enum


class SymmetryPlane(Enum):
    NONE = 1
    TWO_FOLD = 2
    FOUR_FOLD = 4


SYMMETRY_PLANE: dict[ModuleType, SymmetryPlane] = {
    ModuleType.CORE: SymmetryPlane.FOUR_FOLD,
    ModuleType.BRICK: SymmetryPlane.FOUR_FOLD,
    ModuleType.HINGE: SymmetryPlane.TWO_FOLD,
    ModuleType.NONE: SymmetryPlane.NONE,
}


class FacesOrder(Enum):
    LEFT = 0
    BOTTOM = 1
    RIGHT = 2
    TOP = 3

class CoreFacesOrder(Enum):
    LEFT = 0
    RIGHT = 1
    BACK = 2
    FRONT = 3

# class     




# ModuleFaces



from enum import IntEnum

class CubeFace(IntEnum):
    LEFT = 0
    RIGHT = 1
    BOTTOM = 2
    TOP = 3
    BACK = 4
    FRONT = 5
    
    @property
    def opposite(self) -> 'CubeFace':
        return CubeFace(self.value ^ 1)


    def get_rotating_faces(axis_face: CubeFace) -> list[CubeFace]:
        """Get the 4 faces that rotate around the given axis"""
        all_faces = set(CubeFace)
        axis_pair = {axis_face, axis_face.opposite}
        rotating = all_faces - axis_pair
        return sorted(rotating, key=lambda f: f.value)

def get_rotating_faces(axis_face: ModuleFaces) -> list[ModuleFaces]:
    """Get the 4 faces that rotate around the given axis"""
    all_faces = set(ModuleFaces)
    axis_pair = {axis_face, axis_face.opposite}
    rotating = all_faces - axis_pair
    
    return sorted(rotating, key=lambda f: f.value)

def get_face_index_in_rotation(face: ModuleFaces, axis_face: ModuleFaces) -> int:
    """Get the index of a face in the rotating faces list"""
    rotating = get_rotating_faces(axis_face)
    return rotating.index(face)

# Usage:
rotating = get_rotating_faces(ModuleFaces.FRONT)  # [RIGHT, LEFT, TOP, BOTTOM]
idx = rotating.index(ModuleFaces.TOP)  # 2
print(rotating[idx])  # ModuleFaces.TOP

def build_rotation_map(axis_face: ModuleFaces) -> dict[ModuleFaces, int]:
    """Build a dict: face -> index in rotation order"""
    rotating = get_rotating_faces(axis_face)
    return {face: idx for idx, face in enumerate(rotating)}

# Usage:
rotation_map = build_rotation_map(ModuleFaces.FRONT)
# {RIGHT: 0, LEFT: 1, TOP: 2, BOTTOM: 3}
idx = rotation_map[ModuleFaces.TOP]  # 2


#TODO; create dynamic properties!!!!



class ModuleFaces(Enum):
    """Enum for module attachment points."""

    FRONT = 0
    BACK = 1
    RIGHT = 2
    LEFT = 3
    TOP = 4
    BOTTOM = 5

    @property
    def opposite(self) -> 'CubeFace':
        return CubeFace(self.value ^ 1)


CubeFace.rotating_faces = rotating_faces

# Usage:
print(CubeFace.LEFT.opposite)  # CubeFace.RIGHT (0 ^ 1 = 1)
print(CubeFace.TOP.opposite)   # CubeFace.BOTTOM (3 ^ 1 = 2)
print(get_rotating_faces(CubeFace.FRONT))  # [LEFT, BOTTOM, RIGHT, TOP]




class CanonicalNode:
    def __init__(self, id: int, type: ModuleType, rotation: int = 0, axis_face: ModuleFaces = None):
        # Compute geometry first
        axis = axis_face or MODULE_AXIS_FACE[type]
        symmetry = SYMMETRY_PLANE[type].value
        
        # Build face mappings
        all_faces = set(ModuleFaces)
        axis_pair = {axis, axis.opposite}
        rotating = sorted(all_faces - axis_pair, key=lambda f: f.value)
        
        # Store structure on instance first
        self.id = id
        self.type = type
        self.rotation = rotation
        self.axis_face = axis
        self.symmetry_plane = symmetry
        self.axial_faces = [axis, axis.opposite]
        self.sides = [None] * len(rotating)
        self._face_to_index = {face: idx for idx, face in enumerate(rotating)}
        
        # Create properties dynamically for each rotating face
        properties_dict = {}
        for face, idx in self._face_to_index.items():
            properties_dict[face.name] = self._make_property(idx)
        
        NewClass = type(
            f'CanonicalNode_{type.name}',
            (self.__class__,),
            properties_dict
        )
        self.__class__ = NewClass
    
    @staticmethod
    def _make_property(index: int):
        """Factory to create a property for a specific index"""
        def getter(self):
            return self.sides[index]
        
        def setter(self, value):
            self.sides[index] = value
        
        return property(getter, setter)


class BaseCanonicalNode(BaseModel):
    """Base class for all canonical nodes."""
    id: int
    parent_id: int | None = None
    
    type: ModuleType
    rotation: int
    
    symmetry_plane: int 
    axial_faces: list[]
    sides: list[]
    
    attachment_point: ModuleFaces | None = None
    sides: list[ModuleFaces] | None = None
    rotation_axis: ModuleFaces | None = None

    def __init__():
        
    

    def get_attachments(self) -> list[tuple[str, BaseCanonicalNode | None]]:
        """
        Return a list of all attachments as (name, node) tuples.
        Subclasses should override this to define their specific attachments.
        """
        return []

    def depth_first_traversal(self, visit: callable) -> None:
        """
        General depth-first traversal for the tree.

        Args:
            visit (callable): A function to apply to each node during traversal.
        """
        # Visit the current node
        visit(self)

        # Traverse all attachments
        for _, attachment in self.get_attachments():
            if attachment:
                attachment.depth_first_traversal(visit)

    def __repr__(self, level=0) -> str:
        indent = "  " * level
        attachments_repr = "\n".join(
            f"{indent}  {name}={attachment.__repr__(level + 1) if attachment else None}"
            for name, attachment in self.get_attachments()
        )
        return (
            f"{indent}{self.__class__.__name__}(id={self.id}, type={self.type}, rotation={self.rotation},\n"
            f"{attachments_repr}\n{indent})"
        )

class CanonicalCore(BaseModel):
    """Specialized root node with different face handling."""
    id: int = 0
    top: CanonicalPart | None = None
    bottom: CanonicalPart | None = None
    sides: list[CanonicalPart | None] = Field(default_factory=lambda: [None] * len(CoreFacesOrder))

    # def add_child()

    @classmethod
    def build_tree_from_dict(cls, data: dict[int, ModuleInstance]) -> CanonicalCore:
        # Step 1: Create a lookup table for CanonicalPart instances
        data = dict(sorted(data.items(), key=lambda x: x[0]))    
        root = CanonicalCore()
        
        # Step 2: Create CanonicalPart instances for each node
        for node_id, module in data.items():
            # print(module)
                       
            # first fill in with 0. 
            def look(data: dict[int, ModuleType], parent: CanonicalPart):  
                print(parent)              
                for face, id in (data[parent.id].links.items()):                    
                    if face == ModuleFaces.FRONT:
                        if parent.front:
                            parent.front = CanonicalPart(
                                id=id,
                                type=data[id].type,
                                parent_id=parent.id,
                                rotation=module.rotation,
                            )
                            look(data, parent.front)
                    else:
                        if parent.sides:
                            parent.sides[FacesOrder[face.name].value] = CanonicalPart(
                                id=id,
                                type=data[id].type,
                                parent_id=parent.id,
                                rotation=data[id].rotation,
                            )    
                            look(data, parent.sides[FacesOrder[face.name].value])
            
            if node_id == 0:  # Root node
                for face, id in (data[node_id].links.items()):
                    if face == ModuleFaces.BOTTOM:
                        root.bottom = CanonicalPart(
                            id=id,
                            type=data[id].type,
                            parent_id=root.id,
                            rotation=data[id].rotation,
                        )
                        look(data, root.bottom)
                        
                    elif face == ModuleFaces.TOP:
                        root.top = CanonicalPart(
                            id=id,
                            type=data[id].type,
                            parent_id=root.id,
                            rotation=data[id].rotation,
                        )
                        look(data, root.top)                        
                    else:
                        root.sides[CoreFacesOrder[face.name].value] = CanonicalPart(
                            id=id,
                            type=data[id].type,
                            parent_id=root.id,
                            rotation=data[id].rotation,
                        )       
                        look(data, root.sides[CoreFacesOrder[face.name].value])
                      
        return root
        
    def __repr__(self, level=0) -> str:
        indent = "  " * level
        sides_repr = (
            "\n".join(
                side.__repr__(level + 1) if side else f"{indent}  None"
                for side in self.sides
            )
            if self.sides
            else "None"
        )
        return (
            f"{indent}CanonicalCore(id={self.id},\n"
            f"{indent}  top={self.top.__repr__(level + 1) if self.top else None},\n"
            f"{indent}  bottom={self.bottom.__repr__(level + 1) if self.bottom else None},\n"
            f"{indent}  sides=[\n{sides_repr}\n{indent}  ])"
        )
        
        # # Step 3: Link nodes (top, bottom, and sides)
        # for node_id, module in data.items():
        #     print(node_lookup)
            
        #     current_node = node_lookup[node_id]
        #     # Handle the root node separately
        #     if node_id == 0:
        #         # Set the top and bottom children for the root
                
        #         if ModuleFaces.TOP in module.links:
        #             # get the id of the 
        #             top_id = module.links[ModuleFaces.TOP]
        #             current_node.top = node_lookup[top_id]
        #             node_lookup[top_id].parent = current_node

        #         if ModuleFaces.BOTTOM in module.links:
        #             bottom_id = module.links[ModuleFaces.BOTTOM]
        #             current_node.bottom = node_lookup[bottom_id]
        #             node_lookup[bottom_id].parent = current_node

        #         side_faces = [
        #             ModuleFaces.FRONT,
        #             ModuleFaces.LEFT,
        #             ModuleFaces.RIGHT,
        #             ModuleFaces.BACK,
        #         ]
        #         for i, face in enumerate(side_faces):
        #             if face in module.links:
        #                 side_id = module.links[face]
        #                 current_node.sides[i] = node_lookup[side_id]
        #                 node_lookup[side_id].parent = current_node
        #     else:
        #         # Set the sides for non-root nodes
        #         for i, (face, child_id) in enumerate(module.links.items()):
        #             current_node.sides[i] = node_lookup[child_id]
        #             node_lookup[child_id].parent = current_node

        # # Step 4: Return the root node (assume root is always node 0)
        # return node_lookup[0]
        
        
    def to_dict(self) -> Dict[int, ModuleInstance]:
        """
        Convert the CanonicalPart tree back into a dictionary of ModuleInstances.

        Returns:
            Dict[int, ModuleInstance]: A dictionary where keys are node IDs and values are ModuleInstance objects.
        """
        result = {}

        def visit(node: CanonicalPart):
            # Build the links dictionary for the current node
            links = {}
            if node.front:
                links[ModuleFaces.FRONT] = node.front.id
            if node.sides:
                side_faces = [face for face in ALLOWED_FACES[node.type] if face != ModuleFaces.FRONT]
                for face, side in zip(side_faces, node.sides):
                    if side:
                        links[face] = side.id

            # Add the current node to the result dictionary
            result[node.id] = ModuleInstance(
                type=node.type,
                rotation=ModuleRotationsIdx(node.rotation),
                links=links,
            )

        # Use depth-first traversal to visit all nodes
        self.depth_first_traversal(visit)

        return result


    def generate_networkx_graph(self) -> nx.DiGraph:
        """
        Generate a NetworkX graph from the CanonicalCore tree.

        Returns:
            nx.DiGraph: A directed graph representing the CanonicalCore tree.
        """
        graph = nx.DiGraph()

        def visit(node: CanonicalPart):
            # Add the current node to the graph
            graph.add_node(
                node.id,
                type=node.type.name,
                rotation=ModuleRotationsIdx(node.rotation).name,  # Use the enum name
            )

            # Handle CanonicalCore-specific attributes (top, bottom, sides)
            if isinstance(node, CanonicalCore):
                # Add edges for top and bottom
                if node.top:
                    graph.add_node(
                        node.top.id,
                        type=node.top.type.name,
                        rotation=ModuleRotationsIdx(node.top.rotation).name,
                    )
                    graph.add_edge(node.id, node.top.id, face="TOP")

                if node.bottom:
                    graph.add_node(
                        node.bottom.id,
                        type=node.bottom.type.name,
                        rotation=ModuleRotationsIdx(node.bottom.rotation).name,
                    )
                    graph.add_edge(node.id, node.bottom.id, face="BOTTOM")

                # Add edges for sides
                if node.sides:
                    side_faces = list(CoreFacesOrder)  # LEFT, RIGHT, BACK, FRONT
                    for side, face in zip(node.sides, side_faces):
                        if side:
                            graph.add_node(
                                side.id,
                                type=side.type.name,
                                rotation=ModuleRotationsIdx(side.rotation).name,
                            )
                            graph.add_edge(node.id, side.id, face=face.name)
            else:
                # Handle CanonicalPart-specific attributes (front, sides)
                if node.front:
                    graph.add_node(
                        node.front.id,
                        type=node.front.type.name,
                        rotation=ModuleRotationsIdx(node.front.rotation).name,
                    )
                    graph.add_edge(node.id, node.front.id, face="FRONT")

                if node.sides:
                    side_faces = [
                        face for face in ALLOWED_FACES[node.type] if face != ModuleFaces.FRONT
                    ]
                    for side, face in zip(node.sides, side_faces):
                        if side:
                            graph.add_node(
                                side.id,
                                type=side.type.name,
                                rotation=ModuleRotationsIdx(side.rotation).name,
                            )
                            graph.add_edge(node.id, side.id, face=face.name)

        # Traverse the tree using depth-first traversal
        self.depth_first_traversal(visit)
        return graph


class CanonicalPart(BaseModel):
    """Dynamic canonical part that adapts based on module type."""

    id: int | None = None
    type: ModuleType
    rotation: int 
    
    front: CanonicalPart | None = None
    sides: list[CanonicalPart | None] | None = None
    symmetry_plane: int
    parent: CanonicalPart | None = None
    
    
    # model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data):
        # Initialize sides list based on module type if not provided
        if "sides" not in data or not data["sides"]:
            module_type = data.get("type")
            if module_type:
                allowed_faces = [
                    face
                    for face in ALLOWED_FACES[module_type]
                    if face != ModuleFaces.FRONT
                ]
                ordered_allowed = sorted(
                    allowed_faces, key=lambda f: FacesOrder[f.name].value
                )
                             
                if len(ordered_allowed) != 0:
                    data["sides"] = [None] * len(ordered_allowed)
 
        if "symmetry_plane" not in data or data["symmetry_plane"] is None:
            module_type = data.get("type")
            if module_type:
                data["symmetry_plane"] = SYMMETRY_PLANE[module_type].value

        super().__init__(**data)

        
    def depth_first_traversal(self, visit: callable) -> None:
        """
        General depth-first traversal for CanonicalPart tree.

        Args:
            visit (callable): A function to apply to each node during traversal.
        """
        # Visit the current node
        visit(self)

        # Traverse the front child
        if self.front:
            self.front.depth_first_traversal(visit)

        # Traverse the side children
        if self.sides:
            for side in self.sides:
                if side:
                    side.depth_first_traversal(visit)


    def breadth_first_traversal(self, visit: callable) -> None:
        """
        General breadth-first traversal for CanonicalPart tree.

        Args:
            visit (callable): A function to apply to each node during traversal.
        """
        from collections import deque

        queue = deque([self])  # Initialize the queue with the root node (self)

        while queue:
            # Dequeue the next node
            current = queue.popleft()
            # Visit the current node
            visit(current)
            # Enqueue the front child
            if current.front:
                queue.append(current.front)
            # Enqueue the side children
            if current.sides:
                for side in current.sides:
                    if side:
                        queue.append(side)

    def depth_first_traversal_bottom_up(self, visit: callable) -> None:
        """
        Depth-first traversal for CanonicalPart tree, starting from the leaves.

        Args:
            visit (callable): A function to apply to each node during traversal.
        """
        # Traverse the front child first
        if self.front:
            self.front.depth_first_traversal_bottom_up(visit)

        # Traverse the side children
        if self.sides:
            for side in self.sides:
                if side:
                    side.depth_first_traversal_bottom_up(visit)

        # Visit the current node after processing all children
        visit(self)

    def accumulate_rotations_down(self):
        """
        Propagate the rotation value of the current node down to its children.
        """
        def _rotation_down(node: CanonicalPart):
            if node.front:
                node.front.rotation += node.rotation

        self.depth_first_traversal(_rotation_down)
            
        
    def normalize_rotation(self):
        """Normalize rotation to valid range (0-1 or 0-1-2-3 depending on symmetry plane)."""
        before = self.rotation
        max_rotations = len(ModuleRotationsIdx) // self.symmetry_plane
        self.rotation = self.rotation % max_rotations
        change = self.rotation - before

        self._remap_child_position(change)
        self._apply_rotation_to_front(change)

    def _remap_child_position(self, amt: int):
        if self.sides:
            amt = amt % (len(self.sides))
            self.sides[:] = self.sides[amt:] + self.sides[:amt]

    def _apply_rotation_to_front(self, amt: int):
        if self.front:
            self.front.rotation += amt
            self.front.normalize_rotation()

    def save_subtree_hash(self):
        """Save hash of subtree."""
        pass

    # def get_allowed_side_faces(self) -> List[ModuleFaces]:
    #     """Get the list of allowed side faces for this module type."""
    #     return [face for face in ALLOWED_FACES[self.type]
    #             if face != ModuleFaces.FRONT]

    # def get_expected_sides_count(self) -> int:
    #     """Get the expected number of sides for this module type."""
    #     return len(self.get_allowed_side_faces())

    def rotate_self(self, amt: int) -> int:
        before = self.rotation
        max_rotations = len(ModuleRotationsIdx) // self.symmetry_plane
        self.rotation = (self.rotation + amt) % max_rotations

        return self.rotation - before

    def _get_allowed_rotations(self) -> list[ModuleRotationsIdx]:
        """Get the list of allowed rotations for this module type."""
        return ALLOWED_ROTATIONS[self.type]


    def __repr__(self, level=0) -> str:
        indent = "  " * level
        sides_repr = (
            "\n".join(
                side.__repr__(level + 1) if side else f"{indent}  None"
                for side in self.sides
            )
            if self.sides
            else "None"
        )
        return (
            f"{indent}CanonicalPart(id={self.id}, type={self.type}, rotation={self.rotation}, "
            f"symmetry_plane={self.symmetry_plane},\n"
            f"{indent}  front={self.front.__repr__(level + 1) if self.front else None},\n"
            f"{indent}  sides=[\n{sides_repr}\n{indent}  ])"
        )


if __name__ == "__main__":
    import matplotlib
    # matplotlib.use("TkAgg")
    from ariel_experiments.utils.initialize import generate_random_individual
    from ariel_experiments.gui_vis.visualize_tree import (
        visualize_tree_from_graph,
    )

    from ariel.body_phenotypes.robogen_lite.config import (
        NUM_OF_FACES,
        NUM_OF_ROTATIONS,
        NUM_OF_TYPES_OF_MODULES,
    )
    
    from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
    
    num_modules = 20

    rng = np.random.default_rng(42)
    type_probability_space = rng.random(
        size=(num_modules, NUM_OF_TYPES_OF_MODULES),
        dtype=np.float32,
    )
    conn_probability_space = rng.random(
        size=(num_modules, num_modules, NUM_OF_FACES),
        dtype=np.float32,
    )
    rotation_probability_space = rng.random(
        size=(num_modules, NUM_OF_ROTATIONS),
        dtype=np.float32,
    )
    hpd = HighProbabilityDecoder(num_modules)

    graph1 = hpd.probability_matrices_to_graph(
        type_probability_space,
        conn_probability_space,
        rotation_probability_space,
    )
    
    print(hpd.intermediate_graph)
    
    root = CanonicalCore.build_tree_from_dict(hpd.intermediate_graph)
    
    print(root)

    # root.depth_first_traversal(print_node)
    
    # dict_again = root.to_dict()
    
    # print(dict_again)
    
    graph2 = root.generate_networkx_graph()
    
    visualize_tree_from_graph(graph1, save_file="graph1.png")
    visualize_tree_from_graph(graph2, save_file="graph2.png")
    

    # graph = generate_random_individual()

    # visualize_tree_from_graph(graph, save_file="graph.png")

    # Update forward references
    CanonicalPart.model_rebuild()

    # Usage examples:
    print("=== Valid examples ===")
    core = CanonicalPart(id=0, type=ModuleType.CORE, rotation=0)
    print(f"Core: rotation={core.rotation}, sides={len(core.sides)}")

    print(core)

    print(core.rotate_self(2))
    print("rotation after: ", core.rotation)

    brick = CanonicalPart(id=1, type=ModuleType.BRICK, rotation=2)  # DEG_90
    print(f"Brick: rotation={brick.rotation}, sides={len(brick.sides)}")

    print(brick.rotate_self(12))
    print("rotation after: ", brick.rotation)

    print(brick)

    hinge = CanonicalPart(id=2, type=ModuleType.HINGE, rotation=4)  # DEG_180
    print(f"Hinge: rotation={hinge.rotation}, sides={len(hinge.sides)}")

    none = CanonicalPart(id=2, type=ModuleType.NONE, rotation=0)  # DEG_180
    print(f"Hinge: rotation={none.rotation}, sides={len(none.sides)}")

# print("\n=== Invalid examples ===")
# # This will raise a validation error (CORE only allows rotation 0):
# try:
#     invalid_core = CanonicalPart(id=3, type=ModuleType.CORE, rotation=2)
# except ValueError as e:
#     print(f"Validation error: {e}")

# # This will raise a validation error (rotation out of range):
# try:
#     invalid_rotation = CanonicalPart(id=4, type=ModuleType.BRICK, rotation=10)
# except ValueError as e:
#     print(f"Validation error: {e}")

# # This will raise a validation error (wrong number of sides):
# try:
#     invalid_sides = CanonicalPart(id=5, type=ModuleType.HINGE, sides=[None, None])
# except ValueError as e:
#     print(f"Validation error: {e}")

# print("\n=== Allowed rotations by type ===")
# for module_type in [ModuleType.CORE, ModuleType.BRICK, ModuleType.HINGE]:
#     part = CanonicalPart(id=0, type=module_type)
#     allowed = part.get_allowed_rotations()
#     print(f"{module_type.name}: {[f'{r.name}={r.value}' for r in allowed]}")

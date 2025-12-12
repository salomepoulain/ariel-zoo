from __future__ import annotations
from dataclasses import dataclass, field, replace as std_replace
from enum import Enum, auto
from typing import Dict, Any, Union, List, Callable, Iterator, Protocol, Optional
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

# =============================================================================
# 1. ENUMS & CONSTANTS
# =============================================================================

class VectorSpace(Enum):
    """Immutable physical scopes."""
    ENTIRE_ROBOT = 'full'
    FRONT_LIMB = 'front'
    LEFT_LIMB = 'left'
    BACK_LIMB = 'back'
    RIGHT_LIMB = 'right'
    AGGREGATED = 'aggregated'

    @classmethod
    def limb_spaces_only(cls) -> List['VectorSpace']:
        return [cls.FRONT_LIMB, cls.LEFT_LIMB, cls.BACK_LIMB, cls.RIGHT_LIMB]

class MatrixType(Enum):
    FEATURES = auto()
    SIMILARITY = auto()
    EMBEDDING = auto()

# SIMPLIFICATION: Radius is always an integer depth.
RadiusKey = int 

# =============================================================================
# 2. MATRIX INSTANCE (The Cell)
# =============================================================================
@dataclass(frozen=True)
class MatrixInstance:
    """
    A strictly encapsulated wrapper around the heavy matrix data.
    Enforces immutability and strictly checks memory usage types.
    """
    # Internal fields (Hidden from direct write access)
    _matrix: Union[sp.spmatrix, np.ndarray]
    _space: Union[VectorSpace, str]
    _radius: int  # Strict integer radius
    _type: MatrixType = MatrixType.FEATURES
    _meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        STRICT VALIDATION: prevents memory disasters.
        Ensures FEATURES are Sparse and SIMILARITY/EMBEDDING are Dense.
        """
        # Case 1: FEATURES must be Sparse (prevent OOM on large datasets)
        if self._type == MatrixType.FEATURES:
            if not sp.issparse(self._matrix):
                # We raise an error instead of converting silently to alert the dev
                raise TypeError(
                    f"CRITICAL MEMORY ERROR: MatrixType.FEATURES must be a "
                    f"scipy.sparse matrix, but got {type(self._matrix)}. "
                    "This will crash your RAM on large datasets."
                )

        # Case 2: SIMILARITY / EMBEDDING should ideally be Dense
        # (Optional strictness: you can uncomment this if you want to enforce it)
        # elif self._type in [MatrixType.SIMILARITY, MatrixType.EMBEDDING]:
        #     if sp.issparse(self._matrix):
        #         raise TypeError(f"{self._type.name} expected to be a Dense numpy array")


    #TODO
    # @from npz(or whatever it was called)
    
    # to_npz


    # --- Public Read-Only Properties ---

    @property
    def matrix(self) -> Union[sp.spmatrix, np.ndarray]:
        """Read-only access to raw data."""
        return self._matrix

    @property
    def shape(self):
        return self._matrix.shape

    @property
    def space(self) -> Union[VectorSpace, str]:
        return self._space

    @property
    def radius(self) -> int:
        return self._radius

    @property
    def type(self) -> MatrixType:
        return self._type

    @property
    def meta(self) -> Dict[str, Any]:
        """Returns a COPY of metadata to prevent external mutation bugs."""
        return self._meta.copy()

    # --- The Abstraction Gatekeeper ---

    def replace(self, **public_changes) -> 'MatrixInstance':
        """
        Creates a new instance based on this one.
        Maps public arguments ('matrix') to private fields ('_matrix').
        """
        internal_changes = {}
        
        # 1. Map public names to private fields
        if 'matrix' in public_changes: internal_changes['_matrix'] = public_changes.pop('matrix')
        if 'space' in public_changes:  internal_changes['_space'] = public_changes.pop('space')
        if 'radius' in public_changes: internal_changes['_radius'] = public_changes.pop('radius')
        if 'type' in public_changes:   internal_changes['_type'] = public_changes.pop('type')
        
        # 2. Handle Metadata Safety
        # If 'meta' is explicitly provided, use it. Otherwise, copy existing.
        if 'meta' in public_changes:
            internal_changes['_meta'] = public_changes.pop('meta')
        else:
            internal_changes['_meta'] = self._meta.copy()

        # 3. Allow internal names if needed (e.g. from internal methods)
        internal_changes.update(public_changes)

        return std_replace(self, **internal_changes)

    def add_meta(self, **kwargs) -> 'MatrixInstance':
        """
        Returns a NEW instance with updated metadata (since class is frozen).
        """
        new_meta = self._meta.copy()
        new_meta.update(kwargs)
        return std_replace(self, _meta=new_meta)

    # --- Transformations ---

    def cosine_similarity(self) -> 'MatrixInstance':
        if self._type == MatrixType.SIMILARITY:
            raise ValueError("Matrix is already a similarity matrix.")
        
        # Calculate Dense Similarity from Sparse Features
        sim_matrix = cosine_similarity(self._matrix)
        
        return self.replace(
            matrix=sim_matrix,
            type=MatrixType.SIMILARITY
            # meta is auto-copied by replace logic
        )

    def __add__(self, other: 'MatrixInstance') -> 'MatrixInstance':
        """
        Sums two matrices (e.g. Left + Right).
        Checks for radius compatibility.
        """
        if not isinstance(other, MatrixInstance):
            return NotImplemented
        
        if self._radius != other._radius:
            raise ValueError(f"Cannot add different radii: {self._radius} vs {other._radius}")
        
        if self._type != other._type:
            raise ValueError(f"Cannot add different types: {self._type} vs {other._type}")

        # Math happens on private fields directly
        new_matrix = self._matrix + other._matrix
        
        # Return new instance (space naming handled by caller)
        return self.replace(matrix=new_matrix)

# =============================================================================
# 3. MATRIX SERIES (The Column)
# =============================================================================

@dataclass
class MatrixSeries:
    """
    Abstraction for a column of radii.
    """
    _space: Union[VectorSpace, str]
    _instances: Dict[int, MatrixInstance] = field(default_factory=dict)
    _meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def space(self) -> Union[VectorSpace, str]:
        return self._space

    def __getitem__(self, radius: int) -> MatrixInstance:
        return self._instances[radius]

    def __setitem__(self, radius: int, instance: MatrixInstance):
        if instance.radius != radius:
             instance = instance.replace(radius=radius)
        self._instances[radius] = instance

    def __iter__(self) -> Iterator[int]:
        """Iterates over radii in sorted order (0, 1, 2...)."""
        return iter(sorted(self._instances.keys()))

    def replace(self, **public_changes) -> 'MatrixSeries':
        internal_changes = {}
        if 'space' in public_changes: internal_changes['_space'] = public_changes.pop('space')
        if 'meta' in public_changes: 
            internal_changes['_meta'] = public_changes.pop('meta')
        else:
            internal_changes['_meta'] = self._meta.copy()
            
        internal_changes.update(public_changes)
        return std_replace(self, **internal_changes)

    def map(self, func: Callable[[MatrixInstance], MatrixInstance]) -> 'MatrixSeries':
        new_data = {r: func(inst) for r, inst in self._instances.items()}
        return self.replace(_instances=new_data)

    def cosine_similarity(self) -> 'MatrixSeries':
        return self.map(lambda inst: inst.cosine_similarity())

    # --- NEW: Cumulative Logic (Replaces Flattening) ---

    def to_cumulative(self) -> 'MatrixSeries':
        """
        Returns a new Series where index 'i' contains sum(0..i).
        This preserves the integer keys:
        - Series[0] = Matrix[0]
        - Series[1] = Matrix[0] + Matrix[1]
        """
        if not self._instances: return self.replace(_instances={})
        
        new_instances = {}
        sorted_radii = sorted(self._instances.keys())
        
        current_sum = None
        
        for r in sorted_radii:
            mat_inst = self._instances[r]
            
            if current_sum is None:
                # First radius
                new_inst = mat_inst # Copy reference
                current_sum = mat_inst.matrix
            else:
                # Add to previous sum
                current_sum = current_sum + mat_inst.matrix
                # Create new instance with the accumulated matrix at this radius
                new_inst = mat_inst.replace(matrix=current_sum)
            
            new_instances[r] = new_inst
            
        return self.replace(_instances=new_instances)

    def __add__(self, other: 'MatrixSeries') -> 'MatrixSeries':
        if not isinstance(other, MatrixSeries): return NotImplemented
        
        result = self.replace(_instances={})
        common = set(self._instances) & set(other._instances)
        
        for r in common:
            result[r] = self[r] + other[r]
            
        return result


# =============================================================================
# 4. AGGREGATION PROTOCOL
# =============================================================================

class InstanceAggregator(Protocol):
    def __call__(self, instances: List[MatrixInstance]) -> MatrixInstance: ...

def agg_sum_features(instances: List[MatrixInstance]) -> MatrixInstance:
    total = instances[0].matrix
    for x in instances[1:]: total = total + x.matrix
    return instances[0].replace(matrix=total)

def agg_mean_similarity(instances: List[MatrixInstance]) -> MatrixInstance:
    if instances[0].type != MatrixType.SIMILARITY:
         raise ValueError("Must be similarity")
    total = instances[0].matrix
    for x in instances[1:]: total = total + x.matrix
    return instances[0].replace(matrix=total / len(instances))


# =============================================================================
# 5. MATRIX FRAME (The Table)
# =============================================================================

@dataclass
class MatrixFrame:
    _series: Dict[Union[VectorSpace, str], MatrixSeries] = field(default_factory=dict)
    _meta: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: Union[VectorSpace, str]) -> MatrixSeries:
        if key not in self._series:
            self._series[key] = MatrixSeries(_space=key)
        return self._series[key]

    def __setitem__(self, key: Union[VectorSpace, str], val: MatrixSeries):
        self._series[key] = val

    def keys(self):
        return self._series.keys()

    def replace(self, **public_changes) -> 'MatrixFrame':
        internal_changes = {}
        if 'meta' in public_changes: 
            internal_changes['_meta'] = public_changes.pop('meta')
        else:
            internal_changes['_meta'] = self._meta.copy()
        internal_changes.update(public_changes)
        return std_replace(self, **internal_changes)

    def map(self, func: Callable[[MatrixSeries], MatrixSeries]) -> 'MatrixFrame':
        new_series = {s: func(ser) for s, ser in self._series.items()}
        return self.replace(_series=new_series)

    # --- High Level API ---

    def cosine_similarity(self) -> 'MatrixFrame':
        return self.map(lambda s: s.cosine_similarity())
    
    def to_cumulative(self) -> 'MatrixFrame':
        """Convert all series to cumulative sums."""
        return self.map(lambda s: s.to_cumulative())

    def aggregate_series(
        self,
        new_name: str, 
        sources: List[Union[VectorSpace, str]],
        aggregator: InstanceAggregator
    ) -> 'MatrixFrame':
        
        if not sources: raise ValueError("No sources")

        result_series = MatrixSeries(_space=new_name)
        
        first = self._series[sources[0]]
        common = set(first._instances)
        for s in sources[1:]:
            common &= set(self._series[s]._instances)
            
        for r in common:
            inputs = [self._series[s][r] for s in sources]
            res = aggregator(inputs)
            res = res.replace(space=new_name, radius=r)
            result_series[r] = res

        new_frame = self.replace()
        new_frame[new_name] = result_series
        return new_frame




# """
# Subtree Analyzer: Flexible pipeline for structural feature extraction and analysis.

# This module provides an enum-based API for analyzing hierarchical robot structures
# across different vector spaces and aggregation strategies.
# """

# from dataclasses import dataclass
# from enum import Enum, auto
# from typing import Any

# import umap  # type: ignore
# from sklearn.decomposition import PCA
# from sklearn.feature_extraction import FeatureHasher
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.manifold import TSNE
# from sklearn.metrics.pairwise import (
#     cosine_similarity,
#     euclidean_distances,
#     pairwise_distances,
# )


# class VectorSpace(Enum):
#     """
#     Structural scopes for feature extraction.
#     """
#     ENTIRE_ROBOT = 'full'      # Whole robot as one structure
#     FRONT_LIMB = 'front'       # Just front limb
#     LEFT_LIMB = 'left'         # Just left limb
#     BACK_LIMB = 'back'         # Just back limb
#     RIGHT_LIMB = 'right'       # Just right limb
    
#     # The new meta-space
#     AGGREGATED = 'aggregated'  # Sum of all selected spaces

#     @classmethod
#     def individual_spaces(cls) -> list["VectorSpace"]:
#         """All primitive spaces (excludes the computed aggregate)."""
#         return [
#             cls.ENTIRE_ROBOT,
#             cls.FRONT_LIMB,
#             cls.LEFT_LIMB,
#             cls.BACK_LIMB,
#             cls.RIGHT_LIMB,
#         ]

#     @classmethod
#     def limb_spaces_only(cls) -> list["VectorSpace"]:
#         """Only the limb-specific spaces."""
#         return [cls.FRONT_LIMB, cls.LEFT_LIMB, cls.BACK_LIMB, cls.RIGHT_LIMB]


# class MatrixDomain(Enum):
#     """
#     The different matrix types that can be generated for population of size N
#     """
#     FEATURES = auto()      # N x M: The raw treehashes (Counts OR TFIDF)
#     SIMILARITY = auto()    # N x N: The relationships (Cosine)
#     EMBEDDING = auto()     # N x 2: The visualization (UMAP)


# from dataclasses import dataclass, field
# from typing import Dict, Any, Union, List, Optional, Iterator
# from enum import Enum
# import numpy as np
# import scipy.sparse as sp
# from sklearn.metrics.pairwise import cosine_similarity

# # # --- Configuration & Types ---

# # # Use your provided Enum
# # class VectorSpace(Enum):
# #     ENTIRE_ROBOT = 'full'
# #     FRONT_LIMB = 'front'
# #     LEFT_LIMB = 'left'
# #     BACK_LIMB = 'back'
# #     RIGHT_LIMB = 'right'

# #     @classmethod
# #     def individual_spaces(cls) -> list["VectorSpace"]:
# #         return [cls.ENTIRE_ROBOT, cls.FRONT_LIMB, cls.LEFT_LIMB, cls.BACK_LIMB, cls.RIGHT_LIMB]

# #     @classmethod
# #     def limb_spaces_only(cls) -> list["VectorSpace"]:
# #         return [cls.FRONT_LIMB, cls.LEFT_LIMB, cls.BACK_LIMB, cls.RIGHT_LIMB]

# # RadiusKey = Union[int, str]  # Radius depth (0, 1) or 'all'/'cumulative'

# # --- 1. MatrixInstance: The "Cell" (One Radius, One Space) ---




# class MatrixInstance:
#     """
#     Holds a single sparse matrix for a specific Space + Radius combination.
#     Example: (VectorSpace.FRONT_LIMB, radius=2)
#     """
#     def __init__(self) -> None:
#         self._matrix: sp.spmatrix
#         self._radius: int
#         self._space: VectorSpace 
        
#         # important flags/methadata
#         self._is_aggregated: bool = False # can be across radius or across spaces?
#         self._is_cumulative: bool = False # can be across radius or across spaces?
        

#     @classmethod
#     def from_npz(self, path: str | None) -> None:
#         # add auto cache path
#         # give optional file name (can later be generation or something)
#         # add the shape/radius in the filename
#         self._matrix = sp.load_npz(path)
#         self._radius = radius
#         self._space = space

#     @property
#     def radius(self) -> int:
#         return self._radius

#     @property
#     def space(self) -> VectorSpace:
#         return self._space
    
#     @property
#     def is_aggregated(self) -> bool:
#         return self._is_aggregated

#     @property
#     def is_cumulative(self) -> bool:
#         return self._is_cumulative

#     @property
#     def shape(self):
#         return self._matrix.shape

#     def __repr__(self):
#         s_name = self.space.name if isinstance(self.space, VectorSpace) else self.space
#         return f"<MatrixInstance [{s_name} | r={self._radius}] shape={self.shape}>"

#     # def __add__(self, other: 'MatrixInstance') -> 'MatrixInstance':
#     #     """
#     #     Operator overload for +
#     #     Allows: instance1 + instance2
#     #     """
#     #     if not isinstance(other, MatrixInstance):
#     #         return NotImplemented
        
#     #     # if self.radius != other.radius:
#     #     #     raise ValueError(f"Cannot add different radii: {self.radius} vs {other.radius}")

#     #     # Efficient sparse matrix addition
#     #     new_matrix = self._matrix + other._matrix
        
#     #     # Create a descriptive name for the combined space (e.g., "FRONT_PLUS_BACK")
#     #     s1 = self.space.value if isinstance(self.space, VectorSpace) else str(self.space)
#     #     s2 = other.space.value if isinstance(other.space, VectorSpace) else str(other.space)
        
#     #     return MatrixInstance(
#     #         _matrix=new_matrix, 
#     #         _radius=self._radius, 
#     #         space=f"{s1}+{s2}",
#     #         meta={**self.meta, **other.meta}
#     #     )
    
#     def dot_product(self) -> np.ndarray:
#         return self._matrix.dot(self._matrix.T)

#     def cosine_similarity(self) -> np.ndarray:
#         return cosine_similarity(self._matrix)

#     def euclidean_distances(self) -> np.ndarray:
#         return euclidean_distances(self._matrix)
    
#     def to_sum_array(self) -> np.ndarray:
#         return np.array(self._matrix.sum(axis=1)).flatten()
    
#     def get_knn_sum(self, k: int = 5, do_max: bool = True) -> np.ndarray:
#         """
#         Get the sum of the k-nearest neighbors for each row.
#         """
#         knn_indices = np.argpartition(distances, k, axis=1)[:, :k]
#         knn_sums = np.sum(distances[np.arange(distances.shape[0])[:, None], knn_indices], axis=1)
#         return knn_sums

#     def normalize_by_radius(self) -> 'MatrixInstance':
#         """
#         Normalize the matrix by a given radius.
#         """
#         # if vectorspace 
#         pass

#     def to_dense(self) -> np.ndarray:
#         return self._matrix.toarray()
    
#     def to_npz(self, path: str) -> None:
#         # add auto cache path
#         # give optional file name (can later be generation or something)
#         # add the shape/radius in the filename
#         sp.save_npz(path, self._matrix)
        

# # --- 2. MatrixSeries: The "Column" (One Space, All Radii) ---

# @dataclass
# class MatrixSeries:
#     """
#     Holds a collection of matrices for ONE VectorSpace across different radii (int).
#     Behaves like a Dictionary or Pandas Series.
#     """
#     _space: VectorSpace
#     _instances: dict[RadiusKey, MatrixInstance] = field(default_factory=dict)
#     _is_aggregated: bool = False
#     _is_cumulative: bool = False

#     @classmethod
#     def from_npz(cls, path: str, space: VectorSpace) -> 'MatrixSeries':
#         """Load a MatrixSeries from a directory of .npz files."""
#         # Implement loading logic
#         pass
    
#     @classmethod
#     def from_subtree_dict_list(cls, subtree_dict_list: list[dict[int, str]]) -> 'MatrixSeries':
#         """Create a MatrixSeries from a list of subtree dictionaries."""
#         # Implement creation logic
#         pass
    

#     @property
#     def space(self) -> VectorSpace:
#         return self._space
    
#     def __setitem__(self, radius: RadiusKey, matrix: sp.spmatrix):
#         """Allows: series[0] = sparse_matrix"""
#         self._instances[radius] = MatrixInstance(matrix, radius, self._space)

#     def __getitem__(self, radius: RadiusKey) -> MatrixInstance:
#         """Allows: series[0] -> MatrixInstance"""
#         return self._instances[radius]
    
#     def __iter__(self) -> Iterator[MatrixInstance]:
#         """Iterates through instances sorted by radius."""
#         # Sort keys: integers first, then strings
#         keys = sorted(self._instances.keys(), key=lambda x: (isinstance(x, str), x))
#         for k in keys:
#             yield self._instances[k]

#     def __add__(self, other: 'MatrixSeries') -> 'MatrixSeries':
#         """
#         Operator overload for +
#         Allows: frame[VectorSpace.LEFT] + frame[VectorSpace.RIGHT]
#         Returns a NEW Series where every common radius is summed (aggregated).
#         """
#         if not isinstance(other, MatrixSeries):
#             return NotImplemented

#         # Create new name
#         s1 = self._space.value if isinstance(self._space, VectorSpace) else str(self._space)
#         s2 = other._space.value if isinstance(other._space, VectorSpace) else str(other._space)
#         new_space_name = f"{s1}+{s2}"

#         result = MatrixSeries(_space=new_space_name)

#         # Only sum radii that exist in BOTH series
#         common_radii = set(self._instances.keys()) & set(other._instances.keys())
        
#         for r in common_radii:
#             # Delegate to MatrixInstance.__add__
#             result._instances[r] = self[r] + other[r]

#         return result

#     def __repr__(self):
#         s_name = self._space.name if isinstance(self._space, VectorSpace) else self._space
#         keys = sorted(self._instances.keys(), key=lambda x: str(x))
#         return f"<MatrixSeries '{s_name}': {keys}>"

#     # --- Aggregation Logic ---

#     def to_cumulative(self) -> 'MatrixSeries':
#         """
#         Returns a NEW Series where radius 'r' contains sum(0..r).
#         Useful for progressive analysis.
#         """
#         new_series = MatrixSeries(_space=f"{self._space}_cumulative")
#         new_series._is_cumulative = True
#         sorted_radii = sorted([k for k in self._instances.keys() if isinstance(k, int)])
        
#         if not sorted_radii:
#             return new_series

#         current_sum = None
#         for r in sorted_radii:
#             mat = self._instances[r]._matrix
#             if current_sum is None:
#                 current_sum = mat
#             else:
#                 current_sum = current_sum + mat 
            
#             new_series[r] = current_sum # This calls __setitem__ internally
            
#         return new_series

#     def aggregate_all_radii(self) -> MatrixInstance:
#         """
#         Collapses ALL radii in this series into a single 'flattened' MatrixInstance.
#         Resulting radius key is 'all'.
#         """
#         if not self._instances:
#             raise ValueError("Cannot aggregate empty series")
            
#         total_matrix = sum(inst._matrix for inst in self._instances.values())
#         return MatrixInstance(total_matrix, _radius='all', space=self._space)


# # --- 3. MatrixFrame: The "DataFrame" (All Spaces) ---

# @dataclass
# class MatrixFrame:
#     """
#     The main container. Holds 1 to multiple MatrixSeries keyed by VectorSpace.
#     """
#     _series: dict[Vectorspace, MatrixSeries] = field(default_factory=dict)

#     def __getitem__(self, space: Union[VectorSpace, str]) -> MatrixSeries:
#         """
#         Get or create a Series for a specific VectorSpace.
#         Usage: frame[VectorSpace.FRONT_LIMB]
#         """
#         if space not in self._series:
#             self._series[space] = MatrixSeries(_space=space)
#         return self._series[space]
    
#     def __setitem__(self, space: Union[VectorSpace, str], series: MatrixSeries):
#         """Allow manually setting a series: frame['custom_sum'] = my_series"""
#         self._series[space] = series

#     def keys(self):
#         return self._series.keys()
    
#     def get_aggregated_series(
#         self, 
#         spaces: list[VectorSpace] = None, 
#         target_space: VectorSpace = VectorSpace.AGGREGATED,
#         append: bool = False
#     ) -> MatrixSeries:
#         """
#         Sums across multiple VectorSpaces to create the 'AGGREGATED' series.
#         """
#         # Default to summing limbs if nothing specified
#         if spaces is None:
#             spaces = VectorSpace.limb_spaces_only()
            
#         if not spaces:
#             raise ValueError("No spaces provided for aggregation")

#         # Start with the first one
#         result_series = self._series[spaces[0]]

#         # Add the rest
#         for space in spaces[1:]:
#             if space in self._series:
#                 result_series = result_series + self._series[space]

#         # Tag the result with the official Enum
#         result_series._space = target_space
        
#         return result_series

#     def __repr__(self):
#         keys = [k.name if isinstance(k, VectorSpace) else str(k) for k in self._series.keys()]
#         return f"<MatrixFrame columns={keys}>"



# # MatrixFrame.from_treehash_list().tfidf().cosine_similarity().to_cumulative().

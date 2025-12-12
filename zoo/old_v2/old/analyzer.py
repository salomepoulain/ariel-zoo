"""
Subtree Analyzer: Flexible pipeline for structural feature extraction and analysis.

This module provides an enum-based API for analyzing hierarchical robot structures
across different vector spaces and aggregation strategies.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import umap  # type: ignore
from sklearn.decomposition import PCA
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    pairwise_distances,
)

# ==================== ENUMS ====================

class VectorSpace(Enum):
    """
    Structural scopes for feature extraction.

    Each vector space represents a different view of the robot's structure:
    - ENTIRE_ROBOT: Analyzes the whole robot as one unified structure
    - Individual limbs: Analyzes each limb in isolation
    - ALL_COMBINED: Meta-space that combines all other spaces
    """
    ENTIRE_ROBOT = 'full'      # Whole robot as one structure
    FRONT_LIMB = 'front'       # Just front limb
    LEFT_LIMB = 'left'         # Just left limb
    BACK_LIMB = 'back'         # Just back limb
    RIGHT_LIMB = 'right'       # Just right limb

    @classmethod
    def individual_spaces(cls) -> list["VectorSpace"]:
        """All spaces except the combined meta-space."""
        return [
            cls.ENTIRE_ROBOT,
            cls.FRONT_LIMB,
            cls.LEFT_LIMB,
            cls.BACK_LIMB,
            cls.RIGHT_LIMB,
        ]

    @classmethod
    def limb_spaces_only(cls) -> list["VectorSpace"]:
        """Only the limb-specific spaces (excludes entire robot)."""
        return [cls.FRONT_LIMB, cls.LEFT_LIMB, cls.BACK_LIMB, cls.RIGHT_LIMB]


# class RadiusAggregation(Enum):
#     """
#     How to handle hierarchical depth (radius) in feature matrices.

#     - PER_RADIUS: Each depth gets its own matrix {0: mat, 1: mat, 2: mat}
#     - FLATTENED: All depths merged into one matrix {'all': mat}
#     - CUMULATIVE: Progressive aggregation {0: r0, 1: r0+r1, 2: r0+r1+r2}
#     - BOTH: Generate both per-radius AND flattened versions
#     """
#     PER_RADIUS = auto()        # Separate matrix for each depth
#     FLATTENED = auto()         # Single matrix with all depths merged
#     CUMULATIVE = auto()        # Accumulate: r0, r0+r1, r0+r1+r2, ...
#     BOTH = auto()              # Generate both per-radius AND flattened versions


# Per MatrixInstance
class FeatureStrategy(Enum):
    """Feature extraction and weighting strategies."""
    HASH_COUNT = auto()        # FeatureHasher with counts
    HASH_BINARY = auto()       # FeatureHasher binary (presence/absence)
    TFIDF = auto()             # TF-IDF weighting


class SimilarityMetric(Enum):
    """Pairwise similarity/distance metrics."""
    COSINE = auto()            # Cosine similarity (1 = identical, 0 = orthogonal)
    JACCARD = auto()           # Jaccard similarity (set overlap)
    EUCLIDEAN = auto()         # Euclidean distance converted to similarity
    HAMMING = auto()           # Hamming distance (binary features)


class DimReductionMethod(Enum):
    """Dimensionality reduction algorithms for visualization."""
    UMAP = auto()              # Uniform Manifold Approximation and Projection
    PCA = auto()               # Principal Component Analysis
    TSNE = auto()              # t-Distributed Stochastic Neighbor Embedding


# ==================== CONFIG ====================

@dataclass
class AnalysisConfig:
    """Configuration for subtree analysis pipeline."""
    vector_spaces: list[VectorSpace] | None = None
    radius_mode: RadiusAggregation = RadiusAggregation.BOTH
    n_features: int = 2**20

    def __post_init__(self):
        if self.vector_spaces is None:
            # Default: analyze all individual spaces + the combined meta-space
            self.vector_spaces = VectorSpace.individual_spaces() + [VectorSpace.ALL_COMBINED]


# ==================== MAIN API ====================



@dataclass
class MatrixSeries:
    """per radius"""
    _max_radius = #
    _matrix_instances = list[]
    _cumulative: bool = False
    _aggregated: bool = False
    
    # or should aggreagated be its own instance??? i think so.. different from _matrix_instances
    

    def to_cumulative(self) -> self
        pass
        
    def to_aggregate(self)
        # check that its not already aggregated
        # must make sure that the featurehasher was used!
        # can jsut sum across radii?\
        # _aggregated = True
        # len(matrix_instances) == 1
        pass

    # def compute_similarity(self, other) -> MatrixCollection
    #     # make sure len(self) == len(other)
    #     # else raise thing. or Try Except?
        
    #     # returns 
    #     pass
    
    def __len__(self)
        return len(self.matrix_instances)
    

@dataclass
class MatrixInstance:
    key (hashable thing for the indexing)
    VectorSpace 

    
    matrix



class SubtreeAnalyzer:
    """
    Pipeline for structural feature analysis on robot populations.

    Key Concepts:
    - Vector Spaces: Different structural scopes (entire robot, individual limbs, or all combined)
    - Radius Aggregation: Whether to preserve hierarchical depth or flatten it

    Usage:
        # Option 1: Pass pre-collected subtrees (recommended - no circular imports)
        subtrees = {VectorSpace.ENTIRE_ROBOT: [...], VectorSpace.FRONT_LIMB: [...]}
        analyzer = SubtreeAnalyzer.from_subtrees(subtrees)
        results = (analyzer
            .hash_features(radius_mode=RadiusAggregation.BOTH)
            .compute_similarity()
            .reduce_dimensions(method=DimReductionMethod.UMAP)
            .get_results()
        )

        # Option 2: Use with external subtree collection (evaluator class)
        analyzer = SubtreeAnalyzer()
        analyzer.set_subtrees(subtrees_dict)
        results = analyzer.hash_features().compute_similarity().get_results()
    """

    def __init__(self, n_population: int | None = None):
        """
        Initialize analyzer.

        Args:
            n_population: Size of population (optional, inferred from subtrees if not provided)
        """
        self.n_population = n_population
        self.subtrees = None      # Raw token dictionaries
        self.matrices = None      # Feature matrices: matrices[space][radius] = sparse_matrix
        self.similarity = None    # Similarity matrices
        self.embeddings = None    # 2D embeddings for visualization
        self._config = AnalysisConfig()

    @classmethod
    def from_subtrees(
        cls,
        subtrees: dict[VectorSpace, list[dict[int, list[str]] | None]],
        config: AnalysisConfig | None = None,
    ) -> "SubtreeAnalyzer":
        """
        Create analyzer from pre-collected subtrees.

        Args:
            subtrees: Dictionary mapping VectorSpace to list of token dicts
            config: Optional analysis configuration

        Returns:
            Initialized SubtreeAnalyzer ready for feature hashing

        Example:
            >>> subtrees = {
            ...     VectorSpace.ENTIRE_ROBOT: [{0: ['tok1'], 1: ['tok2']}, ...],
            ...     VectorSpace.FRONT_LIMB: [...]
            ... }
            >>> analyzer = SubtreeAnalyzer.from_subtrees(subtrees)
            >>> analyzer.hash_features().compute_similarity()
        """
        # Infer population size from first available space
        n_population = len(next(iter(subtrees.values())))

        instance = cls(n_population=n_population)
        instance.subtrees = subtrees

        if config is not None:
            instance._config = config

        return instance

    def set_subtrees(
        self,
        subtrees: dict[VectorSpace, list[dict[int, list[str]] | None]],
    ) -> "SubtreeAnalyzer":
        """
        Set pre-collected subtrees (alternative to from_subtrees).

        Args:
            subtrees: Dictionary mapping VectorSpace to list of token dicts

        Returns:
            self (for method chaining)
        """
        self.subtrees = subtrees

        # Infer population size if not already set
        if self.n_population is None:
            self.n_population = len(next(iter(subtrees.values())))

        return self

    def hash_features(
        self,
        strategy: FeatureStrategy = FeatureStrategy.HASH_COUNT,
        radius_mode: RadiusAggregation | None = None,
        n_features: int | None = None
    ) -> 'SubtreeAnalyzer':
        """
        Convert tokens to feature matrices.

        Args:
            strategy: How to extract/weight features
            radius_mode: How to aggregate across depth levels
            n_features: Hash dimension (power of 2 recommended)

        Returns:
            self (for method chaining)

        Result Structure:
            self.matrices = {
                VectorSpace.ENTIRE_ROBOT: {
                    0: <sparse_matrix>,      # if PER_RADIUS or BOTH
                    1: <sparse_matrix>,
                    "all": <sparse_matrix>   # if FLATTENED or BOTH
                },
                VectorSpace.FRONT_LIMB: {...},
                VectorSpace.ALL_COMBINED: {...}
            }
        """
        if self.subtrees is None:
            raise ValueError("Must call collect_subtrees() before hash_features()")

        if radius_mode is None:
            radius_mode = self._config.radius_mode
        if n_features is None:
            n_features = self._config.n_features

        hasher = self._get_hasher(strategy, n_features)
        self.matrices = self._build_matrices(hasher, radius_mode)
        return self

    def compute_similarity(
        self,
        metric: SimilarityMetric = SimilarityMetric.COSINE
    ) -> 'SubtreeAnalyzer':
        """
        Compute pairwise similarity matrices.

        Args:
            metric: Distance/similarity metric to use

        Returns:
            self (for method chaining)

        Result Structure:
            self.similarity = {
                VectorSpace.ENTIRE_ROBOT: {
                    0: <NxN matrix>,
                    1: <NxN matrix>,
                    "all": <NxN matrix>
                },
                VectorSpace.FRONT_LIMB: {...},
                ...
            }
        """
        if self.matrices is None:
            raise ValueError("Must call hash_features() before compute_similarity()")

        self.similarity = self._compute_sim(metric)
        return self

    def reduce_dimensions(
        self,
        method: DimReductionMethod = DimReductionMethod.UMAP,
        **kwargs
    ) -> 'SubtreeAnalyzer':
        """
        Create 2D embeddings for visualization.

        Args:
            method: Dimensionality reduction algorithm
            **kwargs: Method-specific parameters (e.g., n_neighbors for UMAP)

        Returns:
            self (for method chaining)

        Result Structure:
            self.embeddings = {
                VectorSpace.ENTIRE_ROBOT: {
                    0: <Nx2 array>,
                    1: <Nx2 array>,
                    "all": <Nx2 array>
                },
                ...
            }
        """
        if self.matrices is None:
            raise ValueError("Must call hash_features() before reduce_dimensions()")

        self.embeddings = self._reduce_dims(method, **kwargs)
        return self

    def get_results(self) -> dict[str, Any]:
        """
        Return all computed artifacts.

        Returns:
            Dictionary containing subtrees, matrices, similarity, and embeddings
        """
        return {
            'subtrees': self.subtrees,
            'matrices': self.matrices,
            'similarity': self.similarity,
            'embeddings': self.embeddings
        }

    # ==================== INTERNAL METHODS ====================

    def _get_hasher(self, strategy: FeatureStrategy, n_features: int):
        """Map enum to actual sklearn object."""
        if strategy == FeatureStrategy.HASH_COUNT:
            return FeatureHasher(n_features=n_features, input_type='string')

        elif strategy == FeatureStrategy.TFIDF:
            return TfidfVectorizer(
                max_features=n_features,
                token_pattern=r'(?u)\b\w+\b',
                lowercase=False
            )

        elif strategy == FeatureStrategy.HASH_BINARY:
            return FeatureHasher(
                n_features=n_features,
                input_type='string',
                alternate_sign=False
            )

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _build_matrices(
        self,
        hasher: FeatureHasher | TfidfVectorizer,
        radius_mode: RadiusAggregation,
    ) -> dict[VectorSpace, dict[int | str, Any]]:
        """
        The key logic: build feature matrices according to radius strategy.

        Returns:
            Dictionary of {space_name: {radius_key: sparse_matrix}}
        """
        matrices = {}
        global_radii = set()

        # Phase 1: Individual vector spaces
        for space, robot_token_dicts in self.subtrees.items():
            if space == VectorSpace.ALL_COMBINED:
                continue  # Handle meta-space separately

            matrices[space] = {}

            # Collect all radii present in this space
            space_radii = set()
            for token_dict in robot_token_dicts:
                if token_dict:
                    space_radii.update(token_dict.keys())
            global_radii.update(space_radii)

            # Build matrices according to strategy
            if radius_mode in [RadiusAggregation.PER_RADIUS, RadiusAggregation.BOTH]:
                # One matrix per radius depth
                for r in sorted(space_radii):
                    corpus = self._extract_corpus(robot_token_dicts, radius=r)
                    matrices[space][r] = hasher.fit_transform(corpus)

            if radius_mode in [RadiusAggregation.FLATTENED, RadiusAggregation.BOTH]:
                # Single matrix with all radii merged
                corpus = self._extract_corpus(robot_token_dicts, radius='all')
                matrices[space]['all'] = hasher.fit_transform(corpus)

            if radius_mode == RadiusAggregation.CUMULATIVE:
                # Progressive: r0, r0+r1, r0+r1+r2, ...
                for r in sorted(space_radii):
                    corpus = self._extract_corpus(
                        robot_token_dicts,
                        radius=list(range(r+1))
                    )
                    matrices[space][f'cumulative_{r}'] = hasher.fit_transform(corpus)

        # Phase 2: Meta-space (ALL_COMBINED)
        if VectorSpace.ALL_COMBINED in self.subtrees:
            matrices[VectorSpace.ALL_COMBINED] = self._build_meta_space_matrices(
                hasher, global_radii, radius_mode
            )

        return matrices

    def _extract_corpus(
        self,
        robot_token_dicts: list[dict[int, list[str]] | None],
        radius: int | str | list[int],
    ) -> list[list[str]]:
        """
        Extract tokens at specific radius (or aggregated across radii).

        Args:
            robot_token_dicts: List of {radius: [tokens]} per robot
            radius: int, 'all', or list of ints

        Returns:
            List of token lists (one per robot)
        """
        corpus = []

        for token_dict in robot_token_dicts:
            if token_dict is None:
                corpus.append([])
                continue

            if radius == 'all':
                # Flatten all radii
                tokens = [t for token_list in token_dict.values() for t in token_list]
            elif isinstance(radius, list):
                # Cumulative: specific radii only
                tokens = [t for r in radius if r in token_dict
                         for t in token_dict[r]]
            elif radius in token_dict:
                # Single radius
                tokens = token_dict[radius]
            else:
                tokens = []

            corpus.append(tokens)

        return corpus

    def _build_meta_space_matrices(
        self,
        hasher: FeatureHasher | TfidfVectorizer,
        global_radii: set[int],
        radius_mode: RadiusAggregation,
    ) -> dict[int | str, Any]:
        """Build matrices for the ALL_COMBINED meta-space."""
        matrices = {}
        n_robots = len(self.population)

        if radius_mode in [RadiusAggregation.PER_RADIUS, RadiusAggregation.BOTH]:
            # Per-radius: combine same radius across all spaces
            for r in sorted(global_radii):
                layer_corpus = []
                for i in range(n_robots):
                    robot_tokens = []
                    for space in self.subtrees:
                        if space == VectorSpace.ALL_COMBINED:
                            continue
                        ind = self.subtrees[space][i]
                        if ind and r in ind:
                            robot_tokens.extend(ind[r])
                    layer_corpus.append(robot_tokens)
                matrices[r] = hasher.fit_transform(layer_corpus)

        if radius_mode in [RadiusAggregation.FLATTENED, RadiusAggregation.BOTH]:
            # Flattened: all tokens from all spaces and radii
            grand_corpus = []
            for i in range(n_robots):
                robot_tokens = []
                for space in self.subtrees:
                    if space == VectorSpace.ALL_COMBINED:
                        continue
                    ind = self.subtrees[space][i]
                    if ind:
                        for r_tokens in ind.values():
                            robot_tokens.extend(r_tokens)
                grand_corpus.append(robot_tokens)
            matrices["all"] = hasher.fit_transform(grand_corpus)

        return matrices

    def _compute_sim(self, metric: SimilarityMetric) -> dict[VectorSpace, dict[int | str, Any]]:
        """Compute similarity matrices for all spaces and radii."""
        similarity = {}

        for space_name, radius_dict in self.matrices.items():
            similarity[space_name] = {}

            for r, feature_matrix in radius_dict.items():
                if metric == SimilarityMetric.COSINE:
                    sim_matrix = cosine_similarity(feature_matrix)

                elif metric == SimilarityMetric.EUCLIDEAN:
                    # Convert distance to similarity
                    dist = euclidean_distances(feature_matrix)
                    sim_matrix = 1.0 / (1.0 + dist)

                elif metric == SimilarityMetric.JACCARD:
                    sim_matrix = pairwise_distances(
                        feature_matrix,
                        metric='jaccard',
                        n_jobs=-1
                    )
                    sim_matrix = 1.0 - sim_matrix  # Distance -> Similarity

                elif metric == SimilarityMetric.HAMMING:
                    sim_matrix = pairwise_distances(
                        feature_matrix.toarray(),
                        metric='hamming'
                    )
                    sim_matrix = 1.0 - sim_matrix

                else:
                    raise ValueError(f"Unknown metric: {metric}")

                similarity[space_name][r] = sim_matrix

        return similarity

    def _reduce_dims(
        self,
        method: DimReductionMethod,
        **kwargs: Any,
    ) -> dict[VectorSpace, dict[int | str, Any]]:
        """Apply dimensionality reduction to create 2D embeddings."""
        embeddings = {}

        for space_name, radius_dict in self.matrices.items():
            embeddings[space_name] = {}

            for r, feature_matrix in radius_dict.items():
                if method == DimReductionMethod.UMAP:
                    reducer = umap.UMAP(
                        n_components=2,
                        n_neighbors=kwargs.get('n_neighbors', 15),
                        min_dist=kwargs.get('min_dist', 0.01),
                        metric=kwargs.get('metric', 'cosine'),
                        random_state=kwargs.get('random_state', 42)
                    )

                elif method == DimReductionMethod.PCA:
                    reducer = PCA(
                        n_components=2,
                        random_state=kwargs.get('random_state', 42)
                    )

                elif method == DimReductionMethod.TSNE:
                    reducer = TSNE(
                        n_components=2,
                        perplexity=kwargs.get('perplexity', 30),
                        random_state=kwargs.get('random_state', 42)
                    )

                else:
                    raise ValueError(f"Unknown method: {method}")

                embedding = reducer.fit_transform(feature_matrix)
                embedding = embedding - embedding.mean(axis=0)  # Center around origin

                embeddings[space_name][r] = embedding

        return embeddings


# ==================== USAGE EXAMPLES ====================

if __name__ == '__main__':
    """
    Example usage of the SubtreeAnalyzer API.

    NOTE: These examples assume you have a population of robots loaded.
    """

    # This is a placeholder - you would load your actual population
    print("SubtreeAnalyzer Examples")
    print("=" * 60)

    # Example 1: Basic usage with all defaults
    print("\n1. Basic Usage (all spaces, both radius modes):")
    print("-" * 60)
    print("""
    analyzer = SubtreeAnalyzer(population)
    results = (analyzer
        .collect_subtrees()
        .hash_features()
        .compute_similarity()
        .get_results()
    )

    # Results will have both per-radius and flattened matrices
    print(results['matrices']['core'].keys())
    # Output: dict_keys([0, 1, 2, ..., 'all'])
    """)

    # Example 2: Only analyze entire robot structure, flatten all depths
    print("\n2. Single Space, Flattened Radii:")
    print("-" * 60)
    print("""
    results = (SubtreeAnalyzer(population)
        .collect_subtrees(spaces=[VectorSpace.ENTIRE_ROBOT])
        .hash_features(radius_mode=RadiusAggregation.FLATTENED)
        .compute_similarity()
        .get_results()
    )

    # Result: matrices = {'core': {'all': <matrix>}}
    """)

    # Example 3: Compare limb-specific spaces, preserve depth
    print("\n3. Limb Comparison with Per-Radius Matrices:")
    print("-" * 60)
    print("""
    results = (SubtreeAnalyzer(population)
        .collect_subtrees(spaces=VectorSpace.limb_spaces_only())
        .hash_features(radius_mode=RadiusAggregation.PER_RADIUS)
        .compute_similarity()
    )

    # Result: matrices = {
    #   'front': {0: <mat>, 1: <mat>, 2: <mat>, ...},
    #   'left': {0: <mat>, 1: <mat>, ...},
    #   'back': {...},
    #   'right': {...}
    # }
    """)

    # Example 4: UMAP embeddings for visualization
    print("\n4. Create UMAP Embeddings:")
    print("-" * 60)
    print("""
    results = (SubtreeAnalyzer(population)
        .collect_subtrees()
        .hash_features(
            strategy=FeatureStrategy.TFIDF,
            radius_mode=RadiusAggregation.BOTH
        )
        .reduce_dimensions(
            method=DimReductionMethod.UMAP,
            n_neighbors=15,
            min_dist=0.01
        )
    )

    # Access embeddings
    embeddings = results.embeddings['core'][0]  # Radius 0 embeddings
    # Shape: (n_robots, 2) for plotting
    """)

    # Example 5: Different similarity metrics
    print("\n5. Try Different Similarity Metrics:")
    print("-" * 60)
    print("""
    for metric in SimilarityMetric:
        results = (SubtreeAnalyzer(population)
            .collect_subtrees()
            .hash_features()
            .compute_similarity(metric=metric)
            .get_results()
        )

        avg_sim = results['similarity']['core'][0].mean()
        print(f"{metric.name}: {avg_sim:.4f}")
    """)

    # Example 6: Cumulative radius aggregation
    print("\n6. Cumulative Radius Aggregation:")
    print("-" * 60)
    print("""
    results = (SubtreeAnalyzer(population)
        .collect_subtrees()
        .hash_features(radius_mode=RadiusAggregation.CUMULATIVE)
        .compute_similarity()
    )

    # Result: matrices = {
    #   'core': {
    #       'cumulative_0': <r0 only>,
    #       'cumulative_1': <r0 + r1>,
    #       'cumulative_2': <r0 + r1 + r2>,
    #       ...
    #   }
    # }
    """)

    print("\n" + "=" * 60)
    print("See the module docstrings for more details!")

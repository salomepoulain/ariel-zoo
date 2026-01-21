"""SimilarityMatrix: Specialized matrix for tree similarity analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from ....base.matrix import MatrixInstance
from ..options import MatrixDomain, Space, UmapConfig

if TYPE_CHECKING:
    from pathlib import Path

__all__ = ["SimilarityMatrix"]


class SimilarityMatrix(MatrixInstance):
    """
    Matrix specialized for tree similarity analysis.

    Adds typed attributes (space, radius, domain) on top of the generic
    MatrixInstance base class.
    """

    def __init__(
        self,
        matrix: sp.spmatrix | np.ndarray,
        space: Space | str,
        radius: int,
        domain: MatrixDomain | str = MatrixDomain.FEATURES,
        tags: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize similarity matrix with typed attributes.

        Args:
            matrix: Sparse or dense matrix data
            space: VectorSpace enum for morphological space (REQUIRED)
            radius: Neighborhood radius (REQUIRED)
            domain: MatrixDomain enum or string (default: FEATURES)
            tags: Optional extra metadata tags
        """
        final_tags = tags.copy() if tags else {}

        domain_str = (
            domain.name if isinstance(domain, MatrixDomain) else str(domain)
        )
        final_tags["domain"] = domain_str
        final_tags["radius"] = radius

        super().__init__(
            matrix=matrix,
            label=space,
            tags=final_tags,
        )

    # --- Similarity Specific Properties ---

    @property
    def space(self) -> str:
        """The morphological space (e.g., 'FRONT')."""
        return self._label

    @property
    def domain(self) -> str:
        """The data domain (FEATURES, SIMILARITY, EMBEDDING)."""
        return self._tags["domain"]

    @property
    def radius(self) -> int:
        """The neighborhood radius."""
        return self._tags["radius"]
    
    def replace(self, **changes) -> SimilarityMatrix:
        """
        Returns a new SimilarityMatrix with updated fields.
        Maps generic keys (label) to specific ones (space).
        """
        # Map generic keys to specific ones
        if "label" in changes:
            changes["space"] = changes.pop("label")

        new_args = {
            "matrix": self._matrix,
            "space": self.space,
            "radius": self.radius,
            "domain": self.domain,
            "tags": self._tags.copy(),
        } | changes
        valid_keys = {"matrix", "space", "radius", "domain", "tags"}
        filtered_args = {k: v for k, v in new_args.items() if k in valid_keys}

        return SimilarityMatrix(**filtered_args)
    
    @classmethod
    def load(cls, file_path: Path | str):
        """Load SimilarityMatrix from disk.

        Overrides base class load() to translate generic format (label, tags)
        to similarity-specific format (space, radius, domain).

        Args:
            file_path: Full path without extension

        Returns
        -------
            Loaded SimilarityMatrix
        """
        import json
        from pathlib import Path

        file_path = Path(file_path)
        folder = file_path.parent
        filename_stem = file_path.name

        # Load metadata
        json_path = folder / f"{filename_stem}.json"
        if not json_path.exists():
            msg = f"Metadata not found: {json_path}"
            raise FileNotFoundError(msg)

        with json_path.open(encoding="utf-8") as f:
            info = json.load(f)

        # Extract similarity-specific attributes
        space = info["label"]  # label → space
        tags = info.get("tags", {})
        radius = tags.get("radius", 0)
        domain = tags.get("domain", MatrixDomain.FEATURES.name)

        # Load matrix
        matrix_path = folder / info["matrix_file"]
        if info["storage"] == "sparse":
            matrix = sp.load_npz(matrix_path)
        else:
            matrix = np.load(matrix_path)

        # Call __init__ with similarity-specific signature
        return cls(
            matrix=matrix,
            space=space,
            radius=radius,
            domain=domain,
            tags=tags,
        )

    # --- Transformation Overrides ---
    
    def __or__(self, other: SimilarityMatrix) -> SimilarityMatrix:
        """
        Vertically stack matrices (concatenate rows). 
        Maps to the '|' operator.
        """
        if not isinstance(other, SimilarityMatrix):
            return NotImplemented

        # Domain check for safety
        if self.domain != "FEATURES" or other.domain != "FEATURES":
            raise ValueError(f"Vertical stacking (|) only allowed for 'FEATURES' domain.")

        # Perform the actual stacking
        # sp.vstack works for both sparse and dense (converts to sparse if needed)
        new_matrix = sp.vstack([self._matrix, other._matrix])
        
        return self.replace(matrix=new_matrix)

    def replace(self, **changes: Any) -> SimilarityMatrix:
        """Returns a new SimilarityMatrix with updated fields."""
        # Map label back to space if necessary
        if "label" in changes:
            changes["space"] = changes.pop("label")

        # Extract current state
        new_args = {
            "matrix": self._matrix,
            "space": self.space,
            "radius": self.radius,
            "domain": self.domain,
            "tags": self._tags.copy(),
        } | changes

        # Only pass valid SimilarityMatrix __init__ keys
        valid_keys = {"matrix", "space", "radius", "domain", "tags"}
        filtered = {k: v for k, v in new_args.items() if k in valid_keys}
        return self.__class__(**filtered)

    # --- Math & Analysis ---

    def normalize_by_radius(self, in_place: bool = True) -> SimilarityMatrix:
        """Normalize matrix by dividing by (radius + 1)."""
        normalized_values = self._matrix / (self.radius + 1)
        if in_place:
            # Note: _matrix is the internal ref, self.matrix (property) is read-only
            self._matrix = normalized_values
            return self
        return self.replace(matrix=normalized_values)

    def cosine_similarity(self) -> SimilarityMatrix:
        """Compute cosine similarity matrix."""
        if self.domain == MatrixDomain.SIMILARITY.name:
            raise ValueError("Matrix is already a similarity matrix")

        # Handle all-zero matrices efficiently
        nnz = self._matrix.nnz if sp.issparse(self._matrix) else np.count_nonzero(self._matrix)
        if nnz == 0:
            n_rows = self.shape[0]
            sim_matrix = np.zeros((n_rows, n_rows))
        else:
            sim_matrix = cosine_similarity(self._matrix)

        return self.replace(matrix=sim_matrix, domain=MatrixDomain.SIMILARITY)

    def sum_to_rows(
        self,
        *,
        zero_diagonal: bool = True,
        k: int | None = None,
        largest: bool = True,
        normalise: bool = True,
    ) -> np.ndarray:
        """Sum similarity scores per row (neighborhood density)."""
        # Efficiently convert to dense only if needed for diagonal filling
        matrix = self._matrix.toarray() if sp.issparse(self._matrix) else self._matrix.copy()
        
        if zero_diagonal:
            np.fill_diagonal(matrix, 0)

        if k is None:
            scores = matrix.sum(axis=1)
            if normalise:
                n_summed = matrix.shape[1] - (1 if zero_diagonal else 0)
                scores /= max(n_summed, 1)
            return scores

        # Top-K Logic
        if largest:
            partitioned = np.partition(matrix, -k, axis=1)[:, -k:]
        else:
            partitioned = np.partition(matrix, k-1, axis=1)[:, :k]
        
        scores = partitioned.sum(axis=1)
        return scores / k if normalise else scores

    def umap_embed(self, *, config: UmapConfig | None = None, **umap_kwargs: Any) -> SimilarityMatrix:
        """Reduce dimensionality using UMAP."""
        try:
            from umap import UMAP
        except ImportError as e:
            raise ImportError("pip install umap-learn to use this method.") from e

        is_similarity = self.domain == MatrixDomain.SIMILARITY.name

        # Merge config and direct kwargs
        final_config = config if config else UmapConfig(
            metric="precomputed" if is_similarity else "cosine"
        )
        kwargs = final_config.get_kwargs() | umap_kwargs

        model = UMAP(**kwargs)
        # Use .matrix property to ensure we follow our own protocols
        embedding = model.fit_transform(self.matrix)

        return self.replace(matrix=embedding, domain=MatrixDomain.EMBEDDING)

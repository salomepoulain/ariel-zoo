
# ===== Similarity-Specific Matrix Subclass =====
from __future__ import annotations

from canonical_toolkit.core.matrix.matrix import MatrixInstance
from canonical_toolkit.core.similarity.options import VectorSpace, MatrixDomain

import io
import json
from pathlib import Path
from typing import Any, Hashable

import numpy as np
import scipy.sparse as sp
from rich.console import Console
from rich.table import Table
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP


class SimilarityMatrix(MatrixInstance):
    """
    Matrix specialized for tree similarity analysis.

    Adds typed attributes (space, radius, domain) on top of the generic
    MatrixInstance base class.
    """

    def __init__(
        self,
        matrix: sp.spmatrix | Any,
        space: VectorSpace,
        radius: int,
        domain: MatrixDomain = MatrixDomain.FEATURES,
    ) -> None:
        """
        Initialize similarity matrix with typed attributes.

        Args:
            matrix: Sparse or dense matrix data
            space: VectorSpace enum for morphological space
            radius: Neighborhood radius (int)
            domain: MatrixDomain enum (FEATURES, SIMILARITY, EMBEDDING)
        """
        # Map domain-specific params to generic structure
        super().__init__(
            matrix=matrix,
            label=space.name,
            index=radius,
            tags={"domain": domain.name.lower()},
        )

    @property
    def radius(self) -> Hashable:
        return self._index

    @property
    def space(self) -> str:
        return self._label

    @property
    def domain(self) -> str:
        return self.tags['domain']

    def cosine_similarity(self) -> SimilarityMatrix:
        """Compute cosine similarity matrix."""
        if self._tags.get("domain") == "similarity":
            raise ValueError("Matrix is already a similarity matrix")
        sim_matrix = cosine_similarity(self._matrix)
        return SimilarityMatrix(sim_matrix, VectorSpace[self.space], self.radius, MatrixDomain.SIMILARITY)

    def sum_rows(self, *, zero_diagonal: bool = True, k: int | None = None, largest: bool = True) -> np.ndarray:
        """Sum similarity scores per row. Optionally use top-k neighbors."""
        if self._tags.get("domain") != "similarity":
            raise ValueError(f"Requires similarity domain, found {self._tags.get('domain')}")

        matrix = self._matrix.toarray() if sp.issparse(self._matrix) else self._matrix.copy()
        if zero_diagonal:
            np.fill_diagonal(matrix, 0)
        if k is None:
            return matrix.sum(axis=1)

        # Top-k or bottom-k per row
        scores = np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            indices = np.argpartition(matrix[i], -k if largest else k)[(-k if largest else slice(k)):]
            scores[i] = matrix[i][indices].sum()
        return scores

    def umap_embed(self, *, n_neighbors: int = 15, n_components: int = 2, metric: str = "cosine",
                   random_state: int | None = 42, **umap_kwargs) -> SimilarityMatrix:
        """Reduce dimensionality using UMAP. Works on features or similarity matrices."""
        if self._tags.get("domain") == "embedding":
            raise ValueError("Matrix is already an embedding")

        matrix = self._matrix.toarray() if sp.issparse(self._matrix) else self._matrix
        is_similarity = self._tags.get("domain") == "similarity"

        umap_model = UMAP(
            n_neighbors=n_neighbors, n_components=n_components,
            metric="precomputed" if is_similarity else metric,
            random_state=random_state, init="random",
            transform_seed=random_state if random_state else None,
            n_jobs=1 if random_state else -1, **umap_kwargs
        )
        embedding = umap_model.fit_transform(1 - matrix if is_similarity else matrix)
        return SimilarityMatrix(embedding, VectorSpace[self.space], self.radius, MatrixDomain.EMBEDDING)


if __name__ == "__main__":
    print("Testing SimilarityMatrix...")

    # Test 1: Create sparse feature matrix
    sparse_features = sp.random(10, 50, density=0.3, format="csr", random_state=42)
    sim_mat = SimilarityMatrix(
        matrix=sparse_features,
        space=VectorSpace.FRONT,
        radius=3,
        domain=MatrixDomain.FEATURES
    )

    print(sim_mat)

    print(f"✓ SimilarityMatrix created: {sim_mat.short_description}")
    print(f"  Space (typed): {sim_mat.space}")
    print(f"  Radius (typed): {sim_mat.radius}")
    print(f"  Domain (typed): {sim_mat.domain}")
    print(f"  Label (generic): {sim_mat.label}")
    print(f"  Index (generic): {sim_mat.index}")

    # Test 2: Cosine similarity transformation
    print("\n✓ Computing cosine similarity...")
    sim_matrix = sim_mat.cosine_similarity()
    print(f"  Result: {sim_matrix.short_description}")
    print(f"  Domain: {sim_matrix.domain}")
    print(f"  Shape: {sim_matrix.shape}")

    # Test 3: Sum similarity scores
    print("\n✓ Summing similarity scores...")
    scores = sim_matrix.sum_rows(zero_diagonal=True)
    print(f"  Scores shape: {scores.shape}")
    print(f"  Mean score: {scores.mean():.3f}")

    # Test 4: Top-k similarity scores
    print("\n✓ Top-3 similarity scores...")
    top3_scores = sim_matrix.sum_rows(k=3, largest=True)
    print(f"  Top-3 mean: {top3_scores.mean():.3f}")

    # Test 5: UMAP embedding
    print("\n✓ Computing UMAP embedding...")
    embedding = sim_matrix.umap_embed(n_neighbors=5, n_components=2, random_state=42)
    print(f"  Embedding: {embedding.short_description}")
    print(f"  Domain: {embedding.domain}")
    print(f"  Shape: {embedding.shape}")

    # Test 6: Pretty printing
    print("\n✓ Display matrix:")
    print(sim_mat)

    print("\n✅ All SimilarityMatrix tests passed!")

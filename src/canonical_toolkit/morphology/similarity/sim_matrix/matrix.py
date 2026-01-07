# ===== Similarity-Specific Matrix Subclass =====
from __future__ import annotations


import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP
from typing import Any, cast

from ....base.matrix import MatrixInstance

from ..options import VectorSpace, MatrixDomain


class SimilarityMatrix(MatrixInstance):
    """
    Matrix specialized for tree similarity analysis.

    Adds typed attributes (space, radius, domain) on top of the generic
    MatrixInstance base class.
    """

    def __init__(
        self,
        matrix: sp.spmatrix | np.ndarray,
        space: VectorSpace | str,
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
            domain: MatrixDomain enum or string
            tags: Optional extra metadata tags
        """
        # Build final tags dict
        final_tags = tags.copy() if tags else {}

        # Resolve domain string
        domain_str = domain.name if isinstance(domain, MatrixDomain) else str(domain)

        # Store domain and radius in tags
        final_tags["domain"] = domain_str
        final_tags["radius"] = radius

        # Map domain-specific params to generic structure
        super().__init__(
            matrix=matrix,
            label=space,  # space becomes label
            tags=final_tags,
        )

    @property
    def space(self) -> str:
        return self._label

    @property
    def domain(self) -> str:
        return self.tags.get('domain', '')

    @property
    def radius(self) -> int:
        return int(self.tags.get('radius', 0))
    
    def replace(self, **changes) -> SimilarityMatrix:
        """
        Returns a new SimilarityMatrix with updated fields.
        Maps generic keys (label) to specific ones (space).
        """
        # Map generic keys to specific ones
        if "label" in changes:
            changes["space"] = changes.pop("label")

        # Prepare arguments
        new_args = {
            "matrix": self._matrix,
            "space": self.space,
            "radius": self.radius,
            "domain": self.domain,
            "tags": self._tags.copy(),
        }
        new_args.update(changes)

        valid_keys = {"matrix", "space", "radius", "domain", "tags"}
        filtered_args = {k: v for k, v in new_args.items() if k in valid_keys}

        return SimilarityMatrix(**filtered_args)

    @classmethod
    def load(cls, file_path):
        """Load SimilarityMatrix from disk.

        Overrides base class load() to translate generic format (label, tags)
        to similarity-specific format (space, radius, domain).

        Args:
            file_path: Full path without extension

        Returns:
            Loaded SimilarityMatrix
        """
        from pathlib import Path
        import json

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

    def normalize_by_radius(self) -> SimilarityMatrix:
        """Normalize matrix by dividing by (radius + 1).

        Uses the instance's stored radius value for normalization.

        Returns:
            New SimilarityMatrix with normalized values
        """
        if self._tags.get("domain") != MatrixDomain.SIMILARITY.name:
            raise ValueError(f"Requires similarity domain, found {self._tags.get('domain')}")
        matrix_as_array = cast(np.ndarray, self._matrix)
        normalized_matrix = matrix_as_array / (self.radius + 1)
        return SimilarityMatrix(
            matrix=normalized_matrix,
            space=self.space,
            radius=self.radius,
            domain=self.domain,
            tags=self.tags,
        )
    
    def cosine_similarity(self) -> SimilarityMatrix:
        """Compute cosine similarity matrix."""
        if self._tags.get("domain") == MatrixDomain.SIMILARITY.name:
            raise ValueError("Matrix is already a similarity matrix")
        sim_matrix = cosine_similarity(self._matrix)
        return SimilarityMatrix(sim_matrix, self.space, self.radius, MatrixDomain.SIMILARITY)

    def sum_to_rows(
        self,
        *,
        zero_diagonal: bool = True,
        k: int | None = None,
        largest: bool = True,
        normalise_by_pop_len: bool = True,
    ) -> np.ndarray:
        """Sum similarity scores per row. Optionally use top-k neighbors.

        Args:
            zero_diagonal: If True, set diagonal to zero before summing
            k: If specified, only sum top-k or bottom-k values
            largest: If True, use k largest values; if False, use k smallest
            normalise_by_pop_len: If True, divide sum by number of values summed

        Returns:
            Array of scores per row (normalized if normalise_by_pop_len=True)
        """
        if self._tags.get("domain") != MatrixDomain.SIMILARITY.name:
            raise ValueError(f"Requires similarity domain, found {self._tags.get('domain')}")

        matrix = self._matrix.toarray() if sp.issparse(self._matrix) else self._matrix.copy()
        if zero_diagonal:
            np.fill_diagonal(matrix, 0)

        if k is None:
            # Sum all values
            scores = matrix.sum(axis=1)
            if normalise_by_pop_len:
                # Divide by number of values summed per row
                n_summed = matrix.shape[1] - (1 if zero_diagonal else 0)
                scores = scores / n_summed
            return scores

        # Top-k or bottom-k per row
        scores = np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            if largest:
                # Get indices of k largest values
                indices = np.argpartition(matrix[i], -k)[-k:]
            else:
                # Get indices of k smallest values
                indices = np.argpartition(matrix[i], k-1)[:k]
            scores[i] = matrix[i][indices].sum()

        if normalise_by_pop_len:
            # Divide by k (number of values summed)
            scores = scores / k

        return scores

    def umap_embed(self, *, n_neighbors: int = 15, n_components: int = 2, metric: str = "cosine",
                   random_state: int | None = 42, **umap_kwargs) -> SimilarityMatrix:
        """Reduce dimensionality using UMAP. Works on features or similarity matrices."""
        if self._tags.get("domain") == MatrixDomain.EMBEDDING.name:
            raise ValueError("Matrix is already an embedding")

        matrix = self._matrix.toarray() if sp.issparse(self._matrix) else self._matrix
        is_similarity = self._tags.get("domain") == MatrixDomain.SIMILARITY.name

        umap_model = UMAP(
            n_neighbors=n_neighbors, n_components=n_components,
            metric="precomputed" if is_similarity else metric,
            random_state=random_state, init="random",
            transform_seed=random_state if random_state else None,
            n_jobs=1 if random_state else -1, **umap_kwargs
        )
        embedding = umap_model.fit_transform(matrix) # wtf explain?
        return SimilarityMatrix(embedding, VectorSpace[self.space], self.radius, MatrixDomain.EMBEDDING)



if __name__ == "__main__":
    import scipy.sparse as sp
    from ..options import VectorSpace, MatrixDomain

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

    # Test 2: Cosine similarity transformation
    print("\n✓ Computing cosine similarity...")
    sim_matrix = sim_mat.cosine_similarity()
    print(f"  Result: {sim_matrix.short_description}")
    print(f"  Domain: {sim_matrix.domain}")
    print(f"  Shape: {sim_matrix.shape}")

    # Test 3: Sum similarity scores
    print("\n✓ Summing similarity scores...")
    scores = sim_matrix.sum_to_rows(zero_diagonal=True)
    print(f"  Scores shape: {scores.shape}")
    print(f"  Mean score: {scores.mean():.3f}")

    # Test 4: Top-k similarity scores
    print("\n✓ Top-3 similarity scores...")
    top3_scores = sim_matrix.sum_to_rows(k=3, largest=True)
    print(f"  Top-3 mean: {top3_scores.mean():.3f}")

    # Test 5: UMAP embedding
    print("\n✓ Computing UMAP embedding...")
    embedding = sim_matrix.umap_embed(n_neighbors=5, n_components=2, random_state=42)
    print(f"  Embedding: {embedding.short_description}")
    print(f"  Domain: {embedding.domain}")
    print(f"  Shape: {embedding.shape}")
    print(embedding)

    # Test 6: Pretty printing
    print("\n✓ Display matrix:")
    print(sim_mat)
    
    
    # Test 7: Chaining:
    print("\n✓ Chaining operations...")
    print(sim_mat.cosine_similarity())
    print(sim_mat.cosine_similarity().sum_to_rows(zero_diagonal=False))
    print(sim_mat.cosine_similarity().sum_to_rows())
    print(sim_mat.cosine_similarity().sum_to_rows().mean())
    

    print("\n✅ All SimilarityMatrix tests passed!")

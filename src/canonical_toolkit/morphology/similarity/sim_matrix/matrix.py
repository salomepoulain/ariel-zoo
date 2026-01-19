from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
# from umap import UMAP

if TYPE_CHECKING:
    from pathlib import Path

from ....base.matrix import MatrixInstance
from ..options import MatrixDomain, Space, UmapConfig

__all__ = [
    "SimilarityMatrix",
]


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

    @property
    def space(self) -> str:
        return self._label

    @property
    def domain(self) -> str:
        return self.tags.get("domain", "")

    @property
    def radius(self) -> int:
        return int(self.tags.get("radius", 0))

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

    def __or__(self, other: SimilarityMatrix) -> SimilarityMatrix:
        """Vertically stack matrices (concatenate rows/individuals).

        Only allowed when domain is FEATURES.
        Use this to combine populations, e.g. current + archive.

        Args:
            other: Another SimilarityMatrix (must have same columns and be FEATURES)

        Returns
        -------
            New SimilarityMatrix with vstacked matrices
        """
        if not isinstance(other, SimilarityMatrix):
            return NotImplemented

        if self.domain != "FEATURES":
            msg = f"| only allowed for FEATURES domain, got {self.domain}"
            raise ValueError(msg)
        if other.domain != "FEATURES":
            msg = f"| only allowed for FEATURES domain, got {other.domain}"
            raise ValueError(msg)

        new_matrix = sp.vstack([self._matrix, other._matrix])
        return self.replace(matrix=new_matrix)

    def normalize_by_radius(self, in_place: bool = True) -> SimilarityMatrix:
        """Normalize matrix by dividing by (radius + 1)."""
        matrix_as_array = cast("np.ndarray", self._matrix)
        normalized_values = matrix_as_array / (self.radius + 1)

        if in_place:
            self._matrix = normalized_values
            return self

        return SimilarityMatrix(
            matrix=normalized_values,
            space=self.space,
            radius=self.radius,
            domain=self.domain,
            tags=self.tags,
        )

    def cosine_similarity(self) -> SimilarityMatrix:
        """Compute cosine similarity matrix."""
        if self._tags.get("domain") == MatrixDomain.SIMILARITY.name:
            msg = "Matrix is already a similarity matrix"
            raise ValueError(msg)
        sim_matrix = cosine_similarity(self._matrix)
        return SimilarityMatrix(
            sim_matrix,
            self.space,
            self.radius,
            MatrixDomain.SIMILARITY,
        )

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

        Returns
        -------
            Array of scores per row (normalized if normalise_by_pop_len=True)
        """
        # if self._tags.get("domain") != MatrixDomain.SIMILARITY.name:
        #     raise ValueError(
        #         f"Requires similarity domain, found {self._tags.get('domain')}"
        #     )

        matrix = (
            self._matrix.toarray()
            if sp.issparse(self._matrix)
            else self._matrix.copy()
        )
        if zero_diagonal:
            np.fill_diagonal(matrix, 0)

        if k is None:
            scores = matrix.sum(axis=1)
            if normalise_by_pop_len:
                n_summed = matrix.shape[1] - (1 if zero_diagonal else 0)
                scores /= n_summed
            return scores

        scores = np.zeros(matrix.shape[0])
        for i in range(matrix.shape[0]):
            if largest:
                indices = np.argpartition(matrix[i], -k)[-k:]
            else:
                indices = np.argpartition(matrix[i], k - 1)[:k]
            scores[i] = matrix[i][indices].sum()

        if normalise_by_pop_len:
            scores /= k

        return scores

    def get_max_indices(self, amt: int = 1) -> list[tuple[int, int]]:
        """Get indices of the highest values in the upper triangle (excluding diagonal)."""
        return self._get_sorted_indices(amt=amt, reverse=True)

    def get_min_indices(self, amt: int = 1) -> list[tuple[int, int]]:
        """Get indices of the lowest values in the upper triangle (excluding diagonal)."""
        return self._get_sorted_indices(amt=amt, reverse=False)

    def _get_sorted_indices(
        self, amt: int, reverse: bool
    ) -> list[tuple[int, int]]:
        """Helper to find top/bottom values in a symmetric matrix."""
        matrix = self._matrix
        if sp.issparse(matrix):
            matrix = matrix.toarray()

        # We only care about the upper triangle to avoid duplicates (A,B and B,A)
        # k=1 excludes the diagonal (self-similarity)
        rows, cols = np.triu_indices(matrix.shape[0], k=1)
        values = matrix[rows, cols]

        # Sort based on values
        if reverse:
            # Get indices of largest values
            idx = np.argsort(values)[-amt:][::-1]
        else:
            # Get indices of smallest values
            idx = np.argsort(values)[:amt]

        # Map back to matrix coordinates
        return list(zip(rows[idx], cols[idx], strict=False))

    def umap_embed(
        self,
        *,
        config: UmapConfig | None = None,
    ) -> SimilarityMatrix:
        """Reduce dimensionality using UMAP. Works on features or similarity matrices."""
        try:
            from umap import UMAP # Type: ignore
        except ImportError as e:
            raise ImportError(
                "umap-learn is required for this method. Install it with `pip install umap-learn`"
            ) from e

        is_similarity = self._tags.get("domain") == MatrixDomain.SIMILARITY.name

        if not config:
            config = UmapConfig(
                metric="precomputed" if is_similarity else "cosine"
            )

        kwargs = config.get_kwargs()

        if self._tags.get("domain") == MatrixDomain.EMBEDDING.name:
            msg = "Matrix is already an embedding"
            raise ValueError(msg)

        if kwargs.get("metric") == "cosine" and is_similarity:
            msg = "Similarity domain and UMAP cosine not compatible. Use cosine on raw Features or set UMAP metric to 'precomputed'"
            raise ValueError(msg)

        matrix = (
            self._matrix.toarray()
            if sp.issparse(self._matrix)
            else self._matrix
        )

        umap_model = UMAP(**kwargs)
        embedding = umap_model.fit_transform(matrix)

        return SimilarityMatrix(
            embedding,
            Space[self.space],
            self.radius,
            MatrixDomain.EMBEDDING,
        )


if __name__ == "__main__":
    import scipy.sparse as sp

    from ..options import MatrixDomain, Space

    # Test 1: Create sparse feature matrix
    sparse_features = sp.random(
        10,
        50,
        density=0.3,
        format="csr",
        random_state=42,
    )
    sim_mat = SimilarityMatrix(
        matrix=sparse_features,
        space=Space.FRONT,
        radius=3,
        domain=MatrixDomain.FEATURES,
    )

    # Test 2: Cosine similarity transformation
    sim_matrix = sim_mat.cosine_similarity()

    # Test 3: Sum similarity scores
    scores = sim_matrix.sum_to_rows(zero_diagonal=True)

    # Test 4: Top-k similarity scores
    top3_scores = sim_matrix.sum_to_rows(k=3, largest=True)

    # Test 5: UMAP embedding
    embedding = sim_matrix.umap_embed(
        n_neighbors=5,
        n_components=2,
        random_state=42,
    )

    # Test 6: Pretty printing

    # Test 7: Chaining:

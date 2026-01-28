"""SimilarityMatrix: Specialized matrix for tree similarity analysis."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Required,
    TypedDict,
    cast,
)

import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

from ....base.matrix import MatrixInstance
from ..options import MatrixDomain, UmapConfig

if TYPE_CHECKING:
    from .transformers import FitTransformer

__all__ = ["SimilarityMatrix"]


class SimilarityMatrixTags(TypedDict, total=False):
    """typed dict to help IDE integration of child class."""

    domain: Required[MatrixDomain]
    radius: Required[int]
    is_gap: bool


class SimilarityMatrix(MatrixInstance):
    """
    Matrix specialized for tree similarity analysis.

    Adds typed attributes (space, radius, domain) on top of the generic
    MatrixInstance base class.
    """

    def __init__(
        self,
        matrix: sp.spmatrix[Any] | np.ndarray[tuple[Any, ...], np.dtype[Any]],
        label: str,
        tags: SimilarityMatrixTags,
    ) -> None:
        assert tags.get("domain", None) is not None, (
            "SimilarityMatrix must contain domain tag"
        )
        assert tags.get("radius", None) is not None, (
            "SimilarityMatrix must contain radius tag"
        )

        super().__init__(matrix, label, tags)

    @property
    def space(self) -> str:
        return self._label

    @property
    def radius(self) -> int:
        return self.tags["radius"]

    @property
    def domain(self) -> str:
        return self.tags["domain"]

    # --- Transformation Overrides ---

    def __or__(self, other: SimilarityMatrix) -> SimilarityMatrix:
        """
        Vertically stack matrices (concatenate rows).
        Maps to the '|' operator.
        """
        if (
            self.domain != MatrixDomain.FEATURES
            or other.domain != MatrixDomain.FEATURES
        ):
            msg = "Vertical stacking (|) only allowed for 'FEATURES' domain."
            raise ValueError(
                msg,
            )

        new_matrix = sp.vstack([self._matrix, other._matrix])
        return self.replace(matrix=new_matrix)

    def normalize_by_radius(self, inplace: bool = True) -> SimilarityMatrix:
        """Normalize matrix by dividing by (radius + 1)."""
        if self.domain in {MatrixDomain.FEATURES, MatrixDomain.EMBEDDING}:
            msg = f"Cannot apply normalisation on {self.domain}"
            raise ValueError(
                msg,
            )

        normalized_values = cast("Any", self._matrix) / (self.radius + 1)

        if inplace:
            self._matrix = normalized_values
            # self._tags["r_normalised"] = True
            return self

        return self.replace(matrix=normalized_values, tags=self.tags)

    def cosine_similarity(self, inplace: bool = True) -> SimilarityMatrix:
        """Compute cosine similarity matrix."""
        if self.domain == MatrixDomain.SIMILARITY:
            msg = "Matrix is already a similarity matrix"
            raise ValueError(msg)

        nnz = (
            self._matrix.nnz
            if sp.issparse(self._matrix)
            else np.count_nonzero(self._matrix)
        )
        if nnz == 0:
            n_rows = self.shape[0]
            sim_matrix = np.zeros((n_rows, n_rows))
        else:
            sim_matrix = cosine_similarity(self._matrix)

        if inplace:
            self._matrix = sim_matrix
            self._tags["domain"] = MatrixDomain.SIMILARITY
            return self

        return self.replace(
            matrix=sim_matrix,
            tags=self._tags | {"domain": MatrixDomain.SIMILARITY},
        )

    def sum_to_rows(
        self,
        *,
        zero_diagonal: bool = True,
        k: int | None = None,
        largest: bool = True,
        normalise: bool = True,
    ) -> np.ndarray:
        """Sum similarity scores per row (neighborhood density)."""
        matrix = (
            self._matrix.toarray()
            if sp.issparse(self._matrix)
            else self._matrix.copy()
        )

        if zero_diagonal:
            np.fill_diagonal(matrix, 0)

        if k is None:
            scores = matrix.sum(axis=1)
            if normalise:
                n_summed = matrix.shape[1] - (1 if zero_diagonal else 0)
                scores /= max(n_summed, 1)
            return scores

        if largest:
            partitioned = np.partition(matrix, -k, axis=1)[:, -k:]
        else:
            partitioned = np.partition(matrix, k - 1, axis=1)[:, :k]

        scores = partitioned.sum(axis=1)
        return scores / k if normalise else scores

    def fit_embed(
        self,
        transformer: FitTransformer,
    ) -> FitTransformer:
        """
        Fit a transformer on this matrix's data.

        Parameters
        ----------
        transformer : FitTransformer
            An unfitted sklearn-style transformer (UMAP, PCA, etc.)

        Returns
        -------
        FitTransformer
            The same transformer, now fitted to this matrix's data.
        """
        transformer.fit(self.matrix)
        return transformer

    def transform_embed(
        self,
        transformer: FitTransformer,
        *,
        inplace: bool = False,
    ) -> SimilarityMatrix:
        """
        Apply a fitted transformer to get an embedding.

        Parameters
        ----------
        transformer : FitTransformer
            A fitted sklearn-style transformer.
        inplace : bool, optional
            If True, modifies this matrix in place. Default is False.

        Returns
        -------
        SimilarityMatrix
            Matrix with embedding data and EMBEDDING domain.
        """
        embedding = transformer.transform(self.matrix)

        if embedding.ndim == 1:
            embedding = embedding.reshape(-1, 1)

        if inplace:
            self._matrix = embedding
            self._tags["domain"] = MatrixDomain.EMBEDDING
            return self

        return self.replace(
            matrix=embedding,
            tags=self._tags | {"domain": MatrixDomain.EMBEDDING},
        )

    def umap_embed(
        self,
        *,
        config: UmapConfig | None = None,
        inplace: bool = True,
        **umap_kwargs: Any,
    ) -> SimilarityMatrix:
        """
        Convenience method: fit and transform UMAP in one step.

        For more control (fit on subset A, transform subset B),
        use fit_embed() and transform_embed() separately.
        """
        try:
            from umap import UMAP  # type: ignore (stub files missing)
        except ImportError as e:
            msg = "pip install umap-learn to use this method."
            raise ImportError(msg) from e

        is_similarity = self.domain == MatrixDomain.SIMILARITY

        final_config = config or UmapConfig(
            metric="precomputed" if is_similarity else "cosine",
        )
        kwargs = final_config.get_kwargs() | umap_kwargs

        model = UMAP(**kwargs)
        self.fit_embed(model)
        return self.transform_embed(model, inplace=inplace)

    def get_max_indices(self, amt: int) -> list[tuple[int, int]]:
        assert self.domain == MatrixDomain.SIMILARITY, (
            "can only get indices from similarity domain"
        )

        flat = self._matrix.ravel()
        indices = np.argsort(flat)[-amt:][::-1]
        rows, cols = np.unravel_index(indices, self.shape)
        return list(zip(rows.tolist(), cols.tolist(), strict=False))

    def remove_idxs(
        self, idxs: list[int], inplace: bool = True
    ) -> SimilarityMatrix:
        """
        Remove individuals (rows) from the matrix by index.

        If the matrix is in the SIMILARITY domain (square), both rows
        and columns are removed.

        Parameters
        ----------
        idxs : list[int]
            Indices of rows (individuals) to remove.
        inplace : bool, optional
            If True, modifies the matrix data in place. Default is True.

        Returns
        -------
        SimilarityMatrix
            Matrix with specified individuals removed.
        """
        n_rows = self.shape[0]
        # Create boolean mask of rows to keep
        mask = np.ones(n_rows, dtype=bool)
        mask[idxs] = False

        # Determine indices to keep
        keep_idxs = np.where(mask)[0]

        # Apply slicing
        is_similarity = self.domain == MatrixDomain.SIMILARITY

        if is_similarity and self.shape[0] == self.shape[1]:
            # For square similarity matrices, remove row and column
            # Use ix_ for cross-product indexing
            new_matrix = self._matrix[np.ix_(keep_idxs, keep_idxs)]
        else:
            # For features/embeddings, remove rows
            new_matrix = self._matrix[keep_idxs, :]

        if inplace:
            self._matrix = new_matrix
            return self

        return self.replace(matrix=new_matrix)

    def get_indices(self) -> list[int] | tuple[tuple[int, int], ...]:
        if self.domain == MatrixDomain.FEATURES:
            return list(range(self.shape[0]))

        if self.domain == MatrixDomain.EMBEDDING:
            return list(range(self.shape[0]))

        if self.domain == MatrixDomain.SIMILARITY:
            N = self.shape[0]
            return tuple((r, c) for r in range(N) for c in range(N))

        msg = f"Unsupported or unknown MatrixDomain: {self.domain}"
        raise ValueError(msg)

    # def __init__(
    #     self,
    #     matrix: sp.spmatrix | np.ndarray,
    #     space: Space | str,
    #     radius: int,
    #     domain: MatrixDomain | str = MatrixDomain.FEATURES,
    #     tags: dict[str, Any] | None = None,
    # ) -> None:
    #     """
    #     Initialize similarity matrix with typed attributes.

    #     Args:
    #         matrix: Sparse or dense matrix data
    #         space: VectorSpace enum for morphological space (REQUIRED)
    #         radius: Neighborhood radius (REQUIRED)
    #         domain: MatrixDomain enum or string (default: FEATURES)
    #         tags: Optional extra metadata tags
    #     """
    #     final_tags = tags.copy() if tags else {}

    #     domain_str = (
    #         domain.name if isinstance(domain, MatrixDomain) else str(domain)
    #     )
    #     final_tags["domain"] = domain_str
    #     final_tags["radius"] = radius

    #     super().__init__(
    #         matrix=matrix,
    #         label=space,
    #         tags=final_tags,
    #     )

    # @classmethod
    # def zeros(
    #     cls,
    #     shape: tuple[int, int],
    #     space: Space | str,
    #     radius: int,
    #     domain: MatrixDomain | str = MatrixDomain.FEATURES,
    #     sparse: bool = True,
    #     **tags: Any,
    # ) -> SimilarityMatrix:
    #     """
    #     Create a zero-filled SimilarityMatrix to fill gaps in a Frame.

    #     Args:
    #         shape: (rows, cols)
    #         space: The morphological space label
    #         radius: The neighborhood radius
    #         domain: The data domain (default: FEATURES)
    #         sparse: Whether to use a sparse CSR matrix or dense numpy array
    #         **tags: Additional metadata
    #     """
    #     # 1. Create the underlying zero data
    #     if sparse:
    #         matrix = sp.csr_matrix(shape)
    #     else:
    #         matrix = np.zeros(shape)

    #     # 2. Ensure we mark it as a gap in the tags
    #     final_tags = tags.copy()
    #     final_tags["is_gap"] = True

    #     # 3. Return a fully typed SimilarityMatrix
    #     return cls(
    #         matrix=matrix,
    #         space=space,
    #         radius=radius,
    #         domain=domain,
    #         tags=final_tags,
    #     )

    # --- Similarity Specific Properties ---

    # @property
    # def space(self) -> str:
    #     """The morphological space (e.g., 'FRONT')."""
    #     return self._label

    # @property
    # def domain(self) -> str:
    #     """The data domain (FEATURES, SIMILARITY, EMBEDDING)."""
    #     return self._tags["domain"]

    # @property
    # def radius(self) -> int:
    #     """The neighborhood radius."""
    #     return self._tags["radius"]

    # def replace(self, **changes) -> SimilarityMatrix:
    #     """
    #     Returns a new SimilarityMatrix with updated fields.
    #     Maps generic keys (label) to specific ones (space).
    #     """
    #     # Map generic keys to specific ones
    #     if "label" in changes:
    #         changes["space"] = changes.pop("label")

    #     new_args = {
    #         "matrix": self._matrix,
    #         "space": self.space,
    #         "radius": self.radius,
    #         "domain": self.domain,
    #         "tags": self._tags.copy(),
    #     } | changes
    #     valid_keys = {"matrix", "space", "radius", "domain", "tags"}
    #     filtered_args = {k: v for k, v in new_args.items() if k in valid_keys}

    #     return SimilarityMatrix(**filtered_args)

    # @classmethod
    # def load(
    #     cls, file_path: Path | str, subset_indices: list[int] | None = None
    # ):
    #     """Load SimilarityMatrix from disk, optionally subsetting.

    #     Overrides base class load() to translate generic format (label, tags)
    #     to similarity-specific format (space, radius, domain).

    #     Args:
    #         file_path: Full path without extension
    #         subset_indices: Optional list of row/column indices to keep.
    #                     If provided, returns matrix[subset_indices, subset_indices].

    #     Returns
    #     -------
    #         Loaded SimilarityMatrix
    #     """
    #     import json
    #     from pathlib import Path

    #     file_path = Path(file_path)
    #     folder = file_path.parent
    #     filename_stem = file_path.name

    #     # Load metadata
    #     json_path = folder / f"{filename_stem}.json"
    #     if not json_path.exists():
    #         msg = f"Metadata not found: {json_path}"
    #         raise FileNotFoundError(msg)

    #     with json_path.open(encoding="utf-8") as f:
    #         info = json.load(f)

    #     # Extract similarity-specific attributes
    #     space = info["label"]  # label → space
    #     tags = info.get("tags", {})
    #     radius = tags.get("radius", 0)
    #     domain = tags.get("domain", MatrixDomain.FEATURES.name)

    #     # Load matrix
    #     matrix_path = folder / info["matrix_file"]
    #     if info["storage"] == "sparse":
    #         matrix = sp.load_npz(matrix_path)
    #     else:
    #         matrix = np.load(matrix_path)

    #     # Apply subset if requested
    #     if subset_indices is not None:
    #         # Use numpy's ix_ for proper row/column subsetting
    #         matrix = matrix[np.ix_(subset_indices, subset_indices)]

    #         # Update shape in tags/metadata
    #         tags = tags.copy()  # Don't modify original
    #         tags["original_shape"] = (
    #             matrix.shape
    #         )  # Store original for reference
    #         # Optional: update radius or other metadata if needed

    #     # Call __init__ with similarity-specific signature
    #     return cls(
    #         matrix=matrix,
    #         space=space,
    #         radius=radius,
    #         domain=domain,
    #         tags=tags,
    #     )

    # def replace(self, **changes: Any) -> SimilarityMatrix:
    #     """Returns a new SimilarityMatrix with updated fields."""
    #     # Map label back to space if necessary
    #     if "label" in changes:
    #         changes["space"] = changes.pop("label")

    #     # Extract current state
    #     new_args = {
    #         "matrix": self._matrix,
    #         "space": self.space,
    #         "radius": self.radius,
    #         "domain": self.domain,
    #         "tags": self._tags.copy(),
    #     } | changes

    #     # Only pass valid SimilarityMatrix __init__ keys
    #     valid_keys = {"matrix", "space", "radius", "domain", "tags"}
    #     filtered = {k: v for k, v in new_args.items() if k in valid_keys}
    #     return self.replace(**filtered)

    # --- Math & Analysis ---

"""MatrixInstance: Single matrix wrapper with validation and I/O."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from rich.console import Console
from rich.table import Table
from sklearn.metrics.pairwise import cosine_similarity
from umap import UMAP

from canonical_toolkit.core.matrix.m_enums import MatrixDomain, VectorSpace


class MatrixInstance:
    """
    A strictly encapsulated wrapper around heavy matrix data.
    Standard Python class enforcing immutability via private properties.
    """

    def __init__(
        self,
        matrix: sp.spmatrix | np.ndarray,
        space: VectorSpace | str,
        radius: int,
        domain: MatrixDomain = MatrixDomain.FEATURES,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize with strict validation.
        Arguments map directly to private fields.
        """
        # 1. Assign to private fields
        self._matrix = matrix
        self._space = space
        self._radius = radius
        self._domain = domain
        self._meta = meta if meta is not None else {}

        # 2. STRICT VALIDATION: Memory Safety
        if self._domain == MatrixDomain.FEATURES:
            if not sp.issparse(self._matrix):
                msg = (
                    f"CRITICAL MEMORY ERROR: MatrixDomain.FEATURES must be a "
                    f"scipy.sparse matrix, but got {domain(self._matrix)}. "
                    "This will crash your RAM on large datasets."
                )
                raise TypeError(
                    msg,
                )

    # --- Public Read-Only Properties (Getters) ---

    @property
    def matrix(self) -> sp.spmatrix | np.ndarray:
        """Read-only access to raw data."""
        return self._matrix

    @property
    def shape(self):
        return self._matrix.shape

    @property
    def space(self) -> VectorSpace | str:
        return self._space

    @property
    def radius(self) -> int:
        return self._radius

    @property
    def domain(self) -> MatrixDomain:
        return self._domain

    @property
    def meta(self) -> dict[str, Any]:
        """Returns a COPY of metadata to prevent mutation bugs."""
        return self._meta.copy()

    @property
    def label(self) -> str:
        """Short label: space + radius (e.g., 'FRONT r2')."""
        space_str = self._space.value if isinstance(self._space, VectorSpace) else str(self._space)
        return f"{space_str} r{self._radius}"

    @property
    def description(self) -> str:
        """Full description: space + radius + domain (e.g., 'FRONT r2 Features')."""
        space_str = self._space.value if isinstance(self._space, VectorSpace) else str(self._space)
        domain_str = self._domain.name.capitalize()
        return f"{space_str} r{self._radius} {domain_str}"

    # --- Indexing ---

    def __getitem__(self, key):
        """Delegate indexing to the underlying matrix.

        Supports both single indexing and tuple indexing:
        - instance[i] -> returns row i
        - instance[i, j] -> returns element at (i, j)
        - instance[:, j] -> returns column j
        - etc. (all standard numpy/scipy indexing)
        """
        return self._matrix[key]

    # --- Visualization ---

    def __repr__(self) -> str:
        s_name = (
            self._space.value
            if isinstance(self._space, VectorSpace)
            else self._space
        )
        rows, cols = self.shape

        # Compact format matching frame cells
        if self._domain == MatrixDomain.FEATURES:
            shape_str = f"{rows}r×{cols}f"
        elif self._domain == MatrixDomain.SIMILARITY:
            shape_str = f"{rows}r×{cols}r"
        elif self._domain == MatrixDomain.EMBEDDING:
            shape_str = f"{rows}r×{cols}d"
        else:
            shape_str = f"{rows}×{cols}"

        "Sp" if sp.issparse(self._matrix) else "Dn"

        # For dense matrices, show rich table with corner values
        if not sp.issparse(self._matrix):
            table = Table(
                title=f"MatrixInstance: {s_name} (r={self._radius})",
                title_style="bold bright_cyan",
                title_justify="left",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Shape", f"[{shape_str}]")
            table.add_row("domain", self._domain.name)
            table.add_row("Storage", "Dense")

            if self._meta:
                table.add_row("Meta", str(self._meta))

            # Extract corner values (2x2 from each corner)
            matrix = self._matrix
            n_rows, n_cols = matrix.shape

            # Build value display
            value_lines = []

            if n_rows <= 4 and n_cols <= 4:
                # Small matrix, show all values
                for i in range(n_rows):
                    row_str = "  ".join(f"{matrix[i, j]:.2f}" for j in range(n_cols))
                    value_lines.append(row_str)
            else:
                # Large matrix, show corners with ...
                # Top 2 rows
                for i in range(min(2, n_rows)):
                    parts = []
                    # Left 2 cols
                    parts.append("  ".join(f"{matrix[i, j]:.2f}" for j in range(min(2, n_cols))))
                    # Middle ellipsis
                    if n_cols > 4:
                        parts.append("...")
                    # Right 2 cols
                    if n_cols > 2:
                        start_col = max(n_cols - 2, 2)
                        parts.append("  ".join(f"{matrix[i, j]:.2f}" for j in range(start_col, n_cols)))
                    value_lines.append("  ".join(parts))

                # Middle row ellipsis
                if n_rows > 4:
                    if n_cols > 4:
                        value_lines.append("...   ...   ...   ...")
                    else:
                        value_lines.append("   ".join("..." for _ in range(n_cols)))

                # Bottom 2 rows
                start_row = max(n_rows - 2, 2)
                for i in range(start_row, n_rows):
                    parts = []
                    # Left 2 cols
                    parts.append("  ".join(f"{matrix[i, j]:.2f}" for j in range(min(2, n_cols))))
                    # Middle ellipsis
                    if n_cols > 4:
                        parts.append("...")
                    # Right 2 cols
                    if n_cols > 2:
                        start_col = max(n_cols - 2, 2)
                        parts.append("  ".join(f"{matrix[i, j]:.2f}" for j in range(start_col, n_cols)))
                    value_lines.append("  ".join(parts))

            # Add value lines to table
            for i, line in enumerate(value_lines):
                if i == 0:
                    table.add_row("Values", line)
                else:
                    table.add_row("", line)

            # Render to string with colors enabled
            string_buffer = io.StringIO()
            temp_console = Console(file=string_buffer, force_terminal=True, width=100)
            temp_console.print(table)
            return string_buffer.getvalue().rstrip()
        # Sparse matrix - pretty table format
        table = Table(
            title=f"MatrixInstance: {s_name} (r={self._radius})",
            title_style="bold bright_cyan",
            title_justify="left",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Shape", f"[{shape_str}]")
        table.add_row("domain", self._domain.name)
        table.add_row("Storage", f"Sparse ({self._matrix.format})")

        # Calculate sparsity information
        n_nonzero = self._matrix.nnz
        n_total = self._matrix.shape[0] * self._matrix.shape[1]
        sparsity = (1 - n_nonzero / n_total) * 100 if n_total > 0 else 0

        table.add_row("Non-zero", f"{n_nonzero:,} / {n_total:,}")
        table.add_row("Sparsity", f"{sparsity:.2f}%")

        if self._meta:
            table.add_row("Meta", str(self._meta))

        # Show sample of non-zero values
        if n_nonzero > 0:
            # Convert to COO format for easy access to values
            coo = self._matrix.tocoo()
            n_samples = min(5, n_nonzero)

            sample_lines = []
            for idx in range(n_samples):
                row, col, val = coo.row[idx], coo.col[idx], coo.data[idx]
                sample_lines.append(f"  [{row},{col}] = {val:.4f}")

            if n_nonzero > n_samples:
                sample_lines.append(f"  ... ({n_nonzero - n_samples} more)")

            for i, line in enumerate(sample_lines):
                if i == 0:
                    table.add_row("Samples", line)
                else:
                    table.add_row("", line)

        # Render to string with colors enabled
        string_buffer = io.StringIO()
        temp_console = Console(file=string_buffer, force_terminal=True, width=100)
        temp_console.print(table)
        return string_buffer.getvalue().rstrip()

    # --- Abstraction & Modification ---

    def replace(self, **changes) -> MatrixInstance:
        """
        Returns a new instance with updated fields.
        Manually constructs new object to ensure validation runs.
        """
        new_args = {
            "matrix": self._matrix,
            "space": self._space,
            "radius": self._radius,
            "domain": self._domain,
            "meta": self._meta.copy(),
        }
        new_args |= changes
        return MatrixInstance(**new_args)

    # def add_meta(self, **kwargs) -> MatrixInstance:
    #     """Returns a new instance with updated metadata."""
    #     new_meta = self._meta.copy()
    #     new_meta.update(kwargs)
    #     return self.replace(meta=new_meta)

    # --- Math Transforms ---

    def cosine_similarity(self) -> MatrixInstance:
        if self._domain == MatrixDomain.SIMILARITY:
            msg = "Matrix is already a similarity matrix."
            raise ValueError(msg)

        sim_matrix = cosine_similarity(self._matrix)

        return self.replace(matrix=sim_matrix, domain=MatrixDomain.SIMILARITY)

    def sum_similarity_scores(
        self,
        *,
        zero_diagonal: bool = True,
        k: int | None = None,
        largest: bool = True,
    ) -> np.ndarray:
        """Compute total similarity score for each individual.

        Collapses the SIMILARITY matrix by summing each row (individual's
        similarities to others). By default, zeros the diagonal first
        to exclude self-similarity. Optionally sum only top-k neighbors.

        ONLY works on MatrixDomain.SIMILARITY matrices.

        Args:
            zero_diagonal: If True, set diagonal to 0 before summing
                          (excludes self-similarity). Default: True.
            k: Number of neighbors to consider. If None, sum all neighbors.
               If specified, sum only the k largest or k smallest values.
            largest: If True, sum k largest similarities (most similar).
                    If False, sum k smallest (least similar/most dissimilar).
                    Only used when k is not None. Default: True.

        Returns
        -------
            1D numpy array of length n_robots, where each value is the
            sum of that individual's similarities to selected others.

        Raises
        ------
            ValueError: If matrix is not SIMILARITY domain

        Example:
            >>> # Similarity matrix (3 robots)
            >>> sim_matrix = np.array([
            ...     [1.0, 0.8, 0.6, 0.2],
            ...     [0.8, 1.0, 0.7, 0.3],
            ...     [0.6, 0.7, 1.0, 0.4],
            ...     [0.2, 0.3, 0.4, 1.0]
            ... ])
            >>> inst = MatrixInstance(sim_matrix, space=..., domain=SIMILARITY)
            >>> # Sum all similarities (excluding diagonal)
            >>> inst.sum_similarity_scores()  # [1.6, 1.8, 1.7, 0.9]
            >>> # Sum top-2 most similar neighbors
            >>> inst.sum_similarity_scores(k=2, largest=True)  # [1.4, 1.5, 1.3, 0.7]
            >>> # Sum top-2 least similar neighbors
            >>> inst.sum_similarity_scores(k=2, largest=False)  # [0.8, 1.0, 1.0, 0.5]
        """
        if self._domain != MatrixDomain.SIMILARITY:
            msg = (
                f"Can only sum similarity scores on SIMILARITY matrices. "
                f"Found {self._domain.name}"
            )
            raise ValueError(msg)

        # Convert to dense if sparse (should already be dense for SIMILARITY)
        matrix = self._matrix.toarray() if sp.issparse(self._matrix) else self._matrix.copy()

        if zero_diagonal:
            # Zero out the diagonal
            np.fill_diagonal(matrix, 0)

        # If k is None, sum all values
        if k is None:
            return matrix.sum(axis=1)

        # Select top-k or bottom-k neighbors per row
        n_robots = matrix.shape[0]
        scores = np.zeros(n_robots)

        for i in range(n_robots):
            row = matrix[i]
            # Get indices of k largest or smallest values
            if largest:
                # k largest: use argpartition and take largest k
                indices = np.argpartition(row, -k)[-k:]
            else:
                # k smallest: use argpartition and take smallest k
                indices = np.argpartition(row, k)[:k]

            # Sum the selected values
            scores[i] = row[indices].sum()

        return scores

    def umap_embed(
        self,
        *,
        n_neighbors: int = 15,
        n_components: int = 2,
        metric: str = "cosine",
        random_state: int | None = 42,
        **umap_kwargs,
    ) -> MatrixInstance:
        """Compute UMAP embedding from FEATURES or SIMILARITY matrix.

        Reduces dimensionality using UMAP. Works on both FEATURES
        (computes distances internally) and SIMILARITY matrices
        (uses precomputed distances).

        Returns EMBEDDING matrix (N×n_components).

        Args:
            n_neighbors: Number of neighbors for UMAP. Default: 15.
            n_components: Dimensionality of embedding. Default: 2.
            metric: Distance metric for FEATURES matrices.
                   Ignored for SIMILARITY (uses precomputed). Default: "cosine".
            random_state: Random seed for reproducibility. Set to None
                         to enable parallelization. Default: 42.
            **umap_kwargs: Additional arguments passed to UMAP constructor.

        Returns
        -------
            MatrixInstance with domain=EMBEDDING, shape (n_robots, n_components)

        Raises
        ------
            ValueError: If matrix is EMBEDDING (already embedded)

        Example:
            >>> # From FEATURES matrix
            >>> features = MatrixInstance(feat_matrix, domain=FEATURES)
            >>> embedding = features.umap_embed(n_neighbors=10, n_components=2)
            >>> embedding.shape  # (n_robots, 2)
            >>>
            >>> # From SIMILARITY matrix (precomputed cosine similarity)
            >>> sim = features.cosine_similarity()
            >>> embedding = sim.umap_embed(n_neighbors=10, n_components=2)
        """
        if self._domain == MatrixDomain.EMBEDDING:
            msg = "Matrix is already an EMBEDDING. Cannot embed again."
            raise ValueError(msg)

        # Convert sparse to dense if needed
        matrix = self._matrix.toarray() if sp.issparse(self._matrix) else self._matrix

        # Handle SIMILARITY vs FEATURES differently
        if self._domain == MatrixDomain.SIMILARITY:
            # Convert similarity to distance (1 - similarity)
            distance_matrix = 1 - matrix
            # Use precomputed metric
            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                metric="precomputed",
                random_state=random_state,
                init="random",
                transform_seed=random_state if random_state is not None else None,
                n_jobs=1 if random_state is not None else -1,
                **umap_kwargs,
            )
            embedding = umap_model.fit_transform(distance_matrix)
        else:
            # FEATURES matrix - compute distances with specified metric
            umap_model = UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                metric=metric,
                random_state=random_state,
                init="random",
                transform_seed=random_state if random_state is not None else None,
                n_jobs=1 if random_state is not None else -1,
                **umap_kwargs,
            )
            embedding = umap_model.fit_transform(matrix)

        return self.replace(matrix=embedding, domain=MatrixDomain.EMBEDDING)

    def __add__(self, other: MatrixInstance) -> MatrixInstance:
        if not isinstance(other, MatrixInstance):
            return NotImplemented

        if self._radius != other._radius:
            msg = f"Radius mismatch: {self._radius} vs {other._radius}"
            raise ValueError(
                msg,
            )

        if self._domain != other._domain:
            msg = f"domain mismatch: {self._domain} vs {other._domain}"
            raise ValueError(msg)

        new_matrix = self._matrix + other._matrix
        return self.replace(matrix=new_matrix)

    # --- I/O Methods ---

    def save(self, folder: Path, filename_stem: str) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        if sp.issparse(self._matrix):
            matrix_path = folder / f"{filename_stem}.sparse.npz"
            sp.save_npz(matrix_path, self._matrix)
            storage_domain = "sparse"
        else:
            matrix_path = folder / f"{filename_stem}.dense.npy"
            np.save(matrix_path, self._matrix)
            storage_domain = "dense"

        meta_payload = {
            "space": self.space.value
            if isinstance(self.space, VectorSpace)
            else self.space,
            "radius": self.radius,
            "domain": self.domain.name,
            "storage": storage_domain,
            "matrix_file": matrix_path.name,
            "user_meta": self._meta,
        }

        with Path(folder / f"{filename_stem}.json").open("w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

    @classmethod
    def load(cls, folder: Path, filename_stem: str) -> MatrixInstance:
        folder = Path(folder)

        json_path = folder / f"{filename_stem}.json"
        if not json_path.exists():
            msg = f"Metadata not found: {json_path}"
            raise FileNotFoundError(msg)

        with Path(json_path).open(encoding="utf-8") as f:
            info = json.load(f)

        try:
            space = VectorSpace(info["space"])
        except ValueError:
            space = info["space"]

        mat_domain = MatrixDomain[info["domain"]]
        radius = int(info["radius"])

        matrix_path = folder / info["matrix_file"]
        if info["storage"] == "sparse":
            matrix = sp.load_npz(matrix_path)
        else:
            matrix = np.load(matrix_path)

        return cls(
            matrix=matrix,
            space=space,
            radius=radius,
            domain=mat_domain,
            meta=info["user_meta"],
        )


if __name__ == "__main__":
    import scipy.sparse as sp

    # Test 1: Create sparse matrix instance
    sparse_mat = sp.random(5, 10, density=0.3, format="csr", random_state=42)
    sparse_inst = MatrixInstance(
        matrix=sparse_mat,
        space=VectorSpace.FRONT_LIMB,
        radius=0,
    )

    # Test 2: Create dense matrix instance with values
    small_dense = np.array([
        [1.00, 0.85, 0.62],
        [0.85, 1.00, 0.73],
        [0.62, 0.73, 1.00],
    ])
    small_inst = MatrixInstance(
        matrix=small_dense,
        space=VectorSpace.BACK_LIMB,
        domain=MatrixDomain.SIMILARITY,
        radius=1,
    )

    # Test 3: Create large dense matrix with corner display
    dense_sim = np.random.rand(10, 10)
    dense_sim = (dense_sim + dense_sim.T) / 2
    np.fill_diagonal(dense_sim, 1.0)
    dense_inst = MatrixInstance(
        matrix=dense_sim,
        space=VectorSpace.FRONT_LIMB,
        domain=MatrixDomain.SIMILARITY,
        radius=0,
        meta={"method": "cosine", "normalized": True},
    )

    # Test 4: Cosine similarity transform
    sim_result = sparse_inst.cosine_similarity()

    # Test 5: Matrix addition
    sparse_mat2 = sp.random(5, 10, density=0.3, format="csr", random_state=43)
    inst2 = MatrixInstance(
        matrix=sparse_mat2,
        space=VectorSpace.FRONT_LIMB,
        radius=0,
    )
    summed = sparse_inst + inst2

    # Test 6: __getitem__ on dense matrices

    # Access single row
    row_1 = small_inst[1]

    # Access single element with tuple notation
    elem = small_inst[0, 2]

    # Slicing - first 2 rows
    slice_rows = small_inst[:2]

    # Slicing - column access
    col_1 = small_inst[:, 1]

    # Test 7: __getitem__ on sparse matrices (NO to_dense conversion!)

    # Access single row - STAYS SPARSE
    sparse_row = sparse_inst[0]

    # Access single element
    sparse_elem = sparse_inst[0, 5]

    # Slicing - STAYS SPARSE
    sparse_slice = sparse_inst[:3]

    # Column access - STAYS SPARSE
    sparse_col = sparse_inst[:, 3]

    # Test 8: Verify NO memory bloat with large sparse matrix
    large_sparse = sp.random(1000, 5000, density=0.01, format="csr", random_state=99)
    large_inst = MatrixInstance(
        matrix=large_sparse,
        space=VectorSpace.FRONT_LIMB,
        radius=2,
    )

    # Index it - should stay sparse, no memory bloat!
    large_row = large_inst[500]

    # Test 9: Demonstrate [i][j] vs [i, j] for sparse matrices

    # Dense matrix - both work

    # Sparse matrix - only tuple indexing works correctly!

    # Now try the WRONG way [i][j]
    try:
        row = sparse_inst[0]  # Returns shape (1, 10) sparse matrix

        # Now try to index [5] on a (1, 10) matrix
        wrong_result = sparse_inst[0][5]
    except IndexError:
        pass

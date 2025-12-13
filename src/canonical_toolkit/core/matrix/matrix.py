"""MatrixInstance: Generic matrix wrapper with flexible metadata."""

from __future__ import annotations

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


class MatrixInstance:
    """
    Generic matrix wrapper with flexible metadata.

    This base class knows nothing about similarity, trees, or radius.
    Domain-specific subclasses (like SimilarityMatrix) add typed meaning.
    """

    def __init__(
        self,
        matrix: sp.spmatrix | np.ndarray,
        label: str,
        index: Hashable,
        tags: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize generic matrix with flexible metadata.

        Args:
            matrix: Sparse or dense matrix data
            label: Generic label (e.g., space name, experiment ID)
            index: Hashable index (e.g., radius, timepoint, condition)
            tags: Flexible metadata dict (e.g., {"domain": "features"})
        """
        self._matrix = matrix
        self._label = label
        self._index = index
        self._tags = tags if tags is not None else {}

    # --- Public Read-Only Properties (Getters) ---

    @property
    def matrix(self) -> sp.spmatrix | np.ndarray:
        """Read-only access to raw matrix data."""
        return self._matrix

    @property
    def shape(self):
        """Shape of the underlying matrix."""
        return self._matrix.shape

    @property
    def label(self) -> str:
        """Generic label (e.g., space name, experiment ID)."""
        return self._label

    @property
    def index(self) -> Hashable:
        """Generic index (e.g., radius, timepoint) - hashable for dict keys."""
        return self._index

    @property
    def tags(self) -> dict[str, Any]:
        """Returns a COPY of tags to prevent mutation bugs."""
        return self._tags.copy()

    @property
    def short_description(self) -> str:
        """Short description: 'label [index]' - e.g., 'FRONT_LIMB [3]'."""
        return f"{self._label} [{self._index}]"

    @property
    def long_description(self) -> str:
        """Long description with tags - e.g., 'FRONT_LIMB [3] (domain: features)'."""
        if self._tags:
            tags_str = ", ".join(f"{k}: {v}" for k, v in self._tags.items())
            return f"{self.short_description} ({tags_str})"
        return self.short_description

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
        rows, cols = self.shape
        shape_str = f"{rows}×{cols}"

        # For dense matrices, show rich table with corner values
        if not sp.issparse(self._matrix):
            table = Table(
                title=f"MatrixInstance: {self.short_description}",
                title_style="bold bright_cyan",
                title_justify="left",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Shape", f"[{shape_str}]")
            table.add_row("Storage", "Dense")

            # Show up to 5 tags
            if self._tags:
                for i, (key, value) in enumerate(self._tags.items()):
                    if i >= 5:
                        table.add_row("...", f"({len(self._tags) - 5} more tags)")
                        break
                    table.add_row(key.capitalize(), str(value))

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
            title=f"MatrixInstance: {self.short_description}",
            title_style="bold bright_cyan",
            title_justify="left",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Shape", f"[{shape_str}]")
        table.add_row("Storage", f"Sparse ({self._matrix.format})")

        # Calculate sparsity information
        n_nonzero = self._matrix.nnz
        n_total = self._matrix.shape[0] * self._matrix.shape[1]
        sparsity = (1 - n_nonzero / n_total) * 100 if n_total > 0 else 0

        table.add_row("Non-zero", f"{n_nonzero:,} / {n_total:,}")
        table.add_row("Sparsity", f"{sparsity:.2f}%")

        # Show up to 5 tags
        if self._tags:
            for i, (key, value) in enumerate(self._tags.items()):
                if i >= 5:
                    table.add_row("...", f"({len(self._tags) - 5} more tags)")
                    break
                table.add_row(key.capitalize(), str(value))

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
            "label": self._label,
            "index": self._index,
            "tags": self._tags.copy(),
        }
        new_args.update(changes)
        return MatrixInstance(**new_args)

    # def add_meta(self, **kwargs) -> MatrixInstance:
    #     """Returns a new instance with updated metadata."""
    #     new_meta = self._meta.copy()
    #     new_meta.update(kwargs)
    #     return self.replace(meta=new_meta)

    # --- Math Transforms ---


    def __add__(self, other: MatrixInstance) -> MatrixInstance:
        if not isinstance(other, MatrixInstance):
            return NotImplemented

        if self._index != other._index:
            msg = f"Index mismatch: {self._index} vs {other._index}"
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
            "label": self._label,
            "index": str(self._index),  # Convert to string for JSON
            "tags": self._tags,
            "storage": storage_domain,
            "matrix_file": matrix_path.name,
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

        label = info["label"]
        # Try to convert index back to int if possible, otherwise keep as string
        try:
            index = int(info["index"])
        except (ValueError, TypeError):
            index = info["index"]
        tags = info.get("tags", {})

        matrix_path = folder / info["matrix_file"]
        if info["storage"] == "sparse":
            matrix = sp.load_npz(matrix_path)
        else:
            matrix = np.load(matrix_path)

        return cls(
            matrix=matrix,
            label=label,
            index=index,
            tags=tags,
        )


if __name__ == "__main__":
    print("Testing Generic MatrixInstance...")

    # Test 1: Generic matrix with custom label/index
    sparse_mat = sp.random(5, 10, density=0.3, format="csr", random_state=42)
    generic_inst = MatrixInstance(
        matrix=sparse_mat,
        label="experiment_A",
        index="timepoint_0",
        tags={"condition": "control", "temp": 37}
    )

    print(generic_inst)

    print(f"✓ Generic matrix: {generic_inst.short_description}")
    print(f"  Tags: {generic_inst.tags}")

    # Test 2: Generic dense matrix with numeric index
    dense_mat = np.random.rand(4, 4)
    generic_dense = MatrixInstance(
        matrix=dense_mat,
        label="sensor_data",
        index=42,
        tags={"sensor": "temp", "location": "lab"}
    )
    print(f"✓ Dense matrix: {generic_dense.long_description}")

    # Test 3: Matrix operations (add)
    mat1 = MatrixInstance(sp.random(3, 5, density=0.5, format="csr"), "test", 1, {"type": "features"})
    mat2 = MatrixInstance(sp.random(3, 5, density=0.5, format="csr"), "test", 1, {"type": "features"})
    mat_sum = mat1 + mat2
    print(f"✓ Matrix addition works: {mat_sum.shape}")

    # Test 4: Save/Load
    from pathlib import Path
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        generic_inst.save(Path(tmpdir), "test_matrix")
        loaded = MatrixInstance.load(Path(tmpdir), "test_matrix")
        print(f"✓ Save/Load: {loaded.label} [{loaded.index}]")

    print("\n✅ All generic MatrixInstance tests passed!")

    # Original test code (commented out - uses old API)
    """
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
        wrong_result = sparse_inst[0][5]
    except IndexError:
        pass
    """

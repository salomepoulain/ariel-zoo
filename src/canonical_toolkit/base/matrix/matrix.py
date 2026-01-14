"""MatrixInstance: Generic matrix wrapper with flexible metadata."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import scipy.sparse as sp
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from collections.abc import Hashable

    from .matrix_types import InstanceProtocol

# Default save directories
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA_INSTANCES = DATA / "instances"
DATA_SERIES = DATA / "series"
DATA_FRAMES = DATA / "frames"


class MatrixInstance:
    """
    Generic matrix wrapper with flexible metadata.

    This base class knows nothing about similarity, trees, or radius.
    Domain-specific subclasses (like SimilarityMatrix) add typed meaning.
    """

    def __init__(
        self,
        matrix: sp.spmatrix | np.ndarray,
        label: Hashable,
        tags: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize generic matrix with flexible metadata.

        Args:
            matrix: Sparse or dense matrix data
            label: Generic label (e.g., space name, experiment ID)
            tags: Flexible metadata dict (e.g., {"domain": "features"})
        """
        self._matrix = matrix
        self._label = getattr(label, "name", str(label))
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
    def tags(self) -> dict[str, Any]:
        """Returns a COPY of tags to prevent mutation bugs. Tags are immutable after creation."""
        return self._tags.copy()

    @property
    def description(self) -> str:
        """Generate descriptive filename: 'classname_rowsxcols_sp/d'."""
        storage = "sp" if sp.issparse(self._matrix) else "d"
        class_name = self.__class__.__name__.lower()
        return f"{class_name}_{self.shape[0]}x{self.shape[1]}_{storage}".lower()

    # --- Indexing ---

    def __getitem__(
        self,
        key: int | tuple[int | slice, ...] | slice,
    ) -> np.ndarray | sp.spmatrix | np.number:
        """Delegate indexing to the underlying matrix.

        Supports both single indexing and tuple indexing:
        - instance[i] -> returns row i
        - instance[i, j] -> returns element at (i, j)
        - instance[:, j] -> returns column j
        - etc. (all standard numpy/scipy indexing)
        """
        # Sparse matrices support __getitem__ but scipy type stubs are incomplete
        return self._matrix[key]  # type: ignore[index]

    # --- Visualization ---

    def _add_tags_to_table(self, table: Table, max_tags: int = 5) -> None:
        """Add tags to table, showing up to max_tags."""
        if not self._tags:
            return
        for i, (key, value) in enumerate(self._tags.items()):
            if i >= max_tags:
                table.add_row(
                    "...",
                    f"({len(self._tags) - max_tags} more tags)",
                )
                break
            table.add_row(key.capitalize(), str(value))

    def _format_dense_matrix_values(self, matrix: np.ndarray) -> list[str]:
        """Format dense matrix values for display with corner preview."""
        n_rows, n_cols = matrix.shape
        value_lines = []

        if n_rows <= 4 and n_cols <= 4:
            for i in range(n_rows):
                row_str = "  ".join(
                    f"{matrix[i, j]:.2f}" for j in range(n_cols)
                )
                value_lines.append(row_str)
        else:
            for i in range(min(2, n_rows)):
                parts = self._format_row_with_corners(matrix, i, n_cols)
                value_lines.append("  ".join(parts))

            if n_rows > 4:
                ellipsis = (
                    "...   ...   ...   ..."
                    if n_cols > 4
                    else "   ".join("..." for _ in range(n_cols))
                )
                value_lines.append(ellipsis)

            start_row = max(n_rows - 2, 2)
            for i in range(start_row, n_rows):
                parts = self._format_row_with_corners(matrix, i, n_cols)
                value_lines.append("  ".join(parts))

        return value_lines

    def _format_row_with_corners(
        self,
        matrix: np.ndarray,
        row: int,
        n_cols: int,
    ) -> list[str]:
        """Format a single row showing left and right corners."""
        parts = []
        parts.append(
            "  ".join(f"{matrix[row, j]:.2f}" for j in range(min(2, n_cols))),
        )

        if n_cols > 4:
            parts.append("...")

        if n_cols > 2:
            start_col = max(n_cols - 2, 2)
            parts.append(
                "  ".join(
                    f"{matrix[row, j]:.2f}" for j in range(start_col, n_cols)
                ),
            )
        return parts

    def _repr_dense(self, shape_str: str) -> str:
        """Render dense matrix as a rich table."""
        table = Table(
            title=f"{self.__class__.__name__}: {self.label}",
            title_style="bold bright_cyan",
            title_justify="left",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Shape", f"[{shape_str}]")
        table.add_row("Storage", "Dense")

        self._add_tags_to_table(table)

        value_lines = self._format_dense_matrix_values(self._matrix)
        for i, line in enumerate(value_lines):
            label = "Values" if i == 0 else ""
            table.add_row(label, line)

        return self._render_table(table)

    def _repr_sparse(self, shape_str: str) -> str:
        """Render sparse matrix as a rich table."""
        table = Table(
            title=f"{self.__class__.__name__}: {self.label}",
            title_style="bold bright_cyan",
            title_justify="left",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Shape", f"[{shape_str}]")
        table.add_row("Storage", f"Sparse ({self._matrix.format})")

        n_nonzero = self._matrix.nnz
        n_total = self._matrix.shape[0] * self._matrix.shape[1]
        sparsity = (1 - n_nonzero / n_total) * 100 if n_total > 0 else 0

        table.add_row("Non-zero", f"{n_nonzero:,} / {n_total:,}")
        table.add_row("Sparsity", f"{sparsity:.2f}%")

        self._add_tags_to_table(table)

        if n_nonzero > 0:
            self._add_sparse_samples(table, n_nonzero)

        return self._render_table(table)

    def _add_sparse_samples(self, table: Table, n_nonzero: int) -> None:
        """Add sparse matrix value samples to table."""
        coo = self._matrix.tocoo()
        n_samples = min(5, n_nonzero)

        sample_lines = []
        for idx in range(n_samples):
            row, col, val = coo.row[idx], coo.col[idx], coo.data[idx]
            sample_lines.append(f"  [{row},{col}] = {val:.4f}")

        if n_nonzero > n_samples:
            sample_lines.append(f"  ... ({n_nonzero - n_samples} more)")

        for i, line in enumerate(sample_lines):
            label = "Samples" if i == 0 else ""
            table.add_row(label, line)

    def _render_table(self, table: Table) -> str:
        """Render rich table to string."""
        string_buffer = io.StringIO()
        temp_console = Console(
            file=string_buffer,
            force_terminal=True,
            width=100,
        )
        temp_console.print(table)
        return string_buffer.getvalue().rstrip()

    def __repr__(self) -> str:
        """Pretty table representation (dense or sparse)."""
        rows, cols = self.shape
        shape_str = f"{rows}×{cols}"

        if sp.issparse(self._matrix):
            return self._repr_sparse(shape_str)
        return self._repr_dense(shape_str)

    # --- Abstraction & Modification ---

    def replace(self, **changes: Any) -> Self:
        """
        Returns a new instance with updated fields.
        Manually constructs new object to ensure validation runs.
        """
        new_args = {
            "matrix": self._matrix,
            "label": self._label,
            "tags": self._tags.copy(),
        } | changes
        return self.__class__(**new_args)

    # --- Math Transforms ---

    def __add__(self, other: InstanceProtocol) -> Self:
        """Add two matrix instances element-wise.

        Args:
            other: Another matrix instance (must have compatible shape)

        Returns
        -------
            New instance with summed matrices
        """
        if not isinstance(other, MatrixInstance):
            return NotImplemented

        new_matrix = self._matrix + other._matrix
        return self.replace(matrix=new_matrix)

    # --- I/O Methods ---

    def save(
        self,
        file_path: Path | str | None = None,
        *,
        overwrite: bool = False,
    ) -> None:
        """Save instance to disk.

        Args:
            file_path: Full path without extension. If None, uses default:
                      __data__/instances/{description}
            overwrite: If False, appends counter to avoid overwriting existing files

        Creates 2 files:
            {file_path}.json - metadata
            {file_path}.npz or {file_path}.npy - matrix data
        """
        if file_path is None:
            file_path = DATA_INSTANCES / self.description

            if not overwrite:
                counter = 2
                original_path = file_path
                while (file_path.with_suffix(".json")).exists():
                    file_path = original_path.with_name(
                        f"{original_path.name}_{counter}",
                    )
                    counter += 1

        file_path = Path(file_path)
        folder = file_path.parent
        filename_stem = file_path.name

        folder.mkdir(parents=True, exist_ok=True)

        if sp.issparse(self._matrix):
            matrix_path = folder / f"{filename_stem}.npz"
            sp.save_npz(matrix_path, self._matrix)
            storage_domain = "sparse"
        else:
            matrix_path = folder / f"{filename_stem}.npy"
            np.save(matrix_path, self._matrix)
            storage_domain = "dense"

        meta_payload = {
            "label": self._label,
            "tags": self._tags,
            "storage": storage_domain,
            "matrix_file": matrix_path.name,
        }

        with Path(folder / f"{filename_stem}.json").open(
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(meta_payload, f, indent=2)

    @classmethod
    def load(cls, file_path: Path | str) -> Self:
        """Load instance from disk.

        Args:
            file_path: Full path without extension

        Returns
        -------
            Loaded MatrixInstance (or subclass)
        """
        file_path = Path(file_path)
        folder = file_path.parent
        filename_stem = file_path.name

        json_path = folder / f"{filename_stem}.json"
        if not json_path.exists():
            msg = f"Metadata not found: {json_path}"
            raise FileNotFoundError(msg)

        with Path(json_path).open(encoding="utf-8") as f:
            info = json.load(f)

        label = info["label"]
        tags = info.get("tags", {})

        matrix_path = folder / info["matrix_file"]
        if info["storage"] == "sparse":
            matrix = sp.load_npz(matrix_path)
        else:
            matrix = np.load(matrix_path)

        return cls(
            matrix=matrix,
            label=label,
            tags=tags,
        )


if __name__ == "__main__":
    # Test 1: Generic matrix with custom label
    sparse_mat = sp.random(5, 10, density=0.3, format="csr", random_state=42)
    generic_inst = MatrixInstance(
        matrix=sparse_mat,
        label="experiment_A",
        tags={"condition": "control", "temp": 37},
    )

    # Test 2: Generic dense matrix
    dense_mat = np.random.rand(4, 4)
    generic_dense = MatrixInstance(
        matrix=dense_mat,
        label="sensor_data",
        tags={"sensor": "temp", "location": "lab"},
    )

    # Test 3: Matrix operations (add)
    mat1 = MatrixInstance(
        sp.random(3, 5, density=0.5, format="csr"),
        "test",
        {"type": "features"},
    )
    mat2 = MatrixInstance(
        sp.random(3, 5, density=0.5, format="csr"),
        "test",
        {"type": "features"},
    )
    mat_sum = mat1 + mat2

    # Test 4: Save/Load
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test_matrix"
        generic_inst.save(test_path)
        loaded = MatrixInstance.load(test_path)

    # Test 5: Save/Load with default path
    generic_inst.save()  # Uses default __data__/instances/{description}
    default_path = DATA_INSTANCES / generic_inst.description

    # Test 6: Load from default path
    loaded_from_default = MatrixInstance.load(default_path)

"""MatrixSeries: Generic collection of Matrixmatrice objects indexed by index."""

from __future__ import annotations

import enum
import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Self, overload

from numpy import shape
import scipy.sparse as sp
from rich.console import Console
from rich.table import Table

from .matrix import DATA_SERIES, MatrixInstance
from .matrix_types import M

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Iterable

__all__ = ["MatrixSeries"]


class MatrixSeries(Generic[M]):
    """Generic collection of matrices sharing the same label."""

    _matrix_class = MatrixInstance

    def __init__(
        self,
        matrices: Iterable[M | None],
        label: str | None = None
    ) -> None:
        # 1. Pre-process into a list and find a 'template' matrice
        raw_list = list(matrices)
        if not raw_list:
            raise ValueError("MatrixSeries requires at least one entry.")

        # Find the first non-None matrice to use as a template for zeros
        template = next((inst for inst in raw_list if inst is not None), None)
        if template is None:
            raise ValueError("MatrixSeries must contain at least one actual Matrixmatrice.")

        # 2. Handle Labeling
        if label is not None:
            self._label = label
        else:
            unique_labels = {inst.label for inst in raw_list if inst is not None}
            if len(unique_labels) > 1:
                raise ValueError(f"Inconsistent labels: {unique_labels}. Provide a label to overwrite.")
            self._label = next(iter(unique_labels))

        # 3. Populate and auto-fill gaps
        self._matrices: list[M] = []
        fill_shape = template.shape
        is_sparse = sp.issparse(template.matrix)
        # Use template tags (excluding radius) to ensure subclasses get required metadata
        base_tags = {k: v for k, v in template.tags.items() if k != "radius"}

        for idx, inst in enumerate(raw_list):
            if inst is None:
                inst = self._matrix_class.zeros(
                    shape=fill_shape,
                    label=self._label,
                    sparse=is_sparse,
                    radius=idx,
                    **base_tags
                )
            else:
                if label and inst.label != label:
                    inst = inst.replace(label=label)

            self._matrices.append(inst)

    @property
    def shape(self) -> int:
        return len(self._matrices)

    @property
    def label(self) -> str:
        return self._label

    @property
    def matrices(self) -> list[M]:
        return self._matrices.copy()

    @property
    def indices(self) -> list[int]:
        return list(range(len(self._matrices)))

    def items(self) -> Iterable[tuple[int, M]]:
        return enumerate(self._matrices)

    def __iter__(self) -> Iterator[M]:
        return iter(self._matrices)

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    @overload
    def __getitem__(self, key: list[int]) -> Self: ...

    @overload
    def __getitem__(self, key: int) -> M: ...

    def __getitem__(self, key: int | slice | list[int]) -> M | Self:
        if isinstance(key, slice):
            sliced_matrices = self._matrices[key]
            if not sliced_matrices:
                raise ValueError("Cannot slice empty series")
            return self.__class__(matrices=sliced_matrices, label=self.label)

        if isinstance(key, list):
            sliced_matrices = [self._matrices[i] for i in key]
            # Allow empty selection? Usually yes for consistent slicing logic
            if not sliced_matrices and key:
                 # If key was not empty but result is empty (out of bounds? no, that would raise IndexError)
                 pass
            return self.__class__(matrices=sliced_matrices, label=self.label)

        # Integer case
        try:
            return self._matrices[key]
        except IndexError:
            raise KeyError(
                f"Index {key} out of range for series '{self.label}' with length {len(self._matrices)}"
            )

    def __setitem__(self, key: int, matrix: M) -> None:
        self._matrices[key] = matrix

    def replace(
        self,
        matrices: Iterable[M] | None = None,
        label: str | None = None
    ) -> Self:
        """Create a new MatrixSeries with updated fields."""
        new_matrices = self._matrices if matrices is None else list(matrices)
        new_label = self.label if label is None else label
        return self.__class__(matrices=new_matrices, label=new_label)

    @property
    def description(self) -> str:
        class_name = self.__class__.__name__.lower()
        num_matrices = len(self._matrices)
        return f"{class_name}_{self.label}_{num_matrices}i".lower()

    def __repr__(self) -> str:
        label = self.label
        table = Table(
            title=f"{self.__class__.__name__}: {label}",
            title_style="bold bright_cyan",
            title_justify="left",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Index", style="cyan", justify="right")
        table.add_column(label, justify="center")

        for idx, inst in enumerate(self._matrices):
            shape_str = f"{inst.shape[0]}×{inst.shape[1]}"
            storage_abbr = "Sparse" if sp.issparse(inst.matrix) else "Dense"
            tags_str = f" {inst.tags}" if inst.tags else ""
            table.add_row(str(idx), f"[{shape_str}] {storage_abbr}{tags_str}")

        string_buffer = io.StringIO()
        Console(file=string_buffer, force_terminal=True, width=80).print(table)
        return string_buffer.getvalue().rstrip()

    def map(self, func: Callable[[M], M] | str, **kwargs: Any) -> Self:
        new_data = []
        for inst in self.matrices:
            if isinstance(func, str):
                new_data.append(getattr(inst, func)(**kwargs))
            else:
                new_data.append(func(inst))

        if kwargs.get('inplace', False):
            self._matrices = new_data
            return self
        return self.replace(matrices=new_data)

    def __add__(self, other: MatrixSeries[M]) -> Self:
        if len(self._matrices) != len(other.matrices):
             raise ValueError("Series must have same length to add")

        # Zip lists and add
        new_matrices = [m1 + m2 for m1, m2 in zip(self._matrices, other.matrices)] # type: ignore
        return self.replace(matrices=new_matrices)


    @staticmethod
    def _resolve_save_path(base_path: Path, overwrite: bool, suffix_number: bool) -> Path:
        """Resolve the final save path, handling overwrites and numbering."""
        if overwrite:
            return base_path

        counter, fmt = (0, "{name}_{i:04d}") if suffix_number else (2, "{name}_{i}")
        path = base_path.with_name(fmt.format(name=base_path.name, i=counter)) if suffix_number else base_path

        while path.exists():
            counter += 1
            path = base_path.with_name(fmt.format(name=base_path.name, i=counter))

        return path

    def save(
        self,
        folder_path: Path | str | None = None,
        series_name: str | None = None,
        *,
        overwrite: bool = False,
        suffix_number: bool = True,
    ) -> None:
        """Save series to disk.

        Args:
            folder_path: Directory to save series. If None, uses default:
                        __data__/series/{series_name}
            series_name: Name for the series. defaults to description
            overwrite: If False, appends counter to avoid overwriting existing folders
            suffix_number: If True, always adds _0000, _0001 suffix. If False, first
                          save has no suffix, subsequent saves get _2, _3, etc.

        Creates:
            {folder_path}/series_meta.json - metadata with matrice order & files
            {folder_path}/{matrice.description}.json - matrice metadata files
            {folder_path}/{matrice.description}.npz - matrice data files
        """
        if folder_path is None:
            folder_path = DATA_SERIES

        if series_name is None:
            series_name = self.description

        base_path = Path(folder_path) / Path(series_name)
        folder_path = self._resolve_save_path(base_path, overwrite, suffix_number)
        folder_path.mkdir(parents=True, exist_ok=True)

        matrice_files = {}
        used_filenames = set()

        for idx in self.indices:
            base_filename = self._matrices[idx].description
            filename = f"{base_filename}_{idx}"
            used_filenames.add(filename)
            matrice_files[str(idx)] = filename  # Store as string for JSON

        meta_payload = {
            "label": self.label,
            "matrice_indices": self.indices,  # Ordered list
            "num_matrices": len(self._matrices),
            "matrice_files": matrice_files,  # index → filename mapping
        }

        with (folder_path / "series_meta.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(meta_payload, f, indent=2)

        for idx in self.indices:
            filename = matrice_files[str(idx)]
            file_path = folder_path / filename
            self._matrices[idx].save(file_path, overwrite=True)

    @classmethod
    def load(
        cls,
        folder_path: Path | str,
        subset_indices: list[int] | None = None,
        subset_matrices: list[int] | slice | None = None
    ) -> Self:
        """Load series from disk with optional subsetting.

        Args:
            folder_path: Directory to load from
            subset_indices: Optional list of matrix row/column indices to keep
            subset_matrices: Optional list/slice of matrix indices (radii) to keep

        Returns
        -------
            Loaded MatrixSeries (or subclass)
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            msg = f"Folder not found: {folder_path}"
            raise FileNotFoundError(msg)

        meta_path = folder_path / "series_meta.json"
        if not meta_path.exists():
            msg = f"Metadata file not found: {meta_path}"
            raise FileNotFoundError(msg)

        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

        matrice_indices = metadata.get("matrice_indices", [])
        matrice_files = metadata.get("matrice_files", {})

        if not matrice_files:
            msg = f"No matrice_files mapping found in metadata: {meta_path}"
            raise ValueError(msg)

        # Filter which matrices to load
        if subset_matrices is not None:
            if isinstance(subset_matrices, slice):
                # Apply slice to the list of indices
                matrice_indices = matrice_indices[subset_matrices]
            else:
                # Keep only specified indices
                matrice_indices = [idx for idx in matrice_indices
                                if idx in subset_matrices]

        matrices_list = []
        matrice_class = cls._matrix_class
        for idx in matrice_indices:
            filename = matrice_files[str(idx)]
            file_path = folder_path / filename

            # Pass subset_indices to matrice load
            inst = matrice_class.load(file_path, subset_indices=subset_indices)
            matrices_list.append(inst)

        if not matrices_list:
            msg = f"No matrices found in {folder_path}"
            raise ValueError(msg)


        return cls(matrices=matrices_list)

    
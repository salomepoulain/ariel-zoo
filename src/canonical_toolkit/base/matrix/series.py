"""MatrixSeries: Generic collection of MatrixInstance objects indexed by index."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Self, overload

import scipy.sparse as sp
from rich.console import Console
from rich.table import Table

from .matrix import DATA_SERIES, MatrixInstance
from .matrix_types import I

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy as np


class MatrixSeries(Generic[I]):
    """Generic collection of matrices sharing the same label.

    Generic over I (the instance type). Defaults to MatrixInstance when not specified.
    Subclasses can specify their instance type: SimilaritySeries(MatrixSeries[SimilarityMatrix])
    """

    _instance_class = MatrixInstance

    def __init__(
        self,
        instances_list: list[I] | None = None,
    ) -> None:
        """Initialize MatrixSeries.

        Args:
            instances_list: List of instance objects (type I). All must have same label.
                           Cannot be empty or None.

        Raises
        ------
            ValueError: If instances_list is empty/None, has duplicate indices,
                       or instances have different labels
        """
        if not instances_list:
            msg = "instances_list cannot be empty or None. MatrixSeries requires at least one instance."
            raise ValueError(msg)

        labels = [inst.label for inst in instances_list]
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            msg = (
                f"All instances must have the same label. "
                f"Found {len(unique_labels)} different labels: {unique_labels}"
            )
            raise ValueError(msg)

        self._instances = dict(enumerate(instances_list))

    def items(self) -> tuple[list[int], list[sp.spmatrix | np.ndarray]]:
        return self.indices, self.matrices

    @property
    def label(self) -> str:
        """Return the label of all instances (guaranteed to be the same)."""
        return next(iter(self._instances.values())).label

    @property
    def instances(self) -> dict[int, I]:
        """Read-only access to instances dict."""
        return self._instances.copy()  # type: ignore[return-value]

    @property
    def indices(self) -> list[int]:
        """Get sorted list of indices."""
        return sorted(self._instances.keys())  # type: ignore[return-value]

    @property
    def matrices(self) -> list[sp.spmatrix | np.ndarray]:
        """Get list of matrices in index order."""
        return [self._instances[idx].matrix for idx in self.indices]

    @property
    def labels(self) -> list[str]:
        """Get list of labels in index order (all same, one per instance)."""
        return [self._instances[idx].label for idx in self.indices]

    @property
    def description(self) -> str:
        """Generate descriptive name: 'classname_label_Ni'."""
        class_name = self.__class__.__name__.lower()
        num_instances = len(self._instances)
        return f"{class_name}_{self.label}_{num_instances}i".lower()

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    @overload
    def __getitem__(self, key: int) -> I: ...

    def __getitem__(self, key: int | slice) -> I | Self:
        """
        Access instances by index or slice.

        Examples
        --------
            series[2]      # Get MatrixInstance at index 2
            series[:3]     # Get MatrixSeries with indices 0,1,2,3
            series[1:4]    # Get MatrixSeries with indices 1,2,3,4
            series[::2]    # Get every other index
        """
        if not isinstance(key, slice):
            return self._instances[key]

        all_indices = sorted(self._instances.keys())
        selected_indices = all_indices[key]
        new_instances_list = [self._instances[idx] for idx in selected_indices]

        return self.__class__(instances_list=new_instances_list)

    def __setitem__(self, index: int, instance: I) -> None:
        """Set instance at given index."""
        self._instances[index] = instance

    def __iter__(self) -> Iterator[int]:
        """Iterate over sorted indices."""
        return iter(sorted(self._instances.keys()))

    def __repr__(self) -> str:
        """Pretty table representation."""
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

        for idx in sorted(self._instances.keys()):
            inst = self._instances[idx]
            rows, cols = inst.shape

            shape_str = f"{rows}×{cols}"
            storage_abbr = "Sparse" if sp.issparse(inst.matrix) else "Dense"

            tags_str = ""
            if inst.tags:
                tags_str = f" {inst.tags}"

            cell_content = f"[{shape_str}] {storage_abbr}{tags_str}"
            table.add_row(str(idx), cell_content)

        string_buffer = io.StringIO()
        temp_console = Console(
            file=string_buffer, force_terminal=True, width=80
        )
        temp_console.print(table)
        return string_buffer.getvalue().rstrip()

    def replace(self, **changes: Any) -> Self:
        """Create a new MatrixSeries with updated fields.

        Args:
            **changes: Fields to update. Can use internal names (_instances)
                      _instances should be a dict[Hashable, MatrixInstance]

        Returns
        -------
            New instance of same type with updated fields
        """
        new_instances_dict = changes.get("_instances", self._instances.copy())
        new_instances_list = list(new_instances_dict.values())

        return self.__class__(instances_list=new_instances_list)

    def map(
        self,
        func: Callable[[I], I],
        *,
        inplace: bool = True,
    ) -> Self:
        """Apply function to all instances in the series.

        Args:
            func: Function to apply to each instance (type I)
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new instance of same type (safer). Default: True.
                    When False, also attempts to pass inplace=False to func if it accepts it.

        Returns
        -------
            Self if inplace=True, new instance of same type if inplace=False
        """
        new_data = {}
        for idx, inst in self._instances.items():
            if not inplace:
                # Try to pass inplace=False to the function if it accepts it
                try:
                    new_data[idx] = func(inst, inplace=False)
                except TypeError:
                    # Function doesn't accept inplace parameter
                    new_data[idx] = func(inst)
            else:
                new_data[idx] = func(inst)

        if inplace:
            self._instances = new_data
            return self

        return self.replace(_instances=new_data)

    def __add__(self, other: MatrixSeries[I]) -> Self:
        """Add two series element-wise at matching indices."""
        common = set(self._instances) & set(other._instances)
        new_instances = {idx: self[idx] + other[idx] for idx in common}

        return self.replace(_instances=new_instances)

    # --- I/O Methods ---

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
            {folder_path}/series_meta.json - metadata with instance order & files
            {folder_path}/{instance.description}.json - instance metadata files
            {folder_path}/{instance.description}.npz - instance data files
        """
        if folder_path is None:
            folder_path = DATA_SERIES

        if series_name is None:
            series_name = self.description

        base_path = Path(folder_path) / Path(series_name)
        folder_path = self._resolve_save_path(base_path, overwrite, suffix_number)
        folder_path.mkdir(parents=True, exist_ok=True)

        instance_files = {}
        used_filenames = set()

        for idx in self.indices:
            base_filename = self._instances[idx].description
            filename = f"{base_filename}_{idx}"
            used_filenames.add(filename)
            instance_files[str(idx)] = filename  # Store as string for JSON

        meta_payload = {
            "label": self.label,
            "instance_indices": self.indices,  # Ordered list
            "num_instances": len(self._instances),
            "instance_files": instance_files,  # index → filename mapping
        }

        with (folder_path / "series_meta.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(meta_payload, f, indent=2)

        for idx in self.indices:
            filename = instance_files[str(idx)]
            file_path = folder_path / filename
            self._instances[idx].save(file_path, overwrite=True)

    @classmethod
    def load(cls, folder_path: Path | str) -> Self:
        """Load series from disk.

        Args:
            folder_path: Directory to load from

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

        instance_indices = metadata.get("instance_indices", [])
        instance_files = metadata.get("instance_files", {})

        if not instance_files:
            msg = f"No instance_files mapping found in metadata: {meta_path}"
            raise ValueError(msg)

        instances_list = []
        instance_class = cls._instance_class or MatrixInstance
        for idx in instance_indices:
            filename = instance_files[str(idx)]
            file_path = folder_path / filename

            inst = instance_class.load(file_path)
            instances_list.append(inst)

        if not instances_list:
            msg = f"No instances found in {folder_path}"
            raise ValueError(msg)

        series = cls(instances_list=instances_list)
        new_instances = dict(zip(instance_indices, instances_list, strict=True))
        series._instances = new_instances

        return series


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    import scipy.sparse as sp

    # Test 1: Create series from instances list
    n_individuals = 10
    n_features = 100
    instances_list = []
    for idx in range(5):
        mat = sp.random(n_individuals, n_features, density=0.3, format="csr")
        inst = MatrixInstance(
            matrix=mat,
            label="experiment_A",
            tags={"type": "features", "index": idx},
        )
        instances_list.append(inst)

    series = MatrixSeries(instances_list=instances_list)

    # Test 2: Series slicing
    sliced = series[:3]

    # Test 5: Save/Load with custom path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test_series"
        series.save(test_path)
        loaded = MatrixSeries.load(test_path)

    # Test 6: Save/Load with default path
    series.save()  # Uses default __data__/series/{description}
    from .matrix import DATA_SERIES

    default_path = DATA_SERIES / series.description

    # Test 7: Load from default path
    loaded_from_default = MatrixSeries.load(default_path)

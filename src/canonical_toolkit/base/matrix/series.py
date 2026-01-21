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
    from collections.abc import Callable, Iterator, Iterable
    import numpy as np

__all__ = ["MatrixSeries"]


class MatrixSeries(Generic[I]):
    """Generic collection of matrices sharing the same label."""

    _instance_class = MatrixInstance

    def __init__(
        self, instances: Iterable[I], label: str | None = None
    ) -> None:
        """Initialize series from a collection of instances."""
        
        self._instances: dict[int, I] = {}

        if not instances:
            msg = "instances cannot be empty or None. MatrixSeries requires at least one instance."
            raise ValueError(msg)

        instances_list = list(instances)

        for i, inst in enumerate(instances_list):
            idx = inst.tags.get("radius", i)
            self._instances[int(idx)] = inst

        if label:
            self._label = label
        elif self._instances:
            self._label = next(iter(self._instances.values())).label
        else:
            self._label = "unlabeled_series"

    def reindex(self, target_indices: Iterable[int], fill_shape: tuple[int, int] | None = None) -> Self:
        """
        Ensure the series contains all target_indices. 
        Missing indices are filled with zero matrices.
        """
        if fill_shape is None:
            # Use the shape of the first existing matrix as a template
            fill_shape = self.instances[0].shape

        new_instances = self._instances.copy()
        is_sparse = sp.issparse(self.instances[0].matrix)

        for idx in target_indices:
            if idx not in new_instances:
                new_instances[idx] = self._instance_class.zeros(
                    shape=fill_shape,
                    label=self.label,
                    sparse=is_sparse
                )
        
        return self.replace(_instances=new_instances)

    @property
    def label(self) -> str:
        return self._label

    @property
    def instances(self) -> list[I]:
        return [self._instances[k] for k in sorted(self._instances)]

    @property
    def indices(self) -> list[int]:
        return sorted(self._instances.keys())

    def items(self) -> Iterable[tuple[int, I]]:
        for k in sorted(self._instances):
            yield k, self._instances[k]

    def __iter__(self) -> Iterator[I]:
        return iter(self.instances)

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    @overload
    def __getitem__(self, key: int) -> I: ...


    def __getitem__(self, key: int | slice) -> I | Self:
        if isinstance(key, slice):
            all_keys = sorted(self._instances.keys())
            sliced_keys = all_keys[key]
            
            if not sliced_keys:
                if not self._instances:
                    raise ValueError("Cannot slice empty series")
                
                # Get any instance for template
                first_key = next(iter(self._instances.keys()))
                dummy_instance = self._instances[first_key]
                
                # Create series with dummy instance
                new_series = self.__class__(instances=[dummy_instance], label=self.label)
                new_series._instances = {}
                return new_series
            new_series = self.__class__(
                instances=[self._instances[sliced_keys[0]]], 
                label=self.label
            )
            new_series._instances = {k: self._instances[k] for k in sliced_keys}
            return new_series
        
        # Integer case
        if key not in self._instances:
            raise KeyError(
                f"Index/Radius {key} not found in series '{self.label}'"
            )
        return self._instances[key]

    def __setitem__(self, key: int, instance: I) -> None:
        self._instances[key] = instance

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

        return self.__class__(instances=new_instances_list)

    @property
    def description(self) -> str:
        class_name = self.__class__.__name__.lower()
        num_instances = len(self._instances)
        return f"{class_name}_{self.label}_{num_instances}i".lower()

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

        for idx in sorted(self._instances.keys()):
            inst = self._instances[idx]
            shape_str = f"{inst.shape[0]}×{inst.shape[1]}"
            storage_abbr = "Sparse" if sp.issparse(inst.matrix) else "Dense"
            tags_str = f" {inst.tags}" if inst.tags else ""
            table.add_row(str(idx), f"[{shape_str}] {storage_abbr}{tags_str}")

        string_buffer = io.StringIO()
        Console(file=string_buffer, force_terminal=True, width=80).print(table)
        return string_buffer.getvalue().rstrip()

    def map(self, func: Callable[[I], I] | str, *, inplace: bool = True, **kwargs: Any) -> Self:
        new_data = {}
        for idx, inst in self._instances.items():
            if isinstance(func, str):
                new_data[idx] = getattr(inst, func)(**kwargs)
            else:
                new_data[idx] = func(inst)

        if inplace:
            self._instances = new_data
            return self
        return self.replace(_instances=new_data)

    def __add__(self, other: MatrixSeries[I]) -> Self:
        common = set(self._instances) & set(other._instances)
        new_instances = {idx: self[idx] + other[idx] for idx in common}
        return self.replace(_instances=new_instances)


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
    def load(cls, folder_path: Path | str, 
            subset_indices: list[int] | None = None,
            subset_matrices: list[int] | slice | None = None) -> Self:
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

        instance_indices = metadata.get("instance_indices", [])
        instance_files = metadata.get("instance_files", {})

        if not instance_files:
            msg = f"No instance_files mapping found in metadata: {meta_path}"
            raise ValueError(msg)

        # Filter which matrices to load
        if subset_matrices is not None:
            if isinstance(subset_matrices, slice):
                # Apply slice to the list of indices
                instance_indices = instance_indices[subset_matrices]
            else:
                # Keep only specified indices
                instance_indices = [idx for idx in instance_indices 
                                if idx in subset_matrices]

        instances_list = []
        instance_class = cls._instance_class or MatrixInstance
        for idx in instance_indices:
            filename = instance_files[str(idx)]
            file_path = folder_path / filename

            # Pass subset_indices to instance load
            inst = instance_class.load(file_path, subset_indices=subset_indices)
            instances_list.append(inst)

        if not instances_list:
            msg = f"No instances found in {folder_path}"
            raise ValueError(msg)

        series = cls(instances=instances_list)
        new_instances = dict(zip(instance_indices, instances_list, strict=True))
        series._instances = new_instances

        return series

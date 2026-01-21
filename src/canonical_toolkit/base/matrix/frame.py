"""MatrixFrame: Collection of MatrixSeries with DataFrame-like interface."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Self, overload

import scipy.sparse as sp
from rich.console import Console
from rich.table import Table

# from ._indexer import FrameLocIndexer
from .matrix import DATA_FRAMES, MatrixInstance
from .series import MatrixSeries
from .matrix_types import S

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Hashable, Iterable

__all__ = ["MatrixFrame"]


class MatrixFrame(Generic[S]):
    """
    The High-Level Controller.
    Manages storage, retrieval, and aggregation of matrix series.
    Fulfills FrameProtocol.
    """

    _series_class = MatrixSeries

    def __init__(self, series: list[S] | None = None) -> None:
        """Initialize MatrixFrame."""
        if series is not None:
            # Duplicate check
            labels = [s.label for s in series]
            duplicates = [label for label in labels if labels.count(label) > 1]
            if duplicates:
                raise ValueError(
                    f"Duplicate labels detected: {set(duplicates)}"
                )

            self._series = {s.label: s for s in series}
            self._ordered_labels = labels
        else:
            self._series = {}
            self._ordered_labels = []

    # --- Protocol Properties ---

    @property
    def series(self) -> list[S]:
        """Read-only access to series objects in ordered sequence."""
        return [self._series[label] for label in self._ordered_labels]

    @property
    def labels(self) -> list[str]:
        """Ordered list of series labels."""
        return self._ordered_labels

    @property
    def instances(self) -> list[list[Any]]:
        """2D list of Matrix objects: [Series_Index][Matrix_Index]."""
        return [s.instances for s in self.series]

    # --- Indexing ---

    @overload
    def __getitem__(self, key: tuple[slice, Any]) -> Any: ...

    @overload
    def __getitem__(self, key: list[str]) -> Self: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    @overload
    def __getitem__(self, key: str) -> S: ...

    def __getitem__(self, key: Any) -> Any:
        """
        Retrieve data via 2D slicing, 1D slicing, or label lookup.
        """
        # 1. 2D Tuple Indexing: frame[idx_slice, series_selector]
        if isinstance(key, tuple) and len(key) == 2:
            idx_slice, series_key = key
            if not isinstance(idx_slice, slice):
                raise TypeError(
                    f"First index must be a slice, got {type(idx_slice).__name__}"
                )

            # Positional slice on series dimension
            if isinstance(series_key, slice):
                selected_labels = self._ordered_labels[series_key]
                new_series = [
                    self._series[lbl][idx_slice] for lbl in selected_labels
                ]
                return self.__class__(series=new_series)

            # List of labels
            if isinstance(series_key, list):
                new_series = [
                    self._series[str(k)][idx_slice]
                    for k in series_key
                    if str(k) in self._series
                ]
                return self.__class__(series=new_series)

            # Single label
            label = getattr(series_key, "name", str(series_key))
            return self._series[label][idx_slice]

        # 2. 1D Slicing (Rows/Indices)
        if isinstance(key, slice):
            new_series = [s[key] for s in self.series]
            return self.__class__(series=new_series)

        # 3. List Selection (Columns/Labels)
        if isinstance(key, list):
            new_series = [
                self._series[str(k)] for k in key if str(k) in self._series
            ]
            return self.__class__(series=new_series)

        # 4. Single Label Lookup
        label = getattr(key, "name", str(key))
        if label in self._series:
            return self._series[label]

        raise KeyError(f"Key not found: {key}")

    def __setitem__(self, key: Hashable, val: S) -> None:
        label = getattr(key, "name", str(key))
        if label not in self._series:
            self._ordered_labels.append(label)
        self._series[label] = val

    # --- Iteration & Dict-like ---

    def __iter__(self) -> Iterator[S]:
        return iter(self.series)

    # def values(self) -> Iterable[S]:
    #     return self.series

    def items(self) -> Iterable[tuple[Hashable, S]]:
        return ((label, self._series[label]) for label in self._ordered_labels)

    # @property
    # def loc(self) -> FrameLocIndexer:
    #     return FrameLocIndexer(self)

    # --- Transformation ---

    def replace(self, **changes: Any) -> Self:
        if "series" in changes:
            return self.__class__(series=changes["series"])

        new_series_dict = changes.get("_series", self._series.copy())
        ordered_series = [
            new_series_dict[lbl]
            for lbl in self._ordered_labels
            if lbl in new_series_dict
        ]
        return self.__class__(series=ordered_series)

    def map(
        self,
        func: Callable[[S], S] | str,
        *,
        inplace: bool = True,
        **kwargs: Any,
    ) -> Self:
        new_series = {}
        for label in self._ordered_labels:
            series = self._series[label]
            if isinstance(func, str):
                if hasattr(series, func):
                    new_series[label] = getattr(series, func)(**kwargs)
                else:
                    new_series[label] = series.map(
                        func, inplace=inplace, **kwargs
                    )
            else:
                new_series[label] = func(series)

        if inplace:
            self._series = new_series
            return self
        return self.replace(_series=new_series)

    # --- Visualization ---

    def __repr__(self) -> str:
        if not self._ordered_labels:
            return f"<{self.__class__.__name__} [Empty]>"

        all_indices = set()
        for s in self.series:
            all_indices.update(s.indices)
        sorted_indices = sorted(all_indices)

        table = Table(
            title=f"{self.__class__.__name__} ({len(self.labels)} series × {len(sorted_indices)} indices)",
            title_style="bold bright_cyan",
            header_style="bold cyan",
        )
        table.add_column("Index", style="cyan", justify="right")
        for label in self.labels:
            table.add_column(label, justify="center")

        for idx in sorted_indices:
            row = [str(idx)]
            for label in self.labels:
                s = self._series[label]
                try:
                    inst = s[idx]
                    row.append(
                        f"[{inst.shape[0]}×{inst.shape[1]}] {'Sp' if sp.issparse(inst.matrix) else 'Dn'}"
                    )
                except KeyError:
                    row.append("...")
            table.add_row(*row)

        string_buffer = io.StringIO()
        Console(file=string_buffer, force_terminal=True, width=140).print(table)
        return string_buffer.getvalue().rstrip()

    # --- I/O ---

    @staticmethod
    def _resolve_save_path(
        base_path: Path, overwrite: bool, suffix_number: bool
    ) -> Path:
        if overwrite:
            return base_path
        counter, fmt = (
            (0, "{name}_{i:04d}") if suffix_number else (2, "{name}_{i}")
        )
        path = (
            base_path.with_name(fmt.format(name=base_path.name, i=counter))
            if suffix_number
            else base_path
        )
        while path.exists():
            counter += 1
            path = base_path.with_name(
                fmt.format(name=base_path.name, i=counter)
            )
        return path

    @property
    def description(self) -> str:
        class_name = self.__class__.__name__.lower()
        max_i = max((len(s.instances) for s in self.series), default=0)
        return f"{class_name}_{len(self.labels)}s_{max_i}i".lower()

    def save(
        self,
        folder_path: Path | str | None = None,
        frame_name: str | None = None,
        *,
        overwrite: bool = False,
        suffix_number: bool = True,
    ) -> None:
        """Save frame to folder using hierarchical structure.

        Args:
            folder_path: Directory to save frame. If None, uses default:
                        __data__/frames/{frame_name}
            frame_name: defautls to description
            overwrite: If False, appends counter to avoid overwriting existing folders
            suffix_number: If True, always adds _0000, _0001 suffix. If False, first
                          save has no suffix, subsequent saves get _2, _3, etc.

        Creates hierarchical structure:
            {folder_path}/
                frame_meta.json
                {series_label_1}/
                    series_meta.json
                    {index}.json
                    {index}.npz
                {series_label_2}/
                    ...
        """
        if folder_path is None:
            folder_path = DATA_FRAMES

        if frame_name is None:
            frame_name = self.description

        base_path = Path(folder_path) / Path(frame_name)
        folder_path = self._resolve_save_path(base_path, overwrite, suffix_number)
        folder_path.mkdir(parents=True, exist_ok=True)

        series_folders = {
            label: self._series[label].description
            for label in self._ordered_labels
        }

        frame_info = {
            "series_labels": self._ordered_labels,
            "series_folders": series_folders,
        }
        with (folder_path / "frame_meta.json").open("w", encoding="utf-8") as f:
            json.dump(frame_info, f, indent=2)

        for label in self._ordered_labels:
            self._series[label].save(folder_path, series_folders[label], overwrite=True)

    @classmethod
    def load(cls, folder_path: Path | str) -> Self:
        """Load frame from folder using hierarchical structure.

        Args:
            folder_path: Directory to load from

        Returns
        -------
            Loaded MatrixFrame (or subclass)
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            msg = f"Folder not found: {folder_path}"
            raise FileNotFoundError(msg)

        # Load metadata to get series order
        meta_path = folder_path / "frame_meta.json"
        if not meta_path.exists():
            msg = f"Frame metadata not found: {meta_path}"
            raise FileNotFoundError(msg)

        with meta_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        series_labels = metadata.get("series_labels", [])
        series_folders = metadata.get("series_folders", {})
        
        # Load each series from its subfolder using the folder mapping
        series_list = []
        # Use the class's specified series class (polymorphic!)
        series_class = cls._series_class or MatrixSeries
        for label in series_labels:
            # Get folder name from mapping (or fallback to label if mapping missing)
            folder_name = series_folders.get(label, str(label))            
            series_folder = folder_path / folder_name

            if series_folder.exists() and series_folder.is_dir():
                try:
                    series = series_class.load(series_folder)
                    series_list.append(series)
                except Exception:
                    # Skip series that fail to load
                    pass
        
        if not series_list:
            msg = f"No series found in {folder_path}"
            raise ValueError(msg)

        # Create frame with loaded series (preserves order!)
        return cls(series=series_list)



# --- Test Sync ---
if __name__ == "__main__":
    # FIXED: Calling constructors with 'instances' instead of 'instances_list'
    s_a = MatrixSeries(
        instances=[MatrixInstance(sp.eye(5), "A")], label="series_A"
    )
    s_b = MatrixSeries(
        instances=[MatrixInstance(sp.eye(5), "B")], label="series_B"
    )

    frame = MatrixFrame(series=[s_a, s_b])
    print(frame)

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

    def __init__(self, series: list[S]) -> None:
        """Initialize MatrixFrame."""
        labels = [s.label for s in series]
        duplicates = [label for label in labels if labels.count(label) > 1]
        if duplicates:
            raise ValueError(
                f"Duplicate labels detected: {set(duplicates)}"
            )

        self._series: list[S] = list(series)
        self._label_map: dict[str, int] = {s.label: i for i, s in enumerate(series)}

        self.align(inplace=True)

    def align(self, inplace: bool = False) -> Self | None:
        """
        Aligns all series to the same length by padding shorter ones with zeros.
        Assumes list index i corresponds to radius i.
        """
        if not self._series:
            return self if not inplace else None

        # 1. Determine the maximum length among all contained series
        max_len = max(len(s.matrices) for s in self._series)

        new_series_list = []
        needs_align = False

        for s in self._series:
            current_len = len(s.matrices)
            if current_len < max_len:
                needs_align = True
                padding = [None] * (max_len - current_len)
                new_s = s.__class__(
                    matrices=s.matrices + padding, #type: ignore
                    label=s.label                    #type: ignore
                )
                new_series_list.append(new_s)
            else:
                new_series_list.append(s)

        if not needs_align:
             return self if not inplace else None

        if inplace:
            self._series = new_series_list
            # Map doesn't change since order/labels are same
            return None

        return self.__class__(series=new_series_list)

    # --- Protocol Properties ---

    @property
    def series(self) -> list[S]:
        """Read-only access to series objects in ordered sequence."""
        return self._series.copy()

    @property
    def labels(self) -> list[str]:
        """Ordered list of series labels."""
        return [s.label for s in self._series]

    @property
    def matrices(self) -> list[list[Any]]:
        """2D list of Matrix objects: [Series_Index][Matrix_Index]."""
        return [s.matrices for s in self._series]

    @property
    def shape(self) -> tuple[int, int]:
        """returns r,c shape"""
        mats = self.matrices
        if not mats:
            return (0, 0)
        return (len(mats[0]), len(mats))

    # --- Indexing ---


    @overload
    def __getitem__(self, key: int) -> S: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    @overload
    def __getitem__(self, key: list[str]) -> Self: ...

    @overload
    def __getitem__(self, key: str) -> S: ...

    @overload
    def __getitem__(self, key: tuple[int | slice, int | slice | str | list[str]]) -> S | Self: ...


    def __getitem__(self, key: Any) -> S | Self:
        """
        Comprehensive indexing for MatrixFrame.

        Indexing Rules:
        - int: Returns Series at that position (Column access)
        - str: Returns Series with that label (Column access)
        - slice: Returns Frame with subset of series (Column slice)
        - list[str]: Returns Frame with selected series (Column subset)
        - tuple: (row_selector, col_selector) -> (radius/index, series)
        """

        # --- 1. Handle 2D Tuple Indexing: frame[radius, series] ---
        if isinstance(key, tuple) and len(key) == 2:
            rad_selector, col_selector = key

            # A. Resolve Column Selection
            if isinstance(col_selector, int):
                cols = [self._series[col_selector]]
                single_col = True
            elif isinstance(col_selector, str):
                cols = [self._series[self._label_map[col_selector]]]
                single_col = True
            elif isinstance(col_selector, slice):
                cols = self._series[col_selector]
                single_col = False
            elif isinstance(col_selector, list):
                indices = [
                    self._label_map[c] if isinstance(c, str) else c
                    for c in col_selector
                ]
                cols = [self._series[i] for i in indices]
                single_col = False
            else:
                raise TypeError(f"Invalid column selector: {type(col_selector)}")

            # B. Resolve Row Selection on the chosen columns
            # Case 1: Single Row, Single Column -> Return MatrixInstance (Leaf)
            if isinstance(rad_selector, int) and single_col:
                return cols[0][rad_selector]

            # Case 2: Multi-Row, Single Column -> Return MatrixSeries (Column)
            if single_col:
                # rad_selector is slice or :
                return cols[0][rad_selector]

            # Case 3: Single Row, Multi-Column -> Return MatrixFrame (height 1)
            if isinstance(rad_selector, int):
                new_series = [
                    s.__class__(matrices=[s[rad_selector]], label=s.label)
                    for s in cols
                ]
                return self.__class__(series=new_series)

            # Case 4: Multi-Row, Multi-Column -> Return MatrixFrame (Sub-grid)
            new_series = [s[rad_selector] for s in cols]
            return self.__class__(series=new_series)


        # --- 2. Handle Single Integer: frame[3] -> Frame (Row 3, all Series) ---
        if isinstance(key, int):
            # Numpy-style: Indexing first dimension (Rows/Radii)
            # Returns a new Frame containing the single row at this index
            new_series_list = []
            valid_idx = False
            for s in self._series:
                try:
                    # Get matrix at this radius index
                    mat = s[key]
                    valid_idx = True
                    # Wrap in new Series of length 1
                    new_s = s.__class__(matrices=[mat], label=s.label)
                    new_series_list.append(new_s)
                except (IndexError, KeyError):
                    # If aligned, this shouldn't happen unless out of bounds for all
                    pass

            if not valid_idx:
                raise IndexError(f"Radius index {key} out of bounds")

            return self.__class__(series=new_series_list)

        # --- 3. Handle Single Slice: frame[:2] -> Frame slice ---
        if isinstance(key, slice):
            return self.__class__(series=self._series[key])

        # --- 4. Handle List of Labels: frame[["A", "B"]] -> Frame subset ---
        if isinstance(key, list):
            indices = []
            for k in key:
                label = getattr(k, "name", str(k))
                if label not in self._label_map:
                     raise KeyError(f"Label '{label}' not found in frame")
                indices.append(self._label_map[label])

            new_series = [self._series[i] for i in indices]
            return self.__class__(series=new_series)

        # --- 5. Handle Single Label: frame["FRONT"] -> Series ---
        label = getattr(key, "name", str(key))
        if label in self._label_map:
            return self._series[self._label_map[label]]

        raise KeyError(f"Key not found: {key}")


    # --- Iteration & Dict-like ---

    def __iter__(self) -> Iterator[S]:
        return iter(self._series)

    # def values(self) -> Iterable[S]:
    #     return self._series

    def items(self) -> Iterable[tuple[Hashable, S]]:
        return ((s.label, s) for s in self._series)

    # @property
    # def loc(self) -> FrameLocIndexer:
    #     return FrameLocIndexer(self)

    # --- Transformation ---

    def replace(self, **changes: Any) -> Self:
        if "series" in changes:
            return self.__class__(series=changes["series"])

        new_series = changes.get("_series", self._series)
        return self.__class__(series=new_series)

    def map(self, func: Callable[[S], S] | str, **kwargs: Any) -> Self:
        new_data = []
        for serie in self.series:
            if isinstance(func, str):
                # If method exists on series, call it directly
                # Otherwise, delegate to series.map() to call on each matrix
                if hasattr(serie, func) and callable(getattr(serie, func)):
                    new_data.append(getattr(serie, func)(**kwargs))
                else:
                    new_data.append(serie.map(func, **kwargs))
            else:
                new_data.append(func(serie))

        if kwargs.get('inplace', False):
            self._series = new_data
            return self
        return self.replace(series=new_data)

    # --- Visualization ---

    def __repr__(self) -> str:
        if not self._series:
            return f"<{self.__class__.__name__} [Empty]>"

        # Assuming aligned series (same indices)
        # Use first series to get indices (radii)
        num_indices = len(self._series[0].indices)
        indices = self._series[0].indices

        table = Table(
            title=f"{self.__class__.__name__} ({len(self._series)} series × {num_indices} indices)",
            title_style="bold bright_cyan",
            header_style="bold cyan",
            # show_lines=True,
            border_style="dim",
        )
        table.add_column("IDX", style="cyan", justify="right")
        for label in self.labels:
            table.add_column(label, justify="center")

        for i, idx in enumerate(indices):
            row = [str(idx)]
            for s in self._series:
                try:
                    inst = s[i] # Direct list index access
                    shape_str = f"{inst.shape[0]}×{inst.shape[1]}"
                    type_str = "Sp" if sp.issparse(inst.matrix) else "Dn"
                    row.append(f"[{shape_str}] {type_str}")
                except IndexError:
                    row.append("...")
            table.add_row(*row)

        string_buffer = io.StringIO()
        Console(file=string_buffer, force_terminal=True, force_jupyter=False, width=100).print(table)
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
        max_i = max((len(s.matrices) for s in self._series), default=0)
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
            s.label: s.description
            for s in self._series
        }

        frame_info = {
            "series_labels": self.labels, # List preserves order
            "series_folders": series_folders,
        }
        with (folder_path / "frame_meta.json").open("w", encoding="utf-8") as f:
            json.dump(frame_info, f, indent=2)

        for s in self._series:
            s.save(folder_path, series_folders[s.label], overwrite=True)

    @classmethod
    def load(cls, folder_path: Path | str,
            subset_indices: list[int] | None = None,
            subset_matrices: list[int] | slice | None = None,
            subset_series: list[str] | list[int] | None = None) -> Self:
        """Load frame from folder using hierarchical structure with optional subsetting.

        Args:
            folder_path: Directory to load from
            subset_indices: Optional list of matrix row/column indices to keep
            subset_matrices: Optional list/slice of matrix indices (radii) to keep
            subset_series: Optional list of series labels or positions to keep

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

        # Filter which series to load
        if subset_series is not None:
            if subset_series and isinstance(subset_series[0], str):
                # Filter by label names
                series_labels = [label for label in series_labels
                            if label in subset_series]
            else:
                # Filter by position indices
                series_labels = [series_labels[i] for i in subset_series
                            if i < len(series_labels)]

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
                    series = series_class.load(
                        series_folder,
                        subset_indices=subset_indices,
                        subset_matrices=subset_matrices
                    )
                    series_list.append(series)
                except Exception as e:
                    # Optional: log the error but continue
                    import warnings
                    warnings.warn(f"Failed to load series '{label}': {e}")
                    # Skip series that fail to load
                    continue

        if not series_list:
            msg = f"No series found in {folder_path}"
            raise ValueError(msg)

        # Create frame with loaded series (preserves order!)
        return cls(series=series_list)

    # def copy(self) -> Self:
    #     """Create a new Frame instance with copies of the underlying series."""
    #     return self.__class__(series=[s.copy() for s in self._series])

# --- Test Sync ---
if __name__ == "__main__":
    # FIXED: Calling constructors with 'instances' instead of 'instances_list'
    s_a = MatrixSeries(
        matrices=[MatrixInstance(sp.eye(5), "A")], label="series_A"
    )
    s_b = MatrixSeries(
        matrices=[MatrixInstance(sp.eye(5), "B")], label="series_B"
    )

    frame = MatrixFrame(series=[s_a, s_b])
    print(frame)

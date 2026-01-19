"""MatrixFrame: Collection of MatrixSeries with DataFrame-like interface."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, Self, overload

import scipy.sparse as sp
from rich.console import Console
from rich.table import Table

from ._indexer import FrameLocIndexer
from .matrix import DATA_FRAMES, MatrixInstance
from .matrix_types import S
from .series import MatrixSeries

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable


class MatrixFrame(Generic[S]):
    """
    The High-Level Controller.
    Manages storage, retrieval, and aggregation of matrix series.

    Generic over S (the series type). Defaults to MatrixSeries when not specified.
    Subclasses can specify their series type: SimilarityFrame(MatrixFrame[SimilaritySeries])
    """

    _series_class = MatrixSeries

    def __init__(
        self,
        series: list[S] | None = None,
    ) -> None:
        """Initialize MatrixFrame.

        Args:
            series: List of series objects (type S) - uses series.label as dict key

        Raises
        ------
            ValueError: If multiple series have the same label
        """
        # Build internal dict from list using series.label as key
        if series is not None:
            # Check for duplicate labels
            labels = [s.label for s in series]
            duplicates = [label for label in labels if labels.count(label) > 1]
            if duplicates:
                dup_names = set(duplicates)
                msg = (
                    f"Multiple series with same label detected: {dup_names}. "
                    f"Each series must have a unique label."
                )
                raise ValueError(msg)

            self._series = {s.label: s for s in series}
            self._ordered_labels = labels  # Preserve insertion order
        else:
            self._series = {}
            self._ordered_labels = []
            
    def items(self) -> tuple[list[str], list[S]]:
        return self.labels, self.series

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    @overload
    def __getitem__(self, key: list[Hashable]) -> Self: ...

    @overload
    def __getitem__(self, key: Hashable) -> S: ...

    @overload
    def __getitem__(self, key: tuple[slice, slice]) -> Self: ...

    @overload
    def __getitem__(self, key: tuple[slice, Hashable]) -> S: ...

    @overload
    def __getitem__(self, key: tuple[slice, list[Hashable]]) -> Self: ...

    def __getitem__(
        self,
        key: Hashable | slice | list[Hashable] | tuple[slice, slice | Hashable | list[Hashable]],
    ) -> S | Self:
        """
        Access series by label, slice by index, or 2D matrix-style indexing.

        Examples
        --------
            frame["FRONT"]              # Get single series by label
            frame[:3]                   # Slice all series to indices 0-2
            frame[["FRONT", "BACK"]]    # Select specific series by label

            # 2D indexing (indices, series):
            frame[:3, 2:]               # Indices 0-2, series positions 2+ → Frame
            frame[:, :2]                # All indices, first 2 series → Frame
            frame[:3, "FRONT"]          # Indices 0-2, single series by label → Series
            frame[:, ["FRONT", "BACK"]] # All indices, multiple series by label → Frame
        """
        # Handle 2D tuple indexing: frame[idx_slice, series_selector]
        if isinstance(key, tuple) and len(key) == 2:
            idx_slice, series_key = key

            if not isinstance(idx_slice, slice):
                msg = f"First index must be a slice, got {type(idx_slice).__name__}"
                raise TypeError(msg)

            # Slice on series dimension → positional selection
            if isinstance(series_key, slice):
                selected_labels = self._ordered_labels[series_key]
                new_series = [
                    self._series[label][idx_slice] for label in selected_labels
                ]
                return self.__class__(series=new_series)

            # List of labels → multiple series by name
            if isinstance(series_key, list):
                new_series = []
                for k in series_key:
                    label = getattr(k, "name", str(k))
                    if label in self._series:
                        new_series.append(self._series[label][idx_slice])
                return self.__class__(series=new_series)

            # Single label → return sliced Series
            label = getattr(series_key, "name", str(series_key))
            if label not in self._series:
                msg = f"Key not found: {series_key} (serialized: {label})"
                raise KeyError(msg)
            return self._series[label][idx_slice]

        if isinstance(key, slice):
            new_series = [
                self._series[label][key] for label in self._ordered_labels
            ]
            return self.__class__(series=new_series)

        if isinstance(key, list):
            new_series = []
            for k in key:
                label = getattr(k, "name", str(k))
                if label in self._series:
                    new_series.append(self._series[label])

            return self.__class__(series=new_series)

        label = getattr(key, "name", str(key))
        if label in self._series:
            return self._series[label]

        msg = f"Key not found: {key} (serialized: {label})"
        raise KeyError(msg)

    def __setitem__(self, key: Hashable, val: S) -> None:
        """Set series at given label."""
        label = getattr(key, "name", str(key))
        if label not in self._series:
            self._ordered_labels.append(label)
        self._series[label] = val

    # def keys(self) -> list[str]:
    #     """Get all series labels in order."""
    #     return self._ordered_labels

    # def values(self) -> list[S]:
    #     """Get all series in order."""
    #     return [self._series[label] for label in self._ordered_labels]

    @property
    def series(self) -> list[S]:
        """Read-only access to series as a list in order."""
        return [self._series[label] for label in self._ordered_labels]

    @property
    def matrices(self) -> list[list[float]]:
        """Get 2D list of matrices organized by series.

        Returns list where each element is a list of matrices from one series,
        in index order.
        """
        return [self._series[label].matrices for label in self._ordered_labels]

    @property
    def indices(self) -> list[list[Hashable]]:
        """Get 2D list of indices organized by series.

        Returns list where each element is a list of indices from one series,
        in sorted order.
        """
        return [self._series[label].indices for label in self._ordered_labels]

    @property
    def labels(self) -> list[str]:
        """Get 2D list of labels organized by series.

        Returns list where each element is a list of labels from one series,
        in index order.
        """
        return [self._series[label].label for label in self._ordered_labels]

    @property
    def loc(self) -> FrameLocIndexer:
        """
        Pandas-style .loc indexer for 2D slicing.

        Examples
        --------
            frame.loc[:3, "series_A"]    # Indices 0-3, single series
            frame.loc[:, ["A", "B"]]     # All indices, multiple series
            frame.loc[2, "series_A"]     # Single index, single series
        """
        return FrameLocIndexer(self)

    def __repr__(self) -> str:
        """Renders Matrix Dashboard using rich tables."""
        # Use ordered labels
        series_keys = self._ordered_labels
        if not series_keys:
            return f"<{self.__class__.__name__} [Empty]>"

        all_indices = set()
        for label in self._ordered_labels:
            all_indices.update(self._series[label].instances.keys())
        sorted_indices = sorted(all_indices)

        title = f"{self.__class__.__name__} ({len(series_keys)} series × {len(sorted_indices)} indices)"
        table = Table(
            title=title,
            title_style="bold bright_cyan",
            title_justify="left",
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Index", style="cyan", justify="right")
        for label in series_keys:
            label_str = getattr(label, "name", str(label))
            table.add_column(label_str, justify="center")

        for idx in sorted_indices:
            row = [str(idx)]
            for label in series_keys:
                inst = self._series[label].instances.get(idx, None)
                if inst is None:
                    row.append("...")
                else:
                    rows, cols = inst.shape

                    shape_str = f"{rows}×{cols}"
                    storage_abbr = "Sp" if sp.issparse(inst.matrix) else "Dn"

                    tags_str = ""
                    if inst.tags:
                        first_key = next(iter(inst.tags))
                        tags_str = f" {first_key}:{inst.tags[first_key]}"

                    row.append(f"[{shape_str}] {storage_abbr}{tags_str}")
            table.add_row(*row)

        string_buffer = io.StringIO()
        temp_console = Console(
            file=string_buffer,
            force_terminal=True,
            width=140,
        )
        temp_console.print(table)
        return string_buffer.getvalue().rstrip()

    def replace(self, **changes: Any) -> Self:
        """Create a new MatrixFrame with updated fields.

        Args:
            **changes: Fields to update. Can use either public names
                      (series) or internal names (_series)

        Returns
        -------
            New instance of same type with updated fields
        """
        if "series" in changes:
            changes["_series"] = changes.pop("series")

        new_series_dict = changes.get("_series", self._series.copy())

        ordered_series = [
            new_series_dict[label]
            for label in self._ordered_labels
            if label in new_series_dict
        ]
        return self.__class__(series=ordered_series)

    def map(
        self,
        func: Callable[[S], S],
        *,
        inplace: bool = True,
    ) -> Self:
        """Apply function to all series in the frame.

        Args:
            func: Function to apply to each series (type S)
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new instance of same type (safer). Default: True.
                    When False, also attempts to pass inplace=False to func if it accepts it.

        Returns
        -------
            Self if inplace=True, new instance of same type if inplace=False
        """
        new_series = {}
        for label in self._ordered_labels:
            series = self._series[label]
            if not inplace:
                # Try to pass inplace=False to the function if it accepts it
                try:
                    new_series[label] = func(series, inplace=False)
                except TypeError:
                    # Function doesn't accept inplace parameter
                    new_series[label] = func(series)
            else:
                new_series[label] = func(series)

        if inplace:
            self._series = new_series
            return self
        return self.replace(_series=new_series)

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

    @property
    def description(self) -> str:
        """Generate a descriptive name: 'classname_Ns_Mi' where M is max indices."""
        class_name = self.__class__.__name__.lower()
        num_series = len(self._ordered_labels)

        max_indices = 0
        for label in self._ordered_labels:
            num_inst = len(self._series[label].instances)
            max_indices = max(max_indices, num_inst)

        return f"{class_name}_{num_series}s_{max_indices}i".lower()

        

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


if __name__ == "__main__":
    import scipy.sparse as sp

    # Test 1: Create frame with multiple series

    # Create series A
    series_a_instances = []
    for _idx in range(5):
        mat = sp.random(10, 20, density=0.3, format="csr")
        inst = MatrixInstance(
            matrix=mat,
            label="series_A",
            tags={"type": "features"},
        )
        series_a_instances.append(inst)
    series_a = MatrixSeries(instances_list=series_a_instances)

    # Create series B
    series_b_instances = []
    for _idx in range(5):
        mat = sp.random(10, 20, density=0.3, format="csr")
        inst = MatrixInstance(
            matrix=mat,
            label="series_B",
            tags={"tag_type": "tag_A"},
        )
        series_b_instances.append(inst)
    series_b = MatrixSeries(instances_list=series_b_instances)

    frame = MatrixFrame(series=[series_a, series_b])

    # Test 2: Frame slicing by index
    sliced = frame[:2]

    # Test 3: Frame selection by label
    selected = frame[["series_A", "series_B"]]

    # Test 4: Frame .loc 2D slicing
    loc_result = frame.loc[:2, "series_A"]

    # Test 5: Frame repr

    # # Test 6: Cumulative
    # print("\n[6] Cumulative frame:")
    # cumulative = frame.to_cumulative(inplace=False)
    # print(f"✓ Cumulative frame created")

    # Test 7: Description

    # Test 8: Save/Load with custom path (hierarchical structure)
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test_frame"
        frame.save(test_path)

        # Verify hierarchical structure exists (using descriptive folder names)
        assert (test_path / "frame_meta.json").exists()
        series_a_folder = test_path / "matrixseries_series_a_5i"
        series_b_folder = test_path / "matrixseries_series_b_5i"
        assert series_a_folder.is_dir()
        assert series_b_folder.is_dir()
        assert (series_a_folder / "series_meta.json").exists()

        # Load it back
        loaded = MatrixFrame.load(test_path)

    # Test 9: Save/Load with default path
    frame.save()  # Uses default __data__/frames/{description}
    default_path = DATA_FRAMES / frame.description

    # Test 10: Load from default path
    loaded_from_default = MatrixFrame.load(default_path)

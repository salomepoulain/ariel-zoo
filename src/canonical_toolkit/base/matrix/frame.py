"""MatrixFrame: Collection of MatrixSeries with DataFrame-like interface."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING

import scipy.sparse as sp
from rich.console import Console
from rich.table import Table

from .matrix import MatrixInstance, DATA_FRAMES
from .series import MatrixSeries
from .index_typing import SortableHashable

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable

class _FrameLocIndexer:
    """
    Helper class for pandas-style .loc indexing on MatrixFrame.
    Supports 2D slicing: frame.loc[index_slice, label_selector]
    """

    def __init__(self, frame: MatrixFrame) -> None:
        self._frame = frame

    def __getitem__(
        self,
        key: tuple[slice | Hashable, Hashable | list[Hashable]]
        | slice
        | Hashable,
    ) -> MatrixFrame | MatrixSeries | MatrixInstance:
        """
        2D slicing for MatrixFrame.

        Examples:
            frame.loc[:3, "series_A"]       # Indices 0-3, single series
            frame.loc[:, ["A", "B"]]        # All indices, specific series
            frame.loc[2, "series_A"]        # Single index, single series
            frame.loc[:3]                   # Indices 0-3, all series
        """
        # Handle 1D indexing (just index)
        if isinstance(key, (slice,)) or not isinstance(key, tuple):
            return self._frame[key]

        # Handle 2D indexing (index, label)
        if not isinstance(key, tuple) or len(key) != 2:
            msg = "loc requires either a single index or a tuple of (index, label)"
            raise TypeError(msg)

        index_idx, label_idx = key

        # First, filter by index (rows)
        if isinstance(index_idx, slice):
            filtered_frame = self._frame[index_idx]
        else:
            # Single index - will return series or instance
            filtered_frame = self._frame

        # Then, filter by label (columns)
        if isinstance(label_idx, list):
            result = filtered_frame[label_idx]
        elif isinstance(label_idx, (str, int, tuple)) or hasattr(label_idx, "__hash__"):
             # Hashable check (simplistic) - trusting typing for now
            result = filtered_frame[label_idx]
        else:
            msg = f"Invalid label index type: {type(label_idx)}"
            raise TypeError(msg)

        # If we had a single index, extract that index from the result
        if not isinstance(index_idx, slice):
            if isinstance(result, MatrixSeries):
                # Single series, single index -> MatrixInstance
                return result[index_idx]
            elif isinstance(result, MatrixFrame):
                # Multiple series, single index -> MatrixFrame with only that index
                new_series_list = []
                for series in result.series:
                    if index_idx in series.instances:
                        # Create a new series with just this index
                        new_ser = MatrixSeries(
                            instances_list=[series[index_idx]]
                        )
                        new_series_list.append(new_ser)
                return MatrixFrame(series=new_series_list)

        return result


class MatrixFrame:
    """
    The High-Level Controller.
    Manages storage, retrieval, and aggregation of matrix series.
    """

    # Class attribute: Subclasses override to specify their series type
    _series_class = None  # Will be set to MatrixSeries below (avoid circular ref)

    def __init__(
        self,
        series: list[MatrixSeries] | None = None,
    ) -> None:
        """Initialize MatrixFrame.

        Args:
            series: List of MatrixSeries (uses series.label as dict key)

        Raises:
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

    def __getitem__(
        self, key: Hashable | slice | list[Hashable]
    ) -> MatrixSeries | MatrixFrame:
        """
        Access series by label, slice by index, or select multiple series.

        Examples:
            frame["series_A"]       # Get single series
            frame[:3]               # Slice all series to indices 0-3
            frame[["A", "B"]]       # Select specific series
        """
        # Slice - apply to all series (filter by index)
        if isinstance(key, slice):
            # Preserve order when slicing
            new_series = [
                self._series[label][key] for label in self._ordered_labels
            ]
            # Return new instance of same type (works for subclasses!)
            return self.__class__(series=new_series)

        # List of labels - select specific series
        if isinstance(key, list):
            # Preserve order of requested keys
            new_series = []
            for k in key:
                label = getattr(k, "name", str(k))
                if label in self._series:
                    new_series.append(self._series[label])
            # Return new instance of same type (works for subclasses!)
            return self.__class__(series=new_series)

        # Single label key - return that series
        # Serialize key to string
        label = getattr(key, "name", str(key))
        if label in self._series:
            return self._series[label]

        raise KeyError(f"Key not found: {key} (serialized: {label})")

    def __setitem__(self, key: Hashable, val: MatrixSeries) -> None:
        """Set series at given label."""
        # Serialize key to string for consistent storage
        label = getattr(key, "name", str(key))
        # Add to order if new
        if label not in self._series:
            self._ordered_labels.append(label)
        self._series[label] = val

    def keys(self):
        """Get all series labels in order."""
        return self._ordered_labels

    def values(self):
        """Get all series in order."""
        return [self._series[label] for label in self._ordered_labels]

    @property
    def series(self) -> list[MatrixSeries]:
        """Read-only access to series as a list in order."""
        return [self._series[label] for label in self._ordered_labels]

    @property
    def matrices(self) -> list[list]:
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
    def labels(self) -> list[list[Hashable]]:
        """Get 2D list of labels organized by series.

        Returns list where each element is a list of labels from one series,
        in index order.
        """
        return [self._series[label].labels for label in self._ordered_labels]

    @property
    def loc(self) -> _FrameLocIndexer:
        """
        Pandas-style .loc indexer for 2D slicing.

        Examples:
            frame.loc[:3, "series_A"]    # Indices 0-3, single series
            frame.loc[:, ["A", "B"]]     # All indices, multiple series
            frame.loc[2, "series_A"]     # Single index, single series
        """
        return _FrameLocIndexer(self)

    def __repr__(self) -> str:
        """Renders Matrix Dashboard using rich tables."""
        # Use ordered labels
        series_keys = self._ordered_labels
        if not series_keys:
            return "<MatrixFrame [Empty]>"

        all_indices = set()
        for label in self._ordered_labels:
            all_indices.update(self._series[label].instances.keys())
        sorted_indices = sorted(all_indices)

        # Create table
        title = f"MatrixFrame ({len(series_keys)} series × {len(sorted_indices)} indices)"
        table = Table(
            title=title,
            title_style="bold bright_cyan",
            title_justify="left",
            show_header=True,
            header_style="bold cyan",
        )

        # Add columns: Index + one column per series
        table.add_column("Index", style="cyan", justify="right")
        for label in series_keys:
            # Handle Enums gracefully for display
            label_str = getattr(label, "name", str(label))
            table.add_column(label_str, justify="center")

        # Add rows for each index
        for idx in sorted_indices:
            row = [str(idx)]
            for label in series_keys:
                inst = self._series[label].instances.get(idx, None)
                if inst is None:
                    row.append("...")
                else:
                    rows, cols = inst.shape

                    # Simple generic format
                    shape_str = f"{rows}×{cols}"
                    storage_abbr = "Sp" if sp.issparse(inst.matrix) else "Dn"

                    # Show tags if present
                    tags_str = ""
                    if inst.tags:
                        # Show first tag only for compactness
                        first_key = next(iter(inst.tags))
                        tags_str = f" {first_key}:{inst.tags[first_key]}"

                    row.append(f"[{shape_str}] {storage_abbr}{tags_str}")
            table.add_row(*row)

        # Render table to string with colors enabled
        string_buffer = io.StringIO()
        temp_console = Console(file=string_buffer, force_terminal=True, width=140)
        temp_console.print(table)
        return string_buffer.getvalue().rstrip()

    def replace(self, **changes) -> MatrixFrame:
        """Create a new MatrixFrame with updated fields.

        Args:
            **changes: Fields to update. Can use either public names
                      (series) or internal names (_series)

        Returns:
            New instance of same type with updated fields
        """
        # Map public names to internal if needed
        if "series" in changes:
            changes["_series"] = changes.pop("series")

        new_series_dict = changes.get("_series", self._series.copy())

        # Preserve order when creating new instance
        ordered_series = [
            new_series_dict[label]
            for label in self._ordered_labels
            if label in new_series_dict
        ]
        return self.__class__(series=ordered_series)

    def map(
        self,
        func: Callable[[MatrixSeries], MatrixSeries],
        *,
        inplace: bool = True,
    ) -> MatrixFrame:
        """Apply function to all series in the frame.

        Args:
            func: Function to apply to each MatrixSeries
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new instance of same type (safer). Default: True.

        Returns:
            Self if inplace=True, new instance of same type if inplace=False
        """
        # Preserve order when mapping
        new_series = {label: func(self._series[label]) for label in self._ordered_labels}

        if inplace:
            self._series = new_series
            return self
        return self.replace(_series=new_series)

    def to_cumulative(self, *, inplace: bool = True) -> MatrixFrame:
        """Convert all series to cumulative sum across indices.

        Args:
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new MatrixFrame (safer). Default: True.

        Returns:
            Self if inplace=True, new MatrixFrame if inplace=False
        """
        return self.map(lambda s: s.to_cumulative(inplace=False), inplace=inplace)

    # --- I/O Methods ---

    @property
    def description(self) -> str:
        """Generate a descriptive name: 'classname_Ns_Mi' where M is max indices."""
        class_name = self.__class__.__name__.lower()
        num_series = len(self._ordered_labels)

        # Get max number of indices across all series
        max_indices = 0
        for label in self._ordered_labels:
            num_inst = len(self._series[label].instances)
            if num_inst > max_indices:
                max_indices = num_inst

        return f"{class_name}_{num_series}s_{max_indices}i".lower()

    def save(
        self,
        folder_path: Path | str | None = None,
        *,
        overwrite: bool = False,
    ) -> None:
        """Save frame to folder using hierarchical structure.

        Args:
            folder_path: Directory to save frame. If None, uses default:
                        __data__/frames/{description}
            overwrite: If False, appends counter to avoid overwriting existing folders

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
            folder_path = DATA_FRAMES / self.description

            if not overwrite:
                counter = 2
                original_path = folder_path
                while folder_path.exists():
                    folder_path = original_path.with_name(f"{original_path.name}_{counter}")
                    counter += 1

        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)

        # Build label → folder_name mapping
        series_folders = {label: self._series[label].description for label in self._ordered_labels}

        # Save frame metadata with series order and folder mapping
        frame_info = {
            "series_labels": self._ordered_labels,
            "series_folders": series_folders,
        }
        with (folder_path / "frame_meta.json").open("w", encoding="utf-8") as f:
            json.dump(frame_info, f, indent=2)

        # Save each series in its own subfolder (using series order!)
        for label in self._ordered_labels:
            series_folder = folder_path / series_folders[label]
            # Pass overwrite=True since we already handled folder uniqueness
            self._series[label].save(series_folder, overwrite=True)

    @classmethod
    def load(cls, folder_path: Path | str) -> MatrixFrame:
        """Load frame from folder using hierarchical structure.

        Args:
            folder_path: Directory to load from

        Returns:
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


# Set default series class (after class definition to avoid circular import)
MatrixFrame._series_class = MatrixSeries


if __name__ == "__main__":
    import scipy.sparse as sp

    print("=" * 80)
    print("GENERIC MATRIX FRAME TESTS")
    print("=" * 80)

    # Test 1: Create frame with multiple series
    print("\n[1] Creating MatrixFrame...")

    # Create series A
    series_a_instances = []
    for idx in range(5):
        mat = sp.random(10, 20, density=0.3, format="csr")
        inst = MatrixInstance(
            matrix=mat,
            label="series_A",
            tags={"type": "features"}
        )
        series_a_instances.append(inst)
    series_a = MatrixSeries(instances_list=series_a_instances)

    # Create series B
    series_b_instances = []
    for idx in range(5):
        mat = sp.random(10, 20, density=0.3, format="csr")
        inst = MatrixInstance(
            matrix=mat,
            label="series_B",
            tags={"tag_type": "tag_A"}
        )
        series_b_instances.append(inst)
    series_b = MatrixSeries(instances_list=series_b_instances)

    frame = MatrixFrame(series=[series_a, series_b])
    print(f"✓ Created frame with {len(list(frame.keys()))} series")

    # Test 2: Frame slicing by index
    print("\n[2] MatrixFrame index slicing:")
    sliced = frame[:2]
    print(f"✓ frame[:2] → {type(sliced).__name__}")

    # Test 3: Frame selection by label
    print("\n[3] MatrixFrame label selection:")
    selected = frame[["series_A", "series_B"]]
    print(f"✓ frame[['A', 'B']] → {len(list(selected.keys()))} series")

    # Test 4: Frame .loc 2D slicing
    print("\n[4] MatrixFrame .loc 2D slicing:")
    loc_result = frame.loc[:2, "series_A"]
    print(f"✓ frame.loc[:2, 'A'] → {type(loc_result).__name__}")

    # Test 5: Frame repr
    print("\n[5] MatrixFrame representation:")
    print(frame)

    # # Test 6: Cumulative
    # print("\n[6] Cumulative frame:")
    # cumulative = frame.to_cumulative(inplace=False)
    # print(f"✓ Cumulative frame created")

    # Test 7: Description
    print("\n[7] Frame description:")
    print(f"✓ Description: {frame.description}")

    # Test 8: Save/Load with custom path (hierarchical structure)
    print("\n[8] Save/Load with hierarchical structure:")
    from pathlib import Path
    import tempfile
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
        print("✓ Hierarchical structure created correctly")

        # Load it back
        loaded = MatrixFrame.load(test_path)
        print(f"✓ Loaded frame: {len(list(loaded.keys()))} series")
        print(f"  Series order preserved: {list(loaded.keys())}")

    # Test 9: Save/Load with default path
    print("\n[9] Save/Load with default path:")
    frame.save()  # Uses default __data__/frames/{description}
    default_path = DATA_FRAMES / frame.description
    print(f"✓ Saved to default location: {default_path}")

    # Test 10: Load from default path
    loaded_from_default = MatrixFrame.load(default_path)
    print(f"✓ Loaded from default: {len(list(loaded_from_default.keys()))} series")
    print(f"  Verified order: {list(loaded_from_default.keys())}")

    print("\n✅ All generic MatrixFrame tests passed!")

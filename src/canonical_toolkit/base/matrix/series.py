"""MatrixSeries: Generic collection of MatrixInstance objects indexed by index."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import scipy.sparse as sp
from rich.console import Console
from rich.table import Table

from .matrix import MatrixInstance
from .index_typing import SortableHashable

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterator


class MatrixSeries:
    """Generic collection of matrices sharing the same label."""

    # Class attribute: Subclasses override to specify their instance type
    _instance_class = None  # Will be set to MatrixInstance below (avoid circular ref)

    def __init__(
        self,
        instances_list: list[MatrixInstance] | None = None,
    ) -> None:
        """Initialize MatrixSeries.

        Args:
            instances_list: List of MatrixInstance objects. All must have same label.
                           Cannot be empty or None.

        Raises:
            ValueError: If instances_list is empty/None, has duplicate indices,
                       or instances have different labels
        """
        # Validate: Cannot be empty
        if not instances_list:
            msg = "instances_list cannot be empty or None. MatrixSeries requires at least one MatrixInstance."
            raise ValueError(msg)

        # Validate: All instances must have same label
        labels = [inst.label for inst in instances_list]
        unique_labels = set(labels)
        if len(unique_labels) > 1:
            msg = (
                f"All instances must have the same label. "
                f"Found {len(unique_labels)} different labels: {unique_labels}"
            )
            raise ValueError(msg)

        # Auto-assign sequential indices (0, 1, 2, ...) based on list position
        # The index is a Series-level concern, not an Instance property
        self._instances = {idx: inst for idx, inst in enumerate(instances_list)}

    @property
    def label(self) -> str:
        """Return the label of all instances (guaranteed to be the same)."""
        # Get label from first instance (all have the same label by construction)
        return next(iter(self._instances.values())).label

    @property
    def instances(self) -> dict[Hashable, MatrixInstance]:
        """Read-only access to instances dict."""
        return self._instances.copy()

    @property
    def indices(self) -> list[SortableHashable]:
        """Get sorted list of indices."""
        return sorted(self._instances.keys())

    @property
    def matrices(self) -> list:
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

    def __getitem__(self, key: Hashable | slice) -> MatrixInstance | MatrixSeries:
        """
        Access instances by index or slice.

        Examples:
            series[2]      # Get MatrixInstance at index 2
            series[:3]     # Get MatrixSeries with indices 0,1,2,3
            series[1:4]    # Get MatrixSeries with indices 1,2,3,4
            series[::2]    # Get every other index
        """
        if not isinstance(key, slice):
            # Direct index access
            return self._instances[key]

        # Slice access
        # Get all indices and sort them
        all_indices = sorted(self._instances.keys())

        # Apply slice to get selected indices
        selected_indices = all_indices[key]

        # Build new instances list with only selected indices
        new_instances_list = [self._instances[idx] for idx in selected_indices]

        # Return new instance of same type (works for subclasses too!)
        return self.__class__(instances_list=new_instances_list)

    def __setitem__(self, index: Hashable, instance: MatrixInstance) -> None:
        """Set instance at given index."""
        # Simply store the instance at the given index
        # The index is managed by the Series, not the Instance
        self._instances[index] = instance

    def __iter__(self) -> Iterator[Hashable]:
        """Iterate over sorted indices."""
        return iter(sorted(self._instances.keys()))

    def __repr__(self) -> str:
        """Pretty table representation."""
        label = self.label

        # Create table with index column and data column
        table = Table(
            title=f"MatrixSeries: {label}",
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

            # Simple generic format
            shape_str = f"{rows}×{cols}"
            storage_abbr = "Sparse" if sp.issparse(inst.matrix) else "Dense"

            # Show tags if present
            tags_str = ""
            if inst.tags:
                tags_str = f" {inst.tags}"

            cell_content = f"[{shape_str}] {storage_abbr}{tags_str}"
            table.add_row(str(idx), cell_content)

        # Render table to string with colors enabled
        string_buffer = io.StringIO()
        temp_console = Console(file=string_buffer, force_terminal=True, width=80)
        temp_console.print(table)
        return string_buffer.getvalue().rstrip()

    def replace(self, **changes) -> MatrixSeries:
        """Create a new MatrixSeries with updated fields.

        Args:
            **changes: Fields to update. Can use internal names (_instances)
                      _instances should be a dict[Hashable, MatrixInstance]

        Returns:
            New instance of same type with updated fields
        """
        # Build new instance with defaults from current instance
        new_instances_dict = changes.get("_instances", self._instances.copy())

        # Convert dict to list for __init__
        new_instances_list = list(new_instances_dict.values())

        # Return new instance of same type (works for subclasses too!)
        return self.__class__(instances_list=new_instances_list)

    def map(
        self,
        func: Callable[[MatrixInstance], MatrixInstance],
        *,
        inplace: bool = True,
    ) -> MatrixSeries:
        """Apply function to all instances in the series.

        Args:
            func: Function to apply to each MatrixInstance
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new instance of same type (safer). Default: True.

        Returns:
            Self if inplace=True, new instance of same type if inplace=False
        """
        new_data = {idx: func(inst) for idx, inst in self._instances.items()}

        if inplace:
            self._instances = new_data
            return self
        # replace() uses self.__class__, so this works for subclasses!
        return self.replace(_instances=new_data)

    def __add__(self, other: MatrixSeries) -> MatrixSeries:
        """Add two series element-wise at matching indices."""
        if not isinstance(other, MatrixSeries):
            return NotImplemented

        # Find common indices and compute sum for each
        common = set(self._instances) & set(other._instances)
        new_instances = {idx: self[idx] + other[idx] for idx in common}

        # Return new series with summed instances
        return self.replace(_instances=new_instances)

    # --- I/O Methods ---

    def save(self, folder_path: Path | str | None = None, *, overwrite: bool = False) -> None:
        """Save series to disk.

        Args:
            folder_path: Directory to save series. If None, uses default:
                        __data__/series/{description}
            overwrite: If False, appends counter to avoid overwriting existing folders

        Creates:
            {folder_path}/series_meta.json - metadata with instance order & files
            {folder_path}/{instance.description}.json - instance metadata files
            {folder_path}/{instance.description}.npz - instance data files
        """
        from pathlib import Path
        from .matrix import DATA_SERIES
        import json

        if folder_path is None:
            folder_path = DATA_SERIES / self.description

            if not overwrite:
                counter = 2
                original_path = folder_path
                while folder_path.exists():
                    folder_path = original_path.with_name(f"{original_path.name}_{counter}")
                    counter += 1

        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)

        # Build index → filename mapping, handling duplicate descriptions
        instance_files = {}
        used_filenames = set()

        for idx in self.indices:
            base_filename = self._instances[idx].description
            filename = f"{base_filename}_{idx}"
            used_filenames.add(filename)
            instance_files[str(idx)] = filename  # Store as string for JSON

        # Save metadata with instance order and file mapping
        meta_payload = {
            "label": self.label,
            "instance_indices": self.indices,  # Ordered list
            "num_instances": len(self._instances),
            "instance_files": instance_files,  # index → filename mapping
        }

        with (folder_path / "series_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

        # Save each instance using the unique filename from mapping
        for idx in self.indices:
            filename = instance_files[str(idx)]
            file_path = folder_path / filename
            # Pass overwrite=True since we already handled uniqueness above
            self._instances[idx].save(file_path, overwrite=True)

    @classmethod
    def load(cls, folder_path: Path | str) -> MatrixSeries:
        """Load series from disk.

        Args:
            folder_path: Directory to load from

        Returns:
            Loaded MatrixSeries (or subclass)
        """
        from pathlib import Path
        from .matrix import MatrixInstance
        import json

        folder_path = Path(folder_path)
        if not folder_path.exists():
            msg = f"Folder not found: {folder_path}"
            raise FileNotFoundError(msg)

        # Load metadata
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

        # Load instances using the index→filename mapping
        instances_list = []
        # Use the class's specified instance class (polymorphic!)
        instance_class = cls._instance_class or MatrixInstance
        for idx in instance_indices:
            # Get filename from mapping (indices stored as strings in JSON)
            filename = instance_files[str(idx)]
            file_path = folder_path / filename

            # Load the instance using the appropriate class
            inst = instance_class.load(file_path)
            instances_list.append(inst)

        if not instances_list:
            msg = f"No instances found in {folder_path}"
            raise ValueError(msg)

        # Create series with loaded instances
        series = cls(instances_list=instances_list)

        # Rebuild _instances dict with correct indices from metadata
        new_instances = {idx: inst for idx, inst in zip(instance_indices, instances_list, strict=True)}
        series._instances = new_instances

        return series


# Set default instance class (after class definition to avoid circular import)
MatrixSeries._instance_class = MatrixInstance


if __name__ == "__main__":
    import scipy.sparse as sp
    from pathlib import Path
    import tempfile

    print("Testing Generic MatrixSeries...")

    # Test 1: Create series from instances list
    n_individuals = 10
    n_features = 100
    instances_list = []
    for idx in range(5):
        mat = sp.random(n_individuals, n_features, density=0.3, format="csr")
        inst = MatrixInstance(
            matrix=mat,
            label="experiment_A",
            tags={"type": "features", "index": idx}
        )
        instances_list.append(inst)

    series = MatrixSeries(instances_list=instances_list)
    print(f"✓ Created series with {len(series.indices)} instances")
    print(f"  Label: {series.label}")
    print(f"  Indices: {series.indices}")
    print(f"  Description: {series.description}")

    # Test 2: Series slicing
    sliced = series[:3]
    print(f"✓ Sliced series[:3]: {sliced.indices}")

    # Test 3: Series repr
    print("\n✓ Series representation:")
    print(series)

    # # Test 4: Cumulative sum
    # cumulative = series.to_cumulative(inplace=False)
    # print(f"\n✓ Cumulative series created")

    # Test 5: Save/Load with custom path
    with tempfile.TemporaryDirectory() as tmpdir:
        test_path = Path(tmpdir) / "test_series"
        series.save(test_path)
        loaded = MatrixSeries.load(test_path)
        print(f"\n✓ Save/Load: {loaded.label}, {len(loaded.indices)} instances")
        print(f"  Loaded indices: {loaded.indices}")

    # Test 6: Save/Load with default path
    series.save()  # Uses default __data__/series/{description}
    from .matrix import DATA_SERIES
    default_path = DATA_SERIES / series.description
    print(f"✓ Saved to default location: {default_path}")

    # Test 7: Load from default path
    loaded_from_default = MatrixSeries.load(default_path)
    print(f"✓ Loaded from default: {loaded_from_default.label}, {len(loaded_from_default.indices)} instances")

    print("\n✅ All generic MatrixSeries tests passed!")

"""MatrixSeries: Generic collection of MatrixInstance objects indexed by index."""

from __future__ import annotations

import contextlib
import io
from typing import TYPE_CHECKING, Hashable

import scipy.sparse as sp
from rich.console import Console
from rich.table import Table

from canonical_toolkit.core.matrix.matrix import MatrixInstance

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


class MatrixSeries:
    """Generic collection of matrices sharing the same label."""

    def __init__(
        self,
        instances_list: list[MatrixInstance] | None = None,
    ) -> None:
        """Initialize MatrixSeries.

        Args:
            instances_list: List of MatrixInstance objects. All must have same label.
                           Cannot be empty or None.

        Raises
        ------
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

        # Validate: No duplicate indices
        indices = [inst.index for inst in instances_list]
        duplicates = [idx for idx in indices if indices.count(idx) > 1]
        if duplicates:
            dup_indices = set(duplicates)
            msg = (
                f"Duplicate indices detected: {dup_indices}. "
                f"Each MatrixInstance must have a unique index."
            )
            raise ValueError(msg)

        # Convert list to dict using index as key
        self._instances = {inst.index: inst for inst in instances_list}

    @property
    def space(self) -> VectorSpace | str:
        """Return the space of all instances (guaranteed to be the same)."""
        # Get space from first instance (all have the same space by construction)
        return next(iter(self._instances.values())).space

    @property
    def instances(self) -> dict[int, MatrixInstance]:
        """Read-only access to instances dict."""
        return self._instances.copy()

    @property
    def radii(self) -> list[int]:
        """Get sorted list of radii."""
        return sorted(self._instances.keys())

    @property
    def matrices(self) -> list:
        """Get list of matrices in radius order."""
        return [self._instances[r].matrix for r in self.radii]

    @property
    def labels(self) -> list[str]:
        """Get list of labels in radius order (e.g., ['FRONT r0', 'FRONT r1', ...])."""
        return [self._instances[r].label for r in self.radii]

    @property
    def descriptions(self) -> list[str]:
        """Get list of descriptions in radius order (e.g., ['FRONT r0 Features', ...])."""
        return [self._instances[r].description for r in self.radii]

    def __getitem__(self, key: int | slice) -> MatrixInstance | MatrixSeries:
        """
        Access instances by radius or slice.

        Examples
        --------
            series[2]      # Get MatrixInstance at radius 2
            series[:3]     # Get MatrixSeries with radii 0,1,2,3
            series[1:4]    # Get MatrixSeries with radii 1,2,3,4
            series[::2]    # Get every other radius
        """
        if isinstance(key, int):
            return self._instances[key]
        if isinstance(key, slice):
            # Get all radii and sort them
            all_radii = sorted(self._instances.keys())

            # Apply slice to get selected radii
            selected_radii = all_radii[key]

            # Build new instances list with only selected radii
            new_instances_list = [self._instances[r] for r in selected_radii]

            # Return new MatrixSeries with selected instances
            return MatrixSeries(instances_list=new_instances_list)  # , meta=self._meta.copy())
        msg = f"Invalid index type: {type(key)}"
        raise TypeError(msg)

    def __setitem__(self, radius: int, instance: MatrixInstance) -> None:
        if instance.radius != radius:
            instance = instance.replace(radius=radius)
        self._instances[radius] = instance

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self._instances.keys()))

    def __repr__(self) -> str:
        title = (
            self.space.value
            if isinstance(self.space, VectorSpace)
            else str(self.space)
        )

        # Create table with radius column and data column (like a frame column)
        table = Table(
            title=f"MatrixSeries: {title}",
            title_style="bold bright_cyan",
            title_justify="left",
            show_header=True,
            header_style="bold cyan",
            caption_justify="left",
        )
        table.add_column("Radius", style="cyan", justify="right")
        table.add_column(title, justify="center")

        # if self._meta:
        #     # Format metadata as pretty JSON with spacing
        #     meta_json = json.dumps(self._meta, indent=2)
        #     table.caption = f"\nMeta:\n{meta_json}"

        for r in sorted(self._instances.keys()):
            inst = self._instances[r]
            rows, cols = inst.shape

            # Compact format matching frame cells
            if inst.domain == MatrixDomain.FEATURES:
                shape_str = f"{rows}r×{cols}f"
            elif inst.domain == MatrixDomain.SIMILARITY:
                shape_str = f"{rows}r×{cols}r"
            elif inst.domain == MatrixDomain.EMBEDDING:
                shape_str = f"{rows}r×{cols}d"
            else:
                shape_str = f"{rows}×{cols}"

            type_abbr = {
                MatrixDomain.FEATURES: "Feat",
                MatrixDomain.SIMILARITY: "Sim",
                MatrixDomain.EMBEDDING: "Emb",
            }.get(inst.domain, "?")
            storage_abbr = "Sp" if sp.issparse(inst.matrix) else "Dn"

            cell_content = f"[{shape_str}] {type_abbr} ({storage_abbr})"
            table.add_row(str(r), cell_content)

        # Render table to string with colors enabled
        string_buffer = io.StringIO()
        temp_console = Console(file=string_buffer, force_terminal=True, width=80)
        temp_console.print(table)
        return string_buffer.getvalue().rstrip()

    def replace(self, **changes) -> MatrixSeries:
        """Create a new MatrixSeries with updated fields.

        Args:
            **changes: Fields to update. Can use internal names (_instances)
                      _instances should be a dict[int, MatrixInstance]
        """
        # Map public names to internal if needed
        # if "meta" in changes:
        #     changes["_meta"] = changes.pop("meta")

        # Build new instance with defaults from current instance
        new_instances_dict = changes.get("_instances", self._instances.copy())
        # new_meta = changes.get("_meta", self._meta.copy())

        # Convert dict to list for __init__
        new_instances_list = list(new_instances_dict.values())

        return MatrixSeries(
            instances_list=new_instances_list,
            # meta=new_meta,
        )

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
                    If False, create new MatrixSeries (safer). Default: True.

        Returns
        -------
            Self if inplace=True, new MatrixSeries if inplace=False
        """
        new_data = {r: func(inst) for r, inst in self._instances.items()}

        if inplace:
            self._instances = new_data
            return self
        return self.replace(_instances=new_data)

    def cosine_similarity(self, *, inplace: bool = True) -> MatrixSeries:
        """Compute cosine similarity for all instances.

        Args:
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new MatrixSeries (safer). Default: True.

        Returns
        -------
            Self if inplace=True, new MatrixSeries if inplace=False
        """
        return self.map(lambda inst: inst.cosine_similarity(), inplace=inplace)

    def to_cumulative(self, *, inplace: bool = True) -> MatrixSeries:
        """Convert series to cumulative sum across radii.

        Args:
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new MatrixSeries (safer). Default: True.

        Returns
        -------
            Self if inplace=True, new MatrixSeries if inplace=False
        """
        if not self._instances:
            if inplace:
                return self
            return self.replace(_instances={})

        new_instances = {}
        sorted_radii = sorted(self._instances.keys())
        current_sum = None
        for r in sorted_radii:
            mat_inst = self._instances[r]
            if current_sum is None:
                new_inst = mat_inst
                current_sum = mat_inst.matrix
            else:
                current_sum += mat_inst.matrix
                new_inst = mat_inst.replace(matrix=current_sum)
            new_instances[r] = new_inst

        if inplace:
            self._instances = new_instances
            return self
        return self.replace(_instances=new_instances)

    def normalize_by_radius(self, *, inplace: bool = True) -> MatrixSeries:
        """Normalize matrix values by dividing by (radius + 1).

        Divides each matrix by (radius + 1) to normalize by the radius level.
        The +1 ensures radius 0 is divided by 1 (avoiding division by zero).

        Args:
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new MatrixSeries (safer). Default: True.

        Returns
        -------
            Self if inplace=True, new MatrixSeries if inplace=False

        Example:
            >>> series[0].matrix  # Original values at radius 0
            >>> series[1].matrix  # Original values at radius 1
            >>> normalized = series.normalize_by_radius()
            >>> normalized[0].matrix  # Divided by 1
            >>> normalized[1].matrix  # Divided by 2
        """
        def normalize(inst: MatrixInstance) -> MatrixInstance:
            normalized_matrix = inst.matrix / (inst.radius + 1)
            return inst.replace(matrix=normalized_matrix)

        return self.map(normalize, inplace=inplace)

    def aggregate(self, radius: int | None = None) -> MatrixInstance:
        """Aggregate all feature matrices by summing across radii.

        Collapses the radius dimension by element-wise summation.
        Changes VectorSpace to AGGREGATED since spatial hierarchy is merged.

        ONLY works on MatrixDomain.FEATURES matrices.

        Args:
            radius: Radius value for aggregated matrix (default: max radius)

        Returns
        -------
            Single MatrixInstance with:
            - space = VectorSpace.AGGREGATED
            - domain = MatrixDomain.FEATURES
            - matrix = sum of all matrices (stays sparse!)

        Raises
        ------
            ValueError: If series is empty or contains non-FEATURES matrices

        Example:
            >>> series[0].shape  # (100, 10000) FEATURES at radius 0
            >>> series[1].shape  # (100, 10000) FEATURES at radius 1
            >>> aggregated = series.aggregate()
            >>> aggregated.shape  # (100, 10000) AGGREGATED
            >>> aggregated.space  # VectorSpace.AGGREGATED
        """
        if not self._instances:
            msg = "Cannot aggregate empty MatrixSeries"
            raise ValueError(msg)

        # Validate all are FEATURES
        for r, inst in self._instances.items():
            if inst.domain != MatrixDomain.FEATURES:
                msg = (
                    f"Can only aggregate FEATURES matrices. "
                    f"Found {inst.domain.name} at radius {r}"
                )
                raise ValueError(msg)

        sorted_radii = sorted(self._instances.keys())

        # Sum matrices element-wise
        agg_matrix = self._instances[sorted_radii[0]].matrix.copy()
        for r in sorted_radii[1:]:
            agg_matrix += self._instances[r].matrix

        result_radius = radius if radius is not None else max(sorted_radii)

        return MatrixInstance(
            matrix=agg_matrix,
            space=VectorSpace.AGGREGATED,
            radius=result_radius,
            domain=MatrixDomain.FEATURES,
            # meta={"aggregated": True, "from_radii": sorted_radii},
        )

    def __add__(self, other: MatrixSeries) -> MatrixSeries:
        if not isinstance(other, MatrixSeries):
            return NotImplemented
        result = self.replace(_instances={})
        common = set(self._instances) & set(other._instances)
        for r in common:
            result[r] = self[r] + other[r]
        return result


if __name__ == "__main__":
    import scipy.sparse as sp

    # Test 1: Create series from instances list

    # Use SAME SHAPE for all radii (like real FeatureHasher output)
    n_individuals = 10
    n_features = 100
    instances_list = []
    for r in range(5):
        mat = sp.random(n_individuals, n_features, density=0.3, format="csr")
        inst = MatrixInstance(matrix=mat, space=VectorSpace.FRONT_LIMB, radius=r)
        instances_list.append(inst)

    front_series = MatrixSeries(instances_list=instances_list)

    # Test 2: Series slicing
    sliced_first_3 = front_series[:3]
    sliced_mid = front_series[1:4]
    sliced_even = front_series[::2]

    # Test 3: Series repr

    # # Test 4: Series with metadata
    # print("\n[4] MatrixSeries with metadata:")
    # series_with_meta = front_series.replace(_meta={"source": "test", "version": 1.0})
    # print(series_with_meta)

    # Test 5: Cosine similarity on series
    sim_series = front_series.cosine_similarity(inplace=False)

    # Test 6: Aggregate feature series
    for r in range(3):
        pass

    aggregated = front_series.aggregate()

    # Test 7: Try to aggregate SIMILARITY series (should fail)
    with contextlib.suppress(ValueError):
        sim_series.aggregate()

    # Test 8: Normalize by radius
    normalized = front_series.normalize_by_radius(inplace=False)

    # Test 9: Cumulative series
    cumulative = front_series.to_cumulative(inplace=False)

    # Test 10: Chaining cumulative + normalize
    cumulative_normalized = front_series.to_cumulative(inplace=False).normalize_by_radius(inplace=False)

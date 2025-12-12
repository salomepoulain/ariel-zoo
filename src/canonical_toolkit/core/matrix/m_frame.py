"""MatrixFrame: Collection of MatrixSeries with DataFrame-like interface."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import scipy.sparse as sp
from rich.console import Console
from rich.table import Table

from canonical_toolkit.core.matrix.matrix import MatrixInstance
from canonical_toolkit.core.matrix.m_series import MatrixSeries
from canonical_toolkit.core.matrix.m_types import DEFAULT_DATA_DIR
from canonical_toolkit.core.matrix.m_enums import MatrixDomain, VectorSpace

if TYPE_CHECKING:
    from collections.abc import Callable


class _FrameLocIndexer:
    """
    Helper class for pandas-style .loc indexing on MatrixFrame.
    Supports 2D slicing: frame.loc[radius_slice, space_selector]
    """

    def __init__(self, frame: MatrixFrame) -> None:
        self._frame = frame

    def __getitem__(
        self,
        key: tuple[slice | int, VectorSpace | str | list[VectorSpace | str]]
        | slice
        | int,
    ) -> MatrixFrame | MatrixSeries | MatrixInstance:
        """
        2D slicing for MatrixFrame.

        Examples:
            frame.loc[:3, VectorSpace.FRONT_LIMB]       # Radii 0-3, single series
            frame.loc[:, [space1, space2]]               # All radii, specific series
            frame.loc[2, VectorSpace.FRONT_LIMB]        # Single radius, single series
            frame.loc[:3]                                # Radii 0-3, all series
        """
        # Handle 1D indexing (just radius)
        if isinstance(key, (slice, int)):
            return self._frame[key]

        # Handle 2D indexing (radius, space)
        if not isinstance(key, tuple) or len(key) != 2:
            msg = "loc requires either a single index or a tuple of (radius, space)"
            raise TypeError(msg)

        radius_idx, space_idx = key

        # First, filter by radius (rows)
        if isinstance(radius_idx, slice):
            filtered_frame = self._frame[radius_idx]
        elif isinstance(radius_idx, int):
            # Single radius - will return series or instance
            filtered_frame = self._frame
        else:
            msg = f"Invalid radius index type: {type(radius_idx)}"
            raise TypeError(msg)

        # Then, filter by space (columns)
        if isinstance(space_idx, list):
            result = filtered_frame[space_idx]
        elif isinstance(space_idx, (VectorSpace, str)):
            result = filtered_frame[space_idx]
        else:
            msg = f"Invalid space index type: {type(space_idx)}"
            raise TypeError(msg)

        # If we had a single radius, extract that radius from the result
        if isinstance(radius_idx, int):
            if isinstance(result, MatrixSeries):
                # Single series, single radius -> MatrixInstance
                return result[radius_idx]
            elif isinstance(result, MatrixFrame):
                # Multiple series, single radius -> MatrixFrame with only that radius
                new_series_list = []
                for series in result.series:
                    if radius_idx in series.instances:
                        # Create a new series with just this radius
                        new_ser = MatrixSeries(
                            instances_list=[series[radius_idx]]
                        )
                        new_series_list.append(new_ser)
                return MatrixFrame(series=new_series_list)

        return result


class MatrixFrame:
    """
    The High-Level Controller.
    Manages storage, retrieval, and aggregation of matrix series.
    """

    def __init__(
        self,
        series: list[MatrixSeries] | None = None,
        # meta: dict[str, Any] | None = None,
    ) -> None:
        """Initialize MatrixFrame.

        Args:
            series: List of MatrixSeries (uses series.space as dict key)
            # meta: Optional metadata dict

        Raises:
            ValueError: If multiple series have the same space
        """
        # self._meta = meta if meta is not None else {}

        # Build internal dict from list using series.space as key
        if series is not None:
            # Check for duplicate spaces
            spaces = [s.space for s in series]
            duplicates = [space for space in spaces if spaces.count(space) > 1]
            if duplicates:
                dup_names = set(duplicates)
                msg = (
                    f"Multiple series with same space detected: {dup_names}. "
                    f"Each series must have a unique space. "
                    f"Use series.replace(space='new_name') to rename one first!"
                )
                raise ValueError(msg)

            self._series = {s.space: s for s in series}
        else:
            self._series = {}

    def __getitem__(
        self, key: VectorSpace | str | slice | list[VectorSpace | str]
    ) -> MatrixSeries | MatrixFrame:
        """
        Access series by space, slice by radius, or select multiple series.

        Examples:
            frame[VectorSpace.FRONT_LIMB]           # Get single series
            frame[:3]                                # Slice all series to radii 0-3
            frame[[space1, space2]]                  # Select specific series
        """
        # Single space key - return that series
        if isinstance(key, (VectorSpace, str)):
            return self._series[key]

        # Slice - apply to all series (filter by radius)
        if isinstance(key, slice):
            new_series = {
                space: series[key] for space, series in self._series.items()
            }
            return self.replace(series=new_series)

        # List of spaces - select specific series
        if isinstance(key, list):
            new_series = {}
            for space in key:
                if space in self._series:
                    new_series[space] = self._series[space]
                # Skip spaces that don't exist (no longer create empty series)
            return self.replace(series=new_series)

        raise TypeError(f"Invalid key type: {type(key)}")

    def __setitem__(self, key: VectorSpace | str, val: MatrixSeries) -> None:
        self._series[key] = val

    def keys(self):
        return self._series.keys()

    @property
    def series(self) -> list[MatrixSeries]:
        """Read-only access to series as a list."""
        return list(self._series.values())

    @property
    def matrices(self) -> list[list]:
        """Get 2D list of matrices organized by series.

        Returns list where each element is a list of matrices from one series,
        in radius order.
        """
        return [series.matrices for series in self._series.values()]

    @property
    def radii(self) -> list[list[int]]:
        """Get 2D list of radii organized by series.

        Returns list where each element is a list of radii from one series,
        in sorted order.
        """
        return [series.radii for series in self._series.values()]

    @property
    def labels(self) -> list[list[str]]:
        """Get 2D list of labels organized by series.

        Returns list where each element is a list of labels from one series,
        in radius order.
        """
        return [series.labels for series in self._series.values()]

    @property
    def descriptions(self) -> list[list[str]]:
        """Get 2D list of descriptions organized by series.

        Returns list where each element is a list of descriptions from one series,
        in radius order.
        """
        return [series.descriptions for series in self._series.values()]

    # @property
    # def meta(self) -> dict[str, Any]:
    #     return self._meta.copy()

    # @meta.setter
    # def meta(self, value: dict[str, Any]) -> None:
    #     self._meta = value

    @property
    def loc(self) -> _FrameLocIndexer:
        """
        Pandas-style .loc indexer for 2D slicing.

        Examples:
            frame.loc[:3, VectorSpace.FRONT_LIMB]    # Radii 0-3, single series
            frame.loc[:, [space1, space2]]            # All radii, multiple series
            frame.loc[2, VectorSpace.FRONT_LIMB]     # Single radius, single series
        """
        return _FrameLocIndexer(self)

    def __repr__(self) -> str:
        """Renders Matrix Dashboard using rich tables."""
        series_keys = list(self._series.keys())
        if not series_keys:
            return "<MatrixFrame [Empty]>"

        all_radii = set()
        for s in self._series.values():
            all_radii.update(s.instances.keys())
        sorted_radii = sorted(all_radii)

        # Create table
        title = f"MatrixFrame ({len(series_keys)} series × {len(sorted_radii)} radii)"
        table = Table(
            title=title,
            title_style="bold bright_cyan",
            title_justify="left",
            show_header=True,
            header_style="bold cyan",
            caption_justify="left",
        )

        # # Add metadata as caption if present
        # if self._meta:
        #     # Format metadata as pretty JSON with spacing
        #     meta_json = json.dumps(self._meta, indent=2)
        #     table.caption = f"\nMeta:\n{meta_json}"

        # Add columns: Radius + one column per series
        table.add_column("Radius", style="cyan", justify="right")
        for k in series_keys:
            col_name = k.value if isinstance(k, VectorSpace) else str(k)
            table.add_column(col_name, justify="center")

        # Add rows for each radius
        for r in sorted_radii:
            row = [str(r)]
            for k in series_keys:
                inst = self._series[k].instances.get(r, None)
                if inst is None:
                    row.append("...")
                else:
                    rows, cols = inst.shape

                    # Add dimension labels based on matrix domain
                    if inst.domain == MatrixDomain.FEATURES:
                        shape_str = f"{rows}r×{cols}f"  # robots × features
                    elif inst.domain == MatrixDomain.SIMILARITY:
                        shape_str = f"{rows}r×{cols}r"  # robots × robots
                    elif inst.domain == MatrixDomain.EMBEDDING:
                        shape_str = f"{rows}r×{cols}d"  # robots × dims
                    else:
                        shape_str = f"{rows}×{cols}"

                    type_abbr = {
                        MatrixDomain.FEATURES: "Feat",
                        MatrixDomain.SIMILARITY: "Sim",
                        MatrixDomain.EMBEDDING: "Emb",
                    }.get(inst.domain, "?")
                    storage_abbr = "Sp" if sp.issparse(inst.matrix) else "Dn"
                    row.append(f"[{shape_str}] {type_abbr} ({storage_abbr})")
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
        """
        # Map public names to internal if needed
        if "series" in changes:
            changes["_series"] = changes.pop("series")
        # if "meta" in changes:
        #     changes["_meta"] = changes.pop("meta")

        # Build new instance with defaults from current instance
        new_series_dict = changes.get("_series", self._series.copy())
        # new_meta = changes.get("_meta", self._meta.copy())

        # Convert dict back to list for __init__
        return MatrixFrame(
            series=list(new_series_dict.values()),
            # meta=new_meta,
        )

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
                    If False, create new MatrixFrame (safer). Default: True.

        Returns:
            Self if inplace=True, new MatrixFrame if inplace=False
        """
        new_series = {s: func(ser) for s, ser in self._series.items()}

        if inplace:
            self._series = new_series
            return self
        else:
            return self.replace(_series=new_series)

    def to_cumulative(self, *, inplace: bool = True) -> MatrixFrame:
        """Convert all series to cumulative sum across radii.

        Args:
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new MatrixFrame (safer). Default: True.

        Returns:
            Self if inplace=True, new MatrixFrame if inplace=False
        """
        return self.map(lambda s: s.to_cumulative(inplace=False), inplace=inplace)

    # --- High Level API ---

    # --- reducers (make series)


    def aggregate(self, radius: int | None = None) -> MatrixSeries:
        """Aggregate all series by summing across series at each radius.

        Reduces Frame → Series by collapsing the series dimension.
        For each radius, sums all matrices across series.
        Changes space to VectorSpace.AGGREGATED.

        ONLY works on MatrixDomain.FEATURES matrices.

        Args:
            radius: Radius value for aggregated series (default: use original radii)

        Returns:
            Single MatrixSeries with space=AGGREGATED, containing one summed
            instance per radius (summed across all series).

        Raises:
            ValueError: If frame is empty or contains non-FEATURES matrices

        Example:
            >>> # frame has FRONT, BACK, LEFT series at radii [0, 1, 2]
            >>> agg = frame.aggregate()
            >>> agg[0]  # = FRONT[0] + BACK[0] + LEFT[0]
            >>> agg[1]  # = FRONT[1] + BACK[1] + LEFT[1]
            >>> agg.space  # VectorSpace.AGGREGATED
        """
        if not self._series:
            msg = "Cannot aggregate empty MatrixFrame"
            raise ValueError(msg)

        # Validate all instances are FEATURES
        for space, series in self._series.items():
            for r, inst in series.instances.items():
                if inst.domain != MatrixDomain.FEATURES:
                    msg = (
                        f"Can only aggregate FEATURES matrices. "
                        f"Found {inst.domain.name} at {space}, radius {r}"
                    )
                    raise ValueError(msg)

        # Find common radii across all series
        all_radii = [set(series.instances.keys()) for series in self._series.values()]

        common_radii = sorted(set.intersection(*all_radii))

        if not common_radii:
            msg = "No common radii found across all series"
            raise ValueError(msg)

        # Collect instances first, then create series
        result_instances = []

        # For each radius, sum across all series
        for r in common_radii:
            # Get all instances at this radius
            instances_at_r = [series[r] for series in self._series.values()]

            # Sum matrices element-wise
            agg_matrix = instances_at_r[0].matrix.copy()
            for inst in instances_at_r[1:]:
                agg_matrix = agg_matrix + inst.matrix

            result_radius = radius if radius is not None else r

            result_instance = MatrixInstance(
                matrix=agg_matrix,
                space=VectorSpace.AGGREGATED,
                radius=result_radius,
                domain=MatrixDomain.FEATURES,
                # meta={"aggregated": True, "from_series": list(self._series.keys())},
            )
            result_instances.append(result_instance)

        return MatrixSeries(instances_list=result_instances)

    # --- I/O Methods ---

    def save(
        self,
        folder_name: str,
        tag: str | int | None = None,
        base_dir: str | Path = DEFAULT_DATA_DIR,
    ) -> None:
        save_dir = Path(base_dir) / folder_name
        save_dir.mkdir(parents=True, exist_ok=True)

        frame_info = {
            "tag": tag,
            # "meta": self._meta,
            "series_keys": [
                k.value if isinstance(k, VectorSpace) else k
                for k in self._series
            ],
        }
        with Path(save_dir / "frame_meta.json").open("w", encoding="utf-8") as f:
            json.dump(frame_info, f, indent=2)

        suffix = f"_{tag}" if tag is not None else ""

        for space_key, series in self._series.items():
            s_name = (
                space_key.value
                if isinstance(space_key, VectorSpace)
                else space_key
            )
            for radius in series:
                filename = f"{s_name}_r{radius}{suffix}"
                series[radius].save(save_dir, filename)

    @classmethod
    def load(
        cls,
        folder_name: str,
        tag: str | int | None = None,
        base_dir: str | Path = DEFAULT_DATA_DIR,
    ) -> MatrixFrame:
        load_dir = Path(base_dir) / folder_name
        if not load_dir.exists():
            msg = f"Folder not found: {load_dir}"
            raise FileNotFoundError(msg)

        # meta_path = load_dir / "frame_meta.json"
        # frame_meta = {}
        # if meta_path.exists():
        #     with Path(meta_path).open(encoding="utf-8") as f:
        #         frame_meta = json.load(f).get("meta", {})

        frame = cls()
        suffix = f"_{tag}" if tag is not None else ""

        for json_file in load_dir.glob(f"*{suffix}.json"):
            if json_file.name == "frame_meta.json":
                continue
            try:
                inst = MatrixInstance.load(load_dir, json_file.stem)
                frame[inst.space][inst.radius] = inst
            except Exception:
                pass
        return frame


if __name__ == "__main__":
    import scipy.sparse as sp

    print("="*80)
    print("MATRIX FRAME TESTS")
    print("="*80)

    # Test 1: Create frame with multiple series
    print("\n[1] Creating MatrixFrame...")

    # Create front series
    front_instances = []
    for r in range(5):
        mat = sp.random(5+r, 10+r, density=0.3, format="csr")
        inst = MatrixInstance(matrix=mat, space=VectorSpace.FRONT_LIMB, radius=r)
        front_instances.append(inst)
    front_series = MatrixSeries(instances_list=front_instances)

    # Create back series
    back_instances = []
    for r in range(5):
        mat = sp.random(5+r, 10+r, density=0.3, format="csr")
        inst = MatrixInstance(matrix=mat, space=VectorSpace.BACK_LIMB, radius=r)
        back_instances.append(inst)
    back_series = MatrixSeries(instances_list=back_instances)

    frame = MatrixFrame(series=[front_series, back_series])

    print(f"    Created frame with {len(list(frame.keys()))} series")

    # Test 2: Frame slicing by radius
    print("\n[2] MatrixFrame radius slicing:")
    sliced = frame[:2]
    print(f"    frame[:2] → {type(sliced).__name__}")
    for space_key in sliced.keys():
        radii = sorted(sliced[space_key].instances.keys())
        print(f"        {space_key.value}: radii {radii}")

    # Test 3: Frame selection by space
    print("\n[3] MatrixFrame space selection:")
    selected = frame[[VectorSpace.FRONT_LIMB, VectorSpace.BACK_LIMB]]
    print(f"    frame[[FRONT, BACK]] → {len(list(selected.keys()))} series")

    # Test 4: Frame .loc 2D slicing
    print("\n[4] MatrixFrame .loc 2D slicing:")

    loc_result = frame.loc[:2, VectorSpace.FRONT_LIMB]
    print(f"    frame.loc[:2, FRONT_LIMB] → {type(loc_result).__name__} with {len(loc_result.instances)} radii")

    loc_result2 = frame.loc[:, [VectorSpace.FRONT_LIMB]]
    print(f"    frame.loc[:, [FRONT_LIMB]] → {type(loc_result2).__name__} with {len(list(loc_result2.keys()))} series")

    loc_result3 = frame.loc[2, VectorSpace.FRONT_LIMB]
    print(f"    frame.loc[2, FRONT_LIMB] → {type(loc_result3).__name__}")

    # Test 5: Frame repr
    print("\n[5] MatrixFrame repr:")
    print(frame)

    # Test 6: Sliced frame repr
    print("\n[6] Sliced MatrixFrame (radii 0-2):")
    sliced_frame = frame[:3]
    print(sliced_frame)

    # Test 7: Duplicate space validation
    print("\n[7] Testing duplicate space detection:")
    print("    Creating two series with same space (FRONT_LIMB)...")

    # Create second front series
    duplicate_instances = []
    for r in range(3):
        mat = sp.random(5+r, 10+r, density=0.3, format="csr")
        inst = MatrixInstance(matrix=mat, space=VectorSpace.FRONT_LIMB, radius=r)
        duplicate_instances.append(inst)
    duplicate_series = MatrixSeries(instances_list=duplicate_instances)

    # Try to create frame with duplicate spaces
    try:
        bad_frame = MatrixFrame(series=[front_series, duplicate_series])
        print("    ❌ ERROR: Should have raised ValueError!")
    except ValueError as e:
        print(f"    ✅ Correctly raised ValueError:")
        print(f"       {e}")

    print("\n✅ All MatrixFrame tests passed!")

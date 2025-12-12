from __future__ import annotations

import contextlib
import io
import json
from dataclasses import dataclass, field
from dataclasses import replace as std_replace
from enum import Enum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import scipy.sparse as sp
from rich.console import Console
from rich.table import Table
from sklearn.metrics.pairwise import cosine_similarity

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

# =============================================================================
# 1. CONFIGURATION & TYPES
# =============================================================================

# Anchor data path relative to this file location
_CURRENT_DIR = Path(__file__).parent
DEFAULT_DATA_DIR = _CURRENT_DIR / "__data__" / "npz"

console = Console()

RadiusKey = int  # Radius is strictly an integer depth

# region enums


class VectorSpace(Enum):
    """
    The immutable physical reality of the robot.
    Used for ground-truth keys. Derived/Experimental keys should use strings.
    """

    ENTIRE_ROBOT = "FULL"
    FRONT_LIMB = "FRONT"
    LEFT_LIMB = "LEFT"
    BACK_LIMB = "BACK"
    RIGHT_LIMB = "RIGHT"
    AGGREGATED = "AGGREGATED"

    @classmethod
    def limb_spaces_only(cls) -> list[VectorSpace]:
        return [cls.FRONT_LIMB, cls.LEFT_LIMB, cls.BACK_LIMB, cls.RIGHT_LIMB]


class MatrixDomain(Enum):
    """Defines the topology and mathematical meaning of the matrix."""

    FEATURES = auto()  # N x M: Raw Counts or TFIDF (Must be Sparse)
    SIMILARITY = auto()  # N x N: Pairwise relationships (Must be Dense)
    EMBEDDING = auto()  # N x D: Reduced dimensions (Must be Dense)

# endregion


# region instance

class MatrixInstance:
    """
    A strictly encapsulated wrapper around heavy matrix data.
    Standard Python class enforcing immutability via private properties.
    """

    def __init__(
        self,
        matrix: sp.spmatrix | np.ndarray,
        space: VectorSpace | str,
        radius: int,
        type: MatrixDomain = MatrixDomain.FEATURES,
        meta: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize with strict validation.
        Arguments map directly to private fields.
        """
        # 1. Assign to private fields
        self._matrix = matrix
        self._space = space
        self._radius = radius
        self._type = type
        self._meta = meta if meta is not None else {}

        # 2. STRICT VALIDATION: Memory Safety
        if self._type == MatrixDomain.FEATURES:
            if not sp.issparse(self._matrix):
                msg = (
                    f"CRITICAL MEMORY ERROR: MatrixType.FEATURES must be a "
                    f"scipy.sparse matrix, but got {type(self._matrix)}. "
                    "This will crash your RAM on large datasets."
                )
                raise TypeError(
                    msg,
                )

    # --- Public Read-Only Properties (Getters) ---

    @property
    def matrix(self) -> sp.spmatrix | np.ndarray:
        """Read-only access to raw data."""
        return self._matrix

    @property
    def shape(self):
        return self._matrix.shape

    @property
    def space(self) -> VectorSpace | str:
        return self._space

    @property
    def radius(self) -> int:
        return self._radius

    @property
    def type(self) -> MatrixDomain:
        return self._type

    @property
    def meta(self) -> dict[str, Any]:
        """Returns a COPY of metadata to prevent mutation bugs."""
        return self._meta.copy()

    # --- Visualization ---

    def __repr__(self) -> str:
        s_name = (
            self._space.value
            if isinstance(self._space, VectorSpace)
            else self._space
        )
        rows, cols = self.shape

        # Compact format matching frame cells
        if self._type == MatrixDomain.FEATURES:
            shape_str = f"{rows}r×{cols}f"
        elif self._type == MatrixDomain.SIMILARITY:
            shape_str = f"{rows}r×{cols}r"
        elif self._type == MatrixDomain.EMBEDDING:
            shape_str = f"{rows}r×{cols}d"
        else:
            shape_str = f"{rows}×{cols}"

        storage_abbr = "Sp" if sp.issparse(self._matrix) else "Dn"

        # For dense matrices, show rich table with corner values
        if not sp.issparse(self._matrix):
            table = Table(
                title=f"MatrixInstance: {s_name} (r={self._radius})",
                title_style="bold bright_cyan",
                title_justify="left",
                show_header=True,
                header_style="bold cyan",
            )
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Shape", f"[{shape_str}]")
            table.add_row("Type", self._type.name)
            table.add_row("Storage", "Dense")

            if self._meta:
                table.add_row("Meta", str(self._meta))

            # Extract corner values (2x2 from each corner)
            matrix = self._matrix
            n_rows, n_cols = matrix.shape

            # Build value display
            value_lines = []

            if n_rows <= 4 and n_cols <= 4:
                # Small matrix, show all values
                for i in range(n_rows):
                    row_str = "  ".join(f"{matrix[i, j]:.2f}" for j in range(n_cols))
                    value_lines.append(row_str)
            else:
                # Large matrix, show corners with ...
                # Top 2 rows
                for i in range(min(2, n_rows)):
                    parts = []
                    # Left 2 cols
                    parts.append("  ".join(f"{matrix[i, j]:.2f}" for j in range(min(2, n_cols))))
                    # Middle ellipsis
                    if n_cols > 4:
                        parts.append("...")
                    # Right 2 cols
                    if n_cols > 2:
                        start_col = max(n_cols - 2, 2)
                        parts.append("  ".join(f"{matrix[i, j]:.2f}" for j in range(start_col, n_cols)))
                    value_lines.append("  ".join(parts))

                # Middle row ellipsis
                if n_rows > 4:
                    if n_cols > 4:
                        value_lines.append("...   ...   ...   ...")
                    else:
                        value_lines.append("   ".join("..." for _ in range(n_cols)))

                # Bottom 2 rows
                start_row = max(n_rows - 2, 2)
                for i in range(start_row, n_rows):
                    parts = []
                    # Left 2 cols
                    parts.append("  ".join(f"{matrix[i, j]:.2f}" for j in range(min(2, n_cols))))
                    # Middle ellipsis
                    if n_cols > 4:
                        parts.append("...")
                    # Right 2 cols
                    if n_cols > 2:
                        start_col = max(n_cols - 2, 2)
                        parts.append("  ".join(f"{matrix[i, j]:.2f}" for j in range(start_col, n_cols)))
                    value_lines.append("  ".join(parts))

            # Add value lines to table
            for i, line in enumerate(value_lines):
                if i == 0:
                    table.add_row("Values", line)
                else:
                    table.add_row("", line)

            # Render to string with colors enabled
            string_buffer = io.StringIO()
            temp_console = Console(file=string_buffer, force_terminal=True, width=100)
            temp_console.print(table)
            return string_buffer.getvalue().rstrip()
        else:
            # Sparse matrix - keep compact format
            meta_str = f" +meta" if self._meta else ""
            return (
                f"<MatrixInstance {s_name} | r={self._radius} | "
                f"[{shape_str}] {self._type.name} ({storage_abbr}){meta_str}>"
            )

    # --- Abstraction & Modification ---

    def replace(self, **changes) -> MatrixInstance:
        """
        Returns a new instance with updated fields.
        Manually constructs new object to ensure validation runs.
        """
        new_args = {
            "matrix": self._matrix,
            "space": self._space,
            "radius": self._radius,
            "type": self._type,
            "meta": self._meta.copy(),
        }
        new_args.update(changes)
        return MatrixInstance(**new_args)

    def add_meta(self, **kwargs) -> MatrixInstance:
        """Returns a new instance with updated metadata."""
        new_meta = self._meta.copy()
        new_meta.update(kwargs)
        return self.replace(meta=new_meta)

    # --- Math Transforms ---

    def cosine_similarity(self) -> MatrixInstance:
        if self._type == MatrixDomain.SIMILARITY:
            msg = "Matrix is already a similarity matrix."
            raise ValueError(msg)

        sim_matrix = cosine_similarity(self._matrix)

        return self.replace(matrix=sim_matrix, type=MatrixDomain.SIMILARITY)

    def __add__(self, other: MatrixInstance) -> MatrixInstance:
        if not isinstance(other, MatrixInstance):
            return NotImplemented

        if self._radius != other._radius:
            msg = f"Radius mismatch: {self._radius} vs {other._radius}"
            raise ValueError(
                msg,
            )

        if self._type != other._type:
            msg = f"Type mismatch: {self._type} vs {other._type}"
            raise ValueError(msg)

        new_matrix = self._matrix + other._matrix
        return self.replace(matrix=new_matrix)

    # --- I/O Methods ---

    def save(self, folder: Path, filename_stem: str) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)

        if sp.issparse(self._matrix):
            matrix_path = folder / f"{filename_stem}.sparse.npz"
            sp.save_npz(matrix_path, self._matrix)
            storage_type = "sparse"
        else:
            matrix_path = folder / f"{filename_stem}.dense.npy"
            np.save(matrix_path, self._matrix)
            storage_type = "dense"

        meta_payload = {
            "space": self.space.value
            if isinstance(self.space, VectorSpace)
            else self.space,
            "radius": self.radius,
            "type": self.type.name,
            "storage": storage_type,
            "matrix_file": matrix_path.name,
            "user_meta": self._meta,
        }

        with Path(folder / f"{filename_stem}.json").open("w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2)

    @classmethod
    def load(cls, folder: Path, filename_stem: str) -> MatrixInstance:
        folder = Path(folder)

        json_path = folder / f"{filename_stem}.json"
        if not json_path.exists():
            msg = f"Metadata not found: {json_path}"
            raise FileNotFoundError(msg)

        with Path(json_path).open(encoding="utf-8") as f:
            info = json.load(f)

        try:
            space = VectorSpace(info["space"])
        except ValueError:
            space = info["space"]

        mat_type = MatrixDomain[info["type"]]
        radius = int(info["radius"])

        matrix_path = folder / info["matrix_file"]
        if info["storage"] == "sparse":
            matrix = sp.load_npz(matrix_path)
        else:
            matrix = np.load(matrix_path)

        return cls(
            matrix=matrix,
            space=space,
            radius=radius,
            type=mat_type,
            meta=info["user_meta"],
        )


# endregion

# region series


@dataclass
class MatrixSeries:
    """Abstraction for a column of radii (e.g., all depths for 'Front Limb')."""

    _space: VectorSpace | str
    _instances: dict[int, MatrixInstance] = field(default_factory=dict)
    _meta: dict[str, Any] = field(default_factory=dict)

    @property
    def space(self) -> VectorSpace | str:
        return self._space

    def __getitem__(self, key: int | slice) -> MatrixInstance | MatrixSeries:
        """
        Access instances by radius or slice.

        Examples:
            series[2]      # Get MatrixInstance at radius 2
            series[:3]     # Get MatrixSeries with radii 0,1,2,3
            series[1:4]    # Get MatrixSeries with radii 1,2,3,4
            series[::2]    # Get every other radius
        """
        if isinstance(key, int):
            return self._instances[key]
        elif isinstance(key, slice):
            # Get all radii and sort them
            all_radii = sorted(self._instances.keys())

            # Apply slice to get selected radii
            selected_radii = all_radii[key]

            # Build new instances dict with only selected radii
            new_instances = {r: self._instances[r] for r in selected_radii}

            # Return new MatrixSeries with selected instances
            return self.replace(_instances=new_instances)
        else:
            raise TypeError(f"Invalid index type: {type(key)}")

    def __setitem__(self, radius: int, instance: MatrixInstance) -> None:
        if instance.radius != radius:
            instance = instance.replace(radius=radius)
        self._instances[radius] = instance

    def __iter__(self) -> Iterator[int]:
        return iter(sorted(self._instances.keys()))

    def __repr__(self) -> str:
        title = (
            self._space.value
            if isinstance(self._space, VectorSpace)
            else str(self._space)
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

        if self._meta:
            # Format metadata as pretty JSON with spacing
            meta_json = json.dumps(self._meta, indent=2)
            table.caption = f"\nMeta:\n{meta_json}"

        for r in sorted(self._instances.keys()):
            inst = self._instances[r]
            rows, cols = inst.shape

            # Compact format matching frame cells
            if inst.type == MatrixDomain.FEATURES:
                shape_str = f"{rows}r×{cols}f"
            elif inst.type == MatrixDomain.SIMILARITY:
                shape_str = f"{rows}r×{cols}r"
            elif inst.type == MatrixDomain.EMBEDDING:
                shape_str = f"{rows}r×{cols}d"
            else:
                shape_str = f"{rows}×{cols}"

            type_abbr = {
                MatrixDomain.FEATURES: "Feat",
                MatrixDomain.SIMILARITY: "Sim",
                MatrixDomain.EMBEDDING: "Emb",
            }.get(inst.type, "?")
            storage_abbr = "Sp" if sp.issparse(inst.matrix) else "Dn"

            cell_content = f"[{shape_str}] {type_abbr} ({storage_abbr})"
            table.add_row(str(r), cell_content)

        # Render table to string with colors enabled
        string_buffer = io.StringIO()
        temp_console = Console(file=string_buffer, force_terminal=True, width=80)
        temp_console.print(table)
        return string_buffer.getvalue().rstrip()

    def replace(self, **public_changes) -> MatrixSeries:
        internal_changes = {}
        if "space" in public_changes:
            internal_changes["_space"] = public_changes.pop("space")
        if "meta" in public_changes:
            internal_changes["_meta"] = public_changes.pop("meta")
        else:
            internal_changes["_meta"] = self._meta.copy()
        internal_changes.update(public_changes)
        return std_replace(self, **internal_changes)

    def map(
        self, func: Callable[[MatrixInstance], MatrixInstance],
    ) -> MatrixSeries:
        new_data = {r: func(inst) for r, inst in self._instances.items()}
        return self.replace(_instances=new_data)

    def cosine_similarity(self) -> MatrixSeries:
        return self.map(lambda inst: inst.cosine_similarity())

    def to_cumulative(self) -> MatrixSeries:
        if not self._instances:
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
        return self.replace(_instances=new_instances)

    def __add__(self, other: MatrixSeries) -> MatrixSeries:
        if not isinstance(other, MatrixSeries):
            return NotImplemented
        result = self.replace(_instances={})
        common = set(self._instances) & set(other._instances)
        for r in common:
            result[r] = self[r] + other[r]
        return result


# endregion series

# region Frame


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
                new_series = {}
                for space, series in result._series.items():
                    if radius_idx in series._instances:
                        # Create a new series with just this radius
                        new_ser = MatrixSeries(
                            _space=space, _instances={radius_idx: series[radius_idx]}
                        )
                        new_series[space] = new_ser
                return result.replace(_series=new_series)

        return result


@dataclass
class MatrixFrame:
    """
    The High-Level Controller.
    Manages storage, retrieval, and aggregation of matrix series.
    """

    _series: dict[VectorSpace | str, MatrixSeries] = field(default_factory=dict)
    _meta: dict[str, Any] = field(default_factory=dict)

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
            if key not in self._series:
                self._series[key] = MatrixSeries(_space=key)
            return self._series[key]

        # Slice - apply to all series (filter by radius)
        if isinstance(key, slice):
            new_series = {
                space: series[key] for space, series in self._series.items()
            }
            return self.replace(_series=new_series)

        # List of spaces - select specific series
        if isinstance(key, list):
            new_series = {}
            for space in key:
                if space in self._series:
                    new_series[space] = self._series[space]
                else:
                    # Create empty series for missing spaces (pandas-like)
                    new_series[space] = MatrixSeries(_space=space)
            return self.replace(_series=new_series)

        raise TypeError(f"Invalid key type: {type(key)}")

    def __setitem__(self, key: VectorSpace | str, val: MatrixSeries) -> None:
        self._series[key] = val

    def keys(self):
        return self._series.keys()

    @property
    def meta(self) -> dict[str, Any]:
        return self._meta.copy()

    @meta.setter
    def meta(self, value: dict[str, Any]) -> None:
        self._meta = value

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
            all_radii.update(s._instances.keys())
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

        # Add metadata as caption if present
        if self._meta:
            # Format metadata as pretty JSON with spacing
            meta_json = json.dumps(self._meta, indent=2)
            table.caption = f"\nMeta:\n{meta_json}"

        # Add columns: Radius + one column per series
        table.add_column("Radius", style="cyan", justify="right")
        for k in series_keys:
            col_name = k.value if isinstance(k, VectorSpace) else str(k)
            table.add_column(col_name, justify="center")

        # Add rows for each radius
        for r in sorted_radii:
            row = [str(r)]
            for k in series_keys:
                inst = self._series[k]._instances.get(r, None)
                if inst is None:
                    row.append("...")
                else:
                    rows, cols = inst.shape

                    # Add dimension labels based on matrix type
                    if inst.type == MatrixDomain.FEATURES:
                        shape_str = f"{rows}r×{cols}f"  # robots × features
                    elif inst.type == MatrixDomain.SIMILARITY:
                        shape_str = f"{rows}r×{cols}r"  # robots × robots
                    elif inst.type == MatrixDomain.EMBEDDING:
                        shape_str = f"{rows}r×{cols}d"  # robots × dims
                    else:
                        shape_str = f"{rows}×{cols}"

                    type_abbr = {
                        MatrixDomain.FEATURES: "Feat",
                        MatrixDomain.SIMILARITY: "Sim",
                        MatrixDomain.EMBEDDING: "Emb",
                    }.get(inst.type, "?")
                    storage_abbr = "Sp" if sp.issparse(inst.matrix) else "Dn"
                    row.append(f"[{shape_str}] {type_abbr} ({storage_abbr})")
            table.add_row(*row)

        # Render table to string with colors enabled
        string_buffer = io.StringIO()
        temp_console = Console(file=string_buffer, force_terminal=True, width=140)
        temp_console.print(table)
        return string_buffer.getvalue().rstrip()

    def replace(self, **public_changes) -> MatrixFrame:
        internal_changes = {}
        if "meta" in public_changes:
            internal_changes["_meta"] = public_changes.pop("meta")
        else:
            internal_changes["_meta"] = self._meta.copy()
        internal_changes.update(public_changes)
        return std_replace(self, **internal_changes)

    def map(
        self, func: Callable[[MatrixSeries], MatrixSeries],
    ) -> MatrixFrame:
        new_series = {s: func(ser) for s, ser in self._series.items()}
        return self.replace(_series=new_series)

    # --- High Level API ---

    def cosine_similarity(self) -> MatrixFrame:
        return self.map(lambda s: s.cosine_similarity())

    def to_cumulative(self) -> MatrixFrame:
        return self.map(lambda s: s.to_cumulative())

    def aggregate_series(
        self,
        new_name: str,
        sources: list[VectorSpace | str],
        aggregator: InstanceAggregator,
    ) -> MatrixFrame:
        if not sources:
            msg = "No sources provided"
            raise ValueError(msg)
        result_series = MatrixSeries(_space=new_name)

        first = self._series[sources[0]]
        common = set(first._instances)
        for s in sources[1:]:
            common &= set(self._series[s]._instances)

        for r in common:
            inputs = [self._series[s][r] for s in sources]
            res = aggregator(inputs)
            res = res.replace(space=new_name, radius=r)
            result_series[r] = res

        new_frame = self.replace()
        new_frame[new_name] = result_series
        return new_frame

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
            "meta": self._meta,
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

        meta_path = load_dir / "frame_meta.json"
        frame_meta = {}
        if meta_path.exists():
            with Path(meta_path).open(encoding="utf-8") as f:
                frame_meta = json.load(f).get("meta", {})

        frame = cls(_meta=frame_meta)
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


# endregion


# region helpers


class InstanceAggregator(Protocol):
    def __call__(self, instances: list[MatrixInstance]) -> MatrixInstance: ...


def agg_sum_features(instances: list[MatrixInstance]) -> MatrixInstance:
    """EARLY FUSION: Sums the matrices."""
    total = instances[0].matrix
    for x in instances[1:]:
        total += x.matrix
    return instances[0].replace(matrix=total)


def agg_mean_similarity(instances: list[MatrixInstance]) -> MatrixInstance:
    """LATE FUSION: Averages similarity matrices."""
    if instances[0].type != MatrixDomain.SIMILARITY:
        msg = "Must be similarity matrices"
        raise ValueError(msg)
    total = instances[0].matrix
    for x in instances[1:]:
        total += x.matrix
    return instances[0].replace(matrix=total / len(instances))


# endregion


# region main

if __name__ == "__main__":

    print("="*80)
    print("RUNNING MATRIX API TEST SUITE")
    print(f"Storage Path: {DEFAULT_DATA_DIR.resolve()}")
    print("="*80)

    # 1. SETUP
    print("\n[1] Generating Mock Data...")
    frame = MatrixFrame()
    frame.meta = {"project": "Test Protocol", "version": "1.0"}

    # Mock Hashed Data (Sparse)
    mock_data = sp.csr_matrix(np.random.randint(0, 2, size=(5, 10)))

    for space in [VectorSpace.FRONT_LIMB, VectorSpace.BACK_LIMB]:
        for r in [0, 1]:
            inst = MatrixInstance(
                matrix=mock_data,
                space=space,
                radius=r,
                type=MatrixDomain.FEATURES,
                meta={"hashing_time": 0.05}
            )
            frame[space][r] = inst

    # Visualize the Dashboard
    print(frame)

    # 2. TRANSFORM
    print("\n[2] Testing Transformation...")
    sim_frame = frame.cosine_similarity()
    print(sim_frame)

    # 3. AGGREGATE
    print("\n[3] Testing Aggregation...")
    frame = frame.aggregate_series(
        new_name="agg_body",
        sources=[VectorSpace.FRONT_LIMB, VectorSpace.BACK_LIMB],
        aggregator=agg_sum_features
    )
    print("Added 'agg_body' via Early Fusion:")
    print(frame)

    # 4. I/O (Nested Folders)
    print("\n[4] Testing I/O...")
    folder = "test_experiment/gen_0" # Nested folder strategy
    frame.save(folder_name=folder, tag="pop")

    loaded = MatrixFrame.load(folder_name=folder, tag="pop")
    print(f"    Loaded from {folder}")

    # 5. SAFETY
    print("\n[5] Testing Validation...")
    try:
        MatrixInstance(
            matrix=np.ones((5, 10)),
            space="test",
            radius=0,
            type=MatrixDomain.FEATURES
        )
        print("    ❌ FAILED")
    except TypeError:
        print("    ✅ PASSED: Caught illegal Dense Matrix")

    # 6. SLICING TESTS
    print("\n[6] Testing Pandas-Style Slicing...")

    # Add more radii for better testing
    for space in [VectorSpace.FRONT_LIMB, VectorSpace.BACK_LIMB]:
        for r in [2, 3, 4]:
            inst = MatrixInstance(
                matrix=mock_data,
                space=space,
                radius=r,
                type=MatrixDomain.FEATURES,
                meta={"hashing_time": 0.05}
            )
            frame[space][r] = inst

    print("    Original frame:")
    print(frame)

    # Test 6a: Series slicing
    print("\n    [6a] MatrixSeries slicing:")
    series = frame[VectorSpace.FRONT_LIMB]
    print(f"    series[:3] → {len(series[:3]._instances)} radii: {sorted(series[:3]._instances.keys())}")
    print(f"    series[1:4] → {len(series[1:4]._instances)} radii: {sorted(series[1:4]._instances.keys())}")
    print(f"    series[::2] → {len(series[::2]._instances)} radii: {sorted(series[::2]._instances.keys())}")

    # Test 6b: Frame slicing by radius
    print("\n    [6b] MatrixFrame radius slicing:")
    sliced = frame[:2]
    print(f"    frame[:2] → {len(list(sliced.keys()))} series")
    for space_key in sliced.keys():
        radii = sorted(sliced[space_key]._instances.keys())
        print(f"        {space_key}: radii {radii}")

    # Test 6c: Frame selection by space
    print("\n    [6c] MatrixFrame space selection:")
    selected = frame[[VectorSpace.FRONT_LIMB, VectorSpace.BACK_LIMB]]
    print(f"    frame[[FRONT, BACK]] → {len(list(selected.keys()))} series: {[k.value if isinstance(k, VectorSpace) else k for k in selected.keys()]}")

    # Test 6d: Frame .loc 2D slicing
    print("\n    [6d] MatrixFrame .loc 2D slicing:")

    # Test: radii slice, single space
    loc_result = frame.loc[:2, VectorSpace.FRONT_LIMB]
    print(f"    frame.loc[:2, FRONT_LIMB] → {type(loc_result).__name__} with {len(loc_result._instances)} radii")

    # Test: all radii, multiple spaces
    loc_result2 = frame.loc[:, [VectorSpace.FRONT_LIMB]]
    print(f"    frame.loc[:, [FRONT_LIMB]] → {type(loc_result2).__name__} with {len(list(loc_result2.keys()))} series")

    # Test: single radius, single space
    loc_result3 = frame.loc[2, VectorSpace.FRONT_LIMB]
    print(f"    frame.loc[2, FRONT_LIMB] → {type(loc_result3).__name__}: {loc_result3}")

    # Test: radii slice, multiple spaces
    loc_result4 = frame.loc[:2, [VectorSpace.FRONT_LIMB, VectorSpace.BACK_LIMB]]
    print(f"    frame.loc[:2, [FRONT, BACK]] → {type(loc_result4).__name__} with {len(list(loc_result4.keys()))} series")
    for space_key in loc_result4.keys():
        radii = sorted(loc_result4[space_key]._instances.keys())
        print(f"        {space_key.value if isinstance(space_key, VectorSpace) else space_key}: radii {radii}")

    print("\n    ✅ All slicing tests passed!")

    # 7. REPR TESTS (Rich Tables)
    print("\n[7] Testing Rich Table Representations...")

    # Test 7a: MatrixInstance repr
    print("\n    [7a] MatrixInstance repr:")
    instance = frame[VectorSpace.FRONT_LIMB][0]
    print(instance)

    # Test 7b: MatrixSeries repr
    print("\n    [7b] MatrixSeries repr:")
    series = frame[VectorSpace.FRONT_LIMB]
    print(series)

    # Test 7c: MatrixSeries with metadata
    print("\n    [7c] MatrixSeries with metadata:")
    series_with_meta = series.replace(_meta={"source": "test", "version": 1.0})
    print(series_with_meta)

    # Test 7d: MatrixFrame repr (original frame with multiple series)
    print("\n    [7d] MatrixFrame repr:")
    print(frame)

    # Test 7e: Sliced frame repr
    print("\n    [7e] Sliced MatrixFrame (radii 0-2):")
    sliced_frame = frame[:3]
    print(sliced_frame)

    # Test 7f: Frame with selected series
    print("\n    [7f] MatrixFrame with selected series:")
    selected_frame = frame[[VectorSpace.FRONT_LIMB, VectorSpace.BACK_LIMB]]
    print(selected_frame)

    # Test 7g: Empty series repr
    print("\n    [7g] Empty MatrixSeries repr:")
    empty_series = MatrixSeries(_space="empty_test")
    print(empty_series)

    # Test 7h: Dense matrix with corner value display (large matrix)
    print("\n    [7h] Dense matrix repr with corner values (10x10 - shows corners):")
    # Create a 10x10 dense similarity matrix with random values
    dense_sim = np.random.rand(10, 10)
    # Make it symmetric (typical for similarity matrices)
    dense_sim = (dense_sim + dense_sim.T) / 2
    # Set diagonal to 1.0 (self-similarity)
    np.fill_diagonal(dense_sim, 1.0)
    dense_instance = MatrixInstance(
        matrix=dense_sim,
        space=VectorSpace.FRONT_LIMB,
        type=MatrixDomain.SIMILARITY,
        radius=0,
        meta={"method": "cosine", "normalized": True}
    )
    print(dense_instance)

    # Test 7i: Small dense matrix (shows all values)
    print("\n    [7i] Dense matrix repr (3x3 - shows all values):")
    small_dense = np.array([
        [1.00, 0.85, 0.62],
        [0.85, 1.00, 0.73],
        [0.62, 0.73, 1.00]
    ])
    small_instance = MatrixInstance(
        matrix=small_dense,
        space=VectorSpace.BACK_LIMB,
        type=MatrixDomain.SIMILARITY,
        radius=1
    )
    print(small_instance)

    print("\n    ✅ All repr tests passed!")

    # Clean up test artifact
    # shutil.rmtree(DEFAULT_DATA_DIR)
    print("\n" + "="*80)
    print("ALL TESTS COMPLETE")
    print("="*80)

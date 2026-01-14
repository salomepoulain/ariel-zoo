"""MatrixFrame: Collection of MatrixSeries with DataFrame-like interface."""

# TODO FIX TYPING HERE

from __future__ import annotations

from typing import TYPE_CHECKING, overload

from .series import MatrixSeries

if TYPE_CHECKING:
    from collections.abc import Hashable

    from .matrix import MatrixInstance


class FrameLocIndexer:
    """
    Helper class for pandas-style .loc indexing on MatrixFrame.
    Supports 2D slicing: frame.loc[index_slice, label_selector].
    """

    def __init__(self, frame: MatrixFrame) -> None:
        self._frame = frame

    @overload
    def __getitem__(self, key: slice) -> MatrixFrame: ...

    @overload
    def __getitem__(
        self,
        key: tuple[slice, list[Hashable]],
    ) -> MatrixFrame: ...

    @overload
    def __getitem__(self, key: tuple[slice, Hashable]) -> MatrixSeries: ...

    @overload
    def __getitem__(
        self,
        key: tuple[int, list[Hashable]],
    ) -> MatrixFrame: ...

    @overload
    def __getitem__(self, key: tuple[int, Hashable]) -> MatrixInstance: ...

    def __getitem__(
        self,
        key: tuple[slice | int, Hashable | list[Hashable]] | slice,
    ) -> MatrixFrame | MatrixSeries | MatrixInstance:
        """
        2D slicing for MatrixFrame.

        Examples
        --------
            frame.loc[:3, "series_A"]       # Indices 0-3, single series
            frame.loc[:, ["A", "B"]]        # All indices, specific series
            frame.loc[2, "series_A"]        # Single index, single series
            frame.loc[:3]                   # Indices 0-3, all series
        """
        # Handle 1D indexing (just index slice or int)
        if not isinstance(key, tuple):
            return self._frame[key]

        # Handle 2D indexing (index, label)
        if len(key) != 2:
            msg = "loc requires either a single index or a tuple of (index, label)"
            raise TypeError(msg)

        index_key, label_key = key

        # Apply index slicing first (rows)
        if isinstance(index_key, slice):
            filtered_frame = self._frame[index_key]
            return self._apply_label_selection(filtered_frame, label_key)

        # Single integer index - extract specific index from result
        return self._extract_single_index(index_key, label_key)

    def _apply_label_selection(
        self,
        frame: MatrixFrame,
        label_key: Hashable | list[Hashable],
    ) -> MatrixFrame | MatrixSeries:
        """Apply label selection to frame (returns frame or single series)."""
        if isinstance(label_key, list):
            return frame[label_key]
        # Single label - returns a series
        return frame[label_key]

    def _extract_single_index(
        self,
        index: int,
        label_key: Hashable | list[Hashable],
    ) -> MatrixInstance | MatrixFrame:
        """Extract single index from frame after label selection."""
        # First apply label selection
        if isinstance(label_key, list):
            # Multiple labels -> frame with just this index
            selected_frame = self._frame[label_key]
            return self._frame_with_single_index(selected_frame, index)

        # Single label -> single instance
        series = self._frame[label_key]
        return series[index]

    def _frame_with_single_index(
        self,
        frame: MatrixFrame,
        index: int,
    ) -> MatrixFrame:
        """Create frame containing only the specified index across all series."""
        new_series_list = []
        for series in frame.series:
            if index in series.instances:
                new_ser = MatrixSeries(instances_list=[series[index]])
                new_series_list.append(new_ser)
        return MatrixFrame(series=new_series_list)

"""Type protocols and generics for the matrix hierarchy.

This module defines the type structure for the three-level matrix system:
- Instance level: Individual matrix wrappers with metadata.
- Series level: Ordered collections of instances sharing a label (e.g., Radii).
- Frame level: Collections of series organized by distinct labels (e.g., Body Parts).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, Self, overload, runtime_checkable
from collections.abc import Iterable, Iterator

if TYPE_CHECKING:
    import numpy as np
    import scipy.sparse as sp


# --- LEVEL 1: Instance Protocol ---

@runtime_checkable
class InstanceProtocol(Protocol):
    """Protocol for an individual matrix wrapper (the 'Leaf' node)."""

    @property
    def matrix(self) -> sp.spmatrix | np.ndarray:
        """The underlying numeric matrix data (sparse or dense)."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Dimensions of the underlying matrix."""
        ...

    @property
    def label(self) -> str:
        """The identifying label for this specific instance."""
        ...

    @property
    def tags(self) -> dict[str, Any]:
        """Flexible metadata dictionary associated with this matrix."""
        ...

    def replace(self, **changes: Any) -> Self:
        """Create a new instance of the same type with specific fields updated."""
        ...

    def __add__(self, other: InstanceProtocol) -> Self:
        """Element-wise addition of two matrix instances."""
        ...


# TypeVar for instance types (any class implementing InstanceProtocol)
I = TypeVar("I", bound=InstanceProtocol)


# --- LEVEL 2: Series Protocol ---

@runtime_checkable
class SeriesProtocol(Protocol[I]):
    """Protocol for an ordered collection of instances (the 'Column')."""

    @property
    def label(self) -> str:
        """The common label shared by all instances in this series (e.g., 'FRONT')."""
        ...

    @property
    def instances(self) -> list[I]:
        """Ordered list of actual Instance objects."""
        ...

    @property
    def indices(self) -> list[int]:
        """Sorted list of integer keys (e.g., Radii) representing the sequence."""
        ...

    def items(self) -> Iterable[tuple[int, I]]:
        """Dictionary-like iterator yielding (index, InstanceObject) pairs."""
        ...

    def __iter__(self) -> Iterator[I]:
        """Direct iterator yielding InstanceObjects in order."""
        ...

    @overload
    def __getitem__(self, key: slice) -> Self: ...

    @overload
    def __getitem__(self, key: int) -> I: ...

    def __getitem__(self, key: int | slice) -> I | Self:
        """Access a specific instance by index or a subset via slicing."""
        ...

    def __setitem__(self, key: int, instance: I) -> None:
        """Insert or update an instance at a specific index."""
        ...

    def replace(self, **changes: Any) -> Self:
        """Create a new series of the same type with specific fields updated."""
        ...


# TypeVar for series types
S = TypeVar("S", bound=SeriesProtocol[Any])


# --- LEVEL 3: Frame Protocol ---

@runtime_checkable
class FrameProtocol(Protocol[S]):
    """Protocol for a collection of matrix series (the 'Grid')."""

    @property
    def series(self) -> list[S]:
        """Ordered list of Series objects contained within the frame."""
        ...

    @property
    def labels(self) -> list[str]:
        """Ordered list of strings representing the series labels (e.g., ['FRONT', 'BACK'])."""
        ...

    def items(self) -> Iterable[tuple[str, S]]:
        """Iterator yielding (label, SeriesObject) pairs."""
        ...

    def __iter__(self) -> Iterator[S]:
        """Direct iterator yielding SeriesObjects in the frame's order."""
        ...

    @overload
    def __getitem__(self, key: tuple[slice, Any]) -> Any: ...

    @overload
    def __getitem__(self, key: list[str]) -> Self: ...

    @overload
    def __getitem__(self, key: slice) -> Self: ... # type: ignore

    @overload
    def __getitem__(self, key: str) -> S: ...

    def __getitem__(self, key: Any) -> Any:
        """
        Retrieve data from the frame.

        - 2D Tuple: Returns a sub-grid or specific values.
        - List/Slice: Returns a sub-frame (Self).
        - str: Returns a specific series (S).
        """
        ...

    def __setitem__(self, key: str, series: S) -> None:
        """Insert or update a series associated with a specific label."""
        ...

    def replace(self, **changes: Any) -> Self:
        """Create a new frame of the same type with specific fields updated."""
        ...

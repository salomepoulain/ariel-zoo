"""Type protocols and generics for the matrix hierarchy.

This module defines the type structure for the three-level matrix system:
- Instance level: Individual matrices (MatrixInstance, SimilarityMatrix, etc.)
- Series level: Collections of instances (MatrixSeries, SimilaritySeries, etc.)
- Frame level: Collections of series (MatrixFrame, SimilarityFrame, etc.)

The type system uses:
- Protocols: Structural typing contracts (duck typing for type checkers)
- Generics: Type parameters that flow through the hierarchy
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, TypeVar, Self

if TYPE_CHECKING:
    from collections.abc import Hashable
    import numpy as np
    import scipy.sparse as sp


# Utility Protocol

class SortableHashable(Protocol):
    """Protocol for objects that are both Hashable and Sortable (Comparable)."""

    def __hash__(self) -> int: ...

    def __lt__(self, other: Any) -> bool: ...

    def __eq__(self, other: Any) -> bool: ...


# LEVEL 1: Instance Protocol

class InstanceProtocol(Protocol):
    """
    Protocol defining what any matrix instance must provide.

    Any class that has these properties/methods automatically satisfies
    this protocol (structural typing - no explicit inheritance needed).
    """

    @property
    def matrix(self) -> sp.spmatrix | np.ndarray:
        """The underlying matrix data (sparse or dense)."""
        ...

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the matrix (rows, cols)."""
        ...

    @property
    def label(self) -> str:
        """Generic label for this instance."""
        ...

    @property
    def tags(self) -> dict[str, Any]:
        """Metadata tags for this instance."""
        ...

    def replace(self, **changes) -> Self:
        """Create a new instance with updated fields."""
        ...

    def __add__(self, other: InstanceProtocol) -> Self:
        """Add two instances element-wise."""
        ...


# TypeVar for instance types (any class implementing InstanceProtocol)
I = TypeVar("I", bound=InstanceProtocol)


# LEVEL 2: Series Protocol

class SeriesProtocol(Protocol[I]):
    """
    Protocol defining what any series must provide.

    Generic over I (the instance type it contains).
    A series is a collection of instances with the same label.
    """

    @property
    def label(self) -> str:
        """The common label shared by all instances in this series."""
        ...

    @property
    def instances(self) -> dict[Hashable, I]:
        """All instances in this series, indexed by their position."""
        ...

    @property
    def indices(self) -> list[SortableHashable]:
        """Sorted list of indices in this series."""
        ...

    @property
    def matrices(self) -> list[sp.spmatrix | np.ndarray]:
        """List of matrices in index order."""
        ...

    def __getitem__(self, key: Hashable) -> I:
        """Get instance at given index."""
        ...

    def __setitem__(self, key: Hashable, instance: I) -> None:
        """Set instance at given index."""
        ...

    def replace(self, **changes) -> Self:
        """Create a new series with updated fields."""
        ...


# TypeVar for series types (any class implementing SeriesProtocol)
# Note: We don't bind to SeriesProtocol[Any] because it causes type checker issues.
# The structural typing via Protocol provides the contract without needing a bound.
S = TypeVar("S")


# LEVEL 3: Frame Protocol

class FrameProtocol(Protocol[S]):
    """
    Protocol defining what any frame must provide.

    Generic over S (the series type it contains).
    A frame is a collection of series with different labels.
    """

    @property
    def series(self) -> list[S]:
        """All series in this frame, in order."""
        ...

    def keys(self) -> list[Hashable]:
        """Get all series labels in order."""
        ...

    def values(self) -> list[S]:
        """Get all series in order."""
        ...

    def __getitem__(self, key: Hashable) -> S:
        """Get series by label."""
        ...

    def __setitem__(self, key: Hashable, series: S) -> None:
        """Set series at given label."""
        ...

    def replace(self, **changes) -> Self:
        """Create a new frame with updated fields."""
        ...

from __future__ import annotations

from dataclasses import field, dataclass
import numpy as np
from typing import runtime_checkable, Protocol
from sklearn.base import clone


type GridKey = int | slice | tuple[int | slice | list[int], int | slice | list[int]]


__all__ = [
    "TransformerGrid"
]

@runtime_checkable
class FitTransformer(Protocol):
    """Protocol for sklearn-style fit/transform objects (UMAP, PCA, etc.)."""

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> FitTransformer:
        """Learn the parameters from the data."""
        ...

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the learned transformation to new data."""
        ...


@dataclass
class TransformerGrid:
    """Grid of transformers with slice-based assignment."""

    shape: tuple[int, int]
    _transformers: list[list[FitTransformer | None]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._transformers = [
            [None for _ in range(self.shape[1])]
            for _ in range(self.shape[0])
        ]

    @property
    def is_filled(self) -> bool:
        return all(t is not None for row in self._transformers for t in row)

    def _normalize_key(self, key: GridKey) -> tuple[list[int], list[int]]:
        if not isinstance(key, tuple):
            key = (key, slice(None))

        row_key, col_key = key

        rows: list[int] = (
            list(range(self.shape[0])[row_key]) if isinstance(row_key, slice)
            else [row_key] if isinstance(row_key, int) else row_key
        )
        cols: list[int] = (
            list(range(self.shape[1])[col_key]) if isinstance(col_key, slice)
            else [col_key] if isinstance(col_key, int) else col_key
        )
        return rows, cols

    def __getitem__(self, key: GridKey) -> FitTransformer | None | list[list[FitTransformer | None]]:
        rows, cols = self._normalize_key(key)
        if len(rows) == 1 and len(cols) == 1:
            return self._transformers[rows[0]][cols[0]]
        return [[self._transformers[r][c] for c in cols] for r in rows]

    def __setitem__(self, key: GridKey, value: FitTransformer) -> None:
        rows, cols = self._normalize_key(key)
        for r in rows:
            for c in cols:
                self._transformers[r][c] = clone(value)

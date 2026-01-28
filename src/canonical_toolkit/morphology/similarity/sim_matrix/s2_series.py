"""SimilaritySeries: Similarity-specific extension of MatrixSeries."""

from __future__ import annotations

from typing import Any, Self

import numpy as np

from ....base.matrix import MatrixSeries
from ..options import Space
from .s1_matrix import SimilarityMatrix, SimilarityMatrixTags

__all__ = ["SimilaritySeries"]


class SimilaritySeries(MatrixSeries[SimilarityMatrix]):
    """
    Similarity-specific collection of SimilarityMatrix matrices.

    Extends MatrixSeries[SimilarityMatrix] with domain-specific properties and methods for
    morphological similarity analysis.

    Type-safe: Inherits Generic[SimilarityMatrix], so series[2] returns SimilarityMatrix (not Matrixmatrix).

    The series index represents the radius. Use the `radii` property to get all indices as radii.
    """

    _matrix_class = SimilarityMatrix

    @property
    def indices(self) -> list[int]:
        """
        Return the semantic indices (radii) of the matrices.
        Overrides base implementation to use the intrinsic 'radius' property
        of the contained SimilarityMatrix objects. This preserves indices
        even after slicing.
        """
        return [m.radius for m in self._matrices]

    @property
    def space(self) -> str:
        """Return the morphological space (guaranteed to be the same for all)."""
        return self.matrices[0].space

    @property
    def radii(self) -> list[int]:
        """Get sorted list of radii (similarity-specific name for indices)."""
        return self.indices

    # --- Advanced Transformations ---

    def to_cumulative(self, *, inplace: bool = True) -> Self:
        """Convert series to cumulative sum across radii."""
        matrices = self.matrices

        accumulated = []
        running_sum = matrices[0]

        accumulated.append(running_sum)
        for current in matrices[1:]:
            running_sum += current
            running_sum = self._matrix_class(
                matrix=running_sum.matrix,
                label=self.label,
                tags=running_sum.tags | {"radius": current.radius},
            )
            accumulated.append(running_sum)

        if inplace:
            self._matrices = accumulated
            return self

        return self.replace(matrices=accumulated)

    def aggregate(self) -> SimilarityMatrix:
        """Aggregate all feature matrices by summing across radii."""
        agg_matrix: Any = self.matrices[0].matrix.copy()

        for inst in self.matrices[1:]:
            agg_matrix += inst.matrix

        tags = SimilarityMatrixTags(
            domain=f"{self.matrices[0].domain}",
            radius=max(self.radii),
            is_gap=False,
        )
        return SimilarityMatrix(
            matrix=agg_matrix,
            label=f"AGGREGATED_{self.space}",
            tags=tags,
        )

    # --- Operators ---

    def __or__(self, other: SimilaritySeries) -> Self:
        """Vertically stack series (combine populations)."""
        combined_list = [
            m1 | m2
            for m1, m2 in zip(self._matrices, other.matrices, strict=False)
        ]
        return self.replace(matrices=combined_list)

# --- Test Sync ---
if __name__ == "__main__":
    # Ensure tests use the 'matrices' keyword argument
    mats = [SimilarityMatrix(np.eye(3), Space.FRONT, r) for r in range(3)]
    series = SimilaritySeries(matrices=mats)
    print(f"Original indices: {series.indices}")

    # Test Slicing
    sliced = series[1:]
    print(f"Sliced indices (should be [1, 2]): {sliced.indices}")

    # Test aggregation
    agg = series.aggregate()

"""SimilarityFrame: Similarity-specific extension of MatrixFrame."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING
from ....base.matrix import MatrixFrame
from ..options import Space
from .matrix import SimilarityMatrix
from .series import SimilaritySeries
from pathlib import Path
import json
import numpy as np
import scipy.sparse as sp


__all__ = ["SimilarityFrame"]

class SimilarityFrame(MatrixFrame[SimilaritySeries]):
    """
    Similarity-specific collection of SimilaritySeries.

    Extends MatrixFrame[SimilaritySeries] with domain-specific methods for morphological
    similarity analysis.

    Type-safe: Inherits Generic[SimilaritySeries], so frame["FRONT"] returns SimilaritySeries
    (which returns SimilarityMatrix when indexed).
    """

    _series_class = SimilaritySeries

    def to_cumulative(self, *, inplace: bool = True) -> SimilarityFrame:
        """Convert all series to cumulative sum across radii."""
        return self.map(
            lambda s: s.to_cumulative(inplace=False), inplace=inplace
        )

    def aggregate(self) -> SimilaritySeries:
        """
        Aggregate all series by summing across series at each radius.
        Reduces Frame (Grid) -> Series (Column).
        """
        if not self._series:
            raise ValueError("Cannot aggregate empty SimilarityFrame")

        # 1. Determine common radii across all series
        all_radii_sets = [set(s.radii) for s in self.series]
        common_radii = sorted(set.intersection(*all_radii_sets))

        if not common_radii:
            raise ValueError("No common radii found across all series")

        # 2. Extract metadata from first series/instance
        sample_series = self.series[0]
        previous_domain = sample_series[common_radii[0]].domain

        result_instances = []

        # 3. Perform cross-series summation at each radius
        for r in common_radii:
            agg_matrix: Any = self.series[0][r].matrix.copy()
            
            for s in self.series[1:]:
                agg_matrix = agg_matrix + s[r].matrix

            result_instance = SimilarityMatrix(
                matrix=agg_matrix,
                space="AGGREGATED",
                radius=r,
                domain=previous_domain,
            )
            result_instances.append(result_instance)

        return SimilaritySeries(instances=result_instances, label="AGGREGATED")

    def __or__(self, other: SimilarityFrame) -> SimilarityFrame:
        """Vertically stack frames (combine populations for all spaces)."""
        if not isinstance(other, SimilarityFrame):
            return NotImplemented

        common_labels = set(self.labels) & set(other.labels)
        if not common_labels:
            raise ValueError("No common labels between frames")

        new_series = []
        for label in self.labels:
            if label in common_labels:
                combined = self._series[label] | other._series[label]
                new_series.append(combined)

        return SimilarityFrame(series=new_series)


# --- Test Sync ---
if __name__ == "__main__":
    import scipy.sparse as sp
    from ..options import MatrixDomain, Space

    def make_dummy_series(space: Space):
        mats = [SimilarityMatrix(sp.eye(5), space, r) for r in range(3)]
        # FIXED: constructor call uses 'instances'
        return SimilaritySeries(instances=mats)

    frame_a = SimilarityFrame(series=[make_dummy_series(Space.FRONT)])
    frame_b = SimilarityFrame(series=[make_dummy_series(Space.FRONT)])

    # Test Vertical Stack (Population merging)
    combined_frame = frame_a | frame_b
    print(f"Combined Frame Labels: {combined_frame.labels}")

    # Test Aggregate (Space collapsing)
    # If we had FRONT and BACK, this would result in 1 Series named AGGREGATED
    agg_series = combined_frame.aggregate()
    print(f"Aggregated Series Label: {agg_series.label}")

"""SimilarityFrame: Similarity-specific extension of MatrixFrame."""

from __future__ import annotations

from typing import Any

import scipy.sparse as sp

from ....base.matrix import MatrixFrame
from ..options import Space
from .s1_matrix import SimilarityMatrix
from .s2_series import SimilaritySeries

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

    # # TODO
    # def align(self) -> Self:
    #     """
    #     Ensures all series in the frame have the exact same radii.
    #     Fills gaps with zero matrices.
    #     """
    #     if not self.series:
    #         return self

    #     # 1. Find the union of all radii across all series
    #     all_radii = set()
    #     for s in self.series:
    #         all_radii.update(s.radii)
    #     sorted_radii = sorted(list(all_radii))

    #     # 2. Reindex every series (calls the fixed SimilaritySeries.reindex)
    #     new_series = [s.fill_empty(sorted_radii) for s in self.series]

    #     # 3. Update internal state
    #     self._series = {s.label: s for s in new_series}
    #     self._ordered_labels = [s.label for s in new_series]
    #     return self

    def to_cumulative(self, *, inplace: bool = True) -> SimilarityFrame:
        """Convert all series to cumulative sum across radii."""
        return self.map(
            lambda s: s.to_cumulative(inplace=False), inplace=inplace,
        )

    def aggregate(self, normalise: bool = False) -> SimilaritySeries:
        """
        Aggregate all series by summing across series at each radius.
        Reduces Frame (Grid) -> Series (Column).
        """
        if not self._series:
            msg = "Cannot aggregate empty SimilarityFrame"
            raise ValueError(msg)

        # 1. Check all series have the same length
        lengths = [len(s.matrices) for s in self.series]
        if len(set(lengths)) != 1:
            msg = f"Series have different lengths: {lengths}"
            raise ValueError(msg)

        n_matrices = lengths[0]
        if n_matrices == 0:
            msg = "Cannot aggregate series with no matrices"
            raise ValueError(msg)

        # 2. Extract metadata from first series/instance
        sample_series = self.series[0]
        previous_domain = sample_series.matrices[0].domain

        result_instances = []

        num_spaces = len(self.series)
        # print(num_spaces)
        # 3. Perform cross-series summation at each position
        from .s1_matrix import SimilarityMatrixTags  # Ensure imported

        for i in range(n_matrices):
            first_matrix = self.series[0].matrices[i]
            agg_matrix: Any = first_matrix.matrix.copy()

            for s in self.series[1:]:
                agg_matrix += s.matrices[i].matrix

            tags = SimilarityMatrixTags(
                domain=previous_domain,
                radius=first_matrix.radius,
                is_gap=False,
            )
            matrix = SimilarityMatrix(
                matrix=agg_matrix,
                label="AGGREGATED",
                tags=tags,
            )

            result_instance = matrix / float(num_spaces) if normalise else matrix / 1.0
            result_instances.append(result_instance)

        return SimilaritySeries(matrices=result_instances, label="AGGREGATED")

    def __or__(self, other: SimilarityFrame) -> SimilarityFrame:
        """Vertically stack frames (combine populations for all spaces)."""
        common_labels = [
            label for label in self.labels if label in other._label_map
        ]

        if not common_labels:
            msg = "No common labels between frames"
            raise ValueError(msg)

        new_series = []
        for label in common_labels:
            s1 = self[label]
            s2 = other[label]
            combined = s1 | s2
            new_series.append(combined)

        return SimilarityFrame(series=new_series)

    # def get_raw(self):
    #     """Get raw matrices in frame layout: series × radii."""
    #     return [
    #         [inst.matrix for inst in series.instances]
    #         for series in self.series
    #     ]

    # def get_row_raw(self, radius: int):
    #     """Get raw matrices for all series at specific radius."""
    #     return [series[radius].matrix for series in self.series]

    # def get_col_raw(self, series_label: str):
    #     """Get all raw matrices for specific series."""
    #     return [inst.matrix for inst in self[series_label]]


# --- Test Sync ---
if __name__ == "__main__":
    import scipy.sparse as sp

    from ..options import Space

    def make_dummy_series(space: Space):
        mats = [SimilarityMatrix(sp.eye(5), space, r) for r in range(3)]
        # FIXED: constructor call uses 'instances'
        return SimilaritySeries(matrices=mats)

    frame_a = SimilarityFrame(series=[make_dummy_series(Space.FRONT)])
    frame_b = SimilarityFrame(series=[make_dummy_series(Space.FRONT)])

    # Test Vertical Stack (Population merging)
    combined_frame = frame_a | frame_b

    # Test Aggregate (Space collapsing)
    # If we had FRONT and BACK, this would result in 1 Series named AGGREGATED
    agg_series = combined_frame.aggregate()

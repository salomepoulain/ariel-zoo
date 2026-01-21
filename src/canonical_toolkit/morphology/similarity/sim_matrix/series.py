"""SimilaritySeries: Similarity-specific extension of MatrixSeries."""

from __future__ import annotations

import scipy.sparse as sp
import numpy as np
from typing import TYPE_CHECKING, Any, Self

from ....base.matrix import MatrixSeries
from ..options import MatrixDomain, Space, UmapConfig
from .matrix import SimilarityMatrix

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["SimilaritySeries"]

class SimilaritySeries(MatrixSeries[SimilarityMatrix]):
    """
    Similarity-specific collection of SimilarityMatrix instances.

    Extends MatrixSeries[SimilarityMatrix] with domain-specific properties and methods for
    morphological similarity analysis.

    Type-safe: Inherits Generic[SimilarityMatrix], so series[2] returns SimilarityMatrix (not MatrixInstance).

    The series index represents the radius. Use the `radii` property to get all indices as radii.
    """

    _instance_class = SimilarityMatrix

    # --- Similarity-specific properties ---

    @property
    def space(self) -> str:
        """Return the morphological space (guaranteed to be the same for all)."""
        # We know it's not empty because of the base class validation
        return self.instances[0].space

    @property
    def radii(self) -> list[int]:
        """Get sorted list of radii (similarity-specific name for indices)."""
        return self.indices

    # --- Advanced Transformations ---

    def to_cumulative(self, *, inplace: bool = True) -> Self:
        """Convert series to cumulative sum across radii."""
        if not self._instances:
            return self

        new_instances = {}
        sorted_radii = self.radii
        
        # Start with the first instance
        first_radius = sorted_radii[0]
        current_running_instance = self._instances[first_radius]
        new_instances[first_radius] = current_running_instance

        # Iterate through the rest
        for radius in sorted_radii[1:]:
            next_inst = self._instances[radius]
            current_running_instance = current_running_instance + next_inst
            new_instances[radius] = current_running_instance.replace(radius=radius)

        if inplace:
            self._instances = new_instances
            return self
        return self.replace(_instances=new_instances)

    def aggregate(self) -> SimilarityMatrix:
        """Aggregate all feature matrices by summing across radii."""
        # Type the accumulator as Any to bypass the Scipy '+' stub limitation
        agg_matrix: Any = self.instances[0].matrix.copy()
        
        for inst in self.instances[1:]:
            agg_matrix = agg_matrix + inst.matrix

        return SimilarityMatrix(
            matrix=agg_matrix,
            space=f"AGGREGATED_{self.space}",
            radius=max(self.radii),
            domain=f"AGGREGATED_{self.instances[0].domain}",
        )

    # --- Operators ---

    def __or__(self, other: SimilaritySeries) -> Self:
        """Vertically stack series (combine populations)."""
        if not isinstance(other, SimilaritySeries):
            return NotImplemented

        common_radii = set(self.radii) & set(other.radii)
        if not common_radii:
            raise ValueError("No common radii between series")

        # Create combined instances using the SimilarityMatrix | operator
        combined_list = [self._instances[r] | other._instances[r] for r in sorted(common_radii)]

        # Argument name 'instances' must match the base MatrixSeries __init__
        return self.__class__(instances=combined_list)

# --- Test Sync ---
if __name__ == "__main__":
    # Ensure tests use the 'instances' keyword argument
    mats = [SimilarityMatrix(np.eye(3), Space.FRONT, r) for r in range(3)]
    series = SimilaritySeries(instances=mats)
    
    print(f"Space: {series.space}")
    print(f"Radii: {series.radii}")
    
    # Test aggregation
    agg = series.aggregate()
    print(f"Aggregated Label: {agg.label}")

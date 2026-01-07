"""SimilaritySeries: Similarity-specific extension of MatrixSeries."""

from __future__ import annotations

from ....base.matrix import MatrixSeries

from .matrix import SimilarityMatrix
from ..options import MatrixDomain, VectorSpace


class SimilaritySeries(MatrixSeries):
    """
    Similarity-specific collection of SimilarityMatrix instances.

    Extends MatrixSeries with domain-specific properties and methods for
    morphological similarity analysis.

    IMPORTANT: The series index IS the source of truth for radius!
    When instances are added (via __init__ or __setitem__), their radius tag
    is automatically synced to match the series index. This ensures:
    - instance.radius always equals series_index
    - Save/load works automatically via base class (no custom methods needed)
    - Radius stays consistent throughout the series lifecycle
    """

    # Override instance class for polymorphic loading
    _instance_class = SimilarityMatrix

    def __init__(
        self,
        instances_list: list[SimilarityMatrix] | None = None,
    ) -> None:
        """Initialize SimilaritySeries.

        Args:
            instances_list: List of SimilarityMatrix objects. All must have same space.
                           Cannot be empty or None.

        Raises:
            ValueError: If instances_list is empty/None, has duplicate radii,
                       or instances have different spaces
        """
        super().__init__(instances_list=instances_list)

        # Sync radius tags with series indices (index is source of truth!)
        for idx, inst in self._instances.items():
            if inst.radius != idx:
                # Update the radius tag to match the series index
                inst._tags["radius"] = idx

    def __setitem__(self, index, instance: SimilarityMatrix) -> None:
        """Set instance at given index, auto-syncing radius tag with index.

        The series index is the source of truth for radius.
        """
        # Sync radius tag with index before storing
        if instance.radius != index:
            instance._tags["radius"] = index

        super().__setitem__(index, instance)

    # --- Similarity-specific properties ---

    @property
    def space(self) -> VectorSpace | str:
        """Return the morphological space (guaranteed to be the same for all)."""
        # Access the first instance's label (which is the space name)
        return next(iter(self._instances.values())).space

    @property
    def radii(self) -> list[int]:
        """Get sorted list of radii (similarity-specific name for indices)."""
        return sorted(self._instances.keys())

    # --- Similarity-specific methods ---

    def cosine_similarity(self, *, inplace: bool = True) -> SimilaritySeries:
        """Compute cosine similarity for all instances.

        Args:
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new SimilaritySeries (safer). Default: True.

        Returns:
            Self if inplace=True, new SimilaritySeries if inplace=False
        """
        return self.map(lambda inst: inst.cosine_similarity(), inplace=inplace)

    def normalize_by_radius(self, *, inplace: bool = True) -> SimilaritySeries:
        """Normalize matrix values by dividing by (radius + 1).

        Divides each matrix by (radius + 1) to normalize by the radius level.
        The +1 ensures radius 0 is divided by 1 (avoiding division by zero).
        Uses each instance's stored radius value.

        Args:
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new SimilaritySeries (safer). Default: True.

        Returns:
            Self if inplace=True, new SimilaritySeries if inplace=False

        Example:
            >>> series[0].matrix  # Original values at radius 0
            >>> series[1].matrix  # Original values at radius 1
            >>> normalized = series.normalize_by_radius()
            >>> normalized[0].matrix  # Divided by 1
            >>> normalized[1].matrix  # Divided by 2
        """
        return self.map(lambda inst: inst.normalize_by_radius(), inplace=inplace)

    def to_cumulative(self, *, inplace: bool = True) -> SimilaritySeries:
        """Convert series to cumulative sum across indices.

        Args:
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new MatrixSeries (safer). Default: True.

        Returns:
            Self if inplace=True, new MatrixSeries if inplace=False
        """
        if not self._instances:
            return self if inplace else self.replace(_instances={})

        new_instances = {}
        sorted_indices = sorted(self._instances.keys())
        current_sum = None
        for idx in sorted_indices:
            mat_inst = self._instances[idx]
            if current_sum is None:
                new_inst = mat_inst
                current_sum = mat_inst.matrix
            else:
                current_sum = current_sum + mat_inst.matrix
                new_inst = mat_inst.replace(matrix=current_sum)
            new_instances[idx] = new_inst

        if inplace:
            self._instances = new_instances
            return self
        return self.replace(_instances=new_instances)

    def aggregate(self) -> SimilarityMatrix:
        """Aggregate all feature matrices by summing across radii.

        Collapses the radius dimension by element-wise summation.
        Changes space to "AGGREGATED" since spatial hierarchy is merged.

        ONLY works on MatrixDomain.FEATURES matrices.

        Returns:
            Single SimilarityMatrix with:
            - space = "AGGREGATED"
            - domain = MatrixDomain.FEATURES
            - matrix = sum of all matrices (stays sparse!)

        Raises:
            ValueError: If series is empty or contains non-FEATURES matrices

        Example:
            >>> series[0].shape  # (100, 10000) FEATURES at radius 0
            >>> series[1].shape  # (100, 10000) FEATURES at radius 1
            >>> aggregated = series.aggregate()
            >>> aggregated.shape  # (100, 10000) AGGREGATED
            >>> aggregated.space  # "AGGREGATED"
        """
        if not self._instances:
            msg = "Cannot aggregate empty SimilaritySeries"
            raise ValueError(msg)

        # Validate all are FEATURES
        for r, inst in self._instances.items():
            if inst.domain != MatrixDomain.FEATURES.name:
                msg = (
                    f"Can only aggregate FEATURES matrices. "
                    f"Found {inst.domain} at radius {r}"
                )
                raise ValueError(msg)

        sorted_radii = sorted(self._instances.keys())

        # Sum matrices element-wise
        agg_matrix = self._instances[sorted_radii[0]].matrix.copy()
        for r in sorted_radii[1:]:
            agg_matrix = agg_matrix + self._instances[r].matrix

        # Use max radius for aggregated matrix
        max_radius = max(sorted_radii)

        return SimilarityMatrix(
            matrix=agg_matrix,
            space="AGGREGATED",
            radius=max_radius,
            domain=MatrixDomain.FEATURES,
        )


if __name__ == "__main__":
    import scipy.sparse as sp
    from .matrix import SimilarityMatrix
    from ..options import VectorSpace, MatrixDomain

    print("Testing SimilaritySeries...")

    # Test 1: Create series from similarity matrices
    n_individuals = 10
    n_features = 100
    instances_list = []
    for r in range(5):
        mat = sp.random(n_individuals, n_features, density=0.3, format="csr")
        inst = SimilarityMatrix(
            matrix=mat,
            space=VectorSpace.FRONT,
            radius=r,
            domain=MatrixDomain.FEATURES
        )
        instances_list.append(inst)

    series = SimilaritySeries(instances_list=instances_list)
    print(f"✓ Created series with {len(series.radii)} instances")
    print(f"  Space: {series.space}")
    print(f"  Radii: {series.radii}")

    # Test 2: Cosine similarity
    sim_series = series.cosine_similarity(inplace=False)
    print(f"\n✓ Cosine similarity: {sim_series[0].domain}")

    # Test 3: Normalize by radius
    normalized = sim_series.normalize_by_radius(inplace=False)
    print(f"\n✓ Normalized by radius")

    # Test 4: Aggregate
    aggregated = series.aggregate()
    print(f"\n✓ Aggregated: space={aggregated.space}, shape={aggregated.shape}")

    # Test 5: Cumulative
    cumulative = series.to_cumulative(inplace=False)
    print(f"\n✓ Cumulative series created")

    print("\n✅ All SimilaritySeries tests passed!")

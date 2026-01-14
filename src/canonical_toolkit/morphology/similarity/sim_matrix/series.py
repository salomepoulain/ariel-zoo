"""SimilaritySeries: Similarity-specific extension of MatrixSeries."""

from __future__ import annotations

from ....base.matrix import MatrixSeries
from ..options import MatrixDomain, Space, UmapConfig
from .matrix import SimilarityMatrix


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
    def space(self) -> Space | str:
        """Return the morphological space (guaranteed to be the same for all)."""
        return next(iter(self._instances.values())).space

    @property
    def radii(self) -> list[int]:
        """Get sorted list of radii (similarity-specific name for indices)."""
        return sorted(self._instances.keys())       

    def to_cumulative(self, *, inplace: bool = True) -> SimilaritySeries:
        """Convert series to cumulative sum across indices.

        Args:
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new MatrixSeries (safer). Default: True.

        Returns
        -------
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
                current_sum += mat_inst.matrix
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

        Returns
        -------
            Single SimilarityMatrix with:
            - space = "AGGREGATED"
            - domain = MatrixDomain.FEATURES
            - matrix = sum of all matrices (stays sparse!)

        Raises
        ------
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

        for r, inst in self._instances.items():
            if inst.domain != MatrixDomain.FEATURES.name:
                msg = (
                    f"Can only aggregate FEATURES matrices. "
                    f"Found {inst.domain} at radius {r}"
                )
                raise ValueError(msg)

        sorted_radii = sorted(self._instances.keys())

        agg_matrix = self._instances[sorted_radii[0]].matrix.copy()
        for r in sorted_radii[1:]:
            agg_matrix += self._instances[r].matrix

        max_radius = max(sorted_radii)

        return SimilarityMatrix(
            matrix=agg_matrix,
            space="AGGREGATED",
            radius=max_radius,
            domain=MatrixDomain.FEATURES,
        )

    # --- Mappings ---

    def cosine_similarity(self, *, inplace: bool = True) -> SimilaritySeries:
        """Compute cosine similarity for all instances.

        Args:
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new SimilaritySeries (safer). Default: True.

        Returns
        -------
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

        Returns
        -------
            Self if inplace=True, new SimilaritySeries if inplace=False

        Example:
            >>> series[0].matrix  # Original values at radius 0
            >>> series[1].matrix  # Original values at radius 1
            >>> normalized = series.normalize_by_radius()
            >>> normalized[0].matrix  # Divided by 1
            >>> normalized[1].matrix  # Divided by 2
        """
        return self.map(
            lambda inst: inst.normalize_by_radius(), inplace=inplace,
        )
        
    def umap_embed(self, *, config: UmapConfig | None = None, inplace: bool = True) -> SimilaritySeries:
        return self.map(
            lambda inst: inst.umap_embed(config=config), inplace=inplace,
        )
        
if __name__ == "__main__":
    import scipy.sparse as sp

    from ..options import MatrixDomain, Space
    from .matrix import SimilarityMatrix

    # Test 1: Create series from similarity matrices
    n_individuals = 10
    n_features = 100
    instances_list = []
    for r in range(5):
        mat = sp.random(n_individuals, n_features, density=0.3, format="csr")
        inst = SimilarityMatrix(
            matrix=mat,
            space=Space.FRONT,
            radius=r,
            domain=MatrixDomain.FEATURES,
        )
        instances_list.append(inst)

    series = SimilaritySeries(instances_list=instances_list)

    # Test 2: Cosine similarity
    sim_series = series.cosine_similarity(inplace=False)

    # Test 3: Normalize by radius
    normalized = sim_series.normalize_by_radius(inplace=False)

    # Test 4: Aggregate
    aggregated = series.aggregate()

    # Test 5: Cumulative
    cumulative = series.to_cumulative(inplace=False)

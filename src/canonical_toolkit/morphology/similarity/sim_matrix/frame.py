"""SimilarityFrame: Similarity-specific extension of MatrixFrame."""

from __future__ import annotations

from ....base.matrix import MatrixFrame

from .matrix import SimilarityMatrix
from ..options import MatrixDomain, Space
from .series import SimilaritySeries

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
        """Convert all series to cumulative sum across indices.

        Args:
            inplace: If True, modify self and return self (faster, saves memory).
                    If False, create new MatrixFrame (safer). Default: True.

        Returns
        -------
            Self if inplace=True, new MatrixFrame if inplace=False
        """
        return self.map(lambda s: s.to_cumulative(inplace=False), inplace=inplace)

    def aggregate(self) -> SimilaritySeries:
        """Aggregate all series by summing across series at each radius.

        Reduces Frame → Series by collapsing the series dimension.
        For each radius, sums all matrices across series.
        Changes space to "AGGREGATED".

        ONLY works on MatrixDomain.FEATURES matrices.

        Returns:
            Single SimilaritySeries with space=AGGREGATED, containing one summed
            instance per radius (summed across all series).

        Raises:
            ValueError: If frame is empty or contains non-FEATURES matrices

        Example:
            >>> # frame has FRONT, BACK, LEFT series at radii [0, 1, 2]
            >>> agg = frame.aggregate()
            >>> agg[0]  # = FRONT[0] + BACK[0] + LEFT[0]
            >>> agg[1]  # = FRONT[1] + BACK[1] + LEFT[1]
            >>> agg.space  # "AGGREGATED"
        """
        if not self._series:
            msg = "Cannot aggregate empty SimilarityFrame"
            raise ValueError(msg)

        for space, series in self._series.items():
            for r, inst in series.instances.items():
                if inst.domain != MatrixDomain.FEATURES.name:
                    msg = (
                        f"Can only aggregate FEATURES matrices. "
                        f"Found {inst.domain} at {space}, radius {r}"
                    )
                    raise ValueError(msg)

        all_radii = [set(series.instances.keys()) for series in self._series.values()]
        common_radii = sorted(set.intersection(*all_radii))

        if not common_radii:
            msg = "No common radii found across all series"
            raise ValueError(msg)

        result_instances = []

        for r in common_radii:
            instances_at_r = [series[r] for series in self._series.values()]

            agg_matrix = instances_at_r[0].matrix.copy()
            for inst in instances_at_r[1:]:
                agg_matrix = agg_matrix + inst.matrix

            result_instance = SimilarityMatrix(
                matrix=agg_matrix,
                space="AGGREGATED",
                radius=r,
                domain=MatrixDomain.FEATURES,
            )
            result_instances.append(result_instance)

        return SimilaritySeries(instances_list=result_instances)


if __name__ == "__main__":
    import scipy.sparse as sp
    from .matrix import SimilarityMatrix
    from .series import SimilaritySeries
    from ..options import Space, MatrixDomain

    print("=" * 80)
    print("SIMILARITY FRAME TESTS")
    print("=" * 80)

    # Test 1: Create frame with multiple series
    print("\n[1] Creating SimilarityFrame...")

    # Create front series
    front_instances = []
    for r in range(5):
        mat = sp.random(10, 100, density=0.3, format="csr")
        inst = SimilarityMatrix(
            matrix=mat,
            space=Space.FRONT,
            radius=r,
            domain=MatrixDomain.FEATURES
        )
        front_instances.append(inst)
    front_series = SimilaritySeries(instances_list=front_instances)

    # Create back series
    back_instances = []
    for r in range(5):
        mat = sp.random(10, 100, density=0.3, format="csr")
        inst = SimilarityMatrix(
            matrix=mat,
            space=Space.BACK,
            radius=r,
            domain=MatrixDomain.FEATURES
        )
        back_instances.append(inst)
    back_series = SimilaritySeries(instances_list=back_instances)

    frame = SimilarityFrame(series=[front_series, back_series])
    print(f"✓ Created frame with {len(list(frame.keys()))} series")

    # Test 2: Frame slicing by radius
    print("\n[2] SimilarityFrame radius slicing:")
    sliced = frame[:2]
    print(f"✓ frame[:2] → {type(sliced).__name__}")

    # Test 3: Frame selection by space
    print("\n[3] SimilarityFrame space selection:")
    selected = frame[[Space.FRONT, Space.BACK]]
    print(f"✓ frame[[FRONT, BACK]] → {len(list(selected.keys()))} series")

    # Test 4: Aggregate
    print("\n[4] SimilarityFrame aggregation:")
    aggregated = frame.aggregate()
    print(f"✓ Aggregated: space={aggregated.space}, {len(aggregated.radii)} radii")

    # Test 5: Frame repr
    print("\n[5] SimilarityFrame representation:")
    print(frame)

    # Test 6: Save/Load
    print("\n[6] SimilarityFrame save/load:")
    # Save with default path using new hierarchical API
    frame.save()
    from ....base.matrix import DATA_FRAMES
    default_path = DATA_FRAMES / frame.description
    print(f"✓ Saved to default location: {default_path}")

    # Load it back
    loaded = SimilarityFrame.load(default_path)
    print(f"✓ Loaded: {len(list(loaded.keys()))} series")
    print(loaded)

    print("\n✅ All SimilarityFrame tests passed!")

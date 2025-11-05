from __future__ import annotations

from collections import Counter
from collections.abc import Callable

import numpy as np
from scipy.special import softmax

type NeighbourhoodDict = dict[int, list[str]]
type ValidData = tuple[list[int], list[float]]

type TanimotoCalculator = Callable[
    [NeighbourhoodDict, NeighbourhoodDict, int],
    float | None,
]
type WeightCalculator = Callable[[list[int]], list[float]]
type SimilarityAggregator = Callable[[list[float], list[float]], float]


class Evaluator:
    """Helper class for calculating similarity metrics between tree neighborhoods."""

    # region TanimotoCalculator -----

    @staticmethod
    def tanimoto_strings_set(
        fp1_dict: NeighbourhoodDict,
        fp2_dict: NeighbourhoodDict,
        radius: int,
    ) -> float | None:
        if radius not in fp1_dict or radius not in fp2_dict:
            return None

        strings1 = set(fp1_dict[radius])
        strings2 = set(fp2_dict[radius])

        intersection = len(strings1 & strings2)
        union = len(strings1 | strings2)

        if union == 0:
            return 0.0

        return intersection / union

    @staticmethod
    def tanimoto_strings_with_counts(
        fp1_dict: NeighbourhoodDict,
        fp2_dict: NeighbourhoodDict,
        radius: int,
    ) -> float | None:
        """Tanimoto with fragment counts (bag of words approach)."""
        if radius not in fp1_dict or radius not in fp2_dict:
            return None

        counts1 = Counter(fp1_dict[radius])
        counts2 = Counter(fp2_dict[radius])

        intersection = sum((counts1 & counts2).values())
        union = sum((counts1 | counts2).values())

        if union == 0:
            return 0.0

        return intersection / union

    # endregion

    # region WeightCalculators -----

    @staticmethod
    def uniform_weights(radii: list[int]) -> list[float]:
        return [1.0 for _ in radii]

    @staticmethod
    def linear_weights(radii: list[int]) -> list[float]:
        """Calculate linear weights for radii (r+1)."""
        return [float(r + 1) for r in radii]

    @staticmethod
    def exponential_weights(radii: list[int]) -> list[float]:
        """Calculate exponential weights for radii (2^r)."""
        return [float(2**r) for r in radii]

    @staticmethod
    def softmax_weights(radii: list[int], beta: float = 1.0) -> list[float]:
        """Calculate softmax weights for radii."""
        radii_array = np.array(radii, dtype=float)
        weights = softmax(beta * radii_array)
        return weights.tolist()

    # endregion

    # region AggregationFunction -----

    @staticmethod
    def calc_tanimoto_power_mean(
        similarities: list[float],
        weights: list[float],
        *,
        p: float = 1.0,
    ) -> float:
        """
        Calculate similarity using power mean (generalized mean).

        Formula: S = (Σ(w_r · s_r^p) / Σ(w_r))^(1/p)

        The power mean family allows different emphasis on high vs low similarities:
        - p = 1: Arithmetic mean (balanced) [DEFAULT]
        - p = 2: Quadratic mean (rewards high similarities)
        - p = 3: Cubic mean (strongly rewards high similarities)
        - p → ∞: Maximum (only best match matters)

        Args:
            p: Power parameter (valid range: p >= 1)
            - p = 1: Standard weighted average [RECOMMENDED]
            - p = 2: Emphasizes good matches
            - p = 3: Strongly emphasizes good matches

        Returns
        -------
            Combined similarity score in [0, 1], or 0.0 if no valid data

        Raises
        ------
            ValueError: If p < 1 (negative p unsuitable for tree similarity)
        """
        if p < 1:
            msg = (
                f"p must be >= 1 for tree similarity (got p={p}). "
                "Negative p values cause issues with zero similarities."
            )
            raise ValueError(
                msg,
            )

        # Power mean formula: (Σ(w_r · s_r^p) / Σ(w_r))^(1/p)
        weighted_sum = sum(
            w * (s**p) for w, s in zip(weights, similarities, strict=False)
        )
        total_weight = sum(weights)

        return (weighted_sum / total_weight) ** (1 / p)

    # endregion

    # region Base similarity calculator

    @classmethod
    def similarity_calculator(
        cls,
        nh1_dict: NeighbourhoodDict,
        nh2_dict: NeighbourhoodDict,
        *,
        tanimoto_fn: TanimotoCalculator,
        weight_fn: WeightCalculator,
        aggregation_fn: SimilarityAggregator,
        skip_missing_radii: bool = True,
    ) -> float:
        """
        Base similarity calculator that uses provided callables for computation.

        This is a generic method that composes the similarity calculation pipeline
        using injected functions. The toolkit's calculate_similarity method uses
        this as a base and provides the specific functions based on config.

        Args:
            nh1_dict: Neighbourhood dictionary for first node
            nh2_dict: Neighbourhood dictionary for second node (comparison is symmetric)
            tanimoto_fn: Function to calculate Tanimoto similarity per radius
            weight_fn: Function to calculate weights for radii
            aggregation_fn: Function to aggregate weighted similarities
            skip_missing_radii: If True, skip radii not in both dicts; else treat as 0

        Returns
        -------
            Similarity score between 0 and 1
        """
        # Step 1: Calculate Tanimoto for all radii
        results_dict = cls._tanimoto_all_radii(nh1_dict, nh2_dict, tanimoto_fn)

        # Step 2: Extract valid data
        radii, similarities = cls._extract_valid_data(
            results_dict,
            skip_none=skip_missing_radii,
        )

        # If no radii overlap, similarity is 0
        if not radii:
            return 0.0

        # Step 3: Calculate weights
        weights = weight_fn(radii)

        # Step 4: Aggregate similarities
        return aggregation_fn(similarities, weights)

    # endregion

    # region helpers -----

    @staticmethod
    def _tanimoto_all_radii(
        fp1_dict: dict[int, list[str]],
        fp2_dict: dict[int, list[str]],
        tanimoto_calculator_fn: TanimotoCalculator,
    ) -> dict[int, float | None]:
        """Calculate Tanimoto for each radius level."""
        results = {}

        all_radii = max(fp1_dict.keys(), fp2_dict.keys())
        for radius in all_radii:
            results[radius] = tanimoto_calculator_fn(fp1_dict, fp2_dict, radius)

        return results

    @staticmethod
    def _extract_valid_data(
        result_dict: dict[int, float | None],
        skip_none: bool = True,
    ) -> tuple[list[int], list[float]]:
        """
        Extract valid radii and similarities from result dictionary.

        Args:
            result_dict: Dictionary mapping radius to similarity score
            skip_none: If True, ignore None values; if False, treat None as 0.0

        Returns
        -------
            Tuple of (radii, similarities) lists with valid data only
        """
        radii = []
        similarities = []

        for radius, similarity in result_dict.items():
            if skip_none and similarity is None:
                continue
            radii.append(radius)
            similarities.append(similarity if similarity is not None else 0.0)

        return radii, similarities

    # endregion

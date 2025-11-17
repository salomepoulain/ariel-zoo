from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.special import softmax

from ariel_experiments.characterize.canonical.core.tools.vector import (
    HashVector,
)

type TreeHash = str
type Vectorizer = Callable[[list[TreeHash]], HashVector]
type Scorer = Callable[[HashVector, HashVector], float]
type WeightCalculator = Callable[[list[int]], list[float]]
type SimilarityAggregator = Callable[[list[float], list[float]], float]


@dataclass
class RadiusData:
    """Document frequency data for TF-IDF calculations."""

    df_counts: HashVector
    N: int = 0

    def get_df(self, tree_hash: TreeHash) -> int:
        """Get document frequency for a fragment."""
        return int(self.df_counts.get(tree_hash, 0))


class Evaluator:
    """Pure vector operations for similarity calculations."""

    # region Vectorizers -----

    @staticmethod
    def compute_binary_vector(fragments: list[TreeHash]) -> HashVector:
        """
        Create binary presence/absence vector.

        Each unique fragment gets value 1.0.

        Args:
            fragments: List of fragment strings

        Returns
        -------
            HashVector with 1.0 for each unique fragment
        """
        return HashVector(dict.fromkeys(set(fragments), 1.0))

    @staticmethod
    def compute_count_vector(fragments: list[TreeHash]) -> HashVector:
        """
        Create count vector (bag-of-words).

        Each fragment mapped to its frequency.

        Args:
            fragments: List of fragment strings

        Returns
        -------
            HashVector with counts for each fragment
        """
        counts = Counter(fragments)
        return HashVector({
            fragment: float(count) for fragment, count in counts.items()
        })

    @staticmethod
    def tfidf_vector(
        tf_vec: HashVector,
        df_vec: HashVector,
        N: int,
        *,
        smooth: bool = True,
    ) -> HashVector:
        """
        Create TF-IDF weighted vector.

        Args:
            tf_vec: Term frequency vector (counts)
            df_vec: Document frequency vector (counts across corpus)
            N: Total number of documents in corpus
            smooth: Whether to use smoothed IDF

        Returns
        -------
            TF-IDF weighted HashVector
        """
        # Only compute IDF for terms present in tf_vec (much smaller set)
        # Extract only the df_counts we need
        terms = tf_vec.keys()
        df_counts = np.array([df_vec.get(term, 0) for term in terms], dtype=float)

        # Vectorized IDF calculation - much faster than individual np.log calls
        if smooth:
            idf_values = np.log((N + 1) / (df_counts + 1)) + 1
        else:
            # Handle df_count == 0 case with np.where
            idf_values = np.where(
                df_counts > 0,
                np.log(N / df_counts),
                0.0,
            )

        # Build IDF vector only for terms in tf_vec
        idf_vec = HashVector(dict(zip(terms, idf_values, strict=True)))
        return tf_vec * idf_vec

    # endregion

    # region Scorers -----

    @staticmethod
    def score_cosine_similarity(
        vec1: HashVector,
        vec2: HashVector,
    ) -> float:
        """
        Cosine similarity between two vectors.

        Measures angle between vectors, ignoring magnitude.

        Returns
        -------
            Similarity in [-1, 1], or 0 if either vector is empty
        """
        if not vec1 or not vec2:
            return 0.0

        dot = vec1 @ vec2
        norm_product = vec1.l2_norm() * vec2.l2_norm()

        if norm_product == 0:
            return 0.0

        return dot / norm_product

    @staticmethod
    def score_tanimoto_similarity(
        vec1: HashVector,
        vec2: HashVector,
    ) -> float:
        """
        Tanimoto similarity (generalized Jaccard).

        Uses min for intersection, max for union.

        Formula: sum(min(v1, v2)) / sum(max(v1, v2))

        Returns
        -------
            Similarity in [0, 1]
        """
        all_keys = vec1.keys() | vec2.keys()

        intersection = sum(min(vec1.get(k), vec2.get(k)) for k in all_keys)
        union = sum(max(vec1.get(k), vec2.get(k)) for k in all_keys)

        return intersection / union if union > 0 else 0.0

    @classmethod
    def score_tfidf_entropy(
        cls,
        tf_vec: HashVector,
        df_vec: HashVector,
        N: int,
        *,
        smooth: bool,
        normalized: bool,
    ) -> float:
        """Value between 0 and 1."""
        tfidf_vec = cls.tfidf_vector(
            tf_vec=tf_vec,
            df_vec=df_vec,
            N=N,
            smooth=smooth,
        )
        if normalized:
            return tfidf_vec.normalized_entropy()

        return tfidf_vec.entropy()

    @classmethod
    def score_tfidf_mean(
        cls, tf_vec: HashVector, df_vec: HashVector, N: int, *, smooth: bool,
    ) -> float:
        tfidf_vec = cls.tfidf_vector(
            tf_vec=tf_vec,
            df_vec=df_vec,
            N=N,
            smooth=smooth,
        )

        return tfidf_vec.mean()

    @classmethod
    def score_tfidf_l1(
        cls, tf_vec: HashVector, df_vec: HashVector, N: int, *, smooth: bool,
    ) -> float:
        tfidf_vec = cls.tfidf_vector(
            tf_vec=tf_vec,
            df_vec=df_vec,
            N=N,
            smooth=smooth,
        )

        return tfidf_vec.l1_norm()

    @classmethod
    def score_tfidf_sum(
        cls, tf_vec: HashVector, df_vec: HashVector, N: int, *, smooth: bool,
    ) -> float:
        tfidf_vec = cls.tfidf_vector(
            tf_vec=tf_vec,
            df_vec=df_vec,
            N=N,
            smooth=smooth,
        )

        return tfidf_vec.sum()

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

    # region Aggregation -----

    @staticmethod
    def power_mean_aggregate(
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

from __future__ import annotations

from collections.abc import Callable
from scipy.sparse import csr_matrix
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from typing import TypeVar, Callable, Any, cast
from scipy.sparse import csr_matrix, spmatrix

T_Output = TypeVar("T_Output")

# --- Types ---
type TreeHash = str
type NeighbourhoodData = dict[int, list[TreeHash]]

type MatrixDict = dict[int, csr_matrix] # matrix per radius

# Changed: Takes a single integer (radius), returns a single float (weight)
type WeightCalculator = Callable[[int], float]

class Evaluator:
    """Pure vector operations for similarity calculations."""

    # region Vectorizers -----

    @staticmethod
    def _get_max_radius(data_list: list[NeighbourhoodData]) -> int:
        if not data_list:
            return 0
        return max(max(d.keys(), default=0) for d in data_list)

    @staticmethod
    def vectorize_with_hashing(
        data_list: list[NeighbourhoodData],
        *,
        is_binary: bool = False,
        n_features: int = 2**20,
    ) -> MatrixDict:
        """
        Returns a dictionary mapping {radius: csr_matrix}.
        Weights are NOT applied here.
        """
        max_radius = Evaluator._get_max_radius(data_list)

        hasher = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            binary=is_binary,
            analyzer=lambda x: x,
            norm=None
        )

        matrix_dict: MatrixDict = {}

        for radius in range(max_radius + 1):

            field_gen = (doc.get(radius, []) for doc in data_list)
            matrix_dict[radius] = hasher.transform(field_gen)

        return matrix_dict

    @staticmethod
    def vectorize_with_vocab(
        data_list: list[NeighbourhoodData],
        *,
        is_binary: bool = False,
        min_df: int = 1,
        max_features: int | None = None,
        return_vectorizer: bool = False
    ) -> tuple[MatrixDict, np.ndarray] | tuple[MatrixDict, CountVectorizer] :
        """
        Returns (MatrixDict, Vocabulary).
        Ensures all radii share the SAME vocabulary space.
        """
        max_radius = Evaluator._get_max_radius(data_list)

        cv = CountVectorizer(
            binary=is_binary,
            min_df=min_df,
            max_features=max_features,
            analyzer=lambda x: x
        )

        # 1. FIT PHASE: Global Vocabulary
        # We must look at EVERYTHING (all radii) to build the dictionary.
        # Otherwise, Radius 0 might have "apple" as col 1,
        # and Radius 1 might have "banana" as col 1.

        def fit_generator():
            keys_to_use = range(max_radius + 1)
            for doc in data_list:
                all_tokens = itertools.chain.from_iterable(
                    doc.get(k, []) for k in keys_to_use
                )
                yield list(all_tokens)

        cv.fit(fit_generator())

        # 2. TRANSFORM PHASE: Local Radii
        matrix_dict: MatrixDict = {}

        for radius in range(max_radius + 1):
            field_gen = (doc.get(radius, []) for doc in data_list)

            # Transform using the global vocabulary
            matrix_dict[radius] = cv.transform(field_gen)

        if return_vectorizer:
            return matrix_dict, cv

        return matrix_dict, cv.get_feature_names_out()

    
    @staticmethod
    def sum_hash_vectors(
        data_list: list[NeighbourhoodData],
        *,
        is_binary: bool = False,
        n_features: int = 2**20,
    ) -> csr_matrix:
        """
        Calls vectorize_with_hashing and sums all resulting matrices.
        Returns a single csr_matrix.
        """
        matrix_dict = Evaluator.vectorize_with_hashing(
            data_list,
            is_binary=is_binary,
            n_features=n_features
        )
        
        if not matrix_dict:
            return csr_matrix((len(data_list), n_features))

        aggregate_matrix = sum(matrix_dict.values())
        
        return aggregate_matrix

    @staticmethod
    def sum_vocab_vectors(
        data_list: list[NeighbourhoodData],
        *,
        is_binary: bool = False,
        min_df: int = 1,
        max_features: int | None = None,
        # return_vectorizer: bool = False
    ) -> tuple[MatrixDict, np.ndarray]:
        """
        Calls vectorize_with_vocab, sums all resulting matrices, and returns
        (Aggregate_Matrix, Vocabulary/Vectorizer).
        """
        result = Evaluator.vectorize_with_vocab(
            data_list,
            is_binary=is_binary,
            min_df=min_df,
            max_features=max_features,
            return_vectorizer=True
        )
        
        matrix_dict, cv = result

        if not matrix_dict:
            n_features = len(cv.get_feature_names_out())
            aggregate_matrix = csr_matrix((len(data_list), n_features))
        else:
            aggregate_matrix = sum(matrix_dict.values())

        # if return_vectorizer:
        #     return aggregate_matrix, cv
        cast(csr_matrix, aggregate_matrix)
        return aggregate_matrix, cv.get_feature_names_out()

    @staticmethod
    def matrix_dict_applier(
        matrix_dict: dict[int, csr_matrix],
        matrix_applier: Callable[[csr_matrix], T_Output]
    ) -> dict[int, T_Output]:
        """
        Applies a function to every matrix in the dictionary independently.
        Returns a new dictionary with the transformed results.
        """
        sorted_keys = sorted(matrix_dict.keys())
        
        return {
            radius: matrix_applier(matrix_dict[radius]) 
            for radius in sorted_keys
        }

    # region WeightCalculators -----

    @staticmethod
    def uniform_weight(radius: int) -> float:
        return 1.0

    @staticmethod
    def linear_weight(radius: int) -> float:
        """Returns r + 1"""
        return float(radius + 1)

    @staticmethod
    def exponential_weight(radius: int) -> float:
        """Returns 2^r"""
        return float(2**radius)

    @staticmethod
    def inverse_weight(radius: int) -> float:
        """Returns 1 / (r + 1). Good for penalizing distance."""
        return 1.0 / (radius + 1)

    # endregion

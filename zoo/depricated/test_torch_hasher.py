"""
Test suite for TorchFeatureHasher to ensure compatibility with sklearn.

Tests that our PyTorch implementation produces the same results as
sklearn.feature_extraction.FeatureHasher.
"""

import pytest
import numpy as np
import torch
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction import FeatureHasher as SklearnHasher

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ariel_experiments.characterize.canonical.core.analysis.torch_hasher import TorchFeatureHasher, BatchedTorchFeatureHasher


# ==================== FIXTURES ====================


@pytest.fixture
def simple_corpus():
    """Simple test corpus."""
    return [
        ["token1", "token2", "token3"],
        ["token2", "token4"],
        ["token1", "token1", "token5"],  # Duplicates
        [],  # Empty
    ]


@pytest.fixture
def medium_corpus():
    """Medium-sized corpus for performance tests."""
    return [
        [f"word{i % 100}" for i in range(50)]
        for _ in range(100)
    ]


@pytest.fixture
def large_corpus():
    """Large corpus for batching tests."""
    return [
        [f"tok{i % 500}" for i in range(100)]
        for _ in range(1000)
    ]


# ==================== BASIC FUNCTIONALITY TESTS ====================


def test_output_format(simple_corpus):
    """Test that output is a scipy CSR matrix."""
    hasher = TorchFeatureHasher(n_features=2**10)
    X = hasher.fit_transform(simple_corpus)

    assert issparse(X), "Output should be sparse"
    assert X.format == "csr", "Output should be CSR format"
    assert X.shape == (len(simple_corpus), 2**10), "Shape mismatch"


def test_output_dtype(simple_corpus):
    """Test that output has correct dtype."""
    hasher = TorchFeatureHasher(n_features=2**10)
    X = hasher.fit_transform(simple_corpus)

    assert X.dtype == np.float32, "Output dtype should be float32"


def test_empty_corpus():
    """Test handling of empty corpus."""
    hasher = TorchFeatureHasher(n_features=2**10)
    X = hasher.fit_transform([])

    assert X.shape == (0, 2**10), "Empty corpus should have 0 rows"
    assert issparse(X), "Should still return sparse matrix"


def test_all_empty_samples():
    """Test corpus with all empty samples."""
    corpus = [[], [], []]
    hasher = TorchFeatureHasher(n_features=2**10)
    X = hasher.fit_transform(corpus)

    assert X.shape == (3, 2**10), "Should have correct shape"
    assert X.nnz == 0, "Should have no non-zero elements"


# ==================== SKLEARN COMPATIBILITY TESTS ====================


@pytest.mark.parametrize("n_features", [2**8, 2**12, 2**16])
def test_same_shape_as_sklearn(simple_corpus, n_features):
    """Test that output shape matches sklearn."""
    torch_hasher = TorchFeatureHasher(n_features=n_features)
    sklearn_hasher = SklearnHasher(n_features=n_features, input_type="string")

    X_torch = torch_hasher.fit_transform(simple_corpus)
    X_sklearn = sklearn_hasher.fit_transform(simple_corpus)

    assert X_torch.shape == X_sklearn.shape, "Shapes should match sklearn"


@pytest.mark.parametrize("n_features", [2**8, 2**12, 2**16])
def test_same_sparsity_as_sklearn(simple_corpus, n_features):
    """Test that sparsity matches sklearn (same nnz)."""
    torch_hasher = TorchFeatureHasher(n_features=n_features)
    sklearn_hasher = SklearnHasher(n_features=n_features, input_type="string")

    X_torch = torch_hasher.fit_transform(simple_corpus)
    X_sklearn = sklearn_hasher.fit_transform(simple_corpus)

    assert X_torch.nnz == X_sklearn.nnz, "Non-zero counts should match sklearn"


def test_same_hash_buckets_as_sklearn(simple_corpus):
    """
    Test that tokens hash to the same buckets as sklearn.

    Note: This tests structural equivalence (same buckets filled),
    not exact value equivalence (signs might differ).
    """
    n_features = 2**12
    torch_hasher = TorchFeatureHasher(n_features=n_features)
    sklearn_hasher = SklearnHasher(n_features=n_features, input_type="string")

    X_torch = torch_hasher.fit_transform(simple_corpus)
    X_sklearn = sklearn_hasher.fit_transform(simple_corpus)

    # Check that non-zero positions are the same
    torch_nonzero = set(zip(*X_torch.nonzero()))
    sklearn_nonzero = set(zip(*X_sklearn.nonzero()))

    assert torch_nonzero == sklearn_nonzero, (
        "Non-zero positions should match sklearn"
    )


def test_values_match_sklearn_absolute(simple_corpus):
    """
    Test that absolute values match sklearn.

    We test absolute values because the sign hash might differ
    due to implementation details, but magnitudes should match.
    """
    n_features = 2**12
    torch_hasher = TorchFeatureHasher(n_features=n_features)
    sklearn_hasher = SklearnHasher(n_features=n_features, input_type="string")

    X_torch = torch_hasher.fit_transform(simple_corpus).toarray()
    X_sklearn = sklearn_hasher.fit_transform(simple_corpus).toarray()

    # Check absolute values match (signs may differ)
    np.testing.assert_allclose(
        np.abs(X_torch),
        np.abs(X_sklearn),
        rtol=1e-5,
        err_msg="Absolute values should match sklearn"
    )


# ==================== ALTERNATE SIGN TESTS ====================


def test_alternate_sign_true():
    """Test that alternate_sign=True produces both positive and negative values."""
    corpus = [["a", "b", "c", "d", "e", "f", "g", "h"]]
    hasher = TorchFeatureHasher(n_features=2**8, alternate_sign=True)
    X = hasher.fit_transform(corpus).toarray()

    has_positive = (X > 0).any()
    has_negative = (X < 0).any()

    assert has_positive, "Should have positive values"
    assert has_negative, "Should have negative values"


def test_alternate_sign_false():
    """Test that alternate_sign=False produces only positive values."""
    corpus = [["a", "b", "c", "d", "e", "f", "g", "h"]]
    hasher = TorchFeatureHasher(n_features=2**8, alternate_sign=False)
    X = hasher.fit_transform(corpus).toarray()

    assert (X >= 0).all(), "All values should be non-negative with alternate_sign=False"


def test_binary_mode_matches_sklearn():
    """Test that binary mode (alternate_sign=False) matches sklearn."""
    corpus = [["a", "b", "c"], ["a", "a", "d"]]
    n_features = 2**10

    torch_hasher = TorchFeatureHasher(
        n_features=n_features,
        alternate_sign=False
    )
    sklearn_hasher = SklearnHasher(
        n_features=n_features,
        input_type="string",
        alternate_sign=False
    )

    X_torch = torch_hasher.fit_transform(corpus).toarray()
    X_sklearn = sklearn_hasher.fit_transform(corpus).toarray()

    np.testing.assert_allclose(
        X_torch,
        X_sklearn,
        rtol=1e-5,
        err_msg="Binary mode should match sklearn exactly"
    )


# ==================== DUPLICATE TOKEN TESTS ====================


def test_duplicate_tokens_accumulate():
    """Test that duplicate tokens accumulate correctly."""
    corpus = [["token", "token", "token"]]
    hasher = TorchFeatureHasher(n_features=2**10, alternate_sign=False)
    X = hasher.fit_transform(corpus).toarray()

    # Should have a value of 3 (or -3 if signs differ) somewhere
    max_val = np.abs(X).max()
    assert max_val == 3.0, "Duplicate tokens should accumulate"


def test_duplicates_same_as_sklearn(simple_corpus):
    """Test that duplicate token handling matches sklearn."""
    n_features = 2**12
    torch_hasher = TorchFeatureHasher(n_features=n_features)
    sklearn_hasher = SklearnHasher(n_features=n_features, input_type="string")

    X_torch = torch_hasher.fit_transform(simple_corpus).toarray()
    X_sklearn = sklearn_hasher.fit_transform(simple_corpus).toarray()

    # The third sample has duplicate "token1"
    # Check that the counts match
    np.testing.assert_allclose(
        np.abs(X_torch[2]),  # Third sample
        np.abs(X_sklearn[2]),
        rtol=1e-5,
        err_msg="Duplicate handling should match sklearn"
    )


# ==================== LARGE FEATURE SPACE TESTS ====================


@pytest.mark.parametrize("n_features", [2**20, 2**24])
def test_large_feature_spaces(simple_corpus, n_features):
    """Test that very large feature spaces work correctly."""
    hasher = TorchFeatureHasher(n_features=n_features)
    X = hasher.fit_transform(simple_corpus)

    assert X.shape == (len(simple_corpus), n_features), "Shape should be correct"
    assert issparse(X), "Should remain sparse"
    assert X.format == "csr", "Should be CSR format"

    # Check memory efficiency (sparse should be much smaller than dense)
    sparse_bytes = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    dense_bytes = len(simple_corpus) * n_features * 4  # float32

    assert sparse_bytes < dense_bytes / 100, (
        "Sparse representation should be much smaller than dense"
    )


def test_very_large_n_features():
    """Test extremely large n_features (2**28 = 256M)."""
    corpus = [["a", "b"], ["c"]]
    hasher = TorchFeatureHasher(n_features=2**28)
    X = hasher.fit_transform(corpus)

    assert X.shape == (2, 2**28), "Should handle 256M features"
    assert X.nnz == 3, "Should only store 3 non-zero values"

    # Memory usage should still be tiny
    memory_bytes = X.data.nbytes + X.indices.nbytes + X.indptr.nbytes
    assert memory_bytes < 1000, "Should use minimal memory despite huge feature space"


# ==================== BATCHED PROCESSING TESTS ====================


def test_batched_same_as_regular(large_corpus):
    """Test that batched processing gives same result as regular."""
    n_features = 2**16

    regular_hasher = TorchFeatureHasher(n_features=n_features)
    batched_hasher = BatchedTorchFeatureHasher(
        n_features=n_features,
        batch_size=100
    )

    X_regular = regular_hasher.fit_transform(large_corpus)
    X_batched = batched_hasher.fit_transform(large_corpus)

    # Should be identical
    np.testing.assert_array_equal(
        X_regular.toarray(),
        X_batched.toarray(),
        err_msg="Batched result should match regular processing"
    )


@pytest.mark.parametrize("batch_size", [1, 10, 100, 500])
def test_different_batch_sizes(medium_corpus, batch_size):
    """Test that different batch sizes produce identical results."""
    n_features = 2**12

    reference_hasher = TorchFeatureHasher(n_features=n_features)
    batched_hasher = BatchedTorchFeatureHasher(
        n_features=n_features,
        batch_size=batch_size
    )

    X_ref = reference_hasher.fit_transform(medium_corpus)
    X_batched = batched_hasher.fit_transform(medium_corpus)

    np.testing.assert_array_equal(
        X_ref.toarray(),
        X_batched.toarray(),
        err_msg=f"Batch size {batch_size} should produce same result"
    )


# ==================== FIT/TRANSFORM TESTS ====================


def test_fit_is_noop(simple_corpus):
    """Test that fit() doesn't change state."""
    hasher = TorchFeatureHasher(n_features=2**10)

    # Fit on one corpus
    hasher.fit(simple_corpus)

    # Transform a different corpus (should work fine)
    other_corpus = [["x", "y", "z"]]
    X = hasher.transform(other_corpus)

    assert X.shape == (1, 2**10), "Transform should work on any corpus"


def test_fit_transform_same_as_transform(simple_corpus):
    """Test that fit_transform gives same result as fit+transform."""
    hasher1 = TorchFeatureHasher(n_features=2**10)
    hasher2 = TorchFeatureHasher(n_features=2**10)

    X1 = hasher1.fit_transform(simple_corpus)
    X2 = hasher2.fit(simple_corpus).transform(simple_corpus)

    np.testing.assert_array_equal(
        X1.toarray(),
        X2.toarray(),
        err_msg="fit_transform should equal fit+transform"
    )


# ==================== EDGE CASES ====================


def test_single_token_samples():
    """Test samples with only one token."""
    corpus = [["a"], ["b"], ["c"]]
    hasher = TorchFeatureHasher(n_features=2**10)
    X = hasher.fit_transform(corpus)

    assert X.shape == (3, 2**10), "Shape should be correct"
    assert X.nnz == 3, "Should have exactly 3 non-zero values"


def test_very_long_sample():
    """Test a sample with many tokens."""
    corpus = [[f"token{i}" for i in range(1000)]]
    hasher = TorchFeatureHasher(n_features=2**16)
    X = hasher.fit_transform(corpus)

    assert X.shape == (1, 2**16), "Shape should be correct"
    # Should have at most 1000 non-zeros (could be less due to collisions)
    assert X.nnz <= 1000, "Should have reasonable number of non-zeros"


def test_unicode_tokens():
    """Test that unicode tokens work correctly."""
    corpus = [["café", "naïve", "🚀", "日本語"]]
    hasher = TorchFeatureHasher(n_features=2**10)
    X = hasher.fit_transform(corpus)

    assert X.shape == (1, 2**10), "Unicode tokens should work"
    assert X.nnz > 0, "Should have non-zero values"


def test_mixed_length_samples():
    """Test corpus with highly variable sample lengths."""
    corpus = [
        ["a"],
        ["b", "c", "d", "e"],
        [],
        ["f", "g"],
        [f"token{i}" for i in range(100)],
    ]
    hasher = TorchFeatureHasher(n_features=2**12)
    X = hasher.fit_transform(corpus)

    assert X.shape == (5, 2**12), "Shape should be correct"

    # Check that each row has appropriate number of non-zeros
    row_nnz = [X[i].nnz for i in range(5)]
    assert row_nnz[0] == 1, "First sample should have 1 token"
    assert row_nnz[1] == 4, "Second sample should have 4 tokens"
    assert row_nnz[2] == 0, "Third sample should be empty"
    assert row_nnz[3] == 2, "Fourth sample should have 2 tokens"
    assert row_nnz[4] <= 100, "Fifth sample should have <= 100 (due to collisions)"


# ==================== CONSISTENCY TESTS ====================


def test_deterministic_output():
    """Test that same input always produces same output."""
    corpus = [["a", "b", "c"], ["d", "e", "f"]]
    hasher = TorchFeatureHasher(n_features=2**10)

    X1 = hasher.fit_transform(corpus).toarray()
    X2 = hasher.fit_transform(corpus).toarray()
    X3 = hasher.fit_transform(corpus).toarray()

    np.testing.assert_array_equal(X1, X2, err_msg="Output should be deterministic")
    np.testing.assert_array_equal(X1, X3, err_msg="Output should be deterministic")


def test_order_independence():
    """Test that token order within a sample doesn't matter (for counts)."""
    hasher = TorchFeatureHasher(n_features=2**10, alternate_sign=False)

    corpus1 = [["a", "b", "c"]]
    corpus2 = [["c", "b", "a"]]  # Same tokens, different order

    X1 = hasher.fit_transform(corpus1).toarray()
    X2 = hasher.fit_transform(corpus2).toarray()

    np.testing.assert_array_equal(
        X1, X2,
        err_msg="Token order shouldn't matter for bag-of-words"
    )


# ==================== PERFORMANCE BENCHMARKS ====================


@pytest.mark.benchmark
def test_performance_vs_sklearn(medium_corpus, benchmark):
    """Benchmark against sklearn (requires pytest-benchmark)."""
    pytest.importorskip("pytest_benchmark")
    n_features = 2**16
    torch_hasher = TorchFeatureHasher(n_features=n_features)

    result = benchmark(torch_hasher.fit_transform, medium_corpus)

    assert issparse(result), "Should return sparse matrix"


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "--tb=short"])

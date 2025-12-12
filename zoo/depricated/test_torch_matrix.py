"""
Test suite for TorchSparseMatrix to ensure compatibility with sklearn/scipy.

Tests that our PyTorch implementation produces the same results as
sklearn.metrics.pairwise and scipy operations.
"""

import pytest
import numpy as np
import torch
from scipy.sparse import csr_matrix, coo_matrix, issparse
from sklearn.metrics.pairwise import (
    cosine_similarity as sklearn_cosine_similarity,
    euclidean_distances as sklearn_euclidean_distances,
    manhattan_distances as sklearn_manhattan_distances,
)
from sklearn.preprocessing import normalize as sklearn_normalize
from sklearn.feature_extraction.text import TfidfTransformer

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ariel_experiments.characterize.canonical.core.analysis.torch_matrix import TorchSparseMatrix, BatchTorchSparseMatrix


# ==================== FIXTURES ====================


@pytest.fixture
def simple_sparse_matrix():
    """Simple sparse matrix for basic tests."""
    return csr_matrix([[1, 0, 2, 0], [0, 0, 3, 0], [4, 0, 0, 0], [0, 5, 0, 6]])


@pytest.fixture
def simple_dense_matrix():
    """Simple dense matrix for comparison."""
    return np.array([[1, 0, 2, 0], [0, 0, 3, 0], [4, 0, 0, 0], [0, 5, 0, 6]])


@pytest.fixture
def doc_term_matrix():
    """Document-term matrix for TF-IDF tests."""
    return csr_matrix(
        [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 1, 1]]  # doc 1  # doc 2  # doc 3
    )


@pytest.fixture
def medium_sparse_matrix():
    """Medium-sized sparse matrix."""
    np.random.seed(42)
    dense = np.random.rand(100, 500)
    dense[dense < 0.9] = 0  # Make it sparse
    return csr_matrix(dense)


# ==================== INITIALIZATION TESTS ====================


def test_init_from_scipy_sparse(simple_sparse_matrix):
    """Test initialization from scipy sparse matrix."""
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")

    assert matrix.shape == simple_sparse_matrix.shape
    assert matrix.is_sparse
    assert matrix.device == torch.device("cpu")


def test_init_from_numpy(simple_dense_matrix):
    """Test initialization from numpy array."""
    matrix = TorchSparseMatrix(simple_dense_matrix, device="cpu")

    assert matrix.shape == simple_dense_matrix.shape
    assert matrix.is_sparse


def test_init_from_torch_tensor():
    """Test initialization from torch tensor."""
    tensor = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], dtype=torch.float32)
    matrix = TorchSparseMatrix(tensor, device="cpu")

    assert matrix.shape == (2, 3)
    assert matrix.is_sparse


def test_device_auto_detection():
    """Test automatic device detection."""
    matrix = TorchSparseMatrix(csr_matrix([[1, 2], [3, 4]]))

    # Should select best available device
    assert matrix.device.type in ["cpu", "cuda", "mps"]


# ==================== PROPERTY TESTS ====================


def test_nnz_property(simple_sparse_matrix):
    """Test nnz (number of non-zeros) property."""
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")

    assert matrix.nnz == simple_sparse_matrix.nnz
    assert matrix.nnz == 6  # Known value


def test_sparsity_property(simple_sparse_matrix):
    """Test sparsity calculation."""
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")

    expected_sparsity = matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    assert abs(matrix.sparsity - expected_sparsity) < 1e-6


# ==================== CONVERSION TESTS ====================


def test_to_dense(simple_sparse_matrix):
    """Test conversion to dense tensor."""
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")
    dense = matrix.to_dense()

    expected = simple_sparse_matrix.toarray()

    np.testing.assert_array_almost_equal(
        dense.cpu().numpy(),
        expected,
        err_msg="Dense conversion should match scipy"
    )


def test_to_numpy(simple_sparse_matrix):
    """Test conversion to numpy array."""
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")
    numpy_array = matrix.to_numpy()

    expected = simple_sparse_matrix.toarray()

    np.testing.assert_array_almost_equal(
        numpy_array,
        expected,
        err_msg="Numpy conversion should match scipy"
    )


def test_to_scipy(simple_sparse_matrix):
    """Test conversion back to scipy sparse."""
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")
    scipy_matrix = matrix.to_scipy()

    assert issparse(scipy_matrix), "Should return sparse matrix"
    assert scipy_matrix.format == "csr", "Should be CSR format"

    np.testing.assert_array_almost_equal(
        scipy_matrix.toarray(),
        simple_sparse_matrix.toarray(),
        err_msg="Round-trip conversion should preserve values"
    )


def test_round_trip_conversion(simple_sparse_matrix):
    """Test that scipy -> torch -> scipy preserves data."""
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")
    scipy_back = matrix.to_scipy()

    np.testing.assert_array_equal(
        scipy_back.toarray(),
        simple_sparse_matrix.toarray(),
        err_msg="Round-trip should be lossless"
    )


# ==================== NORMALIZATION TESTS ====================


@pytest.mark.parametrize("norm", ["l1", "l2", "max"])
def test_normalize_matches_sklearn(simple_sparse_matrix, norm):
    """Test that normalization matches sklearn."""
    # Our implementation
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")
    normalized = matrix.normalize(norm=norm, axis=1)

    # Sklearn
    sklearn_normalized = sklearn_normalize(
        simple_sparse_matrix,
        norm=norm,
        axis=1
    )

    np.testing.assert_array_almost_equal(
        normalized.to_numpy(),
        sklearn_normalized.toarray(),
        decimal=5,
        err_msg=f"{norm} normalization should match sklearn"
    )


def test_l2_normalize_unit_vectors():
    """Test that L2 normalization produces unit vectors."""
    matrix = TorchSparseMatrix(csr_matrix([[3, 0, 4], [0, 5, 12]]), device="cpu")
    normalized = matrix.normalize(norm="l2", axis=1)

    dense = normalized.to_dense()
    norms = torch.norm(dense, p=2, dim=1)

    np.testing.assert_array_almost_equal(
        norms.cpu().numpy(),
        np.ones(2),
        decimal=5,
        err_msg="L2 normalized rows should have unit norm"
    )


def test_normalize_zero_rows():
    """Test that normalization handles zero rows correctly."""
    matrix = TorchSparseMatrix(csr_matrix([[1, 2], [0, 0], [3, 4]]), device="cpu")
    normalized = matrix.normalize(norm="l2", axis=1)

    # Zero row should remain zero
    row1 = normalized.to_dense()[1].cpu().numpy()
    assert np.allclose(row1, 0), "Zero row should remain zero after normalization"


# ==================== TF-IDF TESTS ====================


def test_tfidf_structure(doc_term_matrix):
    """Test that TF-IDF maintains correct structure."""
    matrix = TorchSparseMatrix(doc_term_matrix, device="cpu")
    tfidf = matrix.tfidf(norm="l2", smooth_idf=True)

    assert tfidf.shape == matrix.shape, "Shape should be preserved"
    assert tfidf.is_sparse, "Should remain sparse"


def test_tfidf_matches_sklearn(doc_term_matrix):
    """Test that TF-IDF matches sklearn's TfidfTransformer."""
    # Our implementation
    matrix = TorchSparseMatrix(doc_term_matrix, device="cpu")
    our_tfidf = matrix.tfidf(norm="l2", smooth_idf=True)

    # Sklearn
    transformer = TfidfTransformer(norm="l2", smooth_idf=True, sublinear_tf=False)
    sklearn_tfidf = transformer.fit_transform(doc_term_matrix)

    np.testing.assert_array_almost_equal(
        our_tfidf.to_numpy(),
        sklearn_tfidf.toarray(),
        decimal=5,
        err_msg="TF-IDF should match sklearn"
    )


def test_tfidf_without_smoothing(doc_term_matrix):
    """Test TF-IDF without IDF smoothing."""
    matrix = TorchSparseMatrix(doc_term_matrix, device="cpu")
    tfidf = matrix.tfidf(norm="l2", smooth_idf=False)

    # Compare with sklearn
    transformer = TfidfTransformer(norm="l2", smooth_idf=False, sublinear_tf=False)
    sklearn_tfidf = transformer.fit_transform(doc_term_matrix)

    np.testing.assert_array_almost_equal(
        tfidf.to_numpy(),
        sklearn_tfidf.toarray(),
        decimal=5,
        err_msg="TF-IDF without smoothing should match sklearn"
    )


def test_tfidf_no_normalization(doc_term_matrix):
    """Test TF-IDF without output normalization."""
    matrix = TorchSparseMatrix(doc_term_matrix, device="cpu")
    tfidf = matrix.tfidf(norm=None, smooth_idf=True)

    # Compare with sklearn
    transformer = TfidfTransformer(norm=None, smooth_idf=True, sublinear_tf=False)
    sklearn_tfidf = transformer.fit_transform(doc_term_matrix)

    np.testing.assert_array_almost_equal(
        tfidf.to_numpy(),
        sklearn_tfidf.toarray(),
        decimal=5,
        err_msg="TF-IDF without norm should match sklearn"
    )


# ==================== COSINE SIMILARITY TESTS ====================


def test_cosine_similarity_self(simple_sparse_matrix):
    """Test self cosine similarity matches sklearn."""
    # Our implementation
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")
    our_sim = matrix.cosine_similarity()

    # Sklearn
    sklearn_sim = sklearn_cosine_similarity(simple_sparse_matrix)

    np.testing.assert_array_almost_equal(
        our_sim.cpu().numpy(),
        sklearn_sim,
        decimal=5,
        err_msg="Cosine similarity should match sklearn"
    )


def test_cosine_similarity_cross():
    """Test cross cosine similarity between two matrices."""
    X = csr_matrix([[1, 0, 2], [0, 3, 0]])
    Y = csr_matrix([[1, 0, 1], [0, 1, 0]])

    # Our implementation
    matrix_X = TorchSparseMatrix(X, device="cpu")
    matrix_Y = TorchSparseMatrix(Y, device="cpu")
    our_sim = matrix_X.cosine_similarity(matrix_Y)

    # Sklearn
    sklearn_sim = sklearn_cosine_similarity(X, Y)

    np.testing.assert_array_almost_equal(
        our_sim.cpu().numpy(),
        sklearn_sim,
        decimal=5,
        err_msg="Cross cosine similarity should match sklearn"
    )


def test_cosine_similarity_diagonal_ones(simple_sparse_matrix):
    """Test that self-similarity diagonal is all ones."""
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")
    similarity = matrix.cosine_similarity()

    diagonal = torch.diag(similarity).cpu().numpy()

    np.testing.assert_array_almost_equal(
        diagonal,
        np.ones(simple_sparse_matrix.shape[0]),
        decimal=5,
        err_msg="Diagonal should be ones (self-similarity = 1)"
    )


def test_cosine_similarity_symmetric(simple_sparse_matrix):
    """Test that similarity matrix is symmetric."""
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")
    similarity = matrix.cosine_similarity()

    similarity_np = similarity.cpu().numpy()

    np.testing.assert_array_almost_equal(
        similarity_np,
        similarity_np.T,
        decimal=5,
        err_msg="Similarity matrix should be symmetric"
    )


# ==================== EUCLIDEAN DISTANCE TESTS ====================


def test_euclidean_distances_matches_sklearn(simple_sparse_matrix):
    """Test that Euclidean distances match sklearn."""
    # Our implementation
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")
    our_dist = matrix.euclidean_distances()

    # Sklearn
    sklearn_dist = sklearn_euclidean_distances(simple_sparse_matrix)

    np.testing.assert_array_almost_equal(
        our_dist.cpu().numpy(),
        sklearn_dist,
        decimal=4,
        err_msg="Euclidean distances should match sklearn"
    )


def test_euclidean_distances_squared():
    """Test squared Euclidean distances."""
    X = csr_matrix([[1, 0, 2], [0, 3, 0]])

    # Our implementation
    matrix = TorchSparseMatrix(X, device="cpu")
    our_dist_sq = matrix.euclidean_distances(squared=True)

    # Sklearn
    sklearn_dist = sklearn_euclidean_distances(X)
    sklearn_dist_sq = sklearn_dist**2

    np.testing.assert_array_almost_equal(
        our_dist_sq.cpu().numpy(),
        sklearn_dist_sq,
        decimal=4,
        err_msg="Squared distances should match sklearn"
    )


def test_euclidean_distances_diagonal_zeros(simple_sparse_matrix):
    """Test that self-distance diagonal is all zeros."""
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")
    distances = matrix.euclidean_distances()

    diagonal = torch.diag(distances).cpu().numpy()

    np.testing.assert_array_almost_equal(
        diagonal,
        np.zeros(simple_sparse_matrix.shape[0]),
        decimal=5,
        err_msg="Diagonal should be zeros (self-distance = 0)"
    )


# ==================== MANHATTAN DISTANCE TESTS ====================


def test_manhattan_distances_matches_sklearn(simple_sparse_matrix):
    """Test that Manhattan distances match sklearn."""
    # Our implementation
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")
    our_dist = matrix.manhattan_distances()

    # Sklearn
    sklearn_dist = sklearn_manhattan_distances(simple_sparse_matrix)

    np.testing.assert_array_almost_equal(
        our_dist.cpu().numpy(),
        sklearn_dist,
        decimal=4,
        err_msg="Manhattan distances should match sklearn"
    )


def test_manhattan_distances_cross():
    """Test cross Manhattan distances."""
    X = csr_matrix([[1, 0, 2], [0, 3, 0]])
    Y = csr_matrix([[1, 0, 1], [0, 1, 0]])

    # Our implementation
    matrix_X = TorchSparseMatrix(X, device="cpu")
    matrix_Y = TorchSparseMatrix(Y, device="cpu")
    our_dist = matrix_X.manhattan_distances(matrix_Y)

    # Sklearn
    sklearn_dist = sklearn_manhattan_distances(X, Y)

    np.testing.assert_array_almost_equal(
        our_dist.cpu().numpy(),
        sklearn_dist,
        decimal=4,
        err_msg="Cross Manhattan distances should match sklearn"
    )


# ==================== JACCARD SIMILARITY TESTS ====================


def test_jaccard_similarity_binary():
    """Test Jaccard similarity with binary data."""
    X = csr_matrix([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0]])

    matrix = TorchSparseMatrix(X, device="cpu")
    jaccard = matrix.jaccard_similarity(binary=True)

    # Manual calculation for verification
    # Sample 0: {0, 2}, Sample 1: {1, 2}, Sample 2: {0, 1}
    # Jaccard(0,1) = |{2}| / |{0,1,2}| = 1/3
    # Jaccard(0,2) = |{0}| / |{0,1,2}| = 1/3
    # Jaccard(1,2) = |{1}| / |{0,1,2}| = 1/3

    jaccard_np = jaccard.cpu().numpy()

    assert abs(jaccard_np[0, 1] - 1/3) < 0.01, "Jaccard should be ~1/3"
    assert abs(jaccard_np[0, 2] - 1/3) < 0.01, "Jaccard should be ~1/3"
    assert abs(jaccard_np[1, 2] - 1/3) < 0.01, "Jaccard should be ~1/3"


def test_jaccard_similarity_diagonal_ones():
    """Test that self-Jaccard is 1.0."""
    X = csr_matrix([[1, 0, 1], [0, 1, 1]])

    matrix = TorchSparseMatrix(X, device="cpu")
    jaccard = matrix.jaccard_similarity(binary=True)

    diagonal = torch.diag(jaccard).cpu().numpy()

    np.testing.assert_array_almost_equal(
        diagonal,
        np.ones(2),
        decimal=5,
        err_msg="Self-Jaccard should be 1.0"
    )


# ==================== MATRIX OPERATIONS TESTS ====================


def test_transpose():
    """Test matrix transpose."""
    X = csr_matrix([[1, 2, 3], [4, 5, 6]])

    matrix = TorchSparseMatrix(X, device="cpu")
    transposed = matrix.transpose()

    expected = X.T.toarray()

    np.testing.assert_array_equal(
        transposed.to_numpy(),
        expected,
        err_msg="Transpose should match scipy"
    )


def test_add():
    """Test element-wise addition."""
    X = csr_matrix([[1, 0, 2], [0, 3, 0]])
    Y = csr_matrix([[0, 1, 0], [2, 0, 1]])

    matrix_X = TorchSparseMatrix(X, device="cpu")
    matrix_Y = TorchSparseMatrix(Y, device="cpu")

    result = matrix_X.add(matrix_Y)

    expected = (X + Y).toarray()

    np.testing.assert_array_equal(
        result.to_numpy(),
        expected,
        err_msg="Addition should match scipy"
    )


def test_multiply():
    """Test scalar multiplication."""
    X = csr_matrix([[1, 0, 2], [0, 3, 0]])

    matrix = TorchSparseMatrix(X, device="cpu")
    result = matrix.multiply(2.5)

    expected = (X * 2.5).toarray()

    np.testing.assert_array_almost_equal(
        result.to_numpy(),
        expected,
        decimal=5,
        err_msg="Scalar multiplication should match scipy"
    )


# ==================== BATCHED OPERATIONS TESTS ====================


def test_batched_cosine_same_as_regular(medium_sparse_matrix):
    """Test that batched cosine similarity matches regular."""
    matrix = TorchSparseMatrix(medium_sparse_matrix, device="cpu")

    # Regular
    regular_sim = matrix.cosine_similarity()

    # Batched
    batched = BatchTorchSparseMatrix(matrix, batch_size=10)
    batched_sim = batched.cosine_similarity_batched()

    np.testing.assert_array_almost_equal(
        regular_sim.cpu().numpy(),
        batched_sim.cpu().numpy(),
        decimal=4,
        err_msg="Batched should match regular"
    )


@pytest.mark.parametrize("batch_size", [1, 10, 50, 200])
def test_different_batch_sizes(medium_sparse_matrix, batch_size):
    """Test that different batch sizes give same result."""
    matrix = TorchSparseMatrix(medium_sparse_matrix, device="cpu")

    reference_sim = matrix.cosine_similarity()

    batched = BatchTorchSparseMatrix(matrix, batch_size=batch_size)
    batched_sim = batched.cosine_similarity_batched()

    np.testing.assert_array_almost_equal(
        reference_sim.cpu().numpy(),
        batched_sim.cpu().numpy(),
        decimal=4,
        err_msg=f"Batch size {batch_size} should match reference"
    )


# ==================== EDGE CASES ====================


def test_empty_matrix():
    """Test handling of empty matrix."""
    X = csr_matrix((0, 5))  # 0 rows, 5 columns

    matrix = TorchSparseMatrix(X, device="cpu")

    assert matrix.shape == (0, 5)
    assert matrix.nnz == 0
    assert matrix.sparsity == 0.0


def test_all_zeros_matrix():
    """Test matrix with all zeros."""
    X = csr_matrix((3, 4))  # All zeros

    matrix = TorchSparseMatrix(X, device="cpu")

    # Cosine similarity should handle gracefully
    similarity = matrix.cosine_similarity()

    # Should be all zeros (or NaN, but we handle it)
    assert similarity.shape == (3, 3)


def test_single_row_matrix():
    """Test matrix with single row."""
    X = csr_matrix([[1, 2, 3, 4, 5]])

    matrix = TorchSparseMatrix(X, device="cpu")

    similarity = matrix.cosine_similarity()

    assert similarity.shape == (1, 1)
    assert abs(similarity[0, 0].item() - 1.0) < 1e-5


# ==================== GPU TESTS (conditional) ====================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_produces_same_as_cpu(simple_sparse_matrix):
    """Test that GPU produces same results as CPU."""
    matrix_cpu = TorchSparseMatrix(simple_sparse_matrix, device="cpu")
    matrix_gpu = TorchSparseMatrix(simple_sparse_matrix, device="cuda")

    sim_cpu = matrix_cpu.cosine_similarity()
    sim_gpu = matrix_gpu.cosine_similarity()

    np.testing.assert_array_almost_equal(
        sim_cpu.cpu().numpy(),
        sim_gpu.cpu().numpy(),
        decimal=4,
        err_msg="GPU should match CPU"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "--tb=short"])

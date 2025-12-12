"""
TorchSparseMatrix: GPU-accelerated sparse matrix operations for ML pipelines.

This module provides a PyTorch-based sparse matrix class with common operations
like cosine similarity, TF-IDF, and more - all with GPU acceleration.
"""

import torch
import numpy as np
from typing import Literal


class TorchSparseMatrix:
    """
    GPU-accelerated sparse matrix with ML operations.

    Wraps PyTorch sparse tensors (CSR format preferred) and provides
    common operations like cosine similarity, TF-IDF, normalization, etc.

    Parameters:
        data: Can be:
            - scipy.sparse matrix (csr, coo, etc.)
            - torch.Tensor (sparse or dense)
            - numpy array (will be converted to sparse if mostly zeros)
        device: torch device ('cpu', 'cuda', 'mps', etc.). Default: auto-detect.
        dtype: torch dtype. Default: torch.float32.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> X = csr_matrix([[1, 0, 2], [0, 0, 3]])
        >>> matrix = TorchSparseMatrix(X, device='cuda')
        >>> similarity = matrix.cosine_similarity()
        >>> print(similarity.shape)  # (2, 2)
    """

    def __init__(
        self,
        data,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.dtype = dtype

        # Convert input to sparse tensor
        self.tensor = self._to_sparse_tensor(data)
        self.shape = tuple(self.tensor.shape)

    def _to_sparse_tensor(self, data) -> torch.Tensor:
        """Convert various input formats to PyTorch sparse tensor."""
        # Already a torch tensor
        if isinstance(data, torch.Tensor):
            if data.is_sparse:
                return data.to(self.device).coalesce()
            else:
                # Convert dense to sparse
                return data.to_sparse().to(self.device).coalesce()

        # Scipy sparse matrix
        try:
            from scipy.sparse import issparse, csr_matrix

            if issparse(data):
                # Convert to COO for PyTorch
                coo = data.tocoo()
                indices = torch.from_numpy(
                    np.vstack([coo.row, coo.col]).astype(np.int64)
                )
                values = torch.from_numpy(coo.data.astype(np.float32))

                sparse_tensor = torch.sparse_coo_tensor(
                    indices,
                    values,
                    size=coo.shape,
                    dtype=self.dtype,
                    device=self.device,
                ).coalesce()

                return sparse_tensor
        except ImportError:
            pass

        # Numpy array
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data).to(dtype=self.dtype, device=self.device)
            return tensor.to_sparse().coalesce()

        raise ValueError(f"Unsupported data type: {type(data)}")

    # ==================== PROPERTIES ====================

    @property
    def nnz(self) -> int:
        """Number of non-zero elements."""
        return self.tensor._nnz()

    @property
    def sparsity(self) -> float:
        """Fraction of non-zero elements."""
        total_elements = self.shape[0] * self.shape[1]
        if total_elements == 0:
            return 0.0
        return self.nnz / total_elements

    @property
    def is_sparse(self) -> bool:
        """Check if underlying tensor is sparse."""
        return self.tensor.is_sparse

    # ==================== CONVERSION METHODS ====================

    def to_dense(self) -> torch.Tensor:
        """Convert to dense tensor. Warning: Can use lots of memory!"""
        return self.tensor.to_dense()

    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array (moves to CPU, densifies)."""
        if self.tensor.is_sparse:
            return self.tensor.to_dense().cpu().numpy()
        return self.tensor.cpu().numpy()

    def to_scipy(self):
        """Convert to scipy sparse CSR matrix."""
        from scipy.sparse import csr_matrix

        tensor_cpu = self.tensor.cpu().coalesce()
        indices = tensor_cpu.indices().numpy()
        values = tensor_cpu.values().numpy()

        rows = indices[0, :]
        cols = indices[1, :]

        return csr_matrix((values, (rows, cols)), shape=self.shape)

    # ==================== NORMALIZATION ====================

    def normalize(
        self, norm: Literal["l1", "l2", "max"] = "l2", axis: int = 1
    ) -> "TorchSparseMatrix":
        """
        Normalize rows or columns (sparse-optimized).

        Args:
            norm: Normalization type ('l1', 'l2', or 'max')
            axis: 0 for column-wise, 1 for row-wise

        Returns:
            New TorchSparseMatrix with normalized values
        """
        # Work directly with sparse tensor - no dense conversion!
        indices = self.tensor._indices()
        values = self.tensor._values()

        # For row-wise normalization (axis=1)
        if axis == 1:
            row_indices = indices[0]

            if norm == "l2":
                # Compute L2 norm per row
                squared_values = values ** 2
                row_norms_squared = torch.zeros(self.shape[0], device=self.device, dtype=self.dtype)
                row_norms_squared.scatter_add_(0, row_indices, squared_values)
                row_norms = torch.sqrt(row_norms_squared)
            elif norm == "l1":
                # Compute L1 norm per row
                abs_values = torch.abs(values)
                row_norms = torch.zeros(self.shape[0], device=self.device, dtype=self.dtype)
                row_norms.scatter_add_(0, row_indices, abs_values)
            elif norm == "max":
                # Compute max norm per row
                abs_values = torch.abs(values)
                row_norms = torch.zeros(self.shape[0], device=self.device, dtype=self.dtype)
                # For max, we need to use scatter_reduce with 'amax'
                row_norms.scatter_reduce_(0, row_indices, abs_values, reduce='amax')
            else:
                raise ValueError(f"Unknown norm: {norm}")

            # Avoid division by zero
            row_norms = torch.clamp(row_norms, min=1e-12)

            # Normalize values
            normalized_values = values / row_norms[row_indices]

        else:  # axis == 0 (column-wise)
            col_indices = indices[1]

            if norm == "l2":
                squared_values = values ** 2
                col_norms_squared = torch.zeros(self.shape[1], device=self.device, dtype=self.dtype)
                col_norms_squared.scatter_add_(0, col_indices, squared_values)
                col_norms = torch.sqrt(col_norms_squared)
            elif norm == "l1":
                abs_values = torch.abs(values)
                col_norms = torch.zeros(self.shape[1], device=self.device, dtype=self.dtype)
                col_norms.scatter_add_(0, col_indices, abs_values)
            elif norm == "max":
                abs_values = torch.abs(values)
                col_norms = torch.zeros(self.shape[1], device=self.device, dtype=self.dtype)
                col_norms.scatter_reduce_(0, col_indices, abs_values, reduce='amax')
            else:
                raise ValueError(f"Unknown norm: {norm}")

            col_norms = torch.clamp(col_norms, min=1e-12)
            normalized_values = values / col_norms[col_indices]

        # Create new sparse tensor with normalized values
        normalized_tensor = torch.sparse_coo_tensor(
            indices, normalized_values, self.shape,
            device=self.device, dtype=self.dtype
        ).coalesce()

        return TorchSparseMatrix(normalized_tensor, device=self.device, dtype=self.dtype)

    # ==================== TF-IDF ====================

    def tfidf(
        self, norm: Literal["l1", "l2", None] = "l2", smooth_idf: bool = True
    ) -> "TorchSparseMatrix":
        """
        Apply TF-IDF transformation matching sklearn's TfidfTransformer.

        Formula (matching sklearn):
        - IDF = log((n_samples + 1) / (df + 1)) + 1  (if smooth_idf=True)
        - IDF = log(n_samples / df) + 1              (if smooth_idf=False)
        - TF-IDF = TF * IDF (where TF is raw term count)
        - Then normalize if requested

        Args:
            norm: Normalize output ('l1', 'l2', or None)
            smooth_idf: Add 1 to document frequencies (prevents zero divisions)

        Returns:
            New TorchSparseMatrix with TF-IDF weighted values
        """
        dense = self.tensor.to_dense()  # (n_samples, n_features)
        n_samples = self.shape[0]

        # Step 1: TF is just the raw counts (sklearn doesn't normalize by doc length)
        tf = dense

        # Step 2: Document Frequency (how many docs contain each term)
        df = (dense != 0).sum(dim=0).float()  # (n_features,)

        # Step 3: IDF calculation (matching sklearn exactly)
        if smooth_idf:
            # sklearn formula with smoothing
            idf = torch.log((n_samples + 1.0) / (df + 1.0)) + 1.0
        else:
            # sklearn formula without smoothing
            idf = torch.log(n_samples / (df + 1e-12)) + 1.0

        # Step 4: TF-IDF = TF * IDF
        tfidf = tf * idf.unsqueeze(0)  # Broadcast IDF across samples

        # Step 5: Normalize (optional)
        if norm is not None:
            norms = torch.norm(tfidf, p=2 if norm == "l2" else 1, dim=1, keepdim=True)
            norms = torch.clamp(norms, min=1e-12)
            tfidf = tfidf / norms

        return TorchSparseMatrix(tfidf, device=self.device, dtype=self.dtype)

    # ==================== SIMILARITY METRICS ====================

    def cosine_similarity(self, other: "TorchSparseMatrix | None" = None) -> torch.Tensor:
        """
        Compute pairwise cosine similarity (sparse-optimized).

        Args:
            other: Another TorchSparseMatrix. If None, computes self-similarity.

        Returns:
            Dense similarity matrix of shape:
            - (n_samples, n_samples) if other is None
            - (n_samples_self, n_samples_other) otherwise
        """
        # Normalize to unit vectors (L2 norm)
        X_normalized = self.normalize(norm="l2", axis=1)

        if other is None:
            # Self-similarity using sparse @ sparse.T
            # This avoids dense conversion during computation
            # Result is sparse but we convert to dense at the end
            similarity_sparse = torch.sparse.mm(X_normalized.tensor, X_normalized.tensor.t())
            similarity = similarity_sparse.to_dense()
        else:
            # Cross-similarity
            Y_normalized = other.normalize(norm="l2", axis=1)
            similarity_sparse = torch.sparse.mm(X_normalized.tensor, Y_normalized.tensor.t())
            similarity = similarity_sparse.to_dense()

        return similarity

    def euclidean_distances(
        self, other: "TorchSparseMatrix | None" = None, squared: bool = False
    ) -> torch.Tensor:
        """
        Compute pairwise Euclidean distances.

        Args:
            other: Another TorchSparseMatrix. If None, computes self-distances.
            squared: If True, return squared distances

        Returns:
            Dense distance matrix
        """
        X = self.to_dense()

        if other is None:
            Y = X
        else:
            Y = other.to_dense()

        # Efficient pairwise distance: ||x - y||^2 = ||x||^2 + ||y||^2 - 2*x*y
        X_norm_sq = (X**2).sum(dim=1, keepdim=True)  # (n, 1)
        Y_norm_sq = (Y**2).sum(dim=1, keepdim=True)  # (m, 1)

        distances_sq = X_norm_sq + Y_norm_sq.T - 2 * torch.mm(X, Y.T)

        # Clamp to avoid numerical issues with sqrt
        distances_sq = torch.clamp(distances_sq, min=0.0)

        if squared:
            return distances_sq
        else:
            return torch.sqrt(distances_sq)

    def manhattan_distances(
        self, other: "TorchSparseMatrix | None" = None
    ) -> torch.Tensor:
        """
        Compute pairwise Manhattan (L1) distances.

        Args:
            other: Another TorchSparseMatrix. If None, computes self-distances.

        Returns:
            Dense distance matrix
        """
        X = self.to_dense()

        if other is None:
            Y = X
        else:
            Y = other.to_dense()

        # Pairwise L1 distance
        distances = torch.cdist(X, Y, p=1)
        return distances

    def jaccard_similarity(
        self, other: "TorchSparseMatrix | None" = None, binary: bool = True
    ) -> torch.Tensor:
        """
        Compute Jaccard similarity (set overlap).

        Args:
            other: Another TorchSparseMatrix. If None, computes self-similarity.
            binary: If True, treat as binary (presence/absence).
                   If False, use actual values.

        Returns:
            Dense similarity matrix with values in [0, 1]
        """
        X = self.to_dense()

        if binary:
            X = (X != 0).float()

        if other is None:
            Y = X
        else:
            Y = other.to_dense()
            if binary:
                Y = (Y != 0).float()

        # Jaccard = |A ∩ B| / |A ∪ B|
        intersection = torch.mm(X, Y.T)  # Element-wise AND (for binary)

        X_sum = X.sum(dim=1, keepdim=True)  # |A|
        Y_sum = Y.sum(dim=1, keepdim=True)  # |B|

        union = X_sum + Y_sum.T - intersection  # |A ∪ B|

        # Avoid division by zero
        jaccard = intersection / torch.clamp(union, min=1e-12)

        return jaccard

    # ==================== MATRIX OPERATIONS ====================

    def dot(self, other: "TorchSparseMatrix") -> torch.Tensor:
        """
        Matrix multiplication.

        Args:
            other: Another TorchSparseMatrix

        Returns:
            Dense result of self @ other
        """
        X = self.to_dense()
        Y = other.to_dense()
        return torch.mm(X, Y)

    def transpose(self) -> "TorchSparseMatrix":
        """Transpose the matrix."""
        transposed = self.tensor.t()
        return TorchSparseMatrix(transposed, device=self.device, dtype=self.dtype)

    def add(self, other: "TorchSparseMatrix") -> "TorchSparseMatrix":
        """Element-wise addition."""
        result = self.tensor + other.tensor
        return TorchSparseMatrix(result, device=self.device, dtype=self.dtype)

    def multiply(self, scalar: float) -> "TorchSparseMatrix":
        """Scalar multiplication."""
        result = self.tensor * scalar
        return TorchSparseMatrix(result, device=self.device, dtype=self.dtype)

    # ==================== MAGIC METHODS ====================

    def __repr__(self) -> str:
        return (
            f"TorchSparseMatrix(shape={self.shape}, nnz={self.nnz}, "
            f"sparsity={self.sparsity:.2%}, device={self.device})"
        )

    def __getitem__(self, idx):
        """Index into the matrix."""
        # Convert to dense for indexing (not ideal but simple)
        return self.to_dense()[idx]

    # ==================== SKLEARN COMPATIBILITY ====================

    @classmethod
    def from_sklearn(cls, sklearn_matrix, device=None):
        """
        Create from sklearn transformer output.

        Args:
            sklearn_matrix: Output from FeatureHasher, TfidfVectorizer, etc.
            device: Target device

        Returns:
            TorchSparseMatrix
        """
        return cls(sklearn_matrix, device=device)


# ==================== BATCH OPERATIONS ====================


class BatchTorchSparseMatrix:
    """
    Efficient batched operations on sparse matrices.

    For very large matrices that don't fit in memory, processes in chunks.
    """

    def __init__(
        self,
        matrix: TorchSparseMatrix,
        batch_size: int = 1000,
    ):
        self.matrix = matrix
        self.batch_size = batch_size

    def cosine_similarity_batched(
        self, other: TorchSparseMatrix | None = None
    ) -> torch.Tensor:
        """
        Compute cosine similarity in batches (memory efficient).

        Args:
            other: Another matrix. If None, self-similarity.

        Returns:
            Full similarity matrix (computed in chunks)
        """
        X = self.matrix.normalize(norm="l2", axis=1).to_dense()

        if other is None:
            Y = X
        else:
            Y = other.normalize(norm="l2", axis=1).to_dense()

        n_samples_X = X.shape[0]
        n_samples_Y = Y.shape[0]

        # Pre-allocate result
        similarity = torch.zeros(
            (n_samples_X, n_samples_Y),
            dtype=self.matrix.dtype,
            device=self.matrix.device,
        )

        # Compute in batches
        for i in range(0, n_samples_X, self.batch_size):
            end_i = min(i + self.batch_size, n_samples_X)
            batch_X = X[i:end_i]

            # Compute this batch against all of Y
            similarity[i:end_i, :] = torch.mm(batch_X, Y.T)

        return similarity


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    """
    Example usage of TorchSparseMatrix.
    """
    from scipy.sparse import csr_matrix
    import time

    print("=" * 60)
    print("TorchSparseMatrix Examples")
    print("=" * 60)

    # Example 1: Basic usage
    print("\n1. Basic Usage:")
    print("-" * 60)

    # Create from scipy sparse
    scipy_matrix = csr_matrix([[1, 0, 2, 0], [0, 0, 3, 0], [4, 0, 0, 0], [0, 5, 0, 6]])

    matrix = TorchSparseMatrix(scipy_matrix, device="cpu")
    print(f"Matrix: {matrix}")
    print(f"Shape: {matrix.shape}")
    print(f"Non-zeros: {matrix.nnz}")
    print(f"Sparsity: {matrix.sparsity:.2%}")

    # Example 2: Cosine Similarity
    print("\n2. Cosine Similarity:")
    print("-" * 60)

    similarity = matrix.cosine_similarity()
    print(f"Similarity matrix shape: {similarity.shape}")
    print(f"Similarity matrix:\n{similarity}")
    print(f"Self-similarity on diagonal: {torch.diag(similarity)}")

    # Example 3: TF-IDF Transformation
    print("\n3. TF-IDF Transformation:")
    print("-" * 60)

    # Create a document-term matrix (3 documents, 5 terms)
    doc_term = csr_matrix(
        [[1, 1, 0, 0, 0], [1, 1, 1, 0, 0], [0, 1, 1, 1, 1]]  # doc 1  # doc 2  # doc 3
    )

    matrix_tf = TorchSparseMatrix(doc_term)
    print(f"Original counts:\n{matrix_tf.to_dense()}")

    matrix_tfidf = matrix_tf.tfidf(norm="l2")
    print(f"\nTF-IDF weighted:\n{matrix_tfidf.to_dense()}")

    # Example 4: Different Distance Metrics
    print("\n4. Different Distance Metrics:")
    print("-" * 60)

    # Create two simple matrices
    X = TorchSparseMatrix(csr_matrix([[1, 0, 2], [0, 0, 3]]))
    Y = TorchSparseMatrix(csr_matrix([[1, 0, 1], [0, 1, 0]]))

    cos_sim = X.cosine_similarity(Y)
    print(f"Cosine similarity:\n{cos_sim}")

    euclidean = X.euclidean_distances(Y)
    print(f"\nEuclidean distances:\n{euclidean}")

    manhattan = X.manhattan_distances(Y)
    print(f"\nManhattan distances:\n{manhattan}")

    jaccard = X.jaccard_similarity(Y, binary=True)
    print(f"\nJaccard similarity:\n{jaccard}")

    # Example 5: Normalization
    print("\n5. Normalization:")
    print("-" * 60)

    matrix_original = TorchSparseMatrix(csr_matrix([[3, 0, 4], [0, 5, 12]]))
    print(f"Original:\n{matrix_original.to_dense()}")

    matrix_l2 = matrix_original.normalize(norm="l2", axis=1)
    print(f"\nL2 normalized (rows):\n{matrix_l2.to_dense()}")
    print(f"Row norms: {torch.norm(matrix_l2.to_dense(), p=2, dim=1)}")

    # Example 6: GPU Acceleration (if available)
    if torch.cuda.is_available():
        print("\n6. GPU Acceleration:")
        print("-" * 60)

        # Create large matrix
        large_scipy = csr_matrix(np.random.rand(1000, 10000) > 0.95)

        # CPU timing
        cpu_matrix = TorchSparseMatrix(large_scipy, device="cpu")
        start = time.time()
        cpu_sim = cpu_matrix.cosine_similarity()
        cpu_time = time.time() - start

        # GPU timing
        gpu_matrix = TorchSparseMatrix(large_scipy, device="cuda")
        start = time.time()
        gpu_sim = gpu_matrix.cosine_similarity()
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"Matrix shape: {large_scipy.shape}")
        print(f"CPU time: {cpu_time:.4f}s")
        print(f"GPU time: {gpu_time:.4f}s")
        print(f"Speedup: {cpu_time / gpu_time:.2f}x")

    # Example 7: Batched operations for huge matrices
    print("\n7. Batched Operations (memory efficient):")
    print("-" * 60)

    # Create moderate-sized matrix
    matrix_big = TorchSparseMatrix(csr_matrix(np.random.rand(5000, 1000) > 0.98))

    batched = BatchTorchSparseMatrix(matrix_big, batch_size=500)
    similarity_batched = batched.cosine_similarity_batched()

    print(f"Input shape: {matrix_big.shape}")
    print(f"Similarity shape: {similarity_batched.shape}")
    print(f"Memory efficient: processed in batches of 500")

    # Example 8: Integration with SubtreeAnalyzer
    print("\n8. Integration Example:")
    print("-" * 60)
    print("""
    # In your analyzer.py:

    from tools.torch_hasher import TorchFeatureHasher
    from tools.torch_matrix import TorchSparseMatrix

    # Hash features
    hasher = TorchFeatureHasher(n_features=2**20, device='cuda')
    scipy_matrix = hasher.fit_transform(corpus)

    # Wrap in TorchSparseMatrix for GPU operations
    matrix = TorchSparseMatrix(scipy_matrix, device='cuda')

    # Fast GPU-accelerated similarity
    similarity = matrix.cosine_similarity()  # On GPU!

    # Or TF-IDF then similarity
    tfidf_matrix = matrix.tfidf(norm='l2')
    similarity = tfidf_matrix.cosine_similarity()
    """)

    print("\n" + "=" * 60)
    print("All examples completed successfully!")

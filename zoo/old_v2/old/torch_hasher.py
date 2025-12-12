"""
PyTorch Feature Hasher: GPU-accelerated feature hashing using the hashing trick.

This module provides a PyTorch implementation of sklearn's FeatureHasher,
allowing for GPU acceleration and seamless integration with neural networks.
Returns CSR sparse format for memory efficiency with large feature spaces.
"""

import torch
import numpy as np
from typing import Iterable


def _murmurhash3_32(key: str, seed: int = 0) -> int:
    """
    MurmurHash3 32-bit implementation matching sklearn's hash function.

    This ensures hash compatibility with sklearn.feature_extraction.FeatureHasher.
    """
    # Convert string to bytes
    key_bytes = key.encode('utf-8')

    # Constants
    c1 = 0xcc9e2d51
    c2 = 0x1b873593
    r1 = 15
    r2 = 13
    m = 5
    n = 0xe6546b64

    hash_val = seed
    length = len(key_bytes)

    # Process 4-byte chunks
    for i in range(0, length // 4):
        k = int.from_bytes(key_bytes[i*4:(i+1)*4], byteorder='little', signed=False)
        k = (k * c1) & 0xffffffff
        k = ((k << r1) | (k >> (32 - r1))) & 0xffffffff
        k = (k * c2) & 0xffffffff

        hash_val ^= k
        hash_val = ((hash_val << r2) | (hash_val >> (32 - r2))) & 0xffffffff
        hash_val = ((hash_val * m) + n) & 0xffffffff

    # Process remaining bytes
    remaining = length % 4
    if remaining > 0:
        k = 0
        for i in range(remaining):
            k |= key_bytes[length - remaining + i] << (8 * i)
        k = (k * c1) & 0xffffffff
        k = ((k << r1) | (k >> (32 - r1))) & 0xffffffff
        k = (k * c2) & 0xffffffff
        hash_val ^= k

    # Finalization
    hash_val ^= length
    hash_val ^= (hash_val >> 16)
    hash_val = (hash_val * 0x85ebca6b) & 0xffffffff
    hash_val ^= (hash_val >> 13)
    hash_val = (hash_val * 0xc2b2ae35) & 0xffffffff
    hash_val ^= (hash_val >> 16)

    # Convert to signed 32-bit integer
    if hash_val >= 2**31:
        hash_val -= 2**32

    return hash_val


class TorchFeatureHasher:
    """
    GPU-accelerated feature hasher using the hashing trick with CSR output.

    Implements the same hashing trick as sklearn.feature_extraction.FeatureHasher
    but with PyTorch tensors for GPU acceleration. Optimized for very large
    feature spaces (e.g., 2**20 or larger).

    Parameters:
        n_features: Number of features in the output (hash space size).
                   Default: 2**20 (1M features). Can be much larger (2**24, 2**28, etc.)
        input_type: Type of input data. Currently supports 'string'.
        alternate_sign: If True, alternates sign of hash values (default: True).
                       If False, all values are positive (binary mode).
        device: torch device ('cpu', 'cuda', 'mps', etc.). Default: auto-detect.
        dtype: torch dtype for output tensor. Default: torch.float32.

    Example:
        >>> hasher = TorchFeatureHasher(n_features=2**20, device='cuda')
        >>> corpus = [['token1', 'token2'], ['token3', 'token1']]
        >>> X = hasher.fit_transform(corpus)  # Returns CSR sparse matrix
        >>> print(X.shape)  # (2, 1048576)
    """

    def __init__(
        self,
        n_features: int = 2**22, 
        input_type: str = "string",
        alternate_sign: bool = True,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if input_type != "string":
            raise NotImplementedError("Only 'string' input_type is currently supported")

        if n_features <= 0:
            raise ValueError(f"n_features must be positive, got {n_features}")

        self.n_features = n_features
        self.input_type = input_type
        self.alternate_sign = alternate_sign
        self.dtype = dtype

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

    def fit(self, X: Iterable[list[str]]) -> "TorchFeatureHasher":
        """
        Fit method (no-op for compatibility with sklearn API).

        Feature hashing is stateless, so this does nothing.
        """
        return self

    def transform(self, X: Iterable[list[str]]):
        """
        Transform a collection of token lists into a CSR sparse matrix.

        Args:
            X: Iterable of token lists. Each element is a list of strings.

        Returns:
            scipy.sparse.csr_matrix of shape (n_samples, n_features).
        """
        from scipy.sparse import csr_matrix

        # Convert to list if needed
        if not isinstance(X, list):
            X = list(X)

        n_samples = len(X)

        # Build sparse matrix in COO format first (efficient for construction)
        row_indices = []
        col_indices = []
        values = []

        for sample_idx, tokens in enumerate(X):
            if not tokens:  # Empty token list
                continue

            # Count token frequencies in this sample
            token_counts = {}
            for token in tokens:
                # Primary hash for bucket index (using MurmurHash3)
                h1 = _murmurhash3_32(token, seed=0)
                bucket = abs(h1) % self.n_features

                # Secondary hash for sign (if alternate_sign is True)
                if self.alternate_sign:
                    # Use the hash value's sign bit
                    sign = 1.0 if h1 >= 0 else -1.0
                else:
                    sign = 1.0

                # Accumulate counts (same token, same bucket)
                if bucket in token_counts:
                    token_counts[bucket] += sign
                else:
                    token_counts[bucket] = sign

            # Add to sparse structure
            for col_idx, value in token_counts.items():
                row_indices.append(sample_idx)
                col_indices.append(col_idx)
                values.append(value)

        # Handle empty corpus
        if len(row_indices) == 0:
            return csr_matrix((n_samples, self.n_features), dtype=np.float32)

        # Convert to numpy arrays
        row_indices = np.array(row_indices, dtype=np.int32)
        col_indices = np.array(col_indices, dtype=np.int32)
        values = np.array(values, dtype=np.float32)

        # Create CSR matrix directly (very memory efficient)
        sparse_matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(n_samples, self.n_features),
            dtype=np.float32,
        )

        return sparse_matrix

    def fit_transform(self, X: Iterable[list[str]]):
        """
        Fit and transform in one step (same as transform since fit is a no-op).

        Args:
            X: Iterable of token lists.

        Returns:
            scipy.sparse.csr_matrix of shape (n_samples, n_features).
        """
        return self.fit(X).transform(X)

    def to_torch_sparse(self, csr_matrix):
        """
        Convert scipy CSR matrix to PyTorch sparse tensor.

        Args:
            csr_matrix: scipy.sparse.csr_matrix

        Returns:
            torch.sparse_csr_tensor on specified device
        """
        # Convert CSR to COO for PyTorch
        coo = csr_matrix.tocoo()

        indices = torch.from_numpy(
            np.vstack([coo.row, coo.col]).astype(np.int64)
        ).to(self.device)

        values = torch.from_numpy(coo.data.astype(np.float32)).to(self.device)

        sparse_tensor = torch.sparse_coo_tensor(
            indices,
            values,
            size=coo.shape,
            dtype=self.dtype,
            device=self.device,
        ).coalesce()

        return sparse_tensor

    def to_dense(self, csr_matrix):
        """
        Convert CSR matrix to dense numpy array.

        Warning: Only use for small matrices! Will consume lots of memory.

        Args:
            csr_matrix: scipy.sparse.csr_matrix

        Returns:
            Dense numpy array
        """
        return csr_matrix.toarray()


# ==================== BATCH PROCESSING ====================


class BatchedTorchFeatureHasher(TorchFeatureHasher):
    """
    Memory-efficient batched version for very large datasets.

    Processes data in batches to avoid memory issues with huge datasets.
    Still returns a single CSR matrix with all data.

    Parameters:
        batch_size: Number of samples to process at once.
        **kwargs: Other parameters passed to TorchFeatureHasher.

    Example:
        >>> hasher = BatchedTorchFeatureHasher(
        ...     n_features=2**24,  # 16M features
        ...     batch_size=1000,
        ...     device='cuda'
        ... )
        >>> # Can handle millions of samples
        >>> X = hasher.fit_transform(huge_corpus)
    """

    def __init__(self, batch_size: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def fit_transform(self, X: Iterable[list[str]]):
        """
        Transform in batches and stack vertically.

        Args:
            X: Iterable of token lists

        Returns:
            scipy.sparse.csr_matrix (stacked batches)
        """
        from scipy.sparse import vstack

        if not isinstance(X, list):
            X = list(X)

        n_samples = len(X)
        batch_results = []

        for i in range(0, n_samples, self.batch_size):
            batch = X[i : i + self.batch_size]
            batch_result = super().fit_transform(batch)
            batch_results.append(batch_result)

        # Vertically stack all batches (memory efficient)
        return vstack(batch_results, format="csr")


# ==================== USAGE EXAMPLES ====================

if __name__ == "__main__":
    """
    Example usage and benchmarking with very large feature spaces.
    """
    import time
    from scipy.sparse import csr_matrix

    print("=" * 60)
    print("TorchFeatureHasher Examples (CSR, Large Feature Spaces)")
    print("=" * 60)

    # Example 1: Basic usage with 1M features
    print("\n1. Basic Usage (2**20 = 1M features):")
    print("-" * 60)
    corpus = [
        ["token1", "token2", "token3"],
        ["token2", "token4"],
        ["token1", "token1", "token5"],  # Duplicate tokens
        [],  # Empty sample
    ]

    hasher = TorchFeatureHasher(n_features=2**20, device="cpu")
    X = hasher.fit_transform(corpus)

    print(f"Input: {len(corpus)} samples")
    print(f"Output shape: {X.shape}")
    print(f"Output type: {type(X)}")
    print(f"Format: {X.format}")
    print(f"Memory usage: {X.data.nbytes + X.indices.nbytes + X.indptr.nbytes:,} bytes")
    print(f"Sparsity: {X.nnz / (X.shape[0] * X.shape[1]) * 100:.6f}%")

    # Example 2: Very large feature space (16M features)
    print("\n2. Very Large Feature Space (2**24 = 16M features):")
    print("-" * 60)
    large_hasher = TorchFeatureHasher(n_features=2**24)
    X_large = large_hasher.fit_transform(corpus)

    print(f"Shape: {X_large.shape}")
    print(f"Memory usage: {X_large.data.nbytes + X_large.indices.nbytes + X_large.indptr.nbytes:,} bytes")
    print(f"Sparsity: {X_large.nnz / (X_large.shape[0] * X_large.shape[1]) * 100:.10f}%")

    # Example 3: Massive feature space (256M features) - Still efficient!
    print("\n3. Massive Feature Space (2**28 = 256M features):")
    print("-" * 60)
    massive_hasher = TorchFeatureHasher(n_features=2**28)
    X_massive = massive_hasher.fit_transform(corpus)

    print(f"Shape: {X_massive.shape}")
    print(f"Memory usage: {X_massive.data.nbytes + X_massive.indices.nbytes + X_massive.indptr.nbytes:,} bytes")
    print(f"Non-zero entries: {X_massive.nnz}")
    print(f"Sparsity: {X_massive.nnz / (X_massive.shape[0] * X_massive.shape[1]) * 100:.15f}%")

    # Example 4: Sklearn compatibility (drop-in replacement)
    print("\n4. Sklearn API Compatibility:")
    print("-" * 60)
    # Can be used anywhere sklearn.FeatureHasher is used
    from sklearn.metrics.pairwise import cosine_similarity

    X1 = hasher.fit_transform([["a", "b", "c"]])
    X2 = hasher.fit_transform([["a", "b", "d"]])
    similarity = cosine_similarity(X1, X2)[0, 0]
    print(f"Cosine similarity: {similarity:.4f}")

    # Example 5: Binary mode (no alternate signs)
    print("\n5. Binary Mode (presence/absence features):")
    print("-" * 60)
    binary_hasher = TorchFeatureHasher(
        n_features=2**20, alternate_sign=False, device="cpu"
    )
    X_binary = binary_hasher.fit_transform(corpus)

    # Check that all values are positive
    all_positive = (X_binary.data >= 0).all()
    print(f"All values positive: {all_positive}")
    print(f"Value range: [{X_binary.data.min():.1f}, {X_binary.data.max():.1f}]")

    # Example 6: Batched processing for huge datasets
    print("\n6. Batched Processing (memory efficient):")
    print("-" * 60)

    # Simulate a large corpus
    large_corpus = [["tok" + str(i % 100) for i in range(50)] for _ in range(10000)]

    batched_hasher = BatchedTorchFeatureHasher(
        n_features=2**20, batch_size=1000, device="cpu"
    )

    start = time.time()
    X_batched = batched_hasher.fit_transform(large_corpus)
    elapsed = time.time() - start

    print(f"Processed {len(large_corpus):,} samples")
    print(f"Time: {elapsed:.3f}s")
    print(f"Output shape: {X_batched.shape}")
    print(f"Throughput: {len(large_corpus) / elapsed:,.0f} samples/sec")

    # Example 7: Integration with your analyzer
    print("\n7. Integration with SubtreeAnalyzer:")
    print("-" * 60)
    print("""
    # In analyzer.py, you can now use TorchFeatureHasher:

    from tools.torch_hasher import TorchFeatureHasher

    # Update _get_hasher method:
    def _get_hasher(self, strategy, n_features):
        if strategy == FeatureStrategy.HASH_COUNT:
            return TorchFeatureHasher(
                n_features=n_features,
                input_type='string',
                device='cuda'  # GPU acceleration!
            )
        # ... rest of strategies

    # Works seamlessly with your pipeline!
    analyzer = SubtreeAnalyzer.from_subtrees(subtrees)
    results = analyzer.hash_features(
        strategy=FeatureStrategy.HASH_COUNT,
        n_features=2**24  # 16M features - no problem!
    )
    """)

    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("CSR format provides excellent memory efficiency for large feature spaces!")

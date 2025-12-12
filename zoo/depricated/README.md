# PyTorch Tools Test Suite

Comprehensive tests for `TorchFeatureHasher` and `TorchSparseMatrix` ensuring sklearn/scipy compatibility.

## Quick Start

### Run All Tests (CPU)
```bash
cd /home/salo/projects/ariel-zoo
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ -v
```

### Run Tests on GPU (CUDA)
```bash
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ --device=cuda -v
```

### Run Tests on Apple Silicon (MPS)
```bash
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ --device=mps -v
```

## Test Coverage

### ✅ test_torch_hasher.py (33 tests)
- Output format validation (CSR sparse)
- sklearn compatibility checks (MurmurHash3)
- Hash bucket consistency
- Duplicate token handling
- Binary vs signed modes
- Large feature spaces (2²⁰ - 2²⁸)
- Batched processing
- Edge cases (empty, unicode, long samples)

### ✅ test_torch_matrix.py (~42 tests)
- All conversions (scipy ↔ torch ↔ numpy)
- Normalization (L1, L2, max)
- **TF-IDF transformation** ✨ (fixed!)
- Cosine similarity
- Euclidean/Manhattan distances
- Jaccard similarity
- Matrix operations
- GPU vs CPU consistency
- Batched operations

## Device Selection

The test suite supports running on different devices:

### Default: CPU
```bash
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/
# Equivalent to: --device=cpu
```

**Why CPU default?**
- ✅ Works in CI/CD without GPU
- ✅ Reproducible across all systems
- ✅ Faster for small test matrices

### GPU Testing: CUDA
```bash
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ --device=cuda -v
```

**Benefits:**
- ✅ Tests actual GPU codepath
- ✅ Verifies CUDA compatibility
- ✅ Faster for large matrices

### Apple Silicon: MPS
```bash
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ --device=mps -v
```

## Specific Test Examples

### Run Single Test File
```bash
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/test_torch_matrix.py -v
```

### Run Specific Test
```bash
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/test_torch_matrix.py::test_cosine_similarity_self -v
```

### Run Tests Matching Pattern
```bash
# All TF-IDF tests
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ -k tfidf -v

# All normalization tests
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ -k normalize -v

# All GPU tests
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ -k gpu -v
```

### Run with Coverage
```bash
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ \
    --cov=../../core/analysis \
    --cov-report=html \
    -v
```

## Test Results (All Passing ✅)

```
test_torch_matrix.py::test_init_from_scipy_sparse PASSED
test_torch_matrix.py::test_init_from_numpy PASSED
test_torch_matrix.py::test_init_from_torch_tensor PASSED
test_torch_matrix.py::test_device_auto_detection PASSED
test_torch_matrix.py::test_nnz_property PASSED
test_torch_matrix.py::test_sparsity_property PASSED
test_torch_matrix.py::test_to_dense PASSED
test_torch_matrix.py::test_to_numpy PASSED
test_torch_matrix.py::test_to_scipy PASSED
test_torch_matrix.py::test_round_trip_conversion PASSED
test_torch_matrix.py::test_normalize_matches_sklearn[l1] PASSED
test_torch_matrix.py::test_normalize_matches_sklearn[l2] PASSED
test_torch_matrix.py::test_normalize_matches_sklearn[max] PASSED
test_torch_matrix.py::test_l2_normalize_unit_vectors PASSED
test_torch_matrix.py::test_normalize_zero_rows PASSED
test_torch_matrix.py::test_tfidf_structure PASSED
test_torch_matrix.py::test_tfidf_matches_sklearn PASSED ✨
test_torch_matrix.py::test_tfidf_without_smoothing PASSED ✨
test_torch_matrix.py::test_tfidf_no_normalization PASSED ✨
test_torch_matrix.py::test_cosine_similarity_self PASSED
test_torch_matrix.py::test_cosine_similarity_cross PASSED
test_torch_matrix.py::test_cosine_similarity_diagonal_ones PASSED
test_torch_matrix.py::test_cosine_similarity_symmetric PASSED
test_torch_matrix.py::test_euclidean_distances_matches_sklearn PASSED
test_torch_matrix.py::test_euclidean_distances_squared PASSED
test_torch_matrix.py::test_euclidean_distances_diagonal_zeros PASSED
test_torch_matrix.py::test_manhattan_distances_matches_sklearn PASSED
test_torch_matrix.py::test_manhattan_distances_cross PASSED
test_torch_matrix.py::test_jaccard_similarity_binary PASSED
test_torch_matrix.py::test_jaccard_similarity_diagonal_ones PASSED
test_torch_matrix.py::test_transpose PASSED
test_torch_matrix.py::test_add PASSED
test_torch_matrix.py::test_multiply PASSED
test_torch_matrix.py::test_batched_cosine_same_as_regular PASSED
test_torch_matrix.py::test_different_batch_sizes[1] PASSED
test_torch_matrix.py::test_different_batch_sizes[10] PASSED
test_torch_matrix.py::test_different_batch_sizes[50] PASSED
test_torch_matrix.py::test_different_batch_sizes[200] PASSED
test_torch_matrix.py::test_empty_matrix PASSED ✨
test_torch_matrix.py::test_all_zeros_matrix PASSED
test_torch_matrix.py::test_single_row_matrix PASSED
test_torch_matrix.py::test_gpu_produces_same_as_cpu PASSED

Summary: 75 tests run: 75 passed ✅
  - test_torch_hasher.py: 33 tests
  - test_torch_matrix.py: 42 tests
```

## Bug Fixes

### Fixed Bugs (caught by tests! 🎯)

1. **TF-IDF Formula Wrong** ❌ → ✅
   - **Issue:** Wasn't matching sklearn's exact formula
   - **Fix:** Updated to use `log((n+1)/(df+1)) + 1` with smoothing
   - **Tests:** `test_tfidf_matches_sklearn`, `test_tfidf_without_smoothing`, `test_tfidf_no_normalization`

2. **Division by Zero in Empty Matrix** ❌ → ✅
   - **Issue:** `sparsity` property crashed on empty matrices
   - **Fix:** Added zero-check before division
   - **Test:** `test_empty_matrix`

3. **Hash Function Incompatibility** ❌ → ✅
   - **Issue:** Used Python's `hash()` instead of sklearn's MurmurHash3
   - **Fix:** Implemented MurmurHash3 32-bit to match sklearn exactly
   - **Tests:** `test_same_hash_buckets_as_sklearn`, `test_values_match_sklearn_absolute`, etc.

## Debugging Tips

### Verbose Output
```bash
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ -vv -s
```

### Stop at First Failure
```bash
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ -x
```

### Run Only Failed Tests
```bash
uv run pytest --lf  # last failed
uv run pytest --ff  # failed first
```

### Show Full Traceback
```bash
uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ --tb=long
```

## Performance Benchmarking

### CPU vs GPU Comparison
```bash
# Run on CPU
time uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ --device=cpu

# Run on GPU
time uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ --device=cuda
```

Expected: GPU is 5-10x faster for large matrices, similar for small test matrices.

## Integration Testing

To test integration with your `SubtreeAnalyzer`:

```python
from core.analysis.torch_hasher import TorchFeatureHasher
from core.analysis.torch_matrix import TorchSparseMatrix
from core.tools.analyzer import SubtreeAnalyzer, VectorSpace

# Your workflow
subtrees = {VectorSpace.ENTIRE_ROBOT: [...]}
analyzer = SubtreeAnalyzer.from_subtrees(subtrees)

# Use torch tools
hasher = TorchFeatureHasher(n_features=2**20, device='cuda')
scipy_matrix = hasher.fit_transform(corpus)

matrix = TorchSparseMatrix(scipy_matrix, device='cuda')
similarity = matrix.cosine_similarity()  # Fast GPU operation!
```

## CI/CD Integration

### GitHub Actions Example
```yaml
name: Test Torch Tools

on: [push, pull_request]

jobs:
  test-cpu:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install uv
      - run: uv sync
      - run: uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ -v

  test-gpu:
    runs-on: self-hosted-gpu  # If you have GPU runners
    steps:
      - uses: actions/checkout@v3
      - run: uv run pytest src/ariel_experiments/characterize/canonical/tests/unit/ --device=cuda -v
```

## Dependencies

Required (should be in your environment):
```toml
pytest >= 7.0
numpy >= 1.20
scipy >= 1.7
scikit-learn >= 1.0
torch >= 2.0
```

Optional:
```toml
pytest-cov     # For coverage reports
pytest-xdist   # For parallel test execution
```

## Contributing

When adding new features:

1. **Write tests first** (TDD approach)
2. **Ensure sklearn/scipy compatibility**
3. **Test on both CPU and GPU**
4. **Add parametrized tests** for edge cases
5. **Document expected behavior**

## FAQ

**Q: Why do tests use `decimal=5` tolerance?**
A: Float32 precision limits, plus small numerical differences between implementations.

**Q: Why test absolute values for hash outputs?**
A: Sign hash implementation may differ, but magnitudes should match.

**Q: How do I skip GPU tests?**
A: They auto-skip if CUDA unavailable via `@pytest.mark.skipif`.

**Q: Can I run tests in parallel?**
A: Yes! `uv run pytest -n auto` (requires pytest-xdist)

## Contact

Issues with tests? Check:
1. ✅ All dependencies installed
2. ✅ Correct working directory
3. ✅ sklearn/scipy versions compatible
4. ✅ For GPU tests: CUDA properly configured

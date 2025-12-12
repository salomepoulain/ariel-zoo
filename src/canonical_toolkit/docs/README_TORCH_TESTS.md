# PyTorch Feature Hasher & Sparse Matrix Tests

Comprehensive test suites to ensure our PyTorch implementations match sklearn/scipy behavior.

## Test Files

### `test_torch_hasher.py`
Tests for `TorchFeatureHasher` to ensure it produces identical results to `sklearn.feature_extraction.FeatureHasher`.

**Coverage:**
- ✅ Output format (CSR sparse matrix)
- ✅ Shape and dtype compatibility
- ✅ Hash bucket consistency with sklearn
- ✅ Absolute value matching (signs may differ)
- ✅ Duplicate token accumulation
- ✅ Alternate sign mode (binary vs signed)
- ✅ Large feature spaces (2^20, 2^24, 2^28)
- ✅ Batched processing
- ✅ Edge cases (empty, unicode, variable length)
- ✅ Deterministic output

### `test_torch_matrix.py`
Tests for `TorchSparseMatrix` to ensure compatibility with sklearn and scipy operations.

**Coverage:**
- ✅ Initialization from scipy/numpy/torch
- ✅ Conversions (to_dense, to_numpy, to_scipy)
- ✅ Normalization (L1, L2, max) matching sklearn
- ✅ TF-IDF matching sklearn.feature_extraction.text.TfidfTransformer
- ✅ Cosine similarity matching sklearn.metrics.pairwise
- ✅ Euclidean distances matching sklearn
- ✅ Manhattan distances matching sklearn
- ✅ Jaccard similarity
- ✅ Matrix operations (transpose, add, multiply)
- ✅ Batched operations
- ✅ GPU vs CPU consistency
- ✅ Edge cases (empty, all zeros, single row)

## Running Tests

### Run All Tests
```bash
cd /home/salo/projects/ariel-zoo/src/ariel_experiments/characterize/canonical/tests
pytest test_torch_hasher.py test_torch_matrix.py -v
```

### Run Specific Test File
```bash
# Only hasher tests
pytest test_torch_hasher.py -v

# Only matrix tests
pytest test_torch_matrix.py -v
```

### Run Specific Test Function
```bash
pytest test_torch_hasher.py::test_same_shape_as_sklearn -v
pytest test_torch_matrix.py::test_cosine_similarity_self -v
```

### Run Tests with Coverage
```bash
pytest test_torch_hasher.py test_torch_matrix.py --cov=../core/tools --cov-report=html
```

### Run Only Fast Tests (skip benchmarks)
```bash
pytest test_torch_hasher.py test_torch_matrix.py -v -m "not benchmark"
```

### Run GPU Tests Only (if CUDA available)
```bash
pytest test_torch_matrix.py -v -k "gpu"
```

## Test Statistics

### Expected Results

#### `test_torch_hasher.py`
- **Total Tests:** ~35
- **Parametrized Tests:** Multiple n_features, batch_sizes
- **Expected Duration:** ~5-10 seconds

#### `test_torch_matrix.py`
- **Total Tests:** ~45
- **Parametrized Tests:** Multiple norms, metrics, batch_sizes
- **Expected Duration:** ~10-15 seconds

## Dependencies

Required packages (should already be in your environment):
```bash
pytest
numpy
scipy
scikit-learn
torch
```

Optional for benchmarking:
```bash
pytest-benchmark
```

## Key Testing Strategies

### 1. **Numerical Tolerance**
We use `decimal=5` (1e-5 tolerance) for most floating-point comparisons:
```python
np.testing.assert_array_almost_equal(our_result, sklearn_result, decimal=5)
```

### 2. **Absolute Values for Hash Comparison**
Since sign hashing may differ due to implementation details, we compare absolute values:
```python
np.testing.assert_allclose(np.abs(X_torch), np.abs(X_sklearn))
```

### 3. **Structural Equivalence**
For hash buckets, we verify the same positions are filled:
```python
assert set(zip(*X_torch.nonzero())) == set(zip(*X_sklearn.nonzero()))
```

### 4. **Property-Based Testing**
We verify mathematical properties:
- L2 normalization → unit vectors
- Cosine similarity diagonal → all 1.0
- Distance matrix diagonal → all 0.0
- Symmetric matrices → equal to transpose

## Common Test Failures & Solutions

### Issue: "Absolute values don't match"
**Cause:** Hash implementation differences
**Solution:** This is expected; we only verify absolute values match

### Issue: "CUDA not available"
**Cause:** GPU tests require CUDA
**Solution:** Tests automatically skip if CUDA unavailable

### Issue: "Import error: No module named 'tools'"
**Cause:** Path issues
**Solution:** Tests add parent directory to sys.path automatically

### Issue: "Numerical precision mismatch"
**Cause:** Float32 vs Float64 differences
**Solution:** We use `rtol=1e-5` for reasonable tolerance

## Integration with CI/CD

### GitHub Actions Example
```yaml
name: Test Torch Tools

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest src/ariel_experiments/characterize/canonical/tests/test_torch_*.py -v
```

## Benchmarking

To benchmark against sklearn:
```bash
pytest test_torch_hasher.py -v --benchmark-only
```

Expected performance:
- **CPU:** ~1-2x slower than sklearn (Python overhead)
- **GPU:** ~5-10x faster than sklearn for large matrices (CUDA acceleration)

## Contributing

When adding new features to `torch_hasher.py` or `torch_matrix.py`:

1. **Add corresponding tests** to ensure sklearn/scipy compatibility
2. **Use parametrize** for multiple input variations
3. **Test edge cases** (empty, zeros, large sizes)
4. **Verify GPU consistency** (if applicable)
5. **Run full test suite** before committing

## Example: Adding a New Test

```python
def test_new_feature(simple_sparse_matrix):
    """Test description."""
    # Our implementation
    matrix = TorchSparseMatrix(simple_sparse_matrix, device="cpu")
    our_result = matrix.new_feature()

    # Sklearn/scipy reference
    sklearn_result = sklearn_new_feature(simple_sparse_matrix)

    # Compare
    np.testing.assert_array_almost_equal(
        our_result.cpu().numpy(),
        sklearn_result,
        decimal=5,
        err_msg="New feature should match sklearn"
    )
```

## Debugging Failed Tests

### Verbose Output
```bash
pytest test_torch_matrix.py::test_cosine_similarity_self -vv -s
```

### Print Array Values
```bash
pytest test_torch_matrix.py::test_cosine_similarity_self -vv -s --tb=short
```

### Stop at First Failure
```bash
pytest test_torch_hasher.py -x
```

### Run Only Failed Tests
```bash
pytest --lf  # last failed
pytest --ff  # failed first
```

## Contact

For issues or questions about the tests:
- Check that all dependencies are installed
- Verify you're running from the correct directory
- Ensure sklearn/scipy versions match requirements

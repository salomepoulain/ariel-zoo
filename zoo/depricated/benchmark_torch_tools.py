"""
Comprehensive Benchmark: TorchFeatureHasher vs sklearn FeatureHasher
GPU vs CPU Performance Comparison

Tests with very large datasets to measure real-world performance.
"""

import time
import numpy as np
import torch
from sklearn.feature_extraction import FeatureHasher as SklearnHasher
from scipy.sparse import issparse
import sys

# Add to path
sys.path.insert(0, 'src/ariel_experiments/characterize/canonical')

from core.analysis.torch_hasher import TorchFeatureHasher, BatchedTorchFeatureHasher
from core.analysis.torch_matrix import TorchSparseMatrix


def generate_large_corpus(n_samples, tokens_per_sample, vocab_size):
    """Generate a large synthetic corpus."""
    print(f"Generating corpus: {n_samples:,} samples, "
          f"{tokens_per_sample} tokens/sample, "
          f"vocab size {vocab_size:,}")

    corpus = []
    for i in range(n_samples):
        # Generate random tokens from vocabulary
        tokens = [f"token_{np.random.randint(0, vocab_size)}"
                  for _ in range(tokens_per_sample)]
        corpus.append(tokens)

    return corpus


def benchmark_hasher(name, hasher, corpus, n_runs=3):
    """Benchmark a feature hasher."""
    times = []
    result = None

    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")

    # Warmup run
    if 'Torch' in name and 'cuda' in str(hasher.device):
        print("Warmup run (GPU)...")
        _ = hasher.fit_transform(corpus[:100])
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # Actual benchmark runs
    for run in range(n_runs):
        start = time.time()
        result = hasher.fit_transform(corpus)

        # Ensure GPU operations complete
        if 'Torch' in name and torch.cuda.is_available() and 'cuda' in str(hasher.device):
            torch.cuda.synchronize()

        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.3f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nAverage: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"Throughput: {len(corpus) / avg_time:,.0f} samples/sec")

    if result is not None:
        print(f"Output shape: {result.shape}")
        print(f"Output sparsity: {result.nnz / (result.shape[0] * result.shape[1]) * 100:.6f}%")
        print(f"Memory (data): {result.data.nbytes / 1024**2:.2f} MB")

    return avg_time, result


def benchmark_similarity(name, matrix, n_runs=3):
    """Benchmark similarity computation."""
    times = []

    print(f"\n{'='*60}")
    print(f"Similarity Benchmark: {name}")
    print(f"{'='*60}")

    for run in range(n_runs):
        start = time.time()

        if isinstance(matrix, TorchSparseMatrix):
            # TorchSparseMatrix
            sim = matrix.cosine_similarity()
            if torch.cuda.is_available() and matrix.device.type == 'cuda':
                torch.cuda.synchronize()
        else:
            # sklearn with scipy
            from sklearn.metrics.pairwise import cosine_similarity
            sim = cosine_similarity(matrix)

        elapsed = time.time() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {elapsed:.3f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\nAverage: {avg_time:.3f}s ± {std_time:.3f}s")
    print(f"Output shape: {sim.shape if hasattr(sim, 'shape') else 'N/A'}")

    return avg_time


def main():
    print("="*80)
    print("COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("TorchFeatureHasher vs sklearn FeatureHasher")
    print("GPU vs CPU Performance")
    print("="*80)

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\nCUDA Available: {cuda_available}")
    if cuda_available:
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Test configurations
    configs = [
        # (n_samples, tokens_per_sample, vocab_size, n_features, name)
        (10_000, 50, 1000, 2**16, "Medium Dataset"),
        (50_000, 100, 5000, 2**20, "Large Dataset"),
        (100_000, 100, 10000, 2**22, "Very Large Dataset"),
    ]

    results = {}

    for n_samples, tokens_per_sample, vocab_size, n_features, config_name in configs:
        print("\n" + "="*80)
        print(f"CONFIGURATION: {config_name}")
        print(f"  Samples: {n_samples:,}")
        print(f"  Tokens/sample: {tokens_per_sample}")
        print(f"  Vocabulary size: {vocab_size:,}")
        print(f"  Feature space: {n_features:,} (2^{int(np.log2(n_features))})")
        print("="*80)

        # Generate corpus
        corpus = generate_large_corpus(n_samples, tokens_per_sample, vocab_size)

        config_results = {}

        # 1. Sklearn FeatureHasher
        print("\n" + "-"*80)
        print("1. SKLEARN FEATUREHASHER (CPU)")
        print("-"*80)
        sklearn_hasher = SklearnHasher(
            n_features=n_features,
            input_type='string'
        )
        sklearn_time, sklearn_result = benchmark_hasher(
            "sklearn FeatureHasher (CPU)",
            sklearn_hasher,
            corpus
        )
        config_results['sklearn_cpu'] = sklearn_time

        # 2. TorchFeatureHasher (CPU)
        print("\n" + "-"*80)
        print("2. TORCHFEATUREHASHER (CPU)")
        print("-"*80)
        torch_cpu_hasher = TorchFeatureHasher(
            n_features=n_features,
            device='cpu'
        )
        torch_cpu_time, torch_cpu_result = benchmark_hasher(
            "TorchFeatureHasher (CPU)",
            torch_cpu_hasher,
            corpus
        )
        config_results['torch_cpu'] = torch_cpu_time

        # 3. TorchFeatureHasher (GPU) if available
        if cuda_available:
            print("\n" + "-"*80)
            print("3. TORCHFEATUREHASHER (GPU/CUDA)")
            print("-"*80)
            torch_gpu_hasher = TorchFeatureHasher(
                n_features=n_features,
                device='cuda'
            )
            torch_gpu_time, torch_gpu_result = benchmark_hasher(
                "TorchFeatureHasher (GPU/CUDA)",
                torch_gpu_hasher,
                corpus
            )
            config_results['torch_gpu'] = torch_gpu_time

        # Similarity benchmarks (only for smaller datasets to avoid OOM)
        if n_samples <= 10_000:
            print("\n" + "="*80)
            print("SIMILARITY COMPUTATION BENCHMARKS")
            print("="*80)

            # sklearn + scipy
            sklearn_sim_time = benchmark_similarity(
                "sklearn cosine_similarity (CPU)",
                sklearn_result
            )
            config_results['similarity_sklearn'] = sklearn_sim_time

            # TorchSparseMatrix (CPU)
            torch_matrix_cpu = TorchSparseMatrix(torch_cpu_result, device='cpu')
            torch_cpu_sim_time = benchmark_similarity(
                "TorchSparseMatrix (CPU)",
                torch_matrix_cpu
            )
            config_results['similarity_torch_cpu'] = torch_cpu_sim_time

            # TorchSparseMatrix (GPU)
            if cuda_available:
                torch_matrix_gpu = TorchSparseMatrix(torch_gpu_result, device='cuda')
                torch_gpu_sim_time = benchmark_similarity(
                    "TorchSparseMatrix (GPU/CUDA)",
                    torch_matrix_gpu
                )
                config_results['similarity_torch_gpu'] = torch_gpu_sim_time

        results[config_name] = config_results

    # Summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    for config_name, config_results in results.items():
        print(f"\n{config_name}:")
        print("-"*80)

        # Feature hashing comparison
        print("\nFeature Hashing:")
        sklearn_time = config_results['sklearn_cpu']
        torch_cpu_time = config_results['torch_cpu']

        print(f"  sklearn (CPU):       {sklearn_time:.3f}s")
        print(f"  Torch (CPU):         {torch_cpu_time:.3f}s  "
              f"[{torch_cpu_time/sklearn_time:.2f}x vs sklearn]")

        if 'torch_gpu' in config_results:
            torch_gpu_time = config_results['torch_gpu']
            print(f"  Torch (GPU):         {torch_gpu_time:.3f}s  "
                  f"[{torch_gpu_time/sklearn_time:.2f}x vs sklearn, "
                  f"{torch_gpu_time/torch_cpu_time:.2f}x vs CPU]")
            print(f"\n  🚀 GPU Speedup:      {torch_cpu_time/torch_gpu_time:.2f}x faster than CPU")
            print(f"  🚀 GPU vs sklearn:   {sklearn_time/torch_gpu_time:.2f}x faster")

        # Similarity comparison
        if 'similarity_sklearn' in config_results:
            print("\nCosine Similarity:")
            sim_sklearn = config_results['similarity_sklearn']
            sim_torch_cpu = config_results['similarity_torch_cpu']

            print(f"  sklearn (CPU):       {sim_sklearn:.3f}s")
            print(f"  Torch (CPU):         {sim_torch_cpu:.3f}s  "
                  f"[{sim_torch_cpu/sim_sklearn:.2f}x vs sklearn]")

            if 'similarity_torch_gpu' in config_results:
                sim_torch_gpu = config_results['similarity_torch_gpu']
                print(f"  Torch (GPU):         {sim_torch_gpu:.3f}s  "
                      f"[{sim_torch_gpu/sim_sklearn:.2f}x vs sklearn, "
                      f"{sim_torch_gpu/sim_torch_cpu:.2f}x vs CPU]")
                print(f"\n  🚀 GPU Speedup:      {sim_torch_cpu/sim_torch_gpu:.2f}x faster than CPU")
                print(f"  🚀 GPU vs sklearn:   {sim_sklearn/sim_torch_gpu:.2f}x faster")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

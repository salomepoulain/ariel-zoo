"""
Realistic Benchmark for Robot Analysis
~1000 robots, ~30 tokens each, HUGE vocabulary

This simulates your actual use case!
"""

import time
import numpy as np
from sklearn.feature_extraction import FeatureHasher as SklearnHasher
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine
import sys

sys.path.insert(0, 'src/ariel_experiments/characterize/canonical')
from core.analysis.torch_matrix import TorchSparseMatrix


def generate_robot_corpus(n_robots, tokens_per_robot, vocab_size):
    """
    Generate corpus mimicking robot subtree representations.

    Args:
        n_robots: Number of robots (~1000)
        tokens_per_robot: Average tokens per robot (~30)
        vocab_size: Size of vocabulary (HUGE - millions possible)
    """
    corpus = []
    for i in range(n_robots):
        # Randomly vary token count (20-40 tokens)
        n_tokens = np.random.randint(
            max(10, tokens_per_robot - 10),
            tokens_per_robot + 10
        )

        # Generate unique tokens from huge vocabulary
        # Simulating subtree structure tokens like "joint_type_revolute_at_depth_2"
        tokens = [
            f"feature_{np.random.randint(0, vocab_size)}"
            for _ in range(n_tokens)
        ]
        corpus.append(tokens)

    return corpus


def benchmark_approach(name, hash_fn, similarity_fn, corpus):
    """Benchmark a complete approach."""
    print(f"\n{'='*70}")
    print(f"{name}")
    print('='*70)

    # Feature hashing
    start = time.time()
    X = hash_fn(corpus)
    hash_time = time.time() - start
    print(f"  Feature Hashing: {hash_time:.4f}s")

    # Similarity computation
    start = time.time()
    sim = similarity_fn(X)
    sim_time = time.time() - start
    print(f"  Cosine Similarity: {sim_time:.4f}s")

    total = hash_time + sim_time
    print(f"  TOTAL: {total:.4f}s")

    # Stats
    print(f"\n  Matrix shape: {X.shape}")
    if hasattr(X, 'nnz'):
        sparsity = X.nnz / (X.shape[0] * X.shape[1]) * 100
        print(f"  Non-zeros: {X.nnz:,}")
        print(f"  Sparsity: {sparsity:.6f}%")
        print(f"  Memory: {(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1024:.2f} KB")

    print(f"  Similarity shape: {sim.shape if hasattr(sim, 'shape') else 'N/A'}")

    return hash_time, sim_time, total


def main():
    print("="*70)
    print("REALISTIC ROBOT ANALYSIS BENCHMARK")
    print("~1000 robots, ~30 features each, HUGE vocabulary")
    print("="*70)

    # Your actual use case
    configs = [
        # (n_robots, tokens_per_robot, vocab_size, feature_space, name)
        (1000, 30, 10_000, 2**16, "Small Vocab (10K)"),
        (1000, 30, 100_000, 2**18, "Medium Vocab (100K)"),
        (1000, 30, 1_000_000, 2**20, "Large Vocab (1M)"),
        (1000, 30, 10_000_000, 2**22, "HUGE Vocab (10M)"),
    ]

    results = []

    for n_robots, tokens_per_robot, vocab_size, n_features, config_name in configs:
        print(f"\n{'='*70}")
        print(f"CONFIGURATION: {config_name}")
        print(f"  Robots: {n_robots:,}")
        print(f"  Tokens/robot: ~{tokens_per_robot}")
        print(f"  Vocabulary: {vocab_size:,}")
        print(f"  Feature space: {n_features:,} (2^{int(np.log2(n_features))})")
        print('='*70)

        # Generate corpus
        print("\nGenerating robot corpus...")
        corpus = generate_robot_corpus(n_robots, tokens_per_robot, vocab_size)

        # Approach 1: sklearn (CPU)
        sklearn_hasher = SklearnHasher(n_features=n_features, input_type='string')

        sklearn_hash, sklearn_sim, sklearn_total = benchmark_approach(
            "APPROACH 1: sklearn (CPU only)",
            lambda c: sklearn_hasher.fit_transform(c),
            lambda X: sklearn_cosine(X),
            corpus
        )

        # Approach 2: Hybrid (sklearn + GPU)
        hybrid_hash_time = None
        hybrid_convert_time = None
        hybrid_sim_time = None

        try:
            print(f"\n{'='*70}")
            print("APPROACH 2: Hybrid (sklearn hash + GPU similarity)")
            print('='*70)

            # Hash with sklearn
            start = time.time()
            X = sklearn_hasher.fit_transform(corpus)
            hybrid_hash_time = time.time() - start
            print(f"  Feature Hashing (sklearn): {hybrid_hash_time:.4f}s")

            # Convert to GPU
            start = time.time()
            matrix = TorchSparseMatrix(X, device='cuda')
            hybrid_convert_time = time.time() - start
            print(f"  GPU Conversion: {hybrid_convert_time:.4f}s")

            # Similarity on GPU
            start = time.time()
            sim = matrix.cosine_similarity()
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            hybrid_sim_time = time.time() - start
            print(f"  Cosine Similarity (GPU): {hybrid_sim_time:.4f}s")

            hybrid_total = hybrid_hash_time + hybrid_convert_time + hybrid_sim_time
            print(f"  TOTAL: {hybrid_total:.4f}s")

            print(f"\n  Similarity shape: {sim.shape}")

        except Exception as e:
            print(f"\n  ⚠️  GPU approach failed: {e}")
            hybrid_total = None

        # Store results
        results.append({
            'config': config_name,
            'n_robots': n_robots,
            'vocab_size': vocab_size,
            'sklearn_total': sklearn_total,
            'sklearn_hash': sklearn_hash,
            'sklearn_sim': sklearn_sim,
            'hybrid_total': hybrid_total,
            'hybrid_hash': hybrid_hash_time,
            'hybrid_convert': hybrid_convert_time,
            'hybrid_sim': hybrid_sim_time,
        })

    # Summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)

    for r in results:
        print(f"\n{r['config']}:")
        print("-"*70)
        print(f"  sklearn (CPU):        {r['sklearn_total']:.4f}s")

        if r['hybrid_total']:
            print(f"  Hybrid (sklearn+GPU): {r['hybrid_total']:.4f}s")
            speedup = r['sklearn_total'] / r['hybrid_total']
            print(f"  🚀 Speedup: {speedup:.2f}x")

            # Breakdown
            print(f"\n  Breakdown:")
            print(f"    sklearn hash:    {r['sklearn_hash']:.4f}s vs {r['hybrid_hash']:.4f}s")
            print(f"    GPU conversion:  N/A           vs {r['hybrid_convert']:.4f}s")
            print(f"    Similarity:      {r['sklearn_sim']:.4f}s vs {r['hybrid_sim']:.4f}s")

            sim_speedup = r['sklearn_sim'] / r['hybrid_sim']
            print(f"    Similarity speedup: {sim_speedup:.2f}x")
        else:
            print(f"  Hybrid: Failed")

    # Final recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION FOR YOUR USE CASE")
    print("="*70)

    if results and results[0]['hybrid_total']:
        best = results[0]
        speedup = best['sklearn_total'] / best['hybrid_total']

        if speedup > 1.5:
            print(f"""
✅ USE HYBRID APPROACH (sklearn + GPU)!

For {best['n_robots']} robots with sparse features:
  - sklearn (CPU):  {best['sklearn_total']:.4f}s
  - Hybrid (GPU):   {best['hybrid_total']:.4f}s
  - Speedup:        {speedup:.2f}x 🚀

Recommended code:
```python
from sklearn.feature_extraction import FeatureHasher
from core.analysis.torch_matrix import TorchSparseMatrix

# Hash features (sklearn is fast for this)
hasher = FeatureHasher(n_features=2**{int(np.log2(configs[0][3]))}, input_type='string')
X = hasher.fit_transform(robot_subtrees)

# Compute similarity on GPU (much faster!)
matrix = TorchSparseMatrix(X, device='cuda')
similarity = matrix.cosine_similarity()
```
""")
        else:
            print(f"""
⚖️  BOTH APPROACHES ARE SIMILAR

For {best['n_robots']} robots:
  - sklearn (CPU):  {best['sklearn_total']:.4f}s
  - Hybrid (GPU):   {best['hybrid_total']:.4f}s
  - Difference:     Only {speedup:.2f}x

With such a small dataset, either approach works fine.
GPU might not be worth the complexity unless you're doing
many similarity computations.
""")
    else:
        print("""
⚠️  GPU APPROACH HAD ISSUES

For very small datasets (~1000 samples), the overhead of GPU
operations may outweigh benefits. Stick with pure sklearn.
""")

    print("="*70)


if __name__ == "__main__":
    main()

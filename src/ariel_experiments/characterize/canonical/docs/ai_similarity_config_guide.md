# Similarity Configuration Guide

Complete guide to `SimilarityConfig` settings for tree similarity and diversity measurement.

---

## Table of Contents

1. [Quick Start: Maximizing Diversity](#quick-start-maximizing-diversity)
2. [Configuration Parameters](#configuration-parameters)
3. [Understanding TF-IDF Modes](#understanding-tfidf-modes)
4. [Vector Modes Explained](#vector-modes-explained)
5. [Weighting Strategies](#weighting-strategies)
6. [Complete Examples](#complete-examples)

---

## Quick Start: Maximizing Diversity

**For evolutionary algorithms optimizing for diversity, use:**

```python
config = SimilarityConfig(
    score_strategy=ScoreStrategy.TFIDF,      # Compare against population
    vector_mode=VectorMode.SET,               # Structural variety (binary)
    tfidf_mode=TFIDFMode.ENTROPY,            # Reward even rare patterns
    entropy_normalised=True,                  # Normalize by log(n)
    weighting_mode=WeightingMode.EXPONENTIAL, # Larger subtrees matter more
    aggregation_mode=AggregationMode.POWER_MEAN,
    power_mean_p=1.0,                        # Arithmetic mean
    tfidf_smooth=True,                       # Smooth IDF calculation
)
```

**Why these settings?** See [Understanding TF-IDF Modes](#understanding-tfidf-modes) below.

---

## Configuration Parameters

### Core Strategy

#### `score_strategy: ScoreStrategy`

Determines the similarity/diversity metric.

- **`TFIDF`**: Compare tree against population statistics (for diversity)
- **`TANIMOTO`**: Pairwise tree similarity (generalized Jaccard)
- **`COSINE`**: Pairwise cosine similarity

**For diversity:** Use `TFIDF` to measure how rare/unique a tree is compared to the population.

---

#### `vector_mode: VectorMode`

How to represent fragment frequencies.

- **`SET`**: Binary (presence/absence) - value = 1.0 for each unique fragment
- **`COUNTS`**: Frequency counts - value = number of times fragment appears

**Example:**

```python
fragments = ["A", "A", "A", "B", "C"]

# SET mode:
vector = {A: 1.0, B: 1.0, C: 1.0}  # Just presence

# COUNTS mode:
vector = {A: 3.0, B: 1.0, C: 1.0}  # Actual frequencies
```

**For diversity:** Use `SET` to reward structural variety (many different patterns), not repetition.

---

#### `tfidf_mode: TFIDFMode`

How to aggregate TF-IDF vector to a single score (only used when `score_strategy=TFIDF`).

- **`ENTROPY`**: Shannon entropy of TF-IDF distribution
- **`MEAN`**: Average TF-IDF value
- **`SUM`**: Sum of all TF-IDF values
- **`L1_NORM`**: Same as SUM (L1 norm)

See [Understanding TF-IDF Modes](#understanding-tfidf-modes) for detailed comparison.

---

### Weighting and Aggregation

#### `weighting_mode: WeightingMode`

How to weight different radii (subtree sizes) when aggregating.

- **`UNIFORM`**: All radii equally important (weight = 1.0)
- **`LINEAR`**: Linear increase (weight = r + 1)
- **`EXPONENTIAL`**: Exponential increase (weight = 2^r)
- **`SOFTMAX`**: Softmax normalization (configurable via `softmax_beta`)

**Example:**

```python
radii = [0, 1, 2]

# UNIFORM:  [1.0, 1.0, 1.0]
# LINEAR:   [1.0, 2.0, 3.0]
# EXPONENTIAL: [1.0, 2.0, 4.0]
```

**For diversity:** Use `EXPONENTIAL` if larger subtree patterns are more structurally significant.

---

#### `aggregation_mode: AggregationMode`

How to combine per-radius scores into final score.

- **`POWER_MEAN`**: Generalized mean with parameter `p`

**Power mean formula:** `(Σ(w_r · s_r^p) / Σ(w_r))^(1/p)`

- `p = 1`: Arithmetic mean (balanced)
- `p = 2`: Quadratic mean (emphasizes high similarities)
- `p → ∞`: Maximum (only best match matters)

**For diversity:** Use `p = 1.0` (arithmetic mean) for balanced aggregation.

---

### Normalization and Smoothing

#### `entropy_normalised: bool`

Whether to normalize entropy by log(n) where n = number of unique fragments.

- **`True`**: Score in [0, 1], where 1 = perfectly even distribution
- **`False`**: Unbounded, increases with more unique fragments

**For diversity:** Use `True` to make scores comparable across different tree sizes.

---

#### `tfidf_smooth: bool`

Use smoothed IDF calculation.

- **`True`**: IDF = log((N+1)/(df+1)) + 1 (avoids division by zero, less extreme values)
- **`False`**: IDF = log(N/df) (standard, can be very large for rare terms)

**For diversity:** Use `True` for more stable scores.

---

### Data Handling

#### `missing_data_mode: MissingDataMode`

How to handle radii not present in both trees.

- **`SKIP_RADIUS`**: Ignore missing radii
- **`TREAT_AS_ZERO`**: Treat missing as similarity = 0

---

## Understanding TF-IDF Modes

TF-IDF (Term Frequency-Inverse Document Frequency) measures how rare/unique fragments are in a tree compared to the population.

- **High TF-IDF** = rare/unique fragments → more diverse
- **Low TF-IDF** = common fragments → less diverse

### Concrete Example

**Population:** 100 trees with mostly common patterns A, B, C

**Individual 1:** Many different rare patterns
```python
fragments = [D, E, F, G, H]  # All rare in population
tfidf_vector = {D: 10, E: 10, F: 10, G: 10, H: 10}
```

**Individual 2:** One extremely rare pattern
```python
fragments = [Z]  # Extremely rare
tfidf_vector = {Z: 50}
```

**Individual 3:** Mixed rare and common
```python
fragments = [A, B, D, E]
tfidf_vector = {A: 1, B: 1, D: 10, E: 10}  # A, B common; D, E rare
```

### How Each Mode Scores Them

#### ENTROPY (Normalized)

Measures how evenly distributed the TF-IDF weights are.

```python
# Individual 1: All equal → perfectly even
probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
entropy = -Σ(p * log(p)) = 1.609
normalized_entropy = 1.609 / log(5) = 1.0  ← Maximum!

# Individual 2: Only one value → no variety
probabilities = [1.0]
entropy = 0
normalized_entropy = 0 / log(1) = 0  ← Minimum

# Individual 3: Uneven (3 common, 2 rare dominate)
probabilities = [0.045, 0.045, 0.455, 0.455]
entropy = 1.04
normalized_entropy = 1.04 / log(4) = 0.75
```

**Ranking:** Individual 1 (1.0) > Individual 3 (0.75) > Individual 2 (0.0)

**Best for diversity:** Rewards having many different rare patterns evenly distributed.

---

#### MEAN

Average TF-IDF value across unique fragments.

```python
# Individual 1:
mean = (10 + 10 + 10 + 10 + 10) / 5 = 10.0

# Individual 2:
mean = 50 / 1 = 50.0  ← Highest!

# Individual 3:
mean = (1 + 1 + 10 + 10) / 4 = 5.5
```

**Ranking:** Individual 2 (50.0) > Individual 1 (10.0) > Individual 3 (5.5)

**Problem:** Ranks single extremely rare pattern highest, ignoring structural variety.

---

#### SUM / L1_NORM

Total TF-IDF across all fragments.

```python
# Individual 1:
sum = 10 + 10 + 10 + 10 + 10 = 50

# Individual 2:
sum = 50

# Individual 3:
sum = 1 + 1 + 10 + 10 = 22
```

**Ranking:** Individual 1 (50) = Individual 2 (50) > Individual 3 (22)

**Ties:** Can't distinguish between "many rare patterns" and "one extremely rare pattern".

---

### Diversity Recommendation

**Use `ENTROPY` with `entropy_normalised=True`** because it:

1. Rewards structural variety (many different patterns)
2. Rewards evenness (not dominated by few patterns)
3. Combined with TF-IDF, patterns that are rare contribute more
4. Normalized → comparable across tree sizes

---

## Vector Modes Explained

### SET vs COUNTS for Diversity

**Tree A:** Repeated rare pattern
```python
fragments = [rare, rare, rare, common]

# SET mode:
vector = {rare: 1.0, common: 1.0}  # 2 unique patterns
tfidf = {rare: 10, common: 1}
sum = 11, mean = 5.5

# COUNTS mode:
vector = {rare: 3.0, common: 1.0}
tfidf = {rare: 30, common: 1}  # Repetition amplified!
sum = 31, mean = 15.5
```

**Tree B:** Many different rare patterns
```python
fragments = [rare1, rare2, rare3, common]

# SET mode:
vector = {rare1: 1.0, rare2: 1.0, rare3: 1.0, common: 1.0}  # 4 unique
tfidf = {rare1: 10, rare2: 10, rare3: 10, common: 1}
sum = 31, mean = 7.75

# COUNTS mode:
vector = {rare1: 1.0, rare2: 1.0, rare3: 1.0, common: 1.0}  # Same
tfidf = {rare1: 10, rare2: 10, rare3: 10, common: 1}
sum = 31, mean = 7.75
```

**For diversity:**

With `SET`:
- Tree B scores higher (more unique patterns)
- Rewards structural variety ✓

With `COUNTS`:
- Tree A scores higher (repetition of rare pattern rewarded)
- Doesn't distinguish variety from repetition ✗

**Use `VectorMode.SET` for diversity.**

---

## Weighting Strategies

### Impact on Final Score

```python
per_radius_scores = {
    0: 0.5,  # Small subtrees
    1: 0.7,
    2: 0.9,  # Large subtrees
}

# UNIFORM weights: [1.0, 1.0, 1.0]
final = (0.5*1 + 0.7*1 + 0.9*1) / 3 = 0.70

# LINEAR weights: [1.0, 2.0, 3.0]
final = (0.5*1 + 0.7*2 + 0.9*3) / 6 = 0.77

# EXPONENTIAL weights: [1.0, 2.0, 4.0]
final = (0.5*1 + 0.7*2 + 0.9*4) / 7 = 0.81
```

**Interpretation:**
- `EXPONENTIAL`: Large subtrees dominate (0.81 closer to 0.9)
- `LINEAR`: Moderate emphasis on large subtrees
- `UNIFORM`: All sizes equally important

**For diversity:**
- Use `EXPONENTIAL` if large structural patterns are more important
- Use `UNIFORM` if all subtree sizes matter equally

---

## Complete Examples

### Example 1: Maximize Diversity in EA

```python
from ariel_experiments.characterize.canonical.core.toolkit import (
    CanonicalToolKit as CTK,
    SimilarityConfig,
    ScoreStrategy,
    VectorMode,
    TFIDFMode,
    WeightingMode,
)

# Configuration for diversity
diversity_config = SimilarityConfig(
    score_strategy=ScoreStrategy.TFIDF,
    vector_mode=VectorMode.SET,
    tfidf_mode=TFIDFMode.ENTROPY,
    entropy_normalised=True,
    weighting_mode=WeightingMode.EXPONENTIAL,
    tfidf_smooth=True,
)

# Build population TF-IDF dictionary
population_hashes = [
    CTK.collect_subtrees(tree, OutputType.STRING)
    for tree in population
]
tfidf_dict = {}
CTK.update_tfidf_dictionary(population_hashes, tfidf_dict)

# Score individual for diversity
individual_hashes = CTK.collect_subtrees(individual, OutputType.STRING)
diversity_score = CTK.calculate_similarity_from_dicts(
    individual_hashes,
    tfidf_dict,
    diversity_config,
)

# Higher score = more diverse individual
print(f"Diversity score: {diversity_score}")
```

---

### Example 2: Pairwise Tree Similarity

```python
# Configuration for pairwise similarity
similarity_config = SimilarityConfig(
    score_strategy=ScoreStrategy.TANIMOTO,
    vector_mode=VectorMode.COUNTS,  # Account for repetition
    weighting_mode=WeightingMode.LINEAR,
    missing_data_mode=MissingDataMode.TREAT_AS_ZERO,
)

# Compare two trees
tree1_hashes = CTK.collect_subtrees(tree1, OutputType.STRING)
tree2_hashes = CTK.collect_subtrees(tree2, OutputType.STRING)

similarity = CTK.calculate_similarity_from_dicts(
    tree1_hashes,
    tree2_hashes,
    similarity_config,
)

# Score in [0, 1]: 1 = identical, 0 = completely different
print(f"Similarity: {similarity}")
```

---

### Example 3: Diversity with Uniform Weighting

```python
# Give equal importance to all subtree sizes
config = SimilarityConfig(
    score_strategy=ScoreStrategy.TFIDF,
    vector_mode=VectorMode.SET,
    tfidf_mode=TFIDFMode.ENTROPY,
    entropy_normalised=True,
    weighting_mode=WeightingMode.UNIFORM,  # All radii equal
    tfidf_smooth=True,
)
```

---

### Example 4: Using MEAN for Size-Invariant Rarity

```python
# If you want average rarity regardless of distribution
config = SimilarityConfig(
    score_strategy=ScoreStrategy.TFIDF,
    vector_mode=VectorMode.SET,
    tfidf_mode=TFIDFMode.MEAN,  # Average TF-IDF
    weighting_mode=WeightingMode.EXPONENTIAL,
    tfidf_smooth=True,
)
```

---

## Summary: Best Settings for Diversity

| Parameter | Recommended Value | Reason |
|-----------|------------------|---------|
| `score_strategy` | `TFIDF` | Measures rarity vs population |
| `vector_mode` | `SET` | Rewards structural variety |
| `tfidf_mode` | `ENTROPY` | Rewards many evenly-distributed rare patterns |
| `entropy_normalised` | `True` | Normalizes for tree size |
| `weighting_mode` | `EXPONENTIAL` | Larger subtrees more significant |
| `aggregation_mode` | `POWER_MEAN` | Standard aggregation |
| `power_mean_p` | `1.0` | Arithmetic mean (balanced) |
| `tfidf_smooth` | `True` | Stable, smooth scores |

**Key Insight:** For diversity, you want individuals with **many different rare structural patterns**, not just average rarity or total rarity. Normalized entropy captures this best.

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.text import Text


console = Console()

type TreeHash = str

@dataclass
class HashVector:
    """
    Vector representation for fragment-based similarity.

    A lightweight wrapper around dict[TreeHash, float] that provides
    convenient methods for vector operations and similarity computations.

    Examples
    --------
        >>> v1 = Vector.from_counts(["A", "A", "B"])
        >>> v2 = Vector.from_counts(["A", "B", "B"])
        >>> v1.cosine(v2)
        0.943
        >>> v1 @ v2  # Dot product
        4.0
        >>> (v1 + v2).data
        {'A': 3.0, 'B': 3.0}
    """

    data: dict[TreeHash, float] = field(default_factory=dict)

    # region Norms and Aggregation -----

    def l1_norm(self) -> float:
        """
        Calculate L1 norm (Manhattan distance).

        Returns
        -------
            Sum of absolute values of all elements
        """
        return sum(abs(v) for v in self.data.values())

    def l2_norm(self) -> float:
        """
        Calculate L2 norm (Euclidean length).

        Returns
        -------
            Square root of sum of squared elements
        """
        return float(np.sqrt(sum(v**2 for v in self.data.values())))

    def richness(self) -> int:
        """Count of non-zero elements."""
        return sum(1 for v in self.data.values() if v != 0.0)

    def entropy(self) -> float:
        """
        Compute Shannon entropy of the vector, treating values as weights.
        The vector is normalized into a probability distribution.

        Returns
        -------
        float
            Shannon entropy H = -sum(p_i * log(p_i))
        """
        total = self.sum()
        if total == 0:
            return 0.0

        ent = 0.0
        for _, v in self.items():
            p = v / total
            if p > 0.0:
                ent -= p * np.log(p)

        return float(ent)

    def normalized_entropy(self) -> float:
        n = self.richness()
        if n == 0:
            return 0.0
        return self.entropy() / np.log(n)

    def perplexity(self) -> float:
        """
        Perplexity of the vector: exp(entropy)
        Intuition: effective number of equally weighted terms.
        """
        return float(np.exp(self.entropy()))

    def gini(self) -> float:
        """
        Gini impurity / Gini index: 1 - sum(p_i^2)
        Measures unevenness of the distribution.
        """
        total = self.sum()
        if total == 0:
            return 0.0
        p = np.array(list(self.values())) / total
        return float(1.0 - np.sum(p**2))

    def sum(self) -> float:
        """Sum of all values."""
        return sum(self.data.values())

    def max(self) -> float:
        """Maximum value, or 0.0 if empty."""
        return max(self.data.values()) if self.data else 0.0

    def min(self) -> float:
        """Minimum value, or 0.0 if empty."""
        return min(self.data.values()) if self.data else 0.0

    def mean(self) -> float:
        """Mean (average) of all values, or 0.0 if empty."""
        return self.sum() / len(self.data) if self.data else 0.0

    def distance(self, other: HashVector) -> float:
        """
        Euclidean distance to another vector.

        Returns
        -------
            Non-negative distance
        """
        all_keys = set(self.data.keys()).union(other.data.keys())
        squared_diff = sum(
            (self.data.get(k, 0.0) - other.data.get(k, 0.0)) ** 2
            for k in all_keys
        )
        return float(np.sqrt(squared_diff))

    # endregion

    # region Operators -----

    def __matmul__(self, other: HashVector) -> float:
        """
        Dot product operator: v1 @ v2.

        Returns
        -------
            Scalar dot product
        """
        all_keys = set(self.data.keys()).union(other.data.keys())
        return sum(
            self.data.get(k, 0.0) * other.data.get(k, 0.0) for k in all_keys
        )

    def __add__(self, other: HashVector) -> HashVector:
        """Vector addition: v1 + v2."""
        all_keys = set(self.data.keys()).union(other.data.keys())
        return HashVector({
            k: self.data.get(k, 0.0) + other.data.get(k, 0.0) for k in all_keys
        })

    def __sub__(self, other: HashVector) -> HashVector:
        """Vector subtraction: v1 - v2."""
        all_keys = set(self.data.keys()).union(other.data.keys())
        return HashVector({
            k: self.data.get(k, 0.0) - other.data.get(k, 0.0) for k in all_keys
        })

    def __mul__(self, other: HashVector | int | float):
        # scalar
        if isinstance(other, (int, float)):
            return HashVector({k: v * other for k, v in self.data.items()})
        # elementwise
        shared_keys = self.data.keys() & other.data.keys()
        return HashVector({
            k: self.data[k] * other.data[k]
            for k in shared_keys
        })

    def __rmul__(self, scalar: float) -> HashVector:
        """Reverse scalar multiplication: 2.0 * v."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> HashVector:
        """Scalar division: v / 2.0."""
        if scalar == 0:
            msg = "Cannot divide vector by zero"
            raise ValueError(msg)
        return HashVector({k: v / scalar for k, v in self.data.items()})

    def normalize(self) -> HashVector:
        """
        Return L2-normalized copy (unit vector).

        Returns
        -------
            New Vector with same direction but length 1.
            Returns empty Vector if original has zero norm.
        """
        norm = self.l2_norm()
        if norm == 0:
            return HashVector()
        return HashVector({k: v / norm for k, v in self.data.items()})

    # endregion

    # region Utility Methods -----

    def __len__(self) -> int:
        """Number of elements."""
        return len(self.data)

    def __bool__(self) -> bool:
        """True if vector is non-empty."""
        return bool(self.data)

    def __getitem__(self, key: TreeHash) -> float:
        """Get value for key using bracket notation: vec[key]."""
        return self.data.get(key, 0.0)

    def __setitem__(self, key: TreeHash, value: float) -> None:
        """Set value for key using bracket notation: vec[key] = value."""
        self.data[key] = value

    def __repr__(self) -> str:
        """String representation."""
        if not self.data:
            return f"{self.__class__.__name__}()"

        sorted_items = sorted(self.data.items(), key=lambda x: str(x[0]))
        max_key_width = max(len(str(k)) for k, _ in sorted_items)

        lines = []
        for k, v in sorted_items:
            key_str = str(k).rjust(max_key_width)
            lines.append(f"    {key_str}: {v:.3f}")

        result = '\n'.join(lines)

        return f"{self.__class__.__name__}(\n{result}\n)"

    # endregion

    # region Access Methods -----

    def keys(self) -> set[TreeHash]:
        """Set of all keys (fragment hashes)."""
        return set(self.data.keys())

    def values(self) -> list[float]:
        """List of all values."""
        return list(self.data.values())

    def items(self) -> list[tuple[TreeHash, float]]:
        """List of (key, value) pairs."""
        return list(self.data.items())

    def get(self, key: TreeHash, default: float = 0.0) -> float:
        """Get value for key, with default."""
        return self.data.get(key, default)

    # endregion

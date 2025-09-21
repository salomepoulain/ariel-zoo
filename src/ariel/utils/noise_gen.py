from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt


class NoiseGenerator(ABC):
    """
    Base interface for 2D noise generators.

    Methods
    -------
    as_grid(width, height, **kwargs) -> np.ndarray
        Generate a full HxW noise grid.
    at_pixel_position(x, y, **kwargs) -> float
        Sample the noise at a single pixel coordinate (x, y).
    """

    @abstractmethod
    def as_grid(self, width: int, height: int, **kwargs) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def at_pixel_position(self, x: int, y: int, **kwargs) -> float:
        raise NotImplementedError


@dataclass
class PerlinNoise(NoiseGenerator):
    """
    Vectorized Perlin noise (2D) using NumPy only.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility.
    """
    seed: int | None = None

    def __post_init__(self):
        # Permutation table (size 256, repeated) for hashing lattice coordinates
        rng = np.random.default_rng(self.seed)
        p = np.arange(256, dtype=np.int32)
        rng.shuffle(p)
        self._perm = np.concatenate([p, p])  # 512 for safe wrap

        # 8 gradient directions (unit vectors)
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False, dtype=np.float32)
        self._grad_lut = np.stack([np.cos(angles), np.sin(angles)], axis=-1).astype(np.float32)

    # ---------- internal helpers (vectorized) ----------

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        # 6t^5 - 15t^4 + 10t^3 (smoothstep^3), stable & vectorized
        return ((6 * t - 15) * t + 10) * (t * t * t)

    @staticmethod
    def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
        return a + t * (b - a)

    def _hash2(self, xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
        """
        Hash (xi, yi) -> [0, 255] via permutation table, vectorized.
        """
        # Ensure non-negative and wrap
        xi = xi & 255
        yi = yi & 255
        return self._perm[(self._perm[xi] + yi) & 255]

    def _grad_at_corner(self, xi: np.ndarray, yi: np.ndarray) -> np.ndarray:
        """
        Lookup 2D unit gradient vector at integer lattice corner (xi, yi).
        Returns array with shape xi.shape + (2,)
        """
        h = self._hash2(xi, yi) % self._grad_lut.shape[0]
        return self._grad_lut[h]

    # ---------- public API ----------

    def as_grid(
        self,
        width: int,
        height: int,
        *,
        scale: float = 64.0,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Generate a (height, width) grid of Perlin noise.

        Parameters
        ----------
        width : int
        height : int
        scale : float
            The number of pixels per noise unit. Larger -> smoother noise.
        normalize : bool
            If True, map result from [-1, 1] to [0, 1].

        Returns
        -------
        np.ndarray
            Array of shape (height, width), dtype float32.
        """
        if scale <= 0:
            raise ValueError("scale must be > 0")

        # Pixel coordinate grid
        xs = np.arange(width, dtype=np.float32)
        ys = np.arange(height, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys, indexing="xy")

        # Noise-space coordinates
        x = X / scale
        y = Y / scale

        # Integer lattice corners
        xi = np.floor(x).astype(np.int32)
        yi = np.floor(y).astype(np.int32)

        # Local offsets inside cell
        xf = (x - xi).astype(np.float32)
        yf = (y - yi).astype(np.float32)

        # Gradients at 4 corners (broadcasts to HxW x 2)
        g00 = self._grad_at_corner(xi,     yi)
        g10 = self._grad_at_corner(xi + 1, yi)
        g01 = self._grad_at_corner(xi,     yi + 1)
        g11 = self._grad_at_corner(xi + 1, yi + 1)

        # Offset vectors to corners (HxW x 2)
        d00 = np.stack([xf,       yf      ], axis=-1)
        d10 = np.stack([xf - 1.0, yf      ], axis=-1)
        d01 = np.stack([xf,       yf - 1.0], axis=-1)
        d11 = np.stack([xf - 1.0, yf - 1.0], axis=-1)

        # Dot products (HxW)
        n00 = np.sum(g00 * d00, axis=-1)
        n10 = np.sum(g10 * d10, axis=-1)
        n01 = np.sum(g01 * d01, axis=-1)
        n11 = np.sum(g11 * d11, axis=-1)

        # Smooth interpolation
        u = self._fade(xf)
        v = self._fade(yf)

        nx0 = self._lerp(n00, n10, u)   # along x at y0
        nx1 = self._lerp(n01, n11, u)   # along x at y1
        nxy = self._lerp(nx0, nx1, v)   # along y

        # Optional normalization to [0, 1]
        if normalize:
            nxy = (nxy * 0.5 + 0.5)

        return nxy.astype(np.float32, copy=False)

    def at_pixel_position(
        self,
        x: int,
        y: int,
        *,
        scale: float = 64.0,
        normalize: bool = True,
    ) -> float:
        """
        Sample Perlin noise at a single pixel (x, y).

        This is still vectorized internally and simply extracts the scalar.

        Parameters
        ----------
        x : int
        y : int
        scale : float
        normalize : bool

        Returns
        -------
        float
        """
        # Use the same path as as_grid but for a 1x1 grid to keep behavior identical.
        grid = self.as_grid(1, 1, scale=scale, normalize=normalize)
        # The (0,0) in that 1x1 grid corresponds to the provided (x,y) by shifting the mesh.
        # To avoid recomputing with offsets, emulate by directly computing at given x,y:

        # Direct scalar path (fully NumPy but tiny arrays):
        X = np.array([[x]], dtype=np.float32)
        Y = np.array([[y]], dtype=np.float32)
        # Reuse the same math with minimal duplication by calling a tiny helper:
        return float(self._sample_points(X, Y, scale=scale, normalize=normalize))

    def _sample_points(self, X: np.ndarray, Y: np.ndarray, *, scale: float, normalize: bool) -> np.ndarray:
        """Vectorized sampling for arbitrary X,Y arrays; returns array with same shape as X."""
        x = (X / scale).astype(np.float32)
        y = (Y / scale).astype(np.float32)

        xi = np.floor(x).astype(np.int32)
        yi = np.floor(y).astype(np.int32)
        xf = (x - xi).astype(np.float32)
        yf = (y - yi).astype(np.float32)

        g00 = self._grad_at_corner(xi,     yi)
        g10 = self._grad_at_corner(xi + 1, yi)
        g01 = self._grad_at_corner(xi,     yi + 1)
        g11 = self._grad_at_corner(xi + 1, yi + 1)

        d00 = np.stack([xf,       yf      ], axis=-1)
        d10 = np.stack([xf - 1.0, yf      ], axis=-1)
        d01 = np.stack([xf,       yf - 1.0], axis=-1)
        d11 = np.stack([xf - 1.0, yf - 1.0], axis=-1)

        n00 = np.sum(g00 * d00, axis=-1)
        n10 = np.sum(g10 * d10, axis=-1)
        n01 = np.sum(g01 * d01, axis=-1)
        n11 = np.sum(g11 * d11, axis=-1)

        u = self._fade(xf)
        v = self._fade(yf)

        nx0 = self._lerp(n00, n10, u)
        nx1 = self._lerp(n01, n11, u)
        nxy = self._lerp(nx0, nx1, v)

        if normalize:
            nxy = (nxy * 0.5 + 0.5)

        return nxy.astype(np.float32, copy=False)


if __name__ == "__main__":
    # Create noise generator
    noise = PerlinNoise(seed=1234)

    # Generate a grid of noise
    width, height = 256, 256
    scale = 50.0
    grid = noise.as_grid(width, height, scale=scale, normalize=True)

    # Plot the noise
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="gray", origin="upper")
    plt.colorbar(label="Noise Value")
    plt.title("Perlin Noise ({}x{}, scale={})".format(width, height, scale))
    plt.tight_layout()
    plt.show()

    # Test at_pixel_position
    points = [(10, 10), (100, 50), (200, 150)]
    for (x, y) in points:
        value = noise.at_pixel_position(x, y, scale=scale, normalize=True)
        print(f"Noise at ({x},{y}) = {value:.4f}")

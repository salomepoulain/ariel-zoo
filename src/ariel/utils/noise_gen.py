"""Module for generating Perlin noise using NumPy."""

# Standard library
from dataclasses import dataclass

# Third-party libraries
import numpy as np
import numpy.typing as npt

# Global constants
float_precision = np.float64
int_precision = np.int32

# Type Aliases
type NpArray = npt.NDArray[np.float64]
type NpIntArray = npt.NDArray[np.int32]


@dataclass
class PerlinNoise:
    """
    Vectorized Perlin noise (2D) using NumPy only.

    Examples
    --------
    >>> noise = PerlinNoise(seed=1234)
    >>> width, height = 256, 256
    >>> scale = 50.0
    >>> grid = noise.as_grid(width, height, scale=scale, normalize=True)
    """

    seed: int | None = None

    def __post_init__(self) -> None:
        """Initialize the Perlin noise generator."""
        # Permutation table (size 256, repeated) for hashing lattice coordinates
        rng = np.random.default_rng(self.seed)
        p = np.arange(256)
        rng.shuffle(p)

        # Duplicate list: len == 512 for safe wrap
        self._perm = np.concatenate([p, p])

        # 8 gradient directions (unit vectors)
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)

        # Gradient lookup table
        self._grad_lut = np.stack(
            [np.cos(angles), np.sin(angles)],
            axis=-1,
        )

    @staticmethod
    def _fade(t: NpArray) -> NpArray:
        # 6t^5 - 15t^4 + 10t^3 (smoothstep^3), stable & vectorized
        #   https://www.wikiwand.com/en/articles/Smoothstep
        return np.array(((6 * t - 15) * t + 10) * (t * t * t))

    @staticmethod
    def _lerp(a: NpArray, b: NpArray, t: NpArray) -> NpArray:
        return a + t * (b - a)

    def _hash2(self, xi: NpIntArray, yi: NpIntArray) -> NpIntArray:
        """
        Hash (xi, yi) -> [0, 255] via permutation table, vectorized.

        Parameters
        ----------
        xi
            x-coordinate of the lattice point
        yi
            y-coordinate of the lattice point

        Returns
        -------
            array with shape xi.shape + (2,)
        """
        # Ensure non-negative and wrap
        xi &= 255
        yi &= 255
        return self._perm[(self._perm[xi] + yi) & 255]

    def _grad_at_corner(self, xi: NpIntArray, yi: NpIntArray) -> NpArray:
        """
        Lookup 2D unit gradient vector at integer lattice corner (xi, yi).

        Parameters
        ----------
        xi
            x-coordinate of the lattice point
        yi
            y-coordinate of the lattice point

        Returns
        -------
            array with shape xi.shape + (2,)
        """
        h = self._hash2(xi, yi) % self._grad_lut.shape[0]
        return self._grad_lut[h]

    # ---------- public API ----------

    def as_grid(  # noqa: PLR0914
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
        width
        height
        scale
            The number of pixels per noise unit. Larger -> smoother noise.
        normalize
            If True, map result from [-1, 1] to [0, 1].

        Raises
        ------
        ValueError
            If scale <= 0.

        Returns
        -------
        np.ndarray
            Array of shape (height, width), dtype float32.
        """
        if scale <= 0:
            msg = "scale must be > 0"
            raise ValueError(msg)

        # Pixel coordinate grid
        xs = np.arange(width, dtype=np.float32)
        ys = np.arange(height, dtype=np.float32)
        x_arr, y_arr = np.meshgrid(xs, ys, indexing="xy")

        # Noise-space coordinates
        x = x_arr / scale
        y = y_arr / scale

        # Integer lattice corners
        xi = np.floor(x).astype(np.int32)
        yi = np.floor(y).astype(np.int32)

        # Local offsets inside cell
        xf = (x - xi).astype(np.float32)
        yf = (y - yi).astype(np.float32)

        # Gradients at 4 corners (broadcasts to HxW x 2)
        g00 = self._grad_at_corner(xi, yi)
        g10 = self._grad_at_corner(xi + 1, yi)
        g01 = self._grad_at_corner(xi, yi + 1)
        g11 = self._grad_at_corner(xi + 1, yi + 1)

        # Offset vectors to corners (HxW x 2)
        d00 = np.stack([xf, yf], axis=-1)
        d10 = np.stack([xf - 1.0, yf], axis=-1)
        d01 = np.stack([xf, yf - 1.0], axis=-1)
        d11 = np.stack([xf - 1.0, yf - 1.0], axis=-1)

        # Dot products (HxW)
        n00 = np.sum(g00 * d00, axis=-1)
        n10 = np.sum(g10 * d10, axis=-1)
        n01 = np.sum(g01 * d01, axis=-1)
        n11 = np.sum(g11 * d11, axis=-1)

        # Smooth interpolation
        u = self._fade(xf)
        v = self._fade(yf)

        nx0 = self._lerp(n00, n10, u)  # along x at y0
        nx1 = self._lerp(n01, n11, u)  # along x at y1
        nxy = self._lerp(nx0, nx1, v)  # along y

        # Optional normalization to [0, 1]
        if normalize:
            nxy = nxy * 0.5 + 0.5

        return nxy.astype(np.float32, copy=False)

"""Heightmap generation functions for simulation environments."""

# Third-party libraries
import cv2
import numpy as np

# Local libraries
from ariel.parameters.ariel_types import (
    ND_FLOAT_PRECISION,
    FloatArray,
)
from ariel.utils.noise_gen import NormMethod, PerlinNoise


def flat_heightmap(
    dims: tuple[int, int],
) -> FloatArray:
    return np.zeros(shape=dims).astype(ND_FLOAT_PRECISION)


def rugged_heightmap(
    dims: tuple[int, int],
    scale_of_noise: int,
    normalize: NormMethod,
) -> FloatArray:
    # Create noise generator
    pnoise = PerlinNoise()

    # Generate a grid of noise
    nrow, ncol = dims
    return pnoise.as_grid(
        ncol,
        nrow,
        scale=scale_of_noise,
        normalize=normalize,
    )


def smooth_edges(
    dims: tuple[int, int],
    edge_width: float,
) -> FloatArray:
    # --- Smooth edge mask (0 at borders -> 1 inside) --- #
    nrow, ncol = dims

    # Create normalized coordinate grid in [0, 1]
    y, x = np.mgrid[0:nrow, 0:ncol].astype(ND_FLOAT_PRECISION)
    x /= max(ncol - 1, 1)
    y /= max(nrow - 1, 1)

    # Distance to the nearest border (0 at border, up to 0.5 in center)
    dist_to_border = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))

    # If edge_width <= 0, treat as hard edge (1 inside, 0 at border)
    if edge_width <= 0.0:
        t = (dist_to_border > 0.0).astype(ND_FLOAT_PRECISION)
    else:
        # edge_width is expected as fraction of the smaller dimension (0..0.5).
        t = np.clip(dist_to_border / float(edge_width), 0.0, 1.0).astype(
            ND_FLOAT_PRECISION,
        )
    # Smoothstep for a soft transition
    return t * t * (3.0 - 2.0 * t)


def amphitheater_heightmap(
    dims: tuple[int, int],
    ring_inner_radius: float,
    ring_outer_radius: float,
    cone_height: float,
) -> FloatArray:
    # Generate grid
    size = np.max(dims)
    y, x = np.mgrid[0:size, 0:size].astype(ND_FLOAT_PRECISION)
    x /= size
    y /= size

    # Radial distance from center
    r = np.array(
        np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2),
    ).astype(ND_FLOAT_PRECISION)

    # Unpack parameters
    r0 = ring_inner_radius
    r1 = ring_outer_radius
    d = cone_height

    # Piecewise slope: flat -> conical rise -> plateau
    heightmap = np.piecewise(
        r,
        [r <= r0, (r > r0) & (r <= r1), r > r1],
        [0.0, lambda r: d * (r - r0) / (r1 - r0), d],
    )

    # Downsample if dims is not square
    if dims[0] != dims[1]:
        heightmap = cv2.resize(
            heightmap,
            dsize=(dims[1], dims[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    return heightmap.astype(ND_FLOAT_PRECISION)


def crater_heightmap(
    dims: tuple[int, int],
    crater_depth: float,
    crater_radius: float,
) -> FloatArray:
    # Generate grid
    size = np.max(dims)
    y, x = np.mgrid[0:size, 0:size].astype(ND_FLOAT_PRECISION)
    x /= size
    y /= size

    # Elliptical cone shape
    a = crater_radius
    b = crater_radius

    # Base conical height
    r = np.sqrt(((x - 0.5) / a) ** 2 + ((y - 0.5) / b) ** 2)
    heightmap = crater_depth * r

    # Downsample if dims is not square
    if dims[0] != dims[1]:
        heightmap = cv2.resize(
            heightmap,
            dsize=(dims[1], dims[0]),
            interpolation=cv2.INTER_CUBIC,
        )
    return heightmap.astype(ND_FLOAT_PRECISION)

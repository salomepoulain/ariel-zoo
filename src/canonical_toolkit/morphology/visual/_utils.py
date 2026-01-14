"""Utility functions for visualization and rendering."""

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

from contextlib import contextmanager

from ariel.body_phenotypes.robogen_lite.modules import hinge


@contextmanager
def visual_dimensions(
    stator_dims: tuple[float] | None = None,
    rotor_dims: tuple[float] | None = None,
):
    """Temporarily override hinge dimensions for visualization."""
    original_stator = hinge.STATOR_DIMENSIONS
    original_rotor = hinge.ROTOR_DIMENSIONS

    try:
        if stator_dims is not None:
            hinge.STATOR_DIMENSIONS = stator_dims
        if rotor_dims is not None:
            hinge.ROTOR_DIMENSIONS = rotor_dims
        yield
    finally:
        hinge.STATOR_DIMENSIONS = original_stator
        hinge.ROTOR_DIMENSIONS = original_rotor
import numpy as np
from PIL import Image

def remove_white_background_and_crop(
    img: Image.Image,
    threshold: int = 245,
) -> Image.Image:
    """
    Remove white background and crop to content.

    Args:
        img: PIL Image
        threshold: Pixel values above this are considered white (0-255)

    Returns
    -------
        Cropped PIL Image with transparent background
    """
    # Convert to RGBA
    img_rgba = img.convert("RGBA")
    img_array = np.array(img_rgba)

    # Create mask: pixels where all RGB channels are ABOVE threshold
    # For pure white, this would be (255, 255, 255)
    is_white = (img_array[:, :, :3] > threshold).all(axis=2)

    # Set alpha to 0 for white pixels (making them transparent)
    img_array[is_white, 3] = 0

    # Find bounding box of non-transparent pixels
    non_transparent = np.where(img_array[:, :, 3] > 0)

    if len(non_transparent[0]) == 0:
        # No content found, return original
        return img_rgba

    # Get bounding box
    y_min, y_max = non_transparent[0].min(), non_transparent[0].max()
    x_min, x_max = non_transparent[1].min(), non_transparent[1].max()

    # Crop to bounding box and return
    return Image.fromarray(img_array[y_min : y_max + 1, x_min : x_max + 1])


def remove_black_background_and_crop(
    img: Image.Image,
    threshold: int = 10,
) -> Image.Image:
    """
    Remove black background and crop to content.

    Args:
        img: PIL Image
        threshold: Pixel values below this are considered black (0-255)

    Returns
    -------
        Cropped PIL Image with transparent background
    """
    # Convert to RGBA
    img_rgba = img.convert("RGBA")
    img_array = np.array(img_rgba)

    # Create mask: pixels where all RGB channels are below threshold
    is_black = (img_array[:, :, :3] < threshold).all(axis=2)

    # Set alpha to 0 for black pixels
    img_array[is_black, 3] = 0

    # Find bounding box of non-transparent pixels
    non_transparent = np.where(img_array[:, :, 3] > 0)

    if len(non_transparent[0]) == 0:
        # No content found, return original
        return img_rgba

    # Get bounding box
    y_min, y_max = non_transparent[0].min(), non_transparent[0].max()
    x_min, x_max = non_transparent[1].min(), non_transparent[1].max()

    # Crop to bounding box
    return Image.fromarray(img_array[y_min : y_max + 1, x_min : x_max + 1])


def look_at(
    cam_pos: tuple[float, float, float],
    target_pos: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    """
    Returns a MuJoCo quaternion (w, x, y, z) for a camera at cam_pos
    looking directly at target_pos.
    """
    # Vector from camera to target
    direction = np.array(target_pos) - np.array(cam_pos)
    # Normalize
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return (1.0, 0.0, 0.0, 0.0)  # Identity if positions are same

    direction /= norm

    # Standard 'Up' vector for the world
    up = np.array([0, 0, 1])

    # Calculate orthogonal axes
    right = np.cross(direction, up)
    # Handle case where looking straight up/down
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1, 0, 0])
    else:
        right /= np.linalg.norm(right)

    new_up = np.cross(right, direction)
    new_up /= np.linalg.norm(new_up)

    # Create rotation matrix: [Right, Up, -Forward]
    # (The camera looks towards -Z, so -Forward is the positive Z axis of the matrix)
    rot_mat = np.column_stack([right, new_up, -direction])

    # Convert to quaternion (Scipy gives x, y, z, w)
    r = R.from_matrix(rot_mat)
    quat = r.as_quat()

    # Return in MuJoCo order: (w, x, y, z)
    return (quat[3], quat[0], quat[1], quat[2])

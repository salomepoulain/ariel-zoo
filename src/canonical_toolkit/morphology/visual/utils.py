"""Utility functions for visualization and rendering."""

from contextlib import contextmanager

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

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


def remove_white_background_and_crop(
    img: Image.Image,
    threshold: int = 200,
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
    threshold: int = 20,
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


def center_on_white(
    img: Image.Image,
    threshold: int = 240,
    padding: float = 0.1,
) -> Image.Image:
    """
    Center the robot content on a white background.

    Finds the bounding box of non-white pixels and places the content
    centered on a new white canvas of the same size.

    Args:
        img: PIL Image with white background
        threshold: Pixel values above this are considered white (0-255)
        padding: Fraction of image size to use as margin (0.1 = 10% on each side)

    Returns
    -------
        PIL Image with centered content on white background
    """
    img_rgb = img.convert("RGB")
    img_array = np.array(img_rgb)
    h, w = img_array.shape[:2]

    # Find non-white pixels
    is_white = (img_array > threshold).all(axis=2)
    non_white = np.where(~is_white)

    if len(non_white[0]) == 0:
        return img  # No content, return as-is

    # Get bounding box of content
    y_min, y_max = non_white[0].min(), non_white[0].max()
    x_min, x_max = non_white[1].min(), non_white[1].max()

    # Crop content
    content = img_rgb.crop((x_min, y_min, x_max + 1, y_max + 1))
    content_w, content_h = content.size

    # Calculate target size with padding
    target_w = int(w * (1 - 2 * padding))
    target_h = int(h * (1 - 2 * padding))

    # Scale content to fit within target area (preserve aspect ratio)
    scale = min(target_w / content_w, target_h / content_h)
    if scale < 1:  # Only shrink, never enlarge
        new_w = int(content_w * scale)
        new_h = int(content_h * scale)
        content = content.resize((new_w, new_h), Image.LANCZOS)
        content_w, content_h = new_w, new_h

    # Create white canvas and paste centered
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    paste_x = (w - content_w) // 2
    paste_y = (h - content_h) // 2
    canvas.paste(content, (paste_x, paste_y))

    return canvas


def fixed_crop_zoom(
    img: Image.Image,
    crop_fraction: float = 0.3,
) -> Image.Image:
    """
    Crop a fixed percentage from edges and resize back to original size.

    This zooms in by the same amount for all images, preserving relative
    robot sizes. A crop_fraction of 0.3 means crop 30% from each edge,
    keeping the center 40%, then resize back to original dimensions.

    Args:
        img: PIL Image
        crop_fraction: Fraction to crop from each edge (0.3 = 30% from each side)

    Returns
    -------
        PIL Image zoomed in and resized to original dimensions
    """
    w, h = img.size

    # Calculate crop box (center region)
    left = int(w * crop_fraction)
    top = int(h * crop_fraction)
    right = int(w * (1 - crop_fraction))
    bottom = int(h * (1 - crop_fraction))

    # Crop center and resize back to original size
    cropped = img.crop((left, top, right, bottom))
    return cropped.resize((w, h), Image.LANCZOS)


def center_on_canvas(img: Image.Image, canvas_size: int = 64) -> Image.Image:
    """High-speed centering on a square transparent canvas."""
    # 1. Faster Resampling: BOX or BILINEAR are much faster than LANCZOS
    # For small thumbnails, BOX is often the most efficient.
    if img.width > canvas_size or img.height > canvas_size:
        img.thumbnail((canvas_size, canvas_size), Image.Resampling.BOX)

    # 2. Optimized Canvas Creation
    # Creating the canvas in RGBA mode once.
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

    # 3. Offsets
    x = (canvas_size - img.width) // 2
    y = (canvas_size - img.height) // 2

    # 4. Fast Paste
    # Only use a mask if the source is already RGBA.
    # If the source is RGB, a simple paste is faster and transparency is preserved on the canvas.
    mask = img if img.mode == "RGBA" else None
    canvas.paste(img, (x, y), mask=mask)

    return canvas


# def center_on_canvas(img: Image.Image, canvas_size: int = 64, bg_color=(255, 255, 255)) -> Image.Image:
#     """Center image on a fixed-size square canvas."""
#     canvas = Image.new("RGB", (canvas_size, canvas_size), bg_color)

#     # Scale down if image exceeds canvas (keeps aspect ratio)
#     if img.width > canvas_size or img.height > canvas_size:
#         img.thumbnail((canvas_size, canvas_size), Image.Resampling.LANCZOS)

#     # Center paste
#     x = (canvas_size - img.width) // 2
#     y = (canvas_size - img.height) // 2
#     canvas.paste(img, (x, y))

#     return canvas


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

"""Utility functions for visualization and rendering."""

import math
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

def remove_black_background_and_crop(img: Image.Image, threshold: int = 10) -> Image.Image:
    """
    Remove black background and crop to content.

    Args:
        img: PIL Image
        threshold: Pixel values below this are considered black (0-255)

    Returns:
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


def look_at(cam_pos: tuple[float, float, float], target_pos: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Returns a MuJoCo quaternion (w, x, y, z) for a camera at cam_pos
    looking directly at target_pos.
    """
    # Vector from camera to target
    direction = np.array(target_pos) - np.array(cam_pos)
    # Normalize
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return (1.0, 0.0, 0.0, 0.0) # Identity if positions are same
    
    direction /= norm

    # MuJoCo cameras look down their local -Z axis.
    # We need a rotation that aligns local -Z with our 'direction' vector.

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


def get_camera_params(angle_deg=45, distance=2.5, height=1.5):
    """
    Get camera position and quaternion for viewing at an angle.

    Args:
        angle_deg: Horizontal angle in degrees (0=front, 90=side, etc)
        distance: Distance from center
        height: Camera height above ground
    """
    raise DeprecationWarning
    angle_rad = math.radians(angle_deg)

    cam_pos = (
        distance * math.cos(angle_rad),
        distance * math.sin(angle_rad),
        height,
    )

    # Quaternion to look at origin with NO ROLL (keeps vertical lines vertical)
    yaw = -angle_rad + math.pi / 2  # Rotate to face center
    pitch = -math.atan2(height, distance)  # Tilt down to see robot
    roll = 0  # Keep camera upright!

    # Convert Euler (ZYX order) to quaternion
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    cam_quat = (
        cr * cp * cy + sr * sp * sy,  # w
        sr * cp * cy - cr * sp * sy,  # x
        cr * sp * cy + sr * cp * sy,  # y
        cr * cp * sy - sr * sp * cy,   # z
    )

    return cam_pos, cam_quat

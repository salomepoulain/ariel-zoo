# Standard library
import math
import traceback
from pathlib import Path

import mujoco
import networkx as nx
import numpy as np

# Third-party libraries
from mujoco import viewer
from PIL import Image
from rich.console import Console
from multiprocessing import Pool

from typing import Any

from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)

# Local libraries
from ariel.simulation.environments._base_world import BaseWorld
from ariel.utils.renderers import single_frame_renderer

# Global constants
# SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__")
DATA.mkdir(exist_ok=True)
SEED = 42

# Global variables
console = Console()
RNG = np.random.default_rng(SEED)
DPI = 300


from IPython.display import display


def view(
    robot: nx.Graph | nx.DiGraph,  # type: ignore
    root: int = 0,
    *,
    title: str = "",
    save_file: Path | str | None = None,
    make_tree: bool = False,
    with_viewer: bool = False,
    remove_background: bool = False,  # TODO change
    tilted: bool = False,
    return_img: bool = False,
):
    """
    Visualize a robot in a MuJoCo simulation environment.

    Parameters
    ----------
    robot : Digraph
        The robot graph to be turn into a robot to visualize in the simulation.
    with_viewer : bool, default False
        Whether to launch an interactive MuJoCo viewer window.
    save_xml : str or None, default None
        Optional filename to save the world specification as XML.
    return_img : bool, default False
        If True, returns the rendered image instead of displaying it.
        Useful for creating side-by-side comparisons.

    Returns
    -------
    Image or MjData
        If return_img=True, returns the rendered image.
        Otherwise, displays the image and returns MuJoCo data.

    Notes
    -----
    - Sets robot geometry transparency to 0.5 for better visualization
    - Enables visualization flags for transparency, actuators, and body BVH
    - Spawns robot in a SimpleFlatWorld environment
    - Logs degrees of freedom and actuator count to console
    - Runs 10 simulation steps before rendering
    - If save_xml provided, saves XML to DATA directory with UTF-8 encoding
    """
    # MuJoCo configuration
    if type(robot) == nx.DiGraph:
        robot = construct_mjspec_from_graph(robot)

    viz_options = mujoco.MjvOption()  # visualization of various elements

    # Visualization of the corresponding model or decoration element
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True

    # MuJoCo basics
    # world = TiltedFlatWorld()
    # world = SimpleFlatWorld(floor_size=(20, 20, 0.1), checker_floor=False)

    world = BaseWorld()

    # Only add floor when using the viewer
    if with_viewer:
        world.spec.add_texture(
            name="custom_grid",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.9, 0.9, 0.9],
            rgb2=[0.95, 0.95, 0.95],
            width=800,
            height=800,
        )
        world.spec.add_material(
            name="custom_floor_material",
            textures=["", "custom_grid"],
            texrepeat=[5, 5],
            texuniform=True,
            reflectance=0.05,
        )

        # Add a floor geom to the world
        world.spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[10, 10, 0.1],
            material="custom_floor_material",
            rgba=[0.9, 0.9, 0.9, 1.0],
        )

    # Save the model to XML
    if save_file:
        xml = world.spec.to_xml()
        with (DATA / f"{save_file}.xml").open("w", encoding="utf-8") as f:
            f.write(xml)

    # Make robot parts more transparant
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.9

    # Spawn the robot at the world
    world.spawn(robot.spec)

    # Compile the model
    model = world.spec.compile()
    data = mujoco.MjData(model)


# --- ADD THIS BLOCK TO BRIGHTEN THE SCENE ---
    # Increase ambient light (overall brightness)
    model.vis.headlight.ambient = [0.17, 0.17, 0.17]  # Default is usually around [0.1, 0.1, 0.1]

    # Increase diffuse light (direct brightness)
    model.vis.headlight.diffuse = [0.8, 0.8, 0.8]  # Default is usually around [0.4, 0.4, 0.4]

    # Increase specular light (shiny highlights)
    model.vis.headlight.specular = [0.3, 0.3, 0.3]

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Render

    if not tilted:
        img = single_frame_renderer(
            model,
            data,
            steps=10,
            cam_fovy=2,
        )  # , cam_pos=(0.0,0.0,0.0)) # camera buggy af
    else:
        distance = 2.5
        angle_deg = 45
        angle_rad = math.radians(angle_deg)

        cam_pos = (
            distance * math.cos(angle_rad),  # x
            distance * math.sin(angle_rad),  # y
            1.5,  # z height
        )
        cam_quat = look_at(cam_pos, [0, 0, 0])

        img = single_frame_renderer(
            model,
            data,
            width=500,
            height=500,
            cam_pos=cam_pos,
            cam_quat=cam_quat,
            cam_fovy=2,
        )

    # View
    if with_viewer:
        # Pass visualization options to ensure ground/floor is visible
        viewer.launch(model=model, data=data, show_left_ui=True, show_right_ui=True)

    if remove_background and return_img:
        img = remove_black_background_and_crop(img)

    if return_img:
        return img

    display(img)
    return data


import numpy as np


def remove_black_background_and_crop(img, threshold=10):
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
    return Image.fromarray(img_array[y_min:y_max + 1, x_min:x_max + 1])


import numpy as np
from scipy.spatial.transform import Rotation as R


def look_at(cam_pos, target_pos):
    """
    Returns a MuJoCo quaternion (w, x, y, z) for a camera at cam_pos
    looking directly at target_pos.
    """
    # Vector from camera to target
    direction = np.array(target_pos) - np.array(cam_pos)
    # Normalize
    direction /= np.linalg.norm(direction)

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


# =============================================================================
# IMAGE CACHING AND THUMBNAIL FUNCTIONS
# =============================================================================

import base64
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from io import BytesIO

import pandas as pd

try:
    from rich.progress import track
except ImportError:
    # Fallback if rich is not installed
    def track(iterable, description="", total=None):
        return iterable


def generate_single_image_worker(args):
    """
    Worker function for parallel image generation.
    Must be at module level for multiprocessing to work.

    Args:
        args: Tuple of (i, robot, cache_dir)

    Returns
    -------
        dict with robot_id and genome_string, or None on error
    """
    i, robot, cache_dir = args
    try:
        # graph = robot.to_graph()

        # Generate image using view function
        img = view(robot, return_img=True, tilted=True, remove_background=True)

        # If it's already a PIL Image, use it directly; otherwise convert from array
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        img.save(f"{cache_dir}/robot_{i:04d}.png", optimize=True, compress_level=6)

        return {
            "robot_id": i,
        }
    except Exception as e:
        console.print(f"[red]Error generating robot {i}: {e}[/red]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return None


def generate_image_cache_with_index(
    population: list[nx.DiGraph],
    robot_names: list[str] | None = None,
    cache_dir: str | Path | None = None,
    parallel: bool = False,
    max_workers: int = 4,
):
    """
    Generate all robot images and create index CSV.

    Args:
        population: List of robot objects
        robot_names: Optional list of string identifiers for each robot
        cache_dir: Directory to save images and index (default: DATA/img)
        parallel: Use multiprocessing for speed (requires macOS/Linux setup)
        max_workers: Number of parallel workers

    Returns
    -------
        DataFrame with robot index
    """
    if cache_dir is None:
        cache_dir = DATA / "img"
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Use robot_names if provided, otherwise use robot string representation
    if robot_names is None:
        robot_names = [None] * len(population)

    if parallel:
        try:
            args = [(i, population[i], cache_dir) for i in range(len(population))]

            with Pool(processes=max_workers) as pool:
                results = list(
                    track(
                        pool.imap(generate_single_image_worker, args),
                        total=len(population),
                        description=(
                            f"Generating images (parallel, {max_workers} "
                            "workers)..."
                        ),
                    ),
                )
            index_data = [r for r in results if r is not None]
        except Exception:
            parallel = False

    if not parallel:
        # Serial generation (slower but reliable in notebooks)
        index_data = []
        for i in track(
            range(len(population)), description="Generating images (serial)...",
        ):
            result = generate_single_image_worker((i, population[i], cache_dir))
            if result:
                index_data.append(result)

    # Add custom names if provided
    for i, data in enumerate(index_data):
        if i < len(robot_names) and robot_names[i] is not None:
            data["robot_name"] = robot_names[i]

    # Save index
    index_df = pd.DataFrame(index_data)
    index_df.to_csv(f"{cache_dir}/index.csv", index=False)

    return index_df


def load_single_thumbnail(
    i: int, cache_dir: str | Path, scale: float,
) -> str:
    """
    Load and scale a single robot image from cache.

    Args:
        i: Robot index
        cache_dir: Directory containing cached images
        scale: Scaling factor (0.25 = 25% size)

    Returns
    -------
        Base64 encoded PNG data URL string
    """
    cache_path = Path(cache_dir)
    img = Image.open(cache_path / f"robot_{i:04d}.png")
    new_size = (int(img.width * scale), int(img.height * scale))
    img_small = img.resize(new_size, Image.Resampling.BICUBIC)
    buffer = BytesIO()
    img_small.save(buffer, format="PNG", optimize=False, compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def load_all_thumbnails(
    n_robots: int,
    cache_dir: str | Path | None = None,
    scale: float = 0.3,
    max_workers: int = 8,
) -> list[str]:
    """
    Load all thumbnails from cache (parallel with threading).

    Args:
        n_robots: Number of robots to load
        cache_dir: Directory containing cached images (default: DATA/img)
        scale: Scaling factor (0.3 = 30% size, ~7MB for 1000 robots)
        max_workers: Number of parallel threads

    Returns
    -------
        List of base64 encoded PNG data URL strings
    """
    if cache_dir is None:
        cache_dir = DATA / "img"
    cache_dir = Path(cache_dir)

    loader = partial(load_single_thumbnail, cache_dir=cache_dir, scale=scale)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(track(
            executor.map(loader, range(n_robots)),
            total=n_robots,
            description=f"Loading {n_robots} thumbnails at {scale * 100:.0f}% scale...",
        ))


def load_index(cache_dir: str | Path | None = None) -> pd.DataFrame:
    """
    Load the robot index CSV.

    Args:
        cache_dir: Directory containing the index.csv (default: DATA/img)

    Returns
    -------
        DataFrame with robot metadata
    """
    if cache_dir is None:
        cache_dir = DATA / "img"
    cache_path = Path(cache_dir)
    return pd.read_csv(cache_path / "index.csv")


def cache_exists(cache_dir: str | Path | None = None) -> bool:
    """
    Check if image cache and index exist.

    Args:
        cache_dir: Directory to check (default: DATA/img)

    Returns
    -------
        bool: True if cache exists and is valid
    """
    if cache_dir is None:
        cache_dir = DATA / "img"
    cache_path = Path(cache_dir)
    index_path = cache_path / "index.csv"

    if not index_path.exists():
        return False

    # Check if at least one image exists
    try:
        index_df = pd.read_csv(index_path)
        if len(index_df) == 0:
            return False

        first_img = cache_path / "robot_0000.png"
        return first_img.exists()
    except Exception:
        return False


def load_or_generate_cache(
    population: list[nx.DiGraph],
    robot_names: list[str] | None = None,
    cache_dir: str | Path | None = None,
    scale: float = 0.3,
    parallel: bool = False,
    max_workers: int = 4,
) -> tuple[list[str], pd.DataFrame]:
    """
    Convenience function: Load cache if it exists, otherwise generate it.

    Args:
        population: List/dict of robot objects (only needed if generating)
        robot_names: Optional list of string identifiers for each robot
        cache_dir: Directory for cache (default: DATA/img)
        scale: Thumbnail scale for loading
        parallel: Use multiprocessing for generation
        max_workers: Number of workers for generation

    Returns
    -------
        Tuple of (thumbnails_list, index_dataframe)
    """
    if cache_dir is None:
        cache_dir = DATA / "img"
    cache_path = Path(cache_dir)

    if cache_exists(cache_path):
        index_df = load_index(cache_path)
        thumbnails = load_all_thumbnails(
            len(index_df), cache_dir=cache_path, scale=scale,
        )
    else:
        index_df = generate_image_cache_with_index(
            population,
            robot_names=robot_names,
            cache_dir=cache_path,
            parallel=parallel,
            max_workers=max_workers,
        )
        thumbnails = load_all_thumbnails(
            len(index_df), cache_dir=cache_path, scale=scale,
        )

    return thumbnails, index_df


# =============================================================================
# END OF IMAGE CACHING FUNCTIONS
# =============================================================================


if __name__ == "__main__":
    from ariel_experiments.utils.initialize import generate_random_individual

    view(generate_random_individual())

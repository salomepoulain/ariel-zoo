"""
Refactored Viewer for Ariel robots.
Provides a clean interface for rendering and interactive visualization.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import networkx as nx
import numpy as np
from mujoco import viewer
from rich.console import Console

# Third-party integration (IPython)
try:
    from IPython.display import display
except ImportError:
    display = print

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.simulation.environments._base_world import BaseWorld
from ariel.utils.renderers import single_frame_renderer

# Relative imports
from .utils import look_at, remove_black_background_and_crop

# Global constants
CWD = Path.cwd()
DATA = Path(CWD / "__data__")
DATA.mkdir(exist_ok=True)
SEED = 42

console = Console()
RNG = np.random.default_rng(SEED)

class RobotViewer:
    """
    Manages the visualization lifecycle for a robot in a MuJoCo environment.
    """

    def __init__(
        self,
        robot: nx.Graph | nx.DiGraph | mujoco.MjSpec | Any,
        world: BaseWorld | None = None,
    ):
        """
        Initialize the viewer with a robot.

        Args:
            robot: The robot to visualize (Graph, MjSpec, or object with to_graph/spec)
            world: Optional custom world. If None, a BaseWorld is created.
        """
        self.robot = self._resolve_robot(robot)
        self.world = world if world is not None else BaseWorld()
        self.model = None
        self.data = None
        
        # Default rendering options
        self.width = 500
        self.height = 500
        self.transparent_parts = False

    def _resolve_robot(self, robot: Any) -> Any:
        """Convert input to MjSpec or compatible object."""
        if isinstance(robot, (nx.Graph, nx.DiGraph)):
            return construct_mjspec_from_graph(robot)
        # Assuming it's already an MjSpec or compatible if not a graph
        return robot

    def configure_scene(
        self,
        add_floor: bool = True,
        transparent_parts: bool = True,
        bright_lights: bool = True
    ):
        """
        Set up the environment (lights, floor, materials).
        """
        if add_floor:
            self._add_floor()
        
        if transparent_parts:
            self.transparent_parts = True
            self._make_robot_transparent()

        # Compile model to allow data modification
        self._compile()

        if bright_lights:
            self._set_bright_lighting()

    def _add_floor(self):
        """Add a checkered floor to the world spec."""
        self.world.spec.add_texture(
            name="custom_grid",
            type=mujoco.mjtTexture.mjTEXTURE_2D,
            builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
            rgb1=[0.9, 0.9, 0.9],
            rgb2=[0.95, 0.95, 0.95],
            width=800,
            height=800,
        )
        self.world.spec.add_material(
            name="custom_floor_material",
            textures=["", "custom_grid"],
            texrepeat=[5, 5],
            texuniform=True,
            reflectance=0.05,
        )
        self.world.spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[10, 10, 0.1],
            material="custom_floor_material",
            rgba=[0.9, 0.9, 0.9, 1.0],
        )

    def _make_robot_transparent(self):
        """Set alpha channel of robot geoms to 0.9."""
        # Note: This assumes self.robot is an MjSpec or has a similar API
        try:
            if hasattr(self.robot, "spec") and hasattr(self.robot.spec, "geoms"):
                 geoms = self.robot.spec.geoms
            elif hasattr(self.robot, "geoms"):
                 geoms = self.robot.geoms
            else:
                 return # Cannot find geoms

            for i in range(len(geoms)):
                geoms[i].rgba[-1] = 0.9
        except Exception:
            pass # Fail silently if structure is unexpected

    def _compile(self):
        """Compile the spec into a model and data."""
        # Add robot to world if not already handled by BaseWorld logic?
        # Ariel's BaseWorld.spawn() seems to be the way.
        
        # Check if already compiled
        if self.model is not None:
            return

        # Spawn robot
        # Check if robot is a spec object or wrapper
        spec_to_spawn = self.robot.spec if hasattr(self.robot, "spec") else self.robot
        self.world.spawn(spec_to_spawn)

        self.model = self.world.spec.compile()
        self.data = mujoco.MjData(self.model)

    def _set_bright_lighting(self):
        """Enhance lighting for better visualization."""
        if self.model is None:
            return

        self.model.vis.headlight.ambient = [0.17, 0.17, 0.17]
        self.model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
        self.model.vis.headlight.specular = [0.3, 0.3, 0.3]

    def render_image(
        self,
        tilted: bool = False,
        remove_background: bool = False,
        width: int | None = None,
        height: int | None = None
    ):
        """
        Render a single frame.

        Args:
            tilted: If True, view from a 45-degree angle. Else front view.
            remove_background: If True, crop and make background transparent.
            width: Custom width override.
            height: Custom height override.

        Returns:
            PIL Image
        """
        if self.model is None:
            self._compile() # Ensure compiled

        # Reset simulation state
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data) # Update geometry

        render_width = width or self.width
        render_height = height or self.height

        if not tilted:
            # Front view (default cam)
            img = single_frame_renderer(
                self.model,
                self.data,
                steps=1, # Don't need to step simulation for static view
                cam_fovy=2,
                width=render_width,
                height=render_height
            )
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
                self.model,
                self.data,
                width=500,
                height=500,
                cam_pos=cam_pos,
                cam_quat=cam_quat,
                cam_fovy=2,
            )

        if remove_background:
            img = remove_black_background_and_crop(img)
            
        return img

    def launch_interactive(self):
        """Launch the native MuJoCo viewer."""
        if self.model is None:
            self._compile()
            
        # Ensure we have a floor for interactive viewing context
        # (Though configure_scene might have added it)
        
        viewer.launch(
            model=self.model,
            data=self.data,
            show_left_ui=True,
            show_right_ui=True
        )

    def save_xml(self, filename: str | Path):
        """Save the scene to an XML file."""
        if hasattr(self.world.spec, "to_xml"):
            xml_str = self.world.spec.to_xml()
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                f.write(xml_str)

# --- Legacy Functional API Wrapper ---

def view(
    robot: nx.Graph | nx.DiGraph | Any,
    *,
    save_file: Path | str | None = None,
    with_viewer: bool = False,
    remove_background: bool = True,
    tilted: bool = True,
    return_img: bool = False,
) -> Any:
    """
    Visualize a robot.
    
    Args:
        robot: Robot graph or spec.
        save_file: If provided, save XML to this path (relative to DATA).
        with_viewer: If True, launch interactive viewer.
        remove_background: If True, return/show transparent cropped image.
        tilted: If True, use 45-degree angle camera.
        return_img: If True, return the PIL Image object instead of displaying it.

    Returns:
        PIL Image (if return_img=True) or MjData (otherwise).
    """
    viewer_inst = RobotViewer(robot)
    
    # Configure based on flags
    # We always add floor for viewer, maybe not for simple render?
    # Legacy logic said "Only add floor when using the viewer"
    viewer_inst.configure_scene(
        add_floor=with_viewer,
        transparent_parts=True,
        bright_lights=True
    )

    if save_file:
        viewer_inst.save_xml(DATA / f"{save_file}.xml")

    # If interactive viewer requested
    if with_viewer:
        viewer_inst.launch_interactive()
        return viewer_inst.data

    # Otherwise render image
    img = viewer_inst.render_image(
        tilted=tilted,
        remove_background=remove_background
    )

    if return_img:
        return img

    display(img)
    return viewer_inst.data

if __name__ == "__main__":
    from ariel_experiments.utils.initialize import generate_random_individual
    view(generate_random_individual())

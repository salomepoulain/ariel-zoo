"""
Refactored Viewer for Ariel robots.

Code could be a bit redundant and could be more cleanly integrated directly into ariel
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

# import mujoco
from mujoco import (
    MjData,
    mj_forward,
    mj_resetData,
    mjtBuiltin,
    mjtGeom,
    mjtTexture,
)

if TYPE_CHECKING:
    from networkx import DiGraph

from mujoco import viewer
from rich.console import Console

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

from .utils import (
    look_at,
    remove_black_background_and_crop,
    remove_white_background_and_crop,
    visual_dimensions,
)

# Global constants
CWD = Path.cwd()
DATA = Path(CWD / "__data__")
SEED = 42

console = Console()

__all__ = [
    "RobotViewer",
    "quick_view",
]


class RobotViewer:
    """Manages the visualization lifecycle for a robot in a MuJoCo environment."""

    def __init__(
        self,
        robot: DiGraph[Any],
        world: BaseWorld | None = None,
        rectangle_hinge: bool = False,
    ) -> None:
        """
        Initialize the viewer with a robot.

        Args:
            robot: The robot to visualize (Graph, MjSpec, or object with to_graph/spec)
            world: Optional custom world. If None, a BaseWorld is created.
        """
        self.rectangle_hinge = rectangle_hinge

        self.robot = self._resolve_robot(robot)
        self.world = world if world is not None else BaseWorld()
        self.model = None
        self.data = None

        # Default rendering options
        self.width = 960
        self.height = 960
        self.transparent_parts = False

    def _resolve_robot(self, robot: DiGraph[Any]) -> Any:
        """Convert input to MjSpec or compatible object."""
        if self.rectangle_hinge:
            with visual_dimensions(
                stator_dims=(
                    0.025,
                    0.01,
                    0.025,
                ),  # VISUALLY CHANGE THE HINGE APPEARANCE
                rotor_dims=(
                    0.005,
                    0.04,
                    0.025,
                ),  # VISUALLY CHANGE THE HINGE APPEARANCE
            ):
                return construct_mjspec_from_graph(robot)

        with visual_dimensions():
            return construct_mjspec_from_graph(robot)

    def configure_scene(
        self,
        add_floor: bool = True,
        transparent_parts: bool = False,
        bright_lights: bool = True,
        camera_lights: bool = False,
        white_background: bool = False,
    ) -> None:
        """Set up the environment (lights, floor, materials)."""
        if camera_lights:
            self._add_camera_lights()

        if add_floor:
            self._add_floor()

        if transparent_parts:
            self.transparent_parts = True

        if white_background:
            self._set_white_background()

        self._set_robot_transparency()

        self._compile()

        if bright_lights:
            self._set_bright_lighting()

    def _add_camera_lights(self) -> None:
        """Add lights positioned around the camera for better depth perception."""
        # Increase shadow quality for smoother shadows
        self.world.spec.visual.quality.shadowsize = 8192

        # Key light - main light from camera-right and slightly above
        self.world.spec.worldbody.add_light(
            name="key_light",
            pos=[2, 2, 2],  # Front-right and above
            dir=[-1, -1, -1],
            diffuse=[0.3, 0.3, 0.3],
            specular=[0.1, 0.1, 0.1],
            castshadow=True,
            attenuation=[1, 0, 0],  # No falloff for consistent lighting
        )

        # Fill light - softer light from camera-left to fill shadows
        self.world.spec.worldbody.add_light(
            name="fill_light",
            pos=[-1.5, 2, 1.5],  # Front-left and above
            dir=[1, -1, -1],
            diffuse=[0.2, 0.2, 0.2],
            specular=[0.05, 0.05, 0.05],
            castshadow=False,
            attenuation=[1, 0, 0],
        )

        # Back light - rim light from behind for edge definition
        self.world.spec.worldbody.add_light(
            name="rim_light",
            pos=[0, -2, 1],  # Behind and slightly above
            dir=[0, 1, -0.5],
            diffuse=[0.1, 0.1, 0.1],
            specular=[0.1, 0.1, 0.1],
            castshadow=False,
            attenuation=[1, 0, 0],
        )

    def _add_floor(self) -> None:
        """Add a checkered floor to the world spec."""
        self.world.spec.add_texture(
            name="custom_grid",
            type=mjtTexture.mjTEXTURE_2D,
            builtin=mjtBuiltin.mjBUILTIN_CHECKER,
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
            type=mjtGeom.mjGEOM_PLANE,
            size=[10, 10, 0.1],
            material="custom_floor_material",
            rgba=[0.9, 0.9, 0.9, 1.0],
        )

    def _set_robot_transparency(self) -> None:
        """Set alpha channel of robot geoms to 0.9. GLITCHY."""
        # Note: This assumes self.robot is an MjSpec or has a similar API
        for i in range(len(self.robot.spec.geoms)):
            self.robot.spec.geoms[i].rgba[-1] = (
                0.9 if self.transparent_parts else 1
            )

    def _compile(self) -> None:
        """Compile the spec into a model and data."""
        # Check if already compiled
        if self.model is not None:
            return

        # Spawn robot
        spec_to_spawn = (
            self.robot.spec if hasattr(self.robot, "spec") else self.robot
        )
        self.world.spawn(spec_to_spawn)

        self.model = self.world.spec.compile()
        self.data = MjData(self.model)

    def _set_bright_lighting(self) -> None:
        """Enhance lighting for better visualization."""
        if self.model is None:
            return

        self.model.vis.headlight.ambient = [0.17, 0.17, 0.17]
        self.model.vis.headlight.diffuse = [0.8, 0.8, 0.8]
        self.model.vis.headlight.specular = [0.3, 0.3, 0.3]

    def _set_white_background(self) -> None:
        """Configures the MuJoCo spec to have a solid white background via textures."""
        self.world.spec.add_texture(
            name="white_sky",
            type=mjtTexture.mjTEXTURE_SKYBOX,
            builtin=mjtBuiltin.mjBUILTIN_FLAT,
            rgb1=[1, 1, 1],
            rgb2=[1, 1, 1],
            width=800,
            height=800,
        )

        self.world.spec.add_material(
            name="white_bg_mat",
            textures=["white_sky"],
            rgba=[1, 1, 1, 1],
        )

    def render_image(
        self,
        tilted: bool = False,
        remove_background: bool = False,
        white_background: bool = True,
        width: int | None = None,
        height: int | None = None,
    ):
        """
        Render a single frame.

        Args:
            tilted: If True, view from a 45-degree angle. Else front view.
            remove_background: If True, crop and make background transparent.
            width: Custom width override.
            height: Custom height override.

        Returns
        -------
            PIL Image
        """
        if self.model is None:
            self._compile()  # Ensure compiled

        # Reset simulation state
        mj_resetData(self.model, self.data)
        mj_forward(self.model, self.data)  # Update geometry

        render_width = width or self.width
        render_height = height or self.height

        if not tilted:
            # Front view (default cam)
            img = single_frame_renderer(
                self.model,
                self.data,
                steps=1,  # Don't need to step simulation for static view
                cam_fovy=2,
                width=render_width,
                height=render_height,
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
                steps=1,
                width=render_width,
                height=render_height,
                cam_pos=cam_pos,
                cam_quat=cam_quat,
                cam_fovy=2,
            )

        if remove_background:
            if white_background:
                img = remove_white_background_and_crop(img)
            else:
                img = remove_black_background_and_crop(img)

        return img

    def launch_interactive(self) -> None:
        """Launch the native MuJoCo viewer."""
        if self.model is None:
            self._compile()

        viewer.launch(
            model=self.model,
            data=self.data,
            show_left_ui=True,
            show_right_ui=True,
        )

    def save_xml(self, filename: str | Path) -> None:
        """Save the scene to an XML file."""
        if hasattr(self.world.spec, "to_xml"):
            xml_str = self.world.spec.to_xml()
            path = Path(filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as f:
                f.write(xml_str)


# --- Legacy Functional API Wrapper ---
# TODO; a mess. turn into data config
def quick_view(
    robot: DiGraph[Any] | Any,
    *,
    save_file: Path | str | None = None,
    with_viewer: bool = False,
    remove_background: bool = True,
    white_background: bool = False,
    width: int | None = None,
    height: int | None = None,
    tilted: bool = True,
    rectangle_hinge: bool = True,
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

    Returns
    -------
        PIL Image (if return_img=True) or MjData (otherwise).
    """
    viewer_inst = RobotViewer(robot, rectangle_hinge=rectangle_hinge)
    # Configure based on flags
    viewer_inst.configure_scene(
        add_floor=with_viewer,
        # transparent_parts=True,
        # bright_lights=True,
        camera_lights=True,
        white_background=white_background,
    )

    if save_file:
        DATA.mkdir(exist_ok=True)
        viewer_inst.save_xml(DATA / f"{save_file}.xml")

    # If interactive viewer requested
    if with_viewer:
        viewer_inst.launch_interactive()
        return viewer_inst.data

    # Otherwise render image
    img = viewer_inst.render_image(
        tilted=tilted,
        remove_background=remove_background,
        white_background=white_background,
        width=width,
        height=height,
    )

    if return_img:
        return img

    display(img)
    return viewer_inst.data


if __name__ == "__main__":
    from ariel_experiments.utils.initialize import generate_random_individual

    quick_view(generate_random_individual())

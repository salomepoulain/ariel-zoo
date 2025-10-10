# Standard library
from pathlib import Path

# Third-party libraries
import mujoco
import numpy as np
from mujoco import viewer
from PIL import Image
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule

# Local libraries
from ariel.simulation.environments.simple_flat_world import SimpleFlatWorld
from ariel.simulation.environments.simple_tilted_world import TiltedFlatWorld
from ariel.utils.renderers import single_frame_renderer

# Global constants
# SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__")
DATA.mkdir(exist_ok=True)
SEED = 42

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)
DPI = 300


def view(
    robot: CoreModule,
    *,
    with_viewer: bool = False,
    save_xml: str | None = None, #TODO: might delete, bit strange
) -> Image.Image:
    """
    Visualize a robot in a MuJoCo simulation environment.

    Parameters
    ----------
    robot : CoreModule
        The robot module to visualize in the simulation.
    with_viewer : bool, default False
        Whether to launch an interactive MuJoCo viewer window.
    save_xml : str or None, default None
        Optional filename to save the world specification as XML.

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
    viz_options = mujoco.MjvOption()  # visualization of various elements

    # Visualization of the corresponding model or decoration element
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True

    # MuJoCo basics
    # world = TiltedFlatWorld()
    world = SimpleFlatWorld(floor_size=(20, 20, 0.1))
    
    for geom in world.spec.worldbody.geoms:
        if geom.name == "floor":
            geom.material = "custom_floor_material"
            break
    

    world.spec.add_texture(
        name="custom_grid",
        type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, 
        rgb1=[0.9, 0.9, 0.9], 
        rgb2=[0.95, 0.95, 0.95],  
        reflectance=0.1,
    )
    
    # Make robot parts more transparant
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.7

    # Spawn the robot at the world
    world.spawn(robot.spec)

    # Compile the model
    model = world.spec.compile()
    data = mujoco.MjData(model)

    if save_xml:
        path = DATA / Path(save_xml)
        console.log(f"saving file to {path}")
        xml = world.spec.to_xml()
        with path.open("w", encoding="utf-8") as f:
            f.write(xml)

    # Number of actuators and DoFs
    console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Render
    img = single_frame_renderer(model, data, steps=10)

    # View
    if with_viewer:
        viewer.launch(model=model, data=data)
        
    return img

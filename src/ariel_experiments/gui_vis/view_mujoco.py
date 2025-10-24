# Standard library
from pathlib import Path

import mujoco
import networkx as nx
import numpy as np
import open3d as o3d

# Third-party libraries
from mujoco import viewer
from rich.console import Console

from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)

# Local libraries
from ariel.body_phenotypes.robogen_lite.modules.core import CORE_DIMENSIONS
from ariel.body_phenotypes.robogen_lite.modules.hinge import ROTOR_DIMENSIONS
from ariel.parameters.ariel_modules import ArielModulesConfig
from ariel.simulation.environments._simple_flat import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer
from ariel_experiments.gui_vis.visualize_tree import VisualizationConfig

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

DIMS = {"hinge": ROTOR_DIMENSIONS ,
"brick": ArielModulesConfig().BRICK_DIMENSIONS,
"core": CORE_DIMENSIONS}

from IPython.display import display


def view(
    robot: nx.Graph | nx.DiGraph,  # type: ignore
    root: int = 0,
    *,
    title: str = "",
    save_file: Path | str | None = None,
    config: VisualizationConfig | None = None,
    make_tree: bool = False,
    with_viewer: bool = False,
) -> None:
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
    robot = construct_mjspec_from_graph(robot)
    
    
    viz_options = mujoco.MjvOption()  # visualization of various elements

    # Visualization of the corresponding model or decoration element
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True

    # MuJoCo basics
    # world = TiltedFlatWorld()
    # world = SimpleFlatWorld(floor_size=(20, 20, 0.1))
        
    world = SimpleFlatWorld(floor_size=(20, 20, 0.1), checker_floor=False)





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
    
    
    # Update floor geom
    for geom in world.spec.worldbody.geoms:
        if geom.name == "simple-flat-world":  # or whatever the floor_name is
            geom.material = "custom_floor_material"
            break

    for geom in world.spec.worldbody.geoms:
        if geom.name == "floor":
            geom.material = "custom_floor_material"
            break
        
    # Save the model to XML
    if save_file:
        xml = world.spec.to_xml()
        with (DATA / f"{save_file}.xml").open("w", encoding="utf-8") as f:
            f.write(xml)

    # Make robot parts more transparant
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.7

    # Spawn the robot at the world
    world.spawn(robot.spec)

    # Compile the model
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Number of actuators and DoFs
    # console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)

    # Render
    img = single_frame_renderer(model, data, steps=10)

    # View
    if with_viewer:
        viewer.launch(model=model, data=data)

    display(img)
    return data


def make_point_cloud(center, type, nr_of_points = 1000):

    width, height,depth = DIMS[type]
    # making a mesh of the component
    component_mesh = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)

    # sampling points on the mesh, not in the mesh
    point_cloud = component_mesh.sample_points_poisson_disk(number_of_points=nr_of_points)

    # shifting all points so the center of the point cloud matches the center using absolute coords
    point_cloud.translate(center, relative=False)

    # to do: rotation, fixing the coords they do not align for some reason

    return point_cloud


def get_cloud_of_robot_from_graph(graph) -> list[list[float]]:

    # first we make mjspecs of the robot
    robot = construct_mjspec_from_graph(graph)

    # make a world for simulation
    world = SimpleFlatWorld(floor_size=(20, 20, 0.1), checker_floor=False)

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

        
    # Update floor geom
    for geom in world.spec.worldbody.geoms:
        if geom.name == "simple-flat-world":  # or whatever the floor_name is
            geom.material = "custom_floor_material"
            break

    for geom in world.spec.worldbody.geoms:
        if geom.name == "floor":
            geom.material = "custom_floor_material"
            break
        


    # Spawn the robot at the world
    world.spawn(robot.spec)

    # Compile the model
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Reset state and time of simulation
    mujoco.mj_resetData(model, data)
    
    # need to run to update the data
    img = single_frame_renderer(model, data, steps=10)
    



    # get the core position
    core = data.body(f"robot1_core").xpos 

    cloud_list = []
    for edge in graph.edges:

        # part id
        id = edge[1]
        
        component_type = graph.nodes(data=True)[id]["type"]

        # we only want to find hinges and bricks, since the core has no edges going towards it we don't check it
        if component_type == None:
            continue

        component_type = component_type.lower()
        # data obj has names for each robot part in the form of robot1_edge-partid-component_type 
        # do make sure the part id is right when errors arise during testing as reusing parts in constructing robots reuses ids which fucks it up
        edge = f"{edge[0]}-{edge[1]}"

        #substracting the coords of the core to make the core the center at 0,0,0
        center = data.body(f"robot1_{edge}-{id}-{component_type}").xpos-core
        point_cloud = make_point_cloud(center,component_type)

        


        cloud_list.append(point_cloud)
        
    
    cloud_list.append(make_point_cloud([0,0,0],"core"))
    

    
    return cloud_list



if __name__ == "__main__":

    num_modules = 20

    type_probability_space = RNG.random(
        size=(num_modules, NUM_OF_TYPES_OF_MODULES),
        dtype=np.float32,
    )

    # "Connection" probability space
    conn_probability_space = RNG.random(
        size=(num_modules, num_modules, NUM_OF_FACES),
        dtype=np.float32,
    )

    # "Rotation" probability space
    rotation_probability_space = RNG.random(
        size=(num_modules, NUM_OF_ROTATIONS),
        dtype=np.float32,
    )

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    robot = hpd.probability_matrices_to_graph(
        type_probability_space,
        conn_probability_space,
        rotation_probability_space,
    )
    
    
    view(robot)

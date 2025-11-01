# Standard library
from pathlib import Path

import mujoco
import networkx as nx
import numpy as np
import quaternion as qnp
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

# coloring for each component
COLOR = { "hinge":[1,0,0] , 
"brick": [0,0,1],
"core": [1,1,0]}

# translation to move component to the correct face
SPACING = {"TOP": [0,0,0.5] ,
"BOTTOM": [0,0,-0.5],
"RIGHT": [0.5,0,0],
"LEFT": [-0.5,0,0] ,
"FRONT": [0,0.5,0],
"BACK": [0,-0.5,0]}

# size of each component
DIMS = {"hinge": [0.025,0.05,0.025] , # place holder dim for now will import later
"brick": ArielModulesConfig().BRICK_DIMENSIONS,
"core": CORE_DIMENSIONS}

# angles associated with the different faces, used to make components face the right direction
BASE_ANGLES = {'TOP': [0.707, 0.707, 0.0, 0.0],
 'BOTTOM': [-0.0, 0.0, -0.707, 0.707],
 'LEFT': [0.707, -0.0, 0.0, 0.707],
 'RIGHT': [-0.707, 0.0, 0.0, 0.707],
 'FRONT': [1.0, 0.0, 0.0, 0.0],
 'BACK': [-0.0, 0.0, 0.0, 1.0]}

def make_point_cloud(center, type, rotation, nr_of_points = 1_000) -> o3d.geometry.PointCloud:

    # making the outline of component
    cube_max = np.array(DIMS[type])     # upper corner

    # sampeling points in the outlined area
    points = np.random.uniform(high=cube_max, size=(nr_of_points, 3))

    # transforming to pointcloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)


    # shifting all points so the center of the point cloud matches the center using absolute coords
    point_cloud.translate(center, relative=False)

    # rotating
    rotation_matrix = point_cloud.get_rotation_matrix_from_quaternion(rotation)
    point_cloud.rotate(rotation_matrix)

    # making the cloud pretty with colors depending on type
    point_cloud.paint_uniform_color(COLOR[type])

    return point_cloud


def get_cloud_of_robot_from_graph(graph:nx.DiGraph, node:int = 0) -> o3d.geometry.PointCloud: 

    """
    graph: is either a graph that can be converted to mjspecs with construct_mjspec_from_graph
            or is an mjspecs of the robot
    
    this will make point clouds and apply translation and rotation to make it match the simulations
    
    """
    # collect orientation and type of parent
    orientation = int(graph.nodes(data=True)[node]["rotation"].lower()[4:])
    
    node_type = graph.nodes(data=True)[node]["type"].lower()
    cloud = make_point_cloud([0,0,0], node_type, [0,0,0,0], nr_of_points=1_000)
    
    for i in graph.edges(node, data=True):

        # collect orientation and type of child component
        component_type = graph.nodes(data=True)[i[1]]["type"].lower()
        direction = i[-1]["face"]

        # check for none types
        if component_type not in DIMS:
            continue

        # makes point cloud
        sub_cloud = get_cloud_of_robot_from_graph(graph,i[1])

        # rotation matrix
        rotation_matrix = sub_cloud.get_rotation_matrix_from_quaternion(BASE_ANGLES[direction])

        # calculating movement to correct face, aka 0.5 own size + 0.5 parent size in a direction
        translation = (np.array(DIMS[component_type][1]) * SPACING[direction] ) + (np.array(DIMS[node_type][1]) * SPACING[direction])

        # core attachment point is not always centered
        if node_type == "core" and direction not in ["TOP","BOTTOM"]:
            # moving component down so it touches the ground
            translation += (np.array(DIMS["core"]) * [0,0,-0.25])

        
        # rotating to face the correct way
        
        sub_cloud.rotate(rotation_matrix, center=[0,0,0])

        # moving cloud to the correct face
        sub_cloud.translate(translation)


        # merging cloud
        cloud += sub_cloud
        
        
    # rotations of components around their parent
    quat = np.roll((qnp.as_float_array(qnp.from_euler_angles([
    np.deg2rad(180),
    -np.deg2rad(180 - orientation),
    np.deg2rad(0),
    ]))), shift=-1)

    rotation_matrix = cloud.get_rotation_matrix_from_quaternion(quat)
    cloud.rotate(rotation_matrix, center=[0,0,0])

    return cloud


def simple_cloud_distance(graph1:nx.DiGraph, graph2:nx.DiGraph) -> float:
    """
    Calculates distance between the pointcloud of 2 robots

    """

    cloud1 = get_cloud_of_robot_from_graph(graph1)
    cloud2 = get_cloud_of_robot_from_graph(graph2)


    # currently returns the Chamfer distance
    return (np.sum(cloud1.compute_point_cloud_distance(cloud2)) + np.sum(cloud2.compute_point_cloud_distance(cloud1)))

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

# Standard library
from pathlib import Path

import networkx as nx
import numpy as np
import open3d as o3d
import quaternion as qnp

# Third-party libraries
from rich.console import Console

# Local libraries
from ariel.body_phenotypes.robogen_lite.modules.core import CORE_DIMENSIONS
from ariel.parameters.ariel_modules import ArielModulesConfig

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


# coloring for each component
COLOR = {"hinge": [1, 0, 0], "brick": [0, 0, 1], "core": [1, 1, 0]}

# translation to move component to the correct face
SPACING = {
    "TOP": [0, 0, 0.5],
    "BOTTOM": [0, 0, -0.5],
    "RIGHT": [0.5, 0, 0],
    "LEFT": [-0.5, 0, 0],
    "FRONT": [0, 0.5, 0],
    "BACK": [0, -0.5, 0],
}

# size of each component
DIMS = {
    "hinge": [0.025, 0.05, 0.025],  # place holder dim for now will import later
    "brick": ArielModulesConfig().BRICK_DIMENSIONS,
    "core": CORE_DIMENSIONS,
}

# angles associated with the different faces, used to make components face the right direction
BASE_ANGLES = {
    "TOP": [0.707, 0.707, 0.0, 0.0],
    "BOTTOM": [-0.0, 0.0, -0.707, 0.707],
    "LEFT": [0.707, -0.0, 0.0, 0.707],
    "RIGHT": [-0.707, 0.0, 0.0, 0.707],
    "FRONT": [1.0, 0.0, 0.0, 0.0],
    "BACK": [-0.0, 0.0, 0.0, 1.0],
}


def make_point_cloud(
    center, type, rotation, nr_of_points=1_000
) -> o3d.geometry.PointCloud:
    # making the outline of component
    cube_max = np.array(DIMS[type])  # upper corner

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


def get_cloud_of_robot_from_graph(
    graph: nx.DiGraph, node: int = 0
) -> o3d.geometry.PointCloud:
    """
    graph: is a DiGraph of a robot
    node: the node id from which to go down to generate the robot, primarily used for recursion of this function
            can be used for to generate point clouds of a specific subtree in the DiGraph.

    this will make point clouds and apply translation and rotation to make it match the simulations

    """
    # collect orientation and type of parent
    orientation = int(graph.nodes(data=True)[node]["rotation"].lower()[4:])

    node_type = graph.nodes(data=True)[node]["type"].lower()
    cloud = make_point_cloud(
        [0, 0, 0], node_type, [0, 0, 0, 0], nr_of_points=1_000
    )

    for i in graph.edges(node, data=True):
        # collect orientation and type of child component
        component_type = graph.nodes(data=True)[i[1]]["type"].lower()
        direction = i[-1]["face"]

        # check for none types
        if component_type not in DIMS:
            continue

        # makes point cloud
        sub_cloud = get_cloud_of_robot_from_graph(graph, i[1])

        # rotation matrix
        rotation_matrix = sub_cloud.get_rotation_matrix_from_quaternion(
            BASE_ANGLES[direction]
        )

        # calculating movement to correct face, aka 0.5 own size + 0.5 parent size in a direction
        translation = (
            np.array(DIMS[component_type][1]) * SPACING[direction]
        ) + (np.array(DIMS[node_type][1]) * SPACING[direction])

        # core attachment point is not always centered
        if node_type == "core" and direction not in {"TOP", "BOTTOM"}:
            # moving component down so it touches the ground
            translation += np.array(DIMS["core"]) * [0, 0, -0.25]

        # rotating to face the correct way

        sub_cloud.rotate(rotation_matrix, center=[0, 0, 0])

        # moving cloud to the correct face
        sub_cloud.translate(translation)

        # merging cloud
        cloud += sub_cloud

    # rotations of components around their parent
    quat = np.roll(
        (
            qnp.as_float_array(
                qnp.from_euler_angles([
                    np.deg2rad(180),
                    -np.deg2rad(180 - orientation),
                    np.deg2rad(0),
                ])
            )
        ),
        shift=-1,
    )

    rotation_matrix = cloud.get_rotation_matrix_from_quaternion(quat)
    cloud.rotate(rotation_matrix, center=[0, 0, 0])

    return cloud


def simple_cloud_distance(
    graph1: nx.DiGraph | o3d.geometry.PointCloud,
    graph2: nx.DiGraph | o3d.geometry.PointCloud,
) -> float:
    """
    Calculates distance between the point cloud of 2 robots
    graph1: either a graph or point cloud of a robot to be compared with graph 2
    graph2: either a graph or point cloud of a robot to be compared with graph 1.
    """
    # checks whether we got point clouds or graphs as input, if graphs are given point clouds are generated
    if type(graph1) == nx.DiGraph:
        cloud1 = get_cloud_of_robot_from_graph(graph1)
    else:
        cloud1 = graph1

    if type(graph2) == nx.DiGraph:
        cloud2 = get_cloud_of_robot_from_graph(graph2)
    else:
        cloud2 = graph2

    # set max distance
    distance = float("inf")

    # compare 24 different orientations to see which orientation has the smallest distance
    for direction in BASE_ANGLES:
        # using base angles to make it all 6 directions
        cloud1.rotate(
            cloud1.get_rotation_matrix_from_quaternion(BASE_ANGLES[direction]),
            center=[0, 0, 0],
        )

        for i in range(4):
            # "rolling" the robot for the other 4 orientations
            quat = np.roll(
                (
                    qnp.as_float_array(
                        qnp.from_euler_angles([
                            np.deg2rad(180),
                            -np.deg2rad(180 - 90 * i),
                            np.deg2rad(0),
                        ])
                    )
                ),
                shift=-1,
            )
            cloud1.rotate(
                cloud1.get_rotation_matrix_from_quaternion(quat),
                center=[0, 0, 0],
            )

            # find smallest distance
            distance2 = np.sum(
                cloud1.compute_point_cloud_distance(cloud2)
            ) + np.sum(cloud2.compute_point_cloud_distance(cloud1))
            distance = min(distance, distance2)

    # currently returns the smallest Chamfer distance
    return distance



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

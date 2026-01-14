"""A2: Template for creating a custom controller using NaCPG."""

# Standard libraries
import csv
from pathlib import Path

# Third-party libraries
import matplotlib.pyplot as plt
import mujoco
import nevergrad as ng
import numpy as np
from mujoco import viewer
import time
from tqdm import tqdm

# import prebuilt robot phenotypes
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.prebuilt_robots.gecko import gecko
from ariel.simulation.controllers import NaCPG
from ariel.simulation.controllers.controller import Controller, Tracker
from ariel.simulation.controllers.na_cpg import create_fully_connected_adjacency
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.video_recorder import VideoRecorder
from ariel_experiments.characterize.canonical.core.toolkit import CanonicalToolKit as ctk

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)
quick_list = []
with open ("sorted_population.csv", "r") as file:
    reader = csv.DictReader(file)
    for line in reader:
        quick_list.append(line["ctk_string"])

# print(quick_list)


ROBOT_LIST = ["C[l(H2H2B)r(H6H2B)f(HBHB[l(H1B)r(H7B)])]","C[l(H2H2B)r(H6H2B)f(HBHB[l(H1B)r(H7B)])]","C[l(H2H2B)r(H6H2B)f(HBHB[l(H1B)r(H7B)])]","C[l(H2H2B)r(H6H2B)f(HBHB[l(H1B)r(H7B)])]","C[l(H2H2B)r(H6H2B)f(HBHB[l(H1B)r(H7B)])]"]

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]


def fitness_function(history: list[list[float]]) -> float:
    return history[-1][1]


def show_xpos_history(history: list[list[float]]) -> None:
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Create figure and axis
    plt.figure(figsize=(10, 6))

    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], "b-", label="Path")
    plt.plot(pos_data[0, 0], pos_data[0, 1], "go", label="Start")
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], "ro", label="End")

    # Add labels and title
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Robot Path in XY Plane")
    plt.legend()
    plt.grid(visible=True)

    # Set equal aspect ratio and center at (0,0)
    plt.axis("equal")
    plt.show()


def check_movement(robot, threshold=0.01):
    # Initialise controller to controller to None, always in the beginning.
        mujoco.set_mjcb_control(None)  # DO NOT REMOVE

        # Initialise world
        # Import environments from ariel.simulation.environments
        world = SimpleFlatWorld()

        # Spawn robot in the world
        # Check docstring for spawn conditions
        world.spawn(robot.spec, position=[0, 0, 0])

        # Generate the model and data
        # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
        model = world.spec.compile()
        data = mujoco.MjData(model)

        # # Initialise data tracking
        # # to_track is automatically updated every time step
        # # You do not need to touch it.
        # mujoco_type_to_find = mujoco.mjtObj.mjOBJ_GEOM
        # name_to_bind = "core"
        # tracker = Tracker(
        #     mujoco_obj_to_find=mujoco_type_to_find,
        #     name_to_bind=name_to_bind,
        # )


        # Reset state and time of simulation
        mujoco.mj_resetData(model, data)
        for _ in range(500):
            mujoco.mj_step(model, data)

        initial_xpos = data.body("robot1_core").xpos.copy()
                
        
        # for id, _ in enumerate(data.ctrl):
        #     data.ctrl[id] = 3.14  # Set target within 0 to 3.14 range
        
        for _ in range(50):
            for id, _ in enumerate(data.ctrl):
                data.ctrl[id] = 3.14  # Set target within 0 to 3.14 range
                mujoco.mj_step(model, data)

            new_pos = data.body("robot1_core").xpos.copy()
            displacement = np.linalg.norm(new_pos - initial_xpos)

            if displacement > threshold:

                return True, displacement
            
            for id, _ in enumerate(data.ctrl):
                data.ctrl[id] = -3.14  # Set target within 0 to 3.14 range
                mujoco.mj_step(model, data)

            new_pos = data.body("robot1_core").xpos.copy()
            displacement = np.linalg.norm(new_pos - initial_xpos)

            if displacement > threshold:
                return True, displacement

        # This opens a viewer window and runs the simulation with your controller
        # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
        # viewer.launch(
        #     model=model,
        #     data=data,
        # )
        return False, 0.0

def check_learnable(robot_list:list[CoreModule], names):
    """
    Trains all robots in robot list, the best weights found are returned and a video is saved to easily see the performance

    robot_list: The list of robots as Coremodule type to be trained
    names: A list of names that will be given to name each robot in the return output, if none is provided they get their index number as name

    """
    return_list = []
    non_learnable_list = []
    movement_list = []
    start = time.time()
    for robot, name in tqdm(zip(robot_list,names)):
        check, movement = check_movement(robot)
        if check:
            return_list.append(name)
        else:
            non_learnable_list.append(name)
        movement_list.append(movement)


    end = time.time()
    console.log(f"complete checking:{end-start} seconds")
    print(np.min(movement_list), np.mean(movement_list),np.median(movement_list))
    return list(return_list), list(non_learnable_list)

        # show_xpos_history(tracker.history["xpos"][0])

def main() -> None:
    """runs the batch training on the global var ROBOT_LIST which contains strings rep of robots and saves results to a csv"""

    learnable, non_learnable = check_learnable(robot_list=[construct_mjspec_from_graph(ctk.to_graph(ctk.from_string(robot))) for robot in quick_list], names=quick_list)

    print(f"Total learnable: {len(learnable)}")
    print(f"Total non-learnable: {len(non_learnable)}")

    # Print first 100 learnable robots
    print("\nFirst 100 learnable robots:")
    for i, robot in enumerate(learnable[:100], 1):
        print(f"'{robot}',")

            
if __name__ == "__main__":
    main()

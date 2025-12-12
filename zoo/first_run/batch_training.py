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


ROBOT_LIST = ['C[f(H6H2B[t(B2[l(B2[t(BH4BB6[r(B6)]H6)])])]HH4)]',
'C[l(B[b(B2B6B[t(B2[r(B6H6BB6[l(B4)]H4H)])])])f(H4)]',
'C[r(B2)l(B6B[b(B4[b(B6)])]H4H2H6)f(H6H2H6H4H)]',
'C[r(B6H4)l(BBHB[l(B6[r(B4)])b(B[b(B2)])])f(H4H2)]',
'C[r(B)f(B[r(B2[l(B6H4H6HH4H)])l(B2[l(B6[l(B6)])]H2H6)])]',
'C[r(B4)f(B4[r(B6[l(B[l(B[l(B[t(B4HBB6)]H4)])]B6)])])]',
'C[f(BH4BB6[l(B6[b(B6H4B[b(B4)]H)]H2)]H6H2H6)]',
'C[f(B4[l(B6[b(B2B6[t(B)]H4HHB2)]BH2)])]',
'C[l(B6[l(B6H4B2)]H)f(H6B4[l(B6H2HHH2H2)])]',
'C[f(B2[r(B6H4BH6B4[b(B)]H2)]H6H2)]',
'C[r(B4H4HBB6[r(B6[b(B2[l(B6)])]H4)]H4H2)f(H6)]',
'C[l(B[t(B6H4B6[b(B2)]HH2)]B[b(B2)])f(H6BH4H6H4H2)]',
'C[r(B)f(BH4B[t(B[l(B2[t(B2[l(B6B2)]B6)])]H4H6H2)])]',
'C[r(B6[r(B6[l(B)b(B2[l(B4[b(B6)]H6B4H2)]H4BB4)]B6H4)])f(H6)]',
'C[f(H6B[l(BB2[r(B2[l(B6H4B6HH4)])])]H6B)]',
'C[r(BB6[r(B6[l(B2[r(B2B6)])]B2H4HH4H4H6H4)]B)f(H6)]',
'C[f(B4[r(B2[l(B2[l(B6[l(B)]B[t(BH4)]H4B)])])l(B4[r(B6)])])]',
'C[f(B2[r(B2H4B[l(B2)]H4H2)t(B2[r(B6B2)])])r(B6H)]',
'C[l(B6[b(B6[b(B[r(B2B6)])]B[l(B2H4)t(B6)])]H4HB)f(H4)]',
'C[r(B4B[r(B)t(B6[l(B2)t(B[b(B6B6)])]H4H)]B6)f(H4)]',
'C[l(B6[b(B6H)]B2)f(H2H4H4H6H)]',
'C[r(BH2)f(H6H2B6[b(B2[l(B)t(B4)]H4B[l(B6)])]BH6)]',
'C[l(B2[r(B2[b(B2[r(B6[t(B2[l(B6)])]H4)]H4)])]B6HB)f(H4)]',
'C[l(B6B2[b(B2)l(B6[t(B2B4)]H4H6BH4)]H6)f(H6)]',
'C[l(B6B[r(B6[b(B2[b(B2[r(B)])t(BH4B[r(B4H4)]H4)])]B)])f(H4)]',
'C[r(B)f(B2[r(B2)b(B4[l(B2B6B[r(B[l(B2)])]B6H6)])]H6H4H)]',
'C[l(B2[l(B6[b(B6[b(B[l(B)]BH6B)]H2)]B)]HH2)f(H6H4)]',
'C[r(B4B6[b(B2[b(B2)])t(BB6)]B4HB)f(H6H4H6H)]',
'C[l(B[l(B2[t(B2)l(B2H2)r(B6B)]H4H4)])f(H6BH6H6)]',
'C[r(BB[t(B6HB[b(B6[b(B2[r(B6H4H)]HH2)]H2)])])f(H4)]',
'C[f(B2[r(B6[t(B6[r(B6BH4B2)])])]HH2)l(B[r(B)]H4)]',
'C[l(B[l(B[l(B2)r(B6B[r(B)]H4B)]H4H6H2)])f(H4)]',
'C[f(H4H6B[l(B6[t(B[t(B6H4)r(B[t(BB2[r(B2H)])])])]B6)]H2)]',
'C[r(B6[r(B4[r(B6HB)]HHB4)])f(H4H2H)]',
'C[f(BH4BB6[l(B6[b(B6[t(B2[r(B4H6)])]H4H6)]B2[l(B2[b(B2)])])])]',
'C[f(B6[l(B6[b(B2[b(B2[r(B6[l(BH4B2)]BH)]H6)])])])r(BHH6)]',
'C[r(B6H6)f(H6B[l(B2[l(B4[l(B6[b(B2)l(BH4)]B2)])])])]',
'C[r(B6H4)f(H6B[t(B[l(B6[b(B2)]B6H)]H)]H6HH2)]',
'C[f(B2[t(B6B2[b(B6[b(B6H2H4B4H4)l(BH4)])])])l(B6H2)]',
'C[f(H6B[l(B6[b(B2[b(B2[r(B6[l(BB)])])]H4BH6)]H4)]H6H4)]',
'C[l(B6H4HB[b(B6)l(B6[l(B2[r(B4[l(B6)])])]H4H6)]H2)f(H4)]',
'C[r(B6[l(B6BH4H4B4B6[t(B)])]H2H4B2H6)f(H6H4)]',
'C[f(H6H6B2B[l(B2[b(B2[r(B2H4)])]B6[r(B[l(B6B6)])]H)])]',
'C[f(H4B4H4H2H4B6[b(B2[b(B2[r(B6)t(B6[b(B[r(B4)])])])])])]',
'C[l(B6[l(B[l(BH4BB6H4)]H2H4H2H6)]B)f(H6)]',
'C[f(B2[r(B2[l(B6[l(B[l(BH4)])r(B[r(B4[l(B[t(B6)])])])]H4H6)])])l(B6B2)]',
'C[f(B[l(B6[b(B2B4[l(B2[l(B2[l(B[b(B6)])])])]H4HH2)])])]',
'C[r(B[t(B)l(B2[r(B2HH6)]BB4B6H6)]H4)f(H4)]',
'C[l(B6[r(BH4B[b(B6[b(B2[b(B2)])l(BH4)]B4)])]B6[b(B)]H2)f(H4)]',
'C[r(B6[t(B6H4HBH6H4H)]B4B[r(B2)b(B6)])f(H6H2)]',
'C[f(B2[r(B2)t(B4H6H6)l(BB2H4H6B[l(B)]H6H)])]',
'C[r(B6[t(B[b(B6[b(B4)])r(BH)]H2)]H4H2)f(H6H4)]',
'C[f(H6H4H6HBB6[l(B6[b(B2[l(B2)])]B6[t(B[l(B6[l(B2)])])]H2)])]',
'C[l(B6[b(B2[l(B2[l(B2[l(B4)])]H4H)]H4B2)t(B6[r(B)])])f(H4)]',
'C[f(B2[r(B2[r(B)l(B[t(B4[l(B6)])]H4H)]HB)])r(B6[r(B2)])]',
'C[r(B4)f(H4H4BB6[l(B6[b(B2[b(B2)t(B[l(B)]H6)])]B2)]H6)]',
'C[r(B6[t(B6H)]B[r(B2[b(B2[r(B2[l(B2BBH)])])])])f(H4H4)]',
'C[l(B[t(B[t(B6[r(B4[l(B2B6HH2)])t(BH2)]H4H6H6H4)])])f(H4)]',
'C[r(BB2[r(B2[t(BB6)]H6B2[r(B6B6[l(B6H4)])])])f(H4)]',
'C[r(B4)f(H4B2HB[t(B2)b(B6[b(B[b(B2)])]H2H4)r(B[t(B4)]B)])]',
'C[f(B2HH2)r(B6[b(B[r(B6B2)l(B6H)]H6H4)])]',
'C[f(B2[r(B2[r(B4[r(B[b(B6[r(B6H)]H4B4[t(B6H)])])]B4)])])]',
'C[f(B4[t(B4[l(B2H4B[r(B6[b(BH)]B6)])]H4H2)])]',
'C[l(B[b(B)r(B6[t(B[t(B6[r(BH4HB[r(B2)])])]H4H2)]H2)])f(H4)]',
'C[l(B[l(B2[r(B[r(B6)])]H4)]H2H4B)f(H6H4B[l(B4)])]',
'C[l(B4B6[l(B)]BH4B[l(B6H2)])f(H6H4H6H2H4)]',
'C[l(B6[l(B[l(B2[r(B2)])t(B[r(B6)])])]H4)f(BH2H6)]',
'C[l(B4[l(B6[b(B2BH)]B2)t(B[r(BH4B2)]HH4)])f(H6)]',
'C[r(B[t(B2B6[b(B2[t(B4)l(B6H4)]H4B)]HH4)])f(H2)]',
'C[r(B6H4H4B2)l(B6[l(B2)]B[r(BH4)]BH6H2)f(H4)]',
'C[f(B2[t(B2[l(B)])r(B6H2)]B2H4H4HB[b(B6[l(B2)])])]',
'C[l(B6[r(B2[r(B2[r(B2)])])b(B[b(B)]H)]B)f(H6H6H4H4)]',
'C[f(B6[l(B6[b(B[t(BB[r(B6H2H6H2)]H4)]H4H4)]B2H4)]B)]',
'C[f(H6H2BB6B6B[t(B[l(B6)]B6[r(B)]H4HH2)]H6H6)]',
'C[f(B6[l(B6[b(B2[b(B2[r(B4B6[l(B)]H4)])])t(B4)]HBH)])]',
'C[r(B6[l(B6[r(B2[r(B2[t(B)l(B6H4)]H6B2)]B)])]H6)f(H4)]',
'C[l(B6B2[l(B6[r(B2[t(B6[r(B6)])r(B6[t(B)]H4H6)])]B)])f(H2)]',
'C[r(B)l(B6B[r(B2[b(B2)]HH2H4H6)])f(H4)]',
'C[f(H2B6[t(B2B)b(B[b(BH4)])]B6H4H6B[b(B4)]H4H2)]',
'C[r(B2[r(B6[l(B)])]H)f(H6B[r(B2)]H4HH4)]',
'C[l(B6[r(B2)b(B4B)])f(H4H4B[r(B4[b(B6)]H4HH2H)]B)]',
'C[f(H6B4[l(B2[l(B[r(B6H4B[t(B6[l(B2[r(B2[r(B)])])])])])])t(B6)]H)]',
'C[f(B2[r(B2[r(B2[l(B4[t(B6[b(B6)])])]H)])l(B6H4B2)]H6)]',
'C[r(B6[l(B6[b(B2[tb(B4)]B[b(B2B4[t(B)]HH)]H2)]B)])f(H6)]',
'C[r(B4)l(B6[b(B2[t(B)])t(B[b(B[b(B6H4)])])]H)f(H2H4)]',
'C[r(B[t(B2[b(B6)])l(B2[r(B2H4)])]HH6)f(H6BH)]',
'C[r(B6[r(B)l(B2[r(B2)b(B4[t(B6[l(B6B6)]B)])])]H)f(H6BH4)]',
'C[l(B6[b(B2[l(B6[r(BH)]H4H4)])]BB6B)f(H6H2)]',
'C[l(B2[l(B2[b(B2[t(B6)r(BH4)l(B[r(B6H)])])])])r(B4[r(B)])f(H6)]',
'C[r(B4[b(B4[r(B6[t(BH4B[l(B6)b(B6B)]B6)])])])f(H4)]',
'C[f(B2[t(B6B6B[l(B[t(B4H)]B6H4H2)]H4H4)])]',
'C[l(B6BH6H4H2H4)r(B6[l(B2)]HB[r(B)l(B6)])f(H4)]',
'C[f(B4[l(B6[t(B6)b(B6[l(B2[r(B2[r(B2)t(BB6)])])r(B[l(B)])]H4)])])]',
'C[r(B2)l(B2[l(B2[b(B2[l(B4BH4H)t(B6B[r(B6B)])])])])f(H4)]',
'C[f(H4B4B[t(B6B2[t(BH4BH4)])]HH)]',
'C[f(H6B6[l(B2[b(B2[r(B6[t(B)]H4H4B4B6H)])])]H4)]',
'C[r(B[t(B2[l(B4H2)]B6H2B)]H4BH6H4)f(H6)]',
'C[f(B4[t(B[l(B6[l(B)r(B2)b(B6H2)]H)])]H4BH6H4H4)]',
'C[f(B6[l(B6[b(BH4H6BH4H6)]B2H2)]B[r(BH)]H2)]',
'C[l(B6[l(B6H4)]BB6[l(B2[r(B2[r(B2[l(B[b(B[r(B6)])])])])])])f(H6)]']

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


def batch_training(robot_list:list[CoreModule], names: None|list[str|int] = None, save_file: str = "robot_data.csv")-> list[(any,list[str|int])]:
    """
    Trains all robots in robot list, the best weights found are returned and a video is saved to easily see the performance

    robot_list: The list of robots as Coremodule type to be trained
    names: A list of names that will be given to name each robot in the return output, if none is provided they get their index number as name
    save_file: Path to CSV file where results will be incrementally saved (default: "robot_data.csv")

    """

    if names is None:
        names = list(range(len(robot_list)))
    elif len(names) != len(robot_list):
        raise Exception("Length of the list of names does not match the length of the robot list")
    start = time.time()
    controller_weights = []

    # Create/clear the CSV file with header
    with open(save_file, "w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        # Optional: write header row
        # writer.writerow(["id", "robot_name", "controller_params"])
    for nr, robot in tqdm(enumerate(robot_list)):
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

        # Initialise data tracking
        # to_track is automatically updated every time step
        # You do not need to touch it.
        mujoco_type_to_find = mujoco.mjtObj.mjOBJ_GEOM
        name_to_bind = "core"
        tracker = Tracker(
            mujoco_obj_to_find=mujoco_type_to_find,
            name_to_bind=name_to_bind,
        )

        # Setup the NaCPG controller
        adj_dict = create_fully_connected_adjacency(len(data.ctrl.copy()))
        na_cpg_mat = NaCPG(adj_dict, angle_tracking=True)

        # Setup Nevergrad optimizer
        params = ng.p.Instrumentation(
            phase=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(
                -2 * np.pi,
                2 * np.pi,
            ),
            w=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(-2 * np.pi, 2 * np.pi),
            amplitudes=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(
                -2 * np.pi,
                2 * np.pi,
            ),
            ha=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(-10, 10),
            b=ng.p.Array(shape=(len(data.ctrl),)).set_bounds(-100, 100),
        )
        num_of_workers = 50
        budget = 500
        optim = ng.optimizers.PSO
        optimizer = optim(
            parametrization=params,
            budget=budget,
            num_workers=num_of_workers,
        )

        # Simulate the robot
        ctrl = Controller(
            controller_callback_function=lambda _, d: na_cpg_mat.forward(d.time),
            tracker=tracker,
        )

        # Set the control callback function
        # This is called every time step to get the next action.
        # Pass the model and data to the tracker
        ctrl.tracker.setup(world.spec, data)

        # Set the control callback function
        mujoco.set_mjcb_control(
            ctrl.set_control,
        )

        # Reset state and time of simulation
        mujoco.mj_resetData(model, data)

        # Run optimization loop
        best_fitness = float("inf")
        best_params = None
        for _ in range(optimizer.budget):
            ctrl.tracker.reset()
            x = optimizer.ask()
            na_cpg_mat.set_param_with_dict(x.kwargs)
            simple_runner(
                model,
                data,
                duration=10,
            )
            loss = fitness_function(tracker.history["xpos"][0])
            optimizer.tell(x, loss)
            if loss < best_fitness:
                best_fitness = loss
                best_params = x.kwargs



        # console.log(
        #     f"finished training robot nr: {nr}, Best loss: {best_fitness}",
        # )
                

        # Rerun best parameters
        # na_cpg_mat.set_param_with_dict(best_params)
        # ctrl.tracker.reset()
        # mujoco.mj_resetData(model, data)

        # This opens a viewer window and runs the simulation with your controller
        # If mujoco.set_mjcb_control(None), then you can control the limbs yourself.
        # viewer.launch(
        #     model=model,
        #     data=data,
        # )
        
        
        # If you want to record a video of your simulation, you can use the video renderer.
        # Non-default VideoRecorder options
        PATH_TO_VIDEO_FOLDER = DATA / "__videos__"
        video_recorder = VideoRecorder(file_name=str(nr), output_folder=PATH_TO_VIDEO_FOLDER)

        # Render with video recorder (wrapped in try-except to continue if video fails)
        try:
            video_renderer(
                model,
                data,
                duration=15,
                video_recorder=video_recorder,
            )
        except Exception as e:
            console.log(f"Warning: Video recording failed for robot {nr}: {e}")
            console.log("Continuing training without video...")

        controller_weights.append(best_params)

        # Incremental save: append this robot's results to CSV immediately
        with open(save_file, "a", newline="", encoding='utf-8') as file:
            writer = csv.writer(file)
            controller_dict = best_params.copy()
            # Convert numpy arrays to lists for CSV serialization
            for key in controller_dict:
                controller_dict[key] = [float(value) for value in controller_dict[key]]
            writer.writerow([nr, names[nr], best_fitness, controller_dict])

        console.log(f"Saved robot {nr} ({names[nr]}) with fitness {best_fitness} to {save_file}")

    end = time.time()
    console.log(f"complete training took:{end-start} seconds")

    return list(zip(controller_weights, names))

        # show_xpos_history(tracker.history["xpos"][0])

def main() -> None:
    """Run batch training on robots from sorted_population.csv."""
    # Read robots from sorted_population.csv
    robot_strings = []
    with open("sorted_population.csv", "r", encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            robot_strings.append(row['ctk_string'])

    # Limit to first 100 robots
    robot_strings = robot_strings[100:]

    # Results are now saved incrementally during batch_training
    robot_graphs = [
        construct_mjspec_from_graph(ctk.to_graph(ctk.from_string(robot)))
        for robot in robot_strings
    ]
    
    results = batch_training(
        robot_list=robot_graphs,
        names=robot_strings,
        save_file="robot_data.csv",
    )
    console.log(
        f"Training complete! All {len(results)} robots saved to robot_data.csv",
    )


if __name__ == "__main__":
    
    main()

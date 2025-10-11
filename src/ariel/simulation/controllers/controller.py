"""TODO(jmdm): description of script."""

# Standard library
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# Third-party libraries
import mujoco as mj
import numpy as np

# Local libraries
from ariel.utils.tracker import Tracker


@dataclass
class Controller:
    # Function that executes the actual control step
    controller_callback_function: Callable[..., Any]

    # How often to call the controller (for every simulation step)
    time_steps_per_ctrl_step: int = 100  # control frequency

    # How often to save the data
    time_steps_per_save: int = 500  # data-sampling frequency

    # How big a step to take towards the output fot the callback function
    alpha: float = 1

    # Optional tracker to save data during simulation
    tracker: Tracker = field(default_factory=Tracker)

    def set_control(
        self,
        model: mj.MjModel,
        data: mj.MjData,
        *args: Any | None,
        **kwargs: dict[Any, Any] | None,
    ) -> None:
        # Calculate current time step
        time = data.time
        time_step = model.opt.timestep
        deduced_time_step = np.ceil(time / time_step)

        # Execute saving only at specific time-steps
        if (deduced_time_step % self.time_steps_per_save) == 0:
            self.tracker.update(data)

        # Execute control strategy only at specific time-steps
        if (deduced_time_step % self.time_steps_per_ctrl_step) == 0:
            # Save the old control values
            old_ctrl = data.ctrl.copy()

            # Execute the custom control function of the user
            output = np.array(
                self.controller_callback_function(
                    model,
                    data,
                    *args,
                    **kwargs,
                ),
            )

            # Calculate the new control values
            new_ctrl = (old_ctrl * (1 - self.alpha)) + (output * self.alpha)

            # Ensure that the new control values are within the servo bounds
            data.ctrl = np.clip(new_ctrl, -np.pi / 2, np.pi / 2)

            # Check if there are any NaN values in the control signal
            if np.any(np.isnan(data.ctrl)):
                msg = "NaN values detected in the control signal.\n"
                msg += f"{data.ctrl}"
                raise ValueError(msg)

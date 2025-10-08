"""TODO(jmdm): description of script."""

# Standard library
from typing import Any

# Third-party libraries
import mujoco as mj

# Local libraries
from ariel import log


class Tracker:
    def __init__(
        self,
        mujoco_obj_to_find: mj.mjtObj | None = None,
        name_to_bind: str | None = None,
        observable_attributes: list[str] | None = None,
    ) -> None:
        # Set default tracking parameters
        if mujoco_obj_to_find is None or name_to_bind is None:
            mujoco_obj_to_find = mj.mjtObj.mjOBJ_GEOM
            name_to_bind = "core"
            msg = "No tracking parameters provided, "
            msg += "defaulting to tracking all geoms with 'core' in their name."
            log.info(msg)

        # Set default observable attributes
        if observable_attributes is None:
            observable_attributes = ["xpos"]

        # Save local variables
        self.mujoco_obj_to_find = mujoco_obj_to_find
        self.name_to_bind = name_to_bind
        self.observable_attributes = observable_attributes
        self.history: dict[str, dict[int, list[Any]]] = {}

    def setup(
        self,
        world: mj.MjSpec,
        data: mj.MjData,
    ) -> None:
        # Find all objects of the specified type and bind them
        self.geoms = world.worldbody.find_all(self.mujoco_obj_to_find)
        self.to_track = [
            data.bind(geom)
            for geom in self.geoms
            if self.name_to_bind in geom.name
        ]

        # Initialize history dictionary
        for attr in self.observable_attributes:
            if attr not in self.history:
                self.history[attr] = {
                    idx: [] for idx in range(len(self.to_track))
                }

    def update(self, data: mj.MjData) -> None:
        # Update the bound objects
        for idx, obj in enumerate(self.to_track):
            for attr in self.observable_attributes:
                self.history[attr][idx].append(getattr(obj, attr).copy())

    def reset(self) -> None:
        # Reset the history dictionary
        for attr in self.observable_attributes:
            for idx in range(len(self.to_track)):
                self.history[attr][idx] = []

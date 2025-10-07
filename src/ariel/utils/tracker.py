"""TODO(jmdm): description of script."""

# Standard library
from typing import Any

# Third-party libraries
import mujoco as mj


class Tracker:
    def __init__(
        self,
        mujoco_obj_to_find: mj.mjtObj,
        name_to_bind: str,
        observable_attributes: list[str] | None = None,
    ) -> None:
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

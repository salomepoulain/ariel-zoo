"""
MuJoCo world: base class for MuJoCo world specifications.

References
----------
    [1] https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mju-euler2quat

Todo
----
    [ ] Document the class methods
"""

# Standard library
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import cast

# Third-party libraries
import mujoco
import numpy as np

# Local libraries
from ariel import log
from ariel.parameters.ariel_types import Position, Rotation
from ariel.parameters.mujoco_params import MujocoConfig
from ariel.utils.mujoco_ops import duplicate_mj_spec, euler_to_quat_conversion


class BaseWorld:
    """Base class for MuJoCo world specifications."""

    name: str = "base-world"

    spawns: int = 0
    spawn_prefix: str = "robot"
    default_spawn_position: Position = (0, 0, 0)  # x, y, z
    default_spawn_rotation: Rotation = (0, 0, 0)  # x, y, z

    is_precompiled: bool = False

    def __init__(
        self,
        name: str | None = None,
        mujoco_config: MujocoConfig | None = None,
        *,
        load_precompiled: bool = True,
    ) -> None:
        # Use default mujoco config if none is provided
        if mujoco_config is None:
            self.mujoco_config = MujocoConfig()

        # Set world name
        if name is not None:
            self.name = name

        # Load precompiled XML if requested
        if load_precompiled:
            self.is_precompiled = self.load_precompiled()
            if self.is_precompiled:
                return

        # Build and save specification
        self.spec: mujoco.MjSpec = self._init_spec()

    @abstractmethod
    def _expand_spec(self) -> None:
        """Expand the world specification with additional elements."""

    def _init_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.modelname = self.name.replace("-", " ").title()

        # --- Option --- #
        spec.option.integrator = self.mujoco_config.integrator

        # --- Compiler --- #
        spec.compiler.autolimits = self.mujoco_config.autolimits
        spec.compiler.balanceinertia = self.mujoco_config.balanceinertia
        spec.compiler.degree = self.mujoco_config.degree

        # --- Visual --- #
        spec.visual.global_.offheight = self.mujoco_config.offheight
        spec.visual.global_.offwidth = self.mujoco_config.offwidth

        # Add a default light source
        spec.worldbody.add_light(
            name="light",
            pos=[0, 0, 1],
            castshadow=False,
            type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
        )

        # Add ortho camera and normal camera
        spec.worldbody.add_camera(
            name="ortho-cam",
            orthographic=True,
            pos=[-5, 0, 5],
            xyaxes=[0, -1, 0, 0.75, 0, 0.75],
            fovy=5,
        )
        return spec

    def _find_contacts(self) -> set[tuple[str, str, float]]:
        # Create a temporary copy of the spec to compile
        temp_spec = duplicate_mj_spec(self.spec)
        model = temp_spec.compile()
        data = mujoco.MjData(model)

        # Step the simulation to ensure everything is stable
        mujoco.mj_forward(model, data)

        # Discover contacts between the world and the spawned robots
        contact_pairs = set()
        for contact in data.contact:
            geom1 = mujoco.mj_id2name(
                m=model,
                type=mujoco.mjtObj.mjOBJ_GEOM,
                id=contact.geom1,
            )
            geom2 = mujoco.mj_id2name(
                m=model,
                type=mujoco.mjtObj.mjOBJ_GEOM,
                id=contact.geom2,
            )
            contact_pairs |= {(geom1, geom2, contact.dist)}

        # Clear the temporary objects
        del model, data, temp_spec

        # Return the set of contact pairs
        return contact_pairs

    def _check_and_correct_spawn(
        self,
        spawn: mujoco.MjsBody,
        spawn_name: str,
    ) -> None:
        validation_steps = 0
        validation_steps_max = 100
        while validation_steps < validation_steps_max:
            # Check for negative contacts
            contact_pairs = self._find_contacts()

            # Handle any negative contacts found
            for contact in contact_pairs:
                # Unpack contact details
                geom1: str | None = cast("str|None", contact[0])
                geom2: str | None = cast("str|None", contact[1])
                dist: float = contact[2]

                # Get the floor name
                floor_name = self.mujoco_config.floor_name

                # Sometimes the spawn name is not given
                if geom1 is None:
                    geom1 = spawn_name
                if geom2 is None:
                    geom2 = spawn_name

                # Check if one of the geoms is the floor and the other is the robot
                one_geom_is_floor = floor_name in geom1 or floor_name in geom2
                one_geom_is_robot = spawn_name in geom1 or spawn_name in geom2
                if one_geom_is_floor and one_geom_is_robot:
                    # Log the collision details
                    msg = "Spawn position causes collision with the world."
                    log.debug(msg)
                    msg = f"Collision details: {geom1=}, {geom2=}, {dist=}"
                    log.debug(msg)

                    # Raise the spawn position by the penetration depth
                    correction = np.round(abs(dist) + 0.01, 5)
                    spawn.pos[2] += correction

                    # Log the new spawn position
                    msg = "Correcting automatically by raising the robot: \n"
                    msg += f"New spawn position: {spawn.pos} (+{correction} m)"
                    log.debug(msg)
                    break
            validation_steps += 1

        # Log the completion of the validation
        msg = f"Validation completed in {validation_steps} steps."
        msg += f" Final spawn position: {spawn.pos}"
        log.debug(msg)

    def spawn(
        self,
        robot_spec: mujoco.MjSpec,
        position: Position | None = None,
        rotation: Rotation | None = None,
        spawn_prefix: str | None = None,
        *,
        correct_spawn_for_collisions: bool = True,
        rotation_sequence: str = "XYZ",  # xyzXYZ, assume intrinsic
    ) -> mujoco.MjSpec:
        # Default spawn position
        if position is None:
            position = self.default_spawn_position
        else:
            position = deepcopy(position)

        # Default spawn orientation
        if rotation is None:
            rotation = self.default_spawn_rotation
        else:
            rotation = deepcopy(rotation)

        # If no prefix is given, use the default one
        if spawn_prefix is None:
            spawn_prefix = self.spawn_prefix

        # Make a copy of the robot and world specs
        temp_robot_spec = duplicate_mj_spec(robot_spec)

        # Convert rotation from Euler angles (degrees) to quaternion
        rotation_as_quat = euler_to_quat_conversion(
            rotation,
            rotation_sequence,
        )

        # Increment the spawn count
        self.spawns += 1

        # Create a spawn site at the specified position
        spawn_site = self.spec.worldbody.add_site(
            pos=np.array(position),
            quat=np.array(rotation_as_quat),
        )

        # Attach the robot body to the spawn site
        spawn_name = f"{spawn_prefix}{self.spawns}-"
        spawn = spawn_site.attach_body(
            body=temp_robot_spec.worldbody,
            prefix=spawn_name,
        )

        # Allow the robot to move freely
        spawn.add_freejoint()

        # Validate the updated world spec
        if correct_spawn_for_collisions is True:
            self._check_and_correct_spawn(spawn, spawn_name)

        # Return a copy of the updated spec
        return duplicate_mj_spec(self.spec)

    def compile_to_xml(self) -> None:
        # Derive save path
        this_script_path = Path(__file__)
        save_dir = this_script_path.parent / "pre_compiled"
        save_path = save_dir / f"{self.name}.xml"
        save_dir.mkdir(parents=True, exist_ok=True)
        xml = self.spec.to_xml()

        # Save file
        with save_path.open("w") as f:
            f.write(xml)

    def load_precompiled(self) -> bool:
        # Derive XML path
        this_script_path = Path(__file__)
        xml_path = this_script_path.parent / "pre_compiled" / f"{self.name}.xml"

        # Check if file exists
        if not xml_path.exists():
            return False

        # Load the XML file
        self.spec = mujoco.MjSpec.from_file(str(xml_path))
        return True

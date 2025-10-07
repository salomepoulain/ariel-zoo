"""TODO(jmdm): description of script.

Notes
-----
    *

References
----------
    [1]

Todo
----
    [ ]

"""

# Third-party libraries
import mujoco
import numpy as np

# Local libraries
from ariel.parameters.ariel_types import FloatArray, Rotation

# Global constants
# Global functions
# Warning Control
# Type Checking
# Type Aliases


def euler_to_quat_conversion(
    rotation: Rotation,
    rotation_sequence: str,
) -> FloatArray:
    """
    Convert Euler angles to a quaternion representation.

    Parameters
    ----------
    rotation
        Euler angles in degrees (x, y, z).
    rotation_sequence
        The sequence of axes for the rotation: "xyzXYZ" (e.g., "XYZ").

    Returns
    -------
        Quaternion representation of the Euler angles (x, y, z, w).
    """
    rotation_as_quat = np.zeros(4)
    mujoco.mju_euler2Quat(
        rotation_as_quat,
        np.deg2rad(rotation),
        rotation_sequence,
    )
    return rotation_as_quat


def duplicate_mj_spec(spec: mujoco.MjSpec) -> mujoco.MjSpec:
    """Create a copy of a MuJoCo specification.

    Parameters
    ----------
    spec
        The original MuJoCo specification to duplicate.

    Returns
    -------
    mujoco.MjSpec
        A deep copy of the provided MuJoCo specification.
    """
    return mujoco.MjSpec.from_string(spec.to_xml())

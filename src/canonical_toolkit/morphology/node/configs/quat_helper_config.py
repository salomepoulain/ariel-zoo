from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import quaternion as qnp

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces


@dataclass(slots=True)
class QuatHolder:
    """Helper class for Quaternion calculations"""
    
    float_array: np.ndarray | Sequence[float | int]

    _SHIFT: int = -1
    _DECIMALS: int = 3
    _ROT_ANGLE = 45 # based on the robotget_lite modules
    _SINGLE_ROTATION_STEP: tuple[int, ...] = (180, -(180 - _ROT_ANGLE), 0)
    _MUJOCO_NORMAL_VECTOR: tuple[int, ...] = (0, 1, 0)

    def __post_init__(self) -> None:
        if isinstance(self.float_array, np.ndarray):
            return

        self.float_array = np.roll(
            qnp.as_float_array(
                qnp.from_euler_angles(np.deg2rad(self.float_array)),
            ),
            shift=self._SHIFT,
        )

    @property
    def quat(self) -> np.quaternion:   # type: ignore
        return qnp.from_float_array(self.float_array)

    @property
    def unit_vectors(self) -> np.ndarray:  # Shape (3, 3)
        quat = self.quat
        local_vectors = np.eye(3)  # rows: X, Y, Z
        rotated = np.array([
            qnp.rotate_vectors(quat, vec) for vec in local_vectors
        ])
        return np.round(rotated, decimals=self._DECIMALS)

    @property
    def normal_vector(self) -> np.ndarray:
        return np.array(self._MUJOCO_NORMAL_VECTOR) @ self.unit_vectors

    def rotate(
        self,
        angle: int | None = None,
        axis: tuple[int, ...] | None = None,
    ) -> QuatHolder:
        if angle is not None or axis is not None:
            angle_to_use = angle if angle is not None else self._ROT_ANGLE
            axis_to_use = (
                axis if axis is not None else self._MUJOCO_NORMAL_VECTOR
            )
            angle_rad = np.deg2rad(angle_to_use)
            axis_normalized = np.array(axis_to_use, dtype=float)
            axis_normalized /= np.linalg.norm(axis_normalized)
            rotation_vector = axis_normalized * angle_rad
            body_rotation_quat = qnp.from_rotation_vector(rotation_vector)
        else:
            body_rotation_quat = QuatHolder(self._SINGLE_ROTATION_STEP).quat

        rotated = body_rotation_quat * self.quat
        return QuatHolder(qnp.as_float_array(rotated))

    def __repr__(self) -> str:
        return f"QuatHolder({np.round(self.float_array, self._DECIMALS)})"

    def __hash__(self) -> int:
        quat_tuple = (
            tuple(self.float_array.flat)
            if isinstance(self.float_array, np.ndarray)
            else tuple(self.quat)
        )
        return hash((quat_tuple, self._SHIFT, self._DECIMALS))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, QuatHolder):
            return False
        q1 = self.quat
        q2 = other.quat
        tolerance = 10 ** (-self._DECIMALS)
        return np.allclose(q1, q2, atol=tolerance) or np.allclose(
            q1,
            -q2,
            atol=tolerance,
        )

# based on the robotget_lite modules
FACE_QUATS: dict[ModuleFaces, QuatHolder] = {
    ModuleFaces.BACK: QuatHolder((0, 0, 0)),
    ModuleFaces.FRONT: QuatHolder((0, 180, 180)),
    ModuleFaces.RIGHT: QuatHolder((90, -90, -90)),
    ModuleFaces.LEFT: QuatHolder((90, 90, -90)),
    ModuleFaces.TOP: QuatHolder((0, 0, -90)),
    ModuleFaces.BOTTOM: QuatHolder((0, 180, 90)),
}

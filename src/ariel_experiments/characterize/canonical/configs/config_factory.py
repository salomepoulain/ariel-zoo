from dataclasses import dataclass
from enum import Enum

import numpy as np

from ariel.body_phenotypes.robogen_lite.config import (
    ALLOWED_ROTATIONS,
    ModuleFaces,
    ModuleRotationsIdx,
    ModuleType,
)
from ariel_experiments.characterize.canonical.configs.canonical_config import (
    CanonicalConfig,
)
from ariel_experiments.characterize.canonical.configs.quat_helper_config import (
    FACE_QUATS,
    QuatHolder,
)


# MARK helper classes

class SymmetryPlane(Enum):
    """Symmetry plane determines how many unique (max allowed) rotations a module has."""

    NONE = 1
    TWO_FOLD = 2
    FOUR_FOLD = 4


class Priority(Enum):
    """Importance for module, used for canonicalizing child order."""

    FIRST = 3
    SECOND = 2
    THIRD = 1
    LAST = 0


# MARK  factory class

@dataclass(slots=True)
class CanonicalConfigFactory:
    """Factory for creating CanonicalConfig"""

    @classmethod
    def create(
        cls,
        axial_faces: set[ModuleFaces] | None,
        radial_faces: set[ModuleFaces] | None,
        attachment_face: ModuleFaces | None,
        face_quats: dict[ModuleFaces, QuatHolder] | None,
        symmetry_plane: SymmetryPlane,
        allowed_rotations: list[ModuleRotationsIdx],
        priority: Priority,
    ) -> CanonicalConfig:
        max_allowed_rotations = cls._compute_max_allowed_rotations(
            allowed_rotations,
            symmetry_plane,
        )

        axial_face_order, rotation_axis = cls._build_axial_configuration(
            axial_faces,
            face_quats,
            attachment_face,
        )

        radial_face_order, radial_shift, radial_adjustments = (
            cls._build_radial_configuration(
                radial_faces,
                face_quats,
                rotation_axis,
                max_allowed_rotations,
            )
        )

        return CanonicalConfig(
            axial_face_order=tuple(axial_face_order),
            radial_face_order=tuple(radial_face_order),
            radial_shift=radial_shift,
            symmetry_plane=symmetry_plane.value,
            max_allowed_rotations=max_allowed_rotations,
            priority=priority.value,
            radial_adjustments=radial_adjustments,
        )

    # region config builders -----

    @classmethod
    def _build_axial_configuration(
        cls,
        axial_faces: set[ModuleFaces] | None,
        face_quats: dict[ModuleFaces, QuatHolder] | None,
        attachment_face: ModuleFaces | None,
    ) -> tuple[list[ModuleFaces], int | None]:
        """Build axial face configuration and determine rotation axis.

        Returns:
            (ordered_axial_faces, rotation_axis)
        """
        if not axial_faces or not face_quats:
            return [], None

        face_list = list(axial_faces)
        rotation_axis = cls._find_rotation_axis(face_list, face_quats)

        if rotation_axis is None:
            return [], None

        axial_face_order = cls._order_axial_faces(
            face_list,
            face_quats,
            rotation_axis,
        )

        # Remove attachment face if present
        if attachment_face in axial_face_order:
            axial_face_order.remove(attachment_face)

        return axial_face_order, rotation_axis

    @classmethod
    def _build_radial_configuration(
        cls,
        radial_faces: set[ModuleFaces] | None,
        face_quats: dict[ModuleFaces, QuatHolder] | None,
        rotation_axis: int | None,
        max_allowed_rotations: int,
    ) -> tuple[list[ModuleFaces], int, tuple[int, ...]]:
        """Build radial face configuration with shift and adjustments.

        Returns:
            (face_order, shift, adjustments)
        """
        if not radial_faces or not face_quats or rotation_axis is None:
            return [], 0, ()

        face_order = cls._order_radial_faces(radial_faces, face_quats)

        shift = cls._calculate_radial_shift(
            face_order,
            face_quats,
            rotation_axis,
            max_allowed_rotations,
        )

        adjustments = cls._calculate_radial_adjustments(
            face_order,
            face_quats,
            rotation_axis,
            shift,
            max_allowed_rotations,
        )

        return face_order, shift, adjustments

    # endregion

    # region axial face operations -----

    @staticmethod
    def _find_rotation_axis(
        faces: list[ModuleFaces],
        face_quats: dict[ModuleFaces, QuatHolder],
    ) -> int | None:
        """Find the axis around which two faces are opposed (normals sum to zero)."""
        if len(faces) != 2:
            return None

        for axis in range(3):
            normal_sum = (
                face_quats[faces[0]].normal_vector[axis]
                + face_quats[faces[1]].normal_vector[axis]
            )
            faces_opposed = (
                normal_sum == 0
                and face_quats[faces[1]].normal_vector[axis] != 0
            )

            if faces_opposed:
                return axis

        return None

    @staticmethod
    def _order_axial_faces(
        faces: list[ModuleFaces],
        face_quats: dict[ModuleFaces, QuatHolder],
        rotation_axis: int,
    ) -> list[ModuleFaces]:
        """Order axial faces from positive to negative along rotation axis."""
        if face_quats[faces[0]].normal_vector[rotation_axis] > 0:
            return [faces[0], faces[1]]
        return [faces[1], faces[0]]

    # endregion

    # region radial face operations -----

    @staticmethod
    def _order_radial_faces(
        radial_faces: set[ModuleFaces],
        face_quats: dict[ModuleFaces, QuatHolder],
    ) -> list[ModuleFaces]:
        """Order radial faces by normal vector orientation (axis, then sign)."""

        def get_sort_key(face: ModuleFaces) -> tuple[int, int]:
            normal = face_quats[face].normal_vector
            axis = int(np.argmax(np.abs(normal)))
            sign = 0 if normal[axis] > 0 else 1
            return (sign, axis)

        return sorted(radial_faces, key=get_sort_key)

    @classmethod
    def _calculate_radial_shift(
        cls,
        face_order: list[ModuleFaces],
        face_quats: dict[ModuleFaces, QuatHolder],
        rotation_axis: int,
        num_rotations: int,
    ) -> int:
        """Calculate how radial face indices shift after rotation."""
        if not face_order:
            return 0

        original_normals = [
            face_quats[face].normal_vector for face in face_order
        ]
        rotated_normals = cls._get_rotated_normals(
            face_order,
            face_quats,
            rotation_axis,
            num_rotations,
        )

        return -cls._find_cyclic_shift(original_normals, rotated_normals)

    @classmethod
    def _calculate_radial_adjustments(
        cls,
        face_order: list[ModuleFaces],
        face_quats: dict[ModuleFaces, QuatHolder],
        rotation_axis: int,
        shift: int,
        num_rotations: int,
    ) -> tuple[int, ...]:
        """Calculate per-face rotation adjustments to maintain alignment after shift.

        After the module rotates around its axis, the radial faces shift positions.
        This calculates how many additional rotations each face needs around its own
        normal to align with the face that was originally in that position.
        """
        original_quats = [face_quats[face] for face in face_order]

        rotated_quats = cls._rotate_quats_around_axis(
            original_quats,
            rotation_axis,
            num_rotations,
        )

        shifted_quats = rotated_quats[shift:] + rotated_quats[:shift]

        return tuple(
            cls._find_face_alignment(original_quats[i], shifted_quats[i])
            for i in range(len(original_quats))
        )

    # endregion

    # region rotation utilities -----

    @staticmethod
    def _compute_max_allowed_rotations(
        allowed_rotations: list[ModuleRotationsIdx],
        symmetry_plane: SymmetryPlane,
    ) -> int:
        """Calculate max allowed rotations based on symmetry."""
        result = len(allowed_rotations) // symmetry_plane.value
        return result if result != 0 else symmetry_plane.value

    @staticmethod
    def _get_rotated_normals(
        faces: list[ModuleFaces],
        face_quats: dict[ModuleFaces, QuatHolder],
        rotation_axis: int,
        num_rotations: int,
    ) -> list[np.ndarray]:
        """Get normal vectors after rotating faces around an axis."""
        axis_vector = [0, 0, 0]
        axis_vector[rotation_axis] = 1
        axis_tuple = tuple(axis_vector)

        quats = [face_quats[face] for face in faces]
        for _ in range(num_rotations):
            quats = [quat.rotate(axis=axis_tuple) for quat in quats]

        return [quat.normal_vector for quat in quats]

    @staticmethod
    def _rotate_quats_around_axis(
        quats: list[QuatHolder],
        rotation_axis: int,
        num_rotations: int,
    ) -> list[QuatHolder]:
        """Rotate quaternions around a cardinal axis."""
        axis_vector = [0, 0, 0]
        axis_vector[rotation_axis] = 1
        axis_tuple = tuple(axis_vector)

        result = quats.copy()
        for _ in range(num_rotations):
            result = [quat.rotate(axis=axis_tuple) for quat in result]

        return result

    @staticmethod
    def _find_cyclic_shift(
        original: list[np.ndarray],
        rotated: list[np.ndarray],
    ) -> int:
        """Find the cyclic shift between two lists of vectors.

        Returns the shift amount where rotated[i] == original[(i + shift) % n]
        """
        n = len(original)

        for shift in range(n):
            if all(
                np.allclose(rotated[i], original[(i + shift) % n])
                for i in range(n)
            ):
                return shift

        msg = "No valid cyclic shift found between vector lists"
        raise ValueError(msg)

    @staticmethod
    def _find_face_alignment(
        target_quat: QuatHolder,
        current_quat: QuatHolder,
        max_attempts: int = 8,
    ) -> int:
        """Find rotations around face normal needed to align quaternions.

        Rotates current_quat around its own normal vector until it matches
        target_quat, or returns 0 if no alignment found.

        Args:
            target_quat: Desired orientation
            current_quat: Current orientation to rotate
            max_attempts: Maximum rotations to try

        Returns:
            Number of rotations needed (0 if alignment impossible)
        """
        test_quat = current_quat

        for rotation_count in range(max_attempts):
            if test_quat == target_quat:
                return rotation_count

            test_quat = test_quat.rotate(axis=tuple(test_quat.normal_vector))

        return 0

        # endregion


CANONICAL_PRE_CONFIG: dict[ModuleType, CanonicalConfig] = {
    ModuleType.CORE: CanonicalConfigFactory.create(
        axial_faces={ModuleFaces.TOP, ModuleFaces.BOTTOM},
        radial_faces={
            ModuleFaces.LEFT,
            ModuleFaces.FRONT,
            ModuleFaces.RIGHT,
            ModuleFaces.BACK,
        },
        attachment_face=None,
        face_quats=FACE_QUATS,
        symmetry_plane=SymmetryPlane.FOUR_FOLD,
        allowed_rotations=ALLOWED_ROTATIONS[
            ModuleType.BRICK
        ],  # !  purposfully changed this. The symmetry of the brick hyptohetically allowes rotations
        priority=Priority.FIRST,
    ),
    ModuleType.BRICK: CanonicalConfigFactory.create(
        axial_faces={ModuleFaces.FRONT, ModuleFaces.BACK},
        radial_faces={
            ModuleFaces.LEFT,
            ModuleFaces.TOP,
            ModuleFaces.RIGHT,
            ModuleFaces.BOTTOM,
        },
        attachment_face=ModuleFaces.BACK,
        face_quats=FACE_QUATS,
        symmetry_plane=SymmetryPlane.FOUR_FOLD,
        allowed_rotations=ALLOWED_ROTATIONS[ModuleType.BRICK],
        priority=Priority.SECOND,
    ),
    ModuleType.HINGE: CanonicalConfigFactory.create(
        axial_faces={ModuleFaces.FRONT, ModuleFaces.BACK},
        radial_faces=None,
        face_quats=FACE_QUATS,
        attachment_face=ModuleFaces.BACK,
        symmetry_plane=SymmetryPlane.TWO_FOLD,
        allowed_rotations=ALLOWED_ROTATIONS[ModuleType.HINGE],
        priority=Priority.THIRD,
    ),
    ModuleType.NONE: CanonicalConfigFactory.create(
        axial_faces=None,
        radial_faces=None,
        face_quats=None,
        attachment_face=ModuleFaces.BACK,
        symmetry_plane=SymmetryPlane.NONE,
        allowed_rotations=ALLOWED_ROTATIONS[ModuleType.NONE],
        priority=Priority.LAST,
    ),
}


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    console.print(CANONICAL_PRE_CONFIG)

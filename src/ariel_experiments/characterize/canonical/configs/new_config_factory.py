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

class RotationalSymmetry(Enum):
    # """Symmetry plane determines how many unique (max allowed) rotations a module has."""
    """How many times the module looks identical during a full 360° rotation."""

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
    """Factory for creating CanonicalConfig

    Key improvements:
    - Radial faces ordered so radial_shift is always 1
    - Uses cumulative adjustments for O(1) lookup instead of loops
    """

    @classmethod
    def create(
        cls,
        axial_faces: set[ModuleFaces] | None,
        radial_faces: set[ModuleFaces] | None,
        attachment_face: ModuleFaces | None,
        face_quats: dict[ModuleFaces, QuatHolder] | None,
        rotational_symmetry: RotationalSymmetry,
        allowed_rotations: list[ModuleRotationsIdx],
        priority: Priority,
    ) -> CanonicalConfig:

        if radial_faces and len(radial_faces) not in (0, rotational_symmetry.value):
            raise ValueError(
                f"Radial faces must match symmetry: "
                f"{rotational_symmetry.name} symmetry requires "
                f"{rotational_symmetry.value} radial faces, got {len(radial_faces)}"
            )

        unique_rotation_amt = cls._compute_unique_allowed_rotations(
            allowed_rotations,
            rotational_symmetry,
        )

        axial_face_order, rotation_axis = cls._build_axial_configuration(
            axial_faces,
            face_quats,
            attachment_face,
        )

        radial_face_order, radial_cumulative_adjustments = (
            cls._build_radial_configuration(
                radial_faces,
                face_quats,
                rotation_axis,
                unique_rotation_amt,
            )
        )

        return CanonicalConfig(
            axial_face_order=tuple(axial_face_order),
            radial_face_order=tuple(radial_face_order),
            # symmetry_plane=rotational_symmetry.value, # TODO: change
            unique_rotation_amt=unique_rotation_amt,
            priority=priority.value,
            radial_adjustments=radial_cumulative_adjustments,
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
        unique_rotation_amt: int,
    ) -> tuple[list[ModuleFaces], tuple[tuple[int, ...], ...]]:
        """Build radial face configuration with cumulative adjustments.

        Returns:
            (face_order, cumulative_adjustments)
        """
        if not radial_faces or not face_quats or rotation_axis is None:
            return [], ()

        # Order faces so that shift is always 1
        face_order = cls._order_radial_faces_for_unit_shift(
            radial_faces,
            face_quats,
            rotation_axis,
            unique_rotation_amt,
        )

        # Calculate per-step adjustments
        per_step_adjustments = cls._calculate_radial_adjustments(
            face_order,
            face_quats,
            rotation_axis,
            shift=1,
            unique_rotation_amt=unique_rotation_amt,
        )

        # Convert to cumulative adjustments for O(1) lookup
        cumulative_adjustments = cls._compute_cumulative_adjustments(
            per_step_adjustments,
            len(radial_faces)
            # max_steps=8,  # Maximum rotation quanta to pre-compute
        )

        return face_order, cumulative_adjustments

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

    @classmethod
    def _order_radial_faces_for_unit_shift(
        cls,
        radial_faces: set[ModuleFaces],
        face_quats: dict[ModuleFaces, QuatHolder],
        rotation_axis: int,
        unique_rotation_amt: int,
    ) -> list[ModuleFaces]:
        """Order radial faces such that one rotation quantum shifts by exactly 1 position.

        Strategy:
        1. Start with an arbitrary ordering
        2. Rotate the module by unique_rotation_amt
        3. Find which position each face moved to
        4. Reorder so the shift is exactly 1
        """
        # Start with basic ordering by normal vector
        initial_order = cls._order_radial_faces_basic(radial_faces, face_quats)

        if not initial_order:
            return []

        # Get rotated normals after one quantum
        original_normals = [
            face_quats[face].normal_vector for face in initial_order
        ]
        rotated_normals = cls._get_rotated_normals(
            initial_order,
            face_quats,
            rotation_axis,
            unique_rotation_amt,
        )

        # Find the natural shift
        natural_shift = -cls._find_cyclic_shift(original_normals, rotated_normals)

        if natural_shift == 1:
            # Already perfect!
            return initial_order

        # Reorder to make shift = 1
        correction = (1 - natural_shift) % len(initial_order)
        reordered = initial_order[correction:] + initial_order[:correction]

        return reordered

    @staticmethod
    def _order_radial_faces_basic(
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
    def _calculate_radial_adjustments(
        cls,
        face_order: list[ModuleFaces],
        face_quats: dict[ModuleFaces, QuatHolder],
        rotation_axis: int,
        shift: int,
        unique_rotation_amt: int,
    ) -> tuple[int, ...]:
        """Calculate per-face rotation adjustments to maintain alignment after shift.

        After the module rotates around its axis, the radial faces shift positions.
        This calculates how many additional rotations each face needs around its own
        normal to align with the face that was originally in that position.

        Returns:
            Tuple where adjustments[i] = rotation needed when moving FROM face[i] TO face[i+1]
        """
        if not face_order:
            return ()

        original_quats = [face_quats[face] for face in face_order]

        rotated_quats = cls._rotate_quats_around_axis(
            original_quats,
            rotation_axis,
            unique_rotation_amt,
        )

        shifted_quats = rotated_quats[shift:] + rotated_quats[:shift]

        return tuple(
            cls._find_face_alignment(original_quats[i], shifted_quats[i])
            for i in range(len(original_quats))
        )

    @classmethod
    def _compute_cumulative_adjustments(
        cls,
        per_step_adjustments: tuple[int, ...],
        radial_face_amt: int,
        max_steps: int = len(ModuleRotationsIdx),
    ) -> tuple[tuple[int, ...], ...]:
        """Compute cumulative rotation adjustments for each starting position.

        This pre-computes the total adjustment needed when a child moves through
        multiple face positions, allowing O(1) lookup instead of looping.

        Args:
            per_step_adjustments: Adjustment needed when moving from face[i] to face[i+1]
            max_steps: Maximum number of rotation quanta to pre-compute

        Returns:
            2D structure where cumulative[starting_position][num_steps] = total adjustment

        Example:
            For 4 faces with per_step_adjustments = (2, 1, 2, 1):
            cumulative[0] = (0, 2, 3, 5, 6, 8, 9, 11, 12)  # Starting from face 0
            cumulative[1] = (0, 1, 3, 4, 6, 7, 9, 10, 12)  # Starting from face 1
            cumulative[2] = (0, 2, 3, 5, 6, 8, 9, 11, 12)  # Starting from face 2
            cumulative[3] = (0, 1, 3, 4, 6, 7, 9, 10, 12)  # Starting from face 3
        """
        if not per_step_adjustments:
            return ()

        num_faces = len(per_step_adjustments)
        cumulative_all_positions = []

        for starting_pos in range(num_faces):
            # Compute cumulative sum starting from this position
            position_cumulative = [0]  # After 0 steps, no adjustment needed

            print(len(ModuleRotationsIdx))

            for step in range(1, radial_face_amt):
                # Which face are we leaving at this step?
                # (We apply the adjustment of the face we're leaving)
                face_leaving = (starting_pos + step - 1) % num_faces

                # Add that face's adjustment to cumulative total
                total = (position_cumulative[-1] + per_step_adjustments[face_leaving]) % max_steps
                position_cumulative.append(total)

            cumulative_all_positions.append(tuple(position_cumulative))

        return tuple(cumulative_all_positions)

    # endregion

    # region rotation utilities -----

    @staticmethod
    def _compute_unique_allowed_rotations(
        allowed_rotations: list[ModuleRotationsIdx],
        rotational_symmetry: RotationalSymmetry,
    ) -> int:
        """Calculate unique rotation increments based on symmetry.

        This is the 'quantum' of rotation - the smallest increment that
        produces a visually distinct orientation.
        """
        result = len(allowed_rotations) // rotational_symmetry.value
        return result if result != 0 else rotational_symmetry.value

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
        max_attempts: int = len(ModuleRotationsIdx),
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
        rotational_symmetry=RotationalSymmetry.FOUR_FOLD,
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
        rotational_symmetry=RotationalSymmetry.FOUR_FOLD,
        allowed_rotations=ALLOWED_ROTATIONS[ModuleType.BRICK],
        priority=Priority.SECOND,
    ),
    ModuleType.HINGE: CanonicalConfigFactory.create(
        axial_faces={ModuleFaces.FRONT, ModuleFaces.BACK},
        radial_faces=None,
        face_quats=FACE_QUATS,
        attachment_face=ModuleFaces.BACK,
        rotational_symmetry=RotationalSymmetry.TWO_FOLD,
        allowed_rotations=ALLOWED_ROTATIONS[ModuleType.HINGE],
        priority=Priority.THIRD,
    ),
    ModuleType.NONE: CanonicalConfigFactory.create(
        axial_faces=None,
        radial_faces=None,
        face_quats=None,
        attachment_face=ModuleFaces.BACK,
        rotational_symmetry=RotationalSymmetry.NONE,
        allowed_rotations=ALLOWED_ROTATIONS[ModuleType.NONE],
        priority=Priority.LAST,
    ),
}


if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    console.print(CANONICAL_PRE_CONFIG)

    # Demonstrate the cumulative adjustments for BRICK
    print("\n=== BRICK Module Cumulative Adjustments ===")
    brick_config = CANONICAL_PRE_CONFIG[ModuleType.BRICK]
    print(f"Radial faces: {brick_config.radial_face_order}")
    print(f"Cumulative adjustments by starting position:")
    for i, cumulative in enumerate(brick_config.radial_adjustments):
        face_name = brick_config.radial_face_order[i].name if i < len(brick_config.radial_face_order) else "N/A"
        print(f"  Starting from {face_name} (pos {i}): {cumulative}")


    # Test HINGE with 2 radial faces
    test_hinge = CanonicalConfigFactory.create(
        axial_faces={ModuleFaces.FRONT, ModuleFaces.BACK},
        radial_faces={ModuleFaces.TOP, ModuleFaces.BOTTOM},
        attachment_face=ModuleFaces.BACK,
        face_quats=FACE_QUATS,
        rotational_symmetry=RotationalSymmetry.TWO_FOLD,
        allowed_rotations=ALLOWED_ROTATIONS[ModuleType.HINGE],
        priority=Priority.THIRD,
    )

    print("\n=== HINGE with 2 radial faces ===")
    print(f"Radial face order: {test_hinge.radial_face_order}")
    print(f"Unique rotation amt: {test_hinge.unique_rotation_amt}")
    print(f"Cumulative adjustments: {test_hinge.radial_adjustments}")

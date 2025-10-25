
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CanonicalConfig:
    """Pre-computed canonical configuration for a module type."""

    axial_face_order: tuple[ModuleFaces, ...]
    radial_face_order: tuple[ModuleFaces, ...]
    radial_shift: int
    symmetry_plane: int
    max_allowed_rotations: int
    priority: int
    radial_adjustments: tuple[int, ...]


# The following configs were obtained using the factory
CANONICAL_CONFIGS: dict[ModuleType, CanonicalConfig] = {
    ModuleType.CORE: CanonicalConfig(
        axial_face_order=(ModuleFaces.BOTTOM, ModuleFaces.TOP),
        radial_face_order=(
            ModuleFaces.LEFT,
            ModuleFaces.FRONT,
            ModuleFaces.RIGHT,
            ModuleFaces.BACK
        ),
        radial_shift=-1,
        symmetry_plane=4,
        max_allowed_rotations=2,
        priority=3,
        radial_adjustments=(0, 0, 0, 0)
    ),
    ModuleType.BRICK: CanonicalConfig(
        axial_face_order=(ModuleFaces.FRONT,),
        radial_face_order=(
            ModuleFaces.LEFT,
            ModuleFaces.BOTTOM,
            ModuleFaces.RIGHT,
            ModuleFaces.TOP
        ),
        radial_shift=-3,
        symmetry_plane=4,
        max_allowed_rotations=2,
        priority=2,
        radial_adjustments=(6, 6, 2, 2)
    ),
    ModuleType.HINGE: CanonicalConfig(
        axial_face_order=(ModuleFaces.FRONT,),
        radial_face_order=(),
        radial_shift=0,
        symmetry_plane=2,
        max_allowed_rotations=4,
        priority=1,
        radial_adjustments=()
    ),
    ModuleType.NONE: CanonicalConfig(
        axial_face_order=(),
        radial_face_order=(),
        radial_shift=0,
        symmetry_plane=1,
        max_allowed_rotations=1,
        priority=0,
        radial_adjustments=()
    )
}

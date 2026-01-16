from dataclasses import dataclass

from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType


@dataclass(frozen=True, slots=True)
class CanonicalConfig:
    """Pre-computed canonical configuration for a module type."""

    module_type: ModuleType
    axial_face_order: tuple[ModuleFaces, ...]
    radial_face_order: tuple[ModuleFaces, ...]
    unique_rotation_amt: int
    # priority: int
    radial_adjustments: tuple[tuple[int, ...]]


# The following configs were obtained using the factory and hardcoded in here for readibility
CANONICAL_CONFIGS: dict[ModuleType, CanonicalConfig] = {
    ModuleType.CORE: CanonicalConfig(
        module_type=ModuleType.CORE,
        axial_face_order=(ModuleFaces.BOTTOM, ModuleFaces.TOP),
        radial_face_order=(
            ModuleFaces.RIGHT,
            ModuleFaces.BACK,
            ModuleFaces.LEFT,
            ModuleFaces.FRONT,
        ),
        unique_rotation_amt=2,
        # priority=3,
        radial_adjustments=(
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
            (0, 0, 0, 0),
        ),
    ),
    ModuleType.BRICK: CanonicalConfig(
        module_type=ModuleType.BRICK,
        axial_face_order=(ModuleFaces.FRONT,),
        radial_face_order=(
            ModuleFaces.LEFT,
            ModuleFaces.BOTTOM,
            ModuleFaces.RIGHT,
            ModuleFaces.TOP,
        ),
        unique_rotation_amt=2,
        # priority=2,
        radial_adjustments=(
            (0, 6, 4, 6),
            (0, 6, 0, 2),
            (0, 2, 4, 2),
            (0, 2, 0, 6),
        ),
    ),
    ModuleType.HINGE: CanonicalConfig(
        module_type=ModuleType.HINGE,
        axial_face_order=(ModuleFaces.FRONT,),
        radial_face_order=(),
        unique_rotation_amt=4,
        # priority=1,
        radial_adjustments=(),
    ),
    ModuleType.NONE: CanonicalConfig(
        module_type=ModuleType.NONE,
        axial_face_order=(),
        radial_face_order=(),
        unique_rotation_amt=1,
        # priority=0,
        radial_adjustments=(),
    ),
}

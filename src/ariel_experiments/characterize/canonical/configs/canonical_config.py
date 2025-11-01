
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces, ModuleType
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CanonicalConfig:
    """Pre-computed canonical configuration for a module type."""

    axial_face_order: tuple[ModuleFaces, ...]
    radial_face_order: tuple[ModuleFaces, ...]
    # radial_shift: int
    # symmetry_plane: int
    unique_rotation_amt: int
    priority: int
    radial_adjustments: tuple[tuple[int, ...]]


# # The following configs were obtained using the factory
# CANONICAL_CONFIGS: dict[ModuleType, CanonicalConfig] = {
#     ModuleType.CORE: CanonicalConfig(
#         axial_face_order=(ModuleFaces.BOTTOM, ModuleFaces.TOP),
#         radial_face_order=(
#             ModuleFaces.LEFT,
#             ModuleFaces.FRONT,
#             ModuleFaces.RIGHT,
#             ModuleFaces.BACK
#         ),
#         # radial_shift=-1,
#         symmetry_plane=4,
#         unique_rotation_amt=2,
#         priority=3,
#         radial_adjustments=(0, 0, 0, 0)
#     ),
#     ModuleType.BRICK: CanonicalConfig(
#         axial_face_order=(ModuleFaces.FRONT,),
#         radial_face_order=(
#             ModuleFaces.LEFT,
#             ModuleFaces.BOTTOM,
#             ModuleFaces.RIGHT,
#             ModuleFaces.TOP
#         ),
#         # radial_shift=-3,
#         symmetry_plane=4,
#         unique_rotation_amt=2,
#         priority=2,
#         radial_adjustments=(6, 6, 2, 2)
#     ),
#     ModuleType.HINGE: CanonicalConfig(
#         axial_face_order=(ModuleFaces.FRONT,),
#         radial_face_order=(),
#         # radial_shift=0,
#         symmetry_plane=2,
#         unique_rotation_amt=4,
#         priority=1,
#         radial_adjustments=()
#     ),
#     ModuleType.NONE: CanonicalConfig(
#         axial_face_order=(),
#         radial_face_order=(),
#         # radial_shift=0,
#         symmetry_plane=1,
#         unique_rotation_amt=1,
#         priority=0,
#         radial_adjustments=()
#     )
# }





# The following configs were obtained using the factory
CANONICAL_CONFIGS: dict[ModuleType, CanonicalConfig] = {
    ModuleType.CORE: CanonicalConfig(
        axial_face_order=(ModuleFaces.BOTTOM, ModuleFaces.TOP),
        radial_face_order=(
            ModuleFaces.RIGHT,
            ModuleFaces.BACK,
            ModuleFaces.LEFT,
            ModuleFaces.FRONT
        ),
        # radial_shift=1,
        # symmetry_plane=4,
        unique_rotation_amt=2,
        priority=3,
        radial_adjustments=((0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0))
    ),
    ModuleType.BRICK: CanonicalConfig(
        axial_face_order=(ModuleFaces.FRONT,),
        radial_face_order=(
            ModuleFaces.LEFT,
            ModuleFaces.BOTTOM,
            ModuleFaces.RIGHT,
            ModuleFaces.TOP
        ),
        # radial_shift=1,
        # symmetry_plane=4,
        unique_rotation_amt=2,
        priority=2,
        radial_adjustments=((0, 6, 4, 6), (0, 6, 0, 2), (0, 2, 4, 2), (0, 2, 0, 6))
        # radial_adjustments=(6, 6, 2, 2)
    ),
    ModuleType.HINGE: CanonicalConfig(
        axial_face_order=(ModuleFaces.FRONT,),
        radial_face_order=(),
        # radial_shift=0,
        # symmetry_plane=2,
        unique_rotation_amt=4,
        priority=1,
        radial_adjustments=()
    ),
    ModuleType.NONE: CanonicalConfig(
        axial_face_order=(),
        radial_face_order=(),
        # radial_shift=0,
        # symmetry_plane=1,
        unique_rotation_amt=1,
        priority=0,
        radial_adjustments=()
    )
}


# def get_config(type: ModuleType):
#     from ariel_experiments.characterize.canonical.configs.config_factory import CANONICAL_PRE_CONFIG
#     return CANONICAL_PRE_CONFIG[type]



# CANONICAL_CONFIGS: dict[ModuleType, CanonicalConfig] = {
#     ModuleType.CORE: get_config(ModuleType.CORE),
#     ModuleType.BRICK: get_config(ModuleType.BRICK),
#     ModuleType.HINGE: get_config(ModuleType.HINGE),
#     ModuleType.NONE: get_config(ModuleType.NONE),
# }


if __name__ == "__main__":
    print(CANONICAL_CONFIGS)

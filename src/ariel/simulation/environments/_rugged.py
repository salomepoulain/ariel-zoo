"""MuJoCo world: rugged heightmap terrain."""

# Standard library
from dataclasses import dataclass

# Local libraries
from ariel.parameters.ariel_types import Dimension
from ariel.simulation.environments._compound_world import CompoundWorld
from ariel.simulation.environments.heightmap_functions import (
    rugged_heightmap,
    smooth_edges,
)
from ariel.utils.noise_gen import NormMethod


@dataclass
class RuggedTerrainWorld(CompoundWorld):
    """A rugged terrain world (CompoundWorld)."""

    name: str = "rugged-world"

    floor_size: Dimension = (10, 10, 1)  # meters (width, height, depth)
    checker_floor: bool = False

    # Rugged heightmap parameters
    dims: tuple[int, int] = (100, 100)
    scale_of_noise: int = 4
    normalize: NormMethod = "none"

    def __post_init__(self) -> None:
        # Create heightmap
        self.floor_heightmap = rugged_heightmap(
            self.dims,
            self.scale_of_noise,
            self.normalize,
        )
        self.floor_heightmap *= smooth_edges(self.dims, edge_width=0.2)

        # Initialize base class
        super().__init__(
            name=self.name,
            floor_size=self.floor_size,
            checker_floor=self.checker_floor,
            floor_heightmap=self.floor_heightmap,
        )

import click
import numpy as np
import torch
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import ariel.ec.a004 as a004
from ariel.ec.a004 import EASettings
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding

from ariel.utils.optimizers.revde import RevDE

import canonical_toolkit as ctk

class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Seed and file handling
    SEED: int = 42
    RUN_NAME: str = "run"
    QUIET: bool = False
    STORE_STRING: bool = True
    
    NUM_MODULES: int = 20
    GENOTYPE_SIZE: int = 64
    SCALE: float = 8192
    REV_SCALING_FACTOR: float = 0.5
    
    POPULATION_SIZE: int = 300
    NUM_GENERATIONS: int = 100
    
    # SURVIVOR_TOURNAMENT_SIZE: int = 10
    IS_MAXIMISATION: bool = True
    
    K_NOVELTY: int = 1
    ARCHIVE_CHANCE: float = 0.1

    # Derived (set in model_validator, excluded from __init__)
    RNG: np.random.Generator = Field(default=None, validate_default=False)  # type: ignore[assignment]
    NDE: NeuralDevelopmentalEncoding = Field(default=None, validate_default=False) # type: ignore[assignment]
    REVDE: RevDE = Field(default=RevDE(-0.5), validate_default=False)
    
    EA_SETTINGS: EASettings = Field(default=EASettings, validate_default=False)  # type: ignore[assignment]
    SIM_CONFIG: ctk.SimilaritySpaceConfig = Field(default=ctk.SimilaritySpaceConfig(), validate_default=False)
    
    OUTPUT_FOLDER: Path = Field(default=Path("__data__"), validate_default=False)


    @model_validator(mode="after")
    def setup(self) -> "Config":
        # Set all random seeds
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        torch.use_deterministic_algorithms(True, warn_only=True)

        object.__setattr__(self, "RNG", np.random.default_rng(self.SEED))

        # Create output folder with auto-incrementing suffix
        base_folder = Path.cwd() / "__data__" / self.RUN_NAME
        counter = 1
        output_folder = base_folder.parent / f"{base_folder.name}_{counter:04d}"
        while output_folder.exists():
            counter += 1
            output_folder = base_folder.parent / f"{base_folder.name}_{counter:04d}"
        output_folder.mkdir(exist_ok=True, parents=True)
        object.__setattr__(self, "OUTPUT_FOLDER", output_folder)
        
        object.__setattr__(self, "NDE", NeuralDevelopmentalEncoding(self.NUM_MODULES, self.GENOTYPE_SIZE))

        revde = RevDE(self.REV_SCALING_FACTOR)
        object.__setattr__(self, "REVDE", revde)
        
        sim_config = ctk.SimilaritySpaceConfig()
        object.__setattr__(self, "SIM_CONFIG", sim_config)

        ea_settings = EASettings(
            output_folder=output_folder,
            db_file_path=output_folder / "database.db",
            quiet=self.QUIET,
            is_maximisation=self.IS_MAXIMISATION,
            target_population_size=self.POPULATION_SIZE,
            num_of_generations=self.NUM_GENERATIONS,
        )
        object.__setattr__(self, "EA_SETTINGS", ea_settings)
        a004.config = ea_settings

        self.save(output_folder / "config.json")
        return self

    def save(self, path: Path) -> None:
        """Save config to JSON file (excludes non-serializable fields)."""
        path.write_text(
            self.model_dump_json(indent=2, exclude={"RNG", "EA_SETTINGS", "REVDE", "NDE", "SIM_CONFIG", "OUTPUT_FOLDER"})
        )

    @field_validator("POPULATION_SIZE", "NUM_GENERATIONS", "NUM_MODULES",)
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("Must be positive")
        return v


config: Config = None  # type: ignore[assignment]

def cli_options(func):
    """Add all config CLI options to a function."""
    @click.command()
    @click.option("--SEED", type=int, help="Random seed for reproducibility (default: 42)")
    @click.option("--POPULATION_SIZE", type=int, help="Population size (default: 300)")
    @click.option("--NUM_GENERATIONS", type=int, help="Number of generations (default: 100)")
    @click.option("--RUN_NAME", type=str, help="Name for output folder (default: ea_run)")
    @click.option("--QUIET", is_flag=True, help="Suppress output")
    def wrapper(**cli_args):
        global config
        # Click lowercases option names, but Config expects uppercase
        overrides = {k.upper(): v for k, v in cli_args.items() if v is not None}
        config = Config(**overrides)
        return func()
    return wrapper

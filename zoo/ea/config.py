from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime

import click
import numpy as np
import torch
from pathlib import Path
from rich.console import Console
from rich.table import Table


import logging
from rich.logging import RichHandler
from rich.console import Console

# Create console and logger
console = Console()

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, show_path=False)]
)

logger = logging.getLogger("ea")

# Detect if running in notebook/interactive mode
def _in_notebook():
    try:
        get_ipython()  # type: ignore
        return True
    except NameError:
        return False

def _is_main_process():
    import sys
    # On macOS with spawn, worker processes have __spec__.name as "__mp_main__"
    # or are invoked via multiprocessing.spawn
    if hasattr(sys.modules.get("__main__", None), "__spec__"):
        spec = sys.modules["__main__"].__spec__
        if spec and spec.name == "__mp_main__":
            return False
    # Also check process name as fallback
    import multiprocessing
    return multiprocessing.current_process().name == "MainProcess"

if _in_notebook() or not _is_main_process():
    # Skip loading animation in notebooks and worker processes
    from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
    import ariel.ec.a004 as a004
    from ariel.ec.a004 import EASettings
    from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
    from ariel.utils.optimizers.revde import RevDE
    import canonical_toolkit as ctk
else:
    console.rule()
    with console.status("[bold cyan]Loading imports and settings...", spinner="dots"):
        from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator, model_validator
        import ariel.ec.a004 as a004
        from ariel.ec.a004 import EASettings
        from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
        from ariel.utils.optimizers.revde import RevDE
        import canonical_toolkit as ctk

from enum import Enum

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class NoveltyMethod(str, Enum):
    CTK = "ctk"
    FUDA = "fuda"
    TED = "ted"

class Config(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    _skip_setup: bool = PrivateAttr(default=False)
    _output_dir: Path | None = PrivateAttr(default=None)

    # ? ------------------------
    SEED: int = 42
    RUN_NAME: str = "run"
    QUIET: bool = False
    LOG_LEVEL: LogLevel = LogLevel.INFO
    STORE_STRING: bool = True
    STORE_IMG: bool = True
    COPY_ANALYSIS: bool = True
    # ? ------------------------
    NUM_MODULES: int = 20
    GENOTYPE_SIZE: int = 64
    SCALE: float = 8192
    # ? ------------------------
    REV_SCALING_FACTOR: float = -0.5
    # ? ------------------------
    POPULATION_SIZE: int = 500
    NUM_GENERATIONS: int = 100
    IS_MAXIMISATION: bool = True
    # ? ------------------------
    PENALTY: bool = False
    FITNESS_NOVELTY: bool = True
    NOVELTY_METHOD: str = NoveltyMethod.TED.value
    FITNESS_SPEED: bool = True
    STORE_NOVELTY: bool | None = None
    STORE_SPEED: bool | None = None
    TOTAL_SIM_DURATION: float = 90.0
    SIM_WARMUP: float = 30.0
    # ? ------------------------
    TOURNAMENT: bool = True
    K_TOURNAMENT: int = 2
    # ? ------------------------
    SAVE_SPACES: list[ctk.Space] | None = ctk.Space.all_spaces()
    NOVELTY_SPACES: list[ctk.Space] | None = ctk.Space.all_spaces()
    MAX_HOP_RADIUS: int | None = 3
    SKIP_EMPTY: bool = False
    # ? ------------------------
    K_NOVELTY: int = 1
    ARCHIVE_CHANCE: float = 0.1
    # ? ------------------------
    # Derived (set in model_validator, excluded from __init__)
    RNG: np.random.Generator = Field(default=None, validate_default=False)  # type: ignore[assignment]
    NDE: NeuralDevelopmentalEncoding = Field(default=None, validate_default=False) # type: ignore[assignment]
    REVDE: RevDE = Field(default=RevDE(-0.5), validate_default=False)
    EA_SETTINGS: EASettings = Field(default=EASettings, validate_default=False)  # type: ignore[assignment]
    SIM_CONFIGS: list[ctk.SimilaritySpaceConfig] = Field(default=[], validate_default=False)
    OUTPUT_FOLDER: Path = Field(default=Path("__data__"), validate_default=False)



    @model_validator(mode="after")
    def validate_fitness_store(self) -> "Config":
        # Auto-inherit from FITNESS_* if STORE_* is None
        if self.STORE_NOVELTY is None:
            object.__setattr__(self, "STORE_NOVELTY", self.FITNESS_NOVELTY)
        if self.STORE_SPEED is None:
            object.__setattr__(self, "STORE_SPEED", self.FITNESS_SPEED)
        if self.NOVELTY_SPACES is None:
            object.__setattr__(self, "NOVELTY_SPACES", self.SAVE_SPACES)

        # Validate: can't use for fitness without storing
        if self.FITNESS_NOVELTY and not self.STORE_NOVELTY:
            raise ValueError("FITNESS_NOVELTY requires STORE_NOVELTY to be True")
        if self.FITNESS_SPEED and not self.STORE_SPEED:
            raise ValueError("FITNESS_SPEED requires STORE_SPEED to be True")
        if not set(self.NOVELTY_SPACES).issubset(set(self.SAVE_SPACES)):
            raise ValueError("NOVELTY_SPACES must be a subset of SAVE_SPACES")

        if not self.STORE_STRING:
            raise ValueError("Current ea relies heavily on string storage")
        return self

    def _initialize(self, folder: Path | None = None, output_dir: Path | None = None) -> None:
        """Initialize derived fields. If folder is None, creates a new output folder.

        Args:
            folder: Existing folder to load from (used by Config.load())
            output_dir: CLI-specified output directory (overwrites if exists)
        """
        np.random.seed(self.SEED)
        torch.manual_seed(self.SEED)
        # torch.use_deterministic_algorithms(True, warn_only=True)

        object.__setattr__(self, "RNG", np.random.default_rng(self.SEED))

        if folder is not None:
            # Loading from existing config
            save_config = False
        elif output_dir is not None:
            # CLI-specified output dir - use as-is, overwrite if exists
            folder = output_dir
            folder.mkdir(exist_ok=True, parents=True)
            save_config = True
        else:
            # Auto-increment logic
            base_folder = Path.cwd() / "__data__" / self.RUN_NAME
            counter = 1
            folder = base_folder.parent / f"{base_folder.name}_{counter:04d}"
            while folder.exists():
                counter += 1
                folder = base_folder.parent / f"{base_folder.name}_{counter:04d}"
            folder.mkdir(exist_ok=True, parents=True)
            save_config = True

        object.__setattr__(self, "OUTPUT_FOLDER", folder)
        object.__setattr__(self, "NDE", NeuralDevelopmentalEncoding(self.NUM_MODULES, self.GENOTYPE_SIZE))
        object.__setattr__(self, "REVDE", RevDE(self.REV_SCALING_FACTOR))
        object.__setattr__(self, "SIM_CONFIGS", [
            ctk.SimilaritySpaceConfig(space=space, max_hop_radius=self.MAX_HOP_RADIUS, skip_empty=self.SKIP_EMPTY)
            for space in self.SAVE_SPACES
        ])

        ea_settings = EASettings(
            output_folder=folder,
            db_file_path=folder / "database.db",
            quiet=self.QUIET,
            is_maximisation=self.IS_MAXIMISATION,
            target_population_size=self.POPULATION_SIZE,
            num_of_generations=self.NUM_GENERATIONS,
        )
        object.__setattr__(self, "EA_SETTINGS", ea_settings)
        a004.config = ea_settings

        if save_config:
            self.save(folder / "config.json")

            # Copy analysis folder if it exists and flag is set
            if self.COPY_ANALYSIS:
                analysis_src = Path.cwd() / "analysis"
                if analysis_src.exists():
                    shutil.copytree(analysis_src, folder / "analysis")

    @model_validator(mode="after")
    def setup(self) -> "Config":
        if self._skip_setup:
            return self
        self._initialize(output_dir=self._output_dir)
        return self

    def save(self, path: Path) -> None:
        """Save config to JSON file (excludes non-serializable fields)."""
        path.write_text(
            self.model_dump_json(indent=2, exclude={"RNG", "EA_SETTINGS", "REVDE", "NDE", "SIM_CONFIGS", "OUTPUT_FOLDER"})
        )

    def __rich__(self) -> Table:
        """Rich representation for console.print(config)."""
        # table = Table(title="EA Configuration", show_header=True)
        # table.add_column("Setting", style="cyan")
        # table.add_column("Value", style="green")

        # try:
        #     folder_str = str(self.OUTPUT_FOLDER.relative_to(Path.cwd()))
        # except ValueError:
        #     folder_str = str(self.OUTPUT_FOLDER)
        # table.add_row("Output Folder", folder_str)
        # table.add_row("Seed", str(self.SEED))
        # table.add_row("Population Size", str(self.POPULATION_SIZE))
        # table.add_row("Generations", str(self.NUM_GENERATIONS))
        # table.add_row("Fitness: Novelty", "✓" if self.FITNESS_NOVELTY else "✗")
        # table.add_row("Fitness: Speed", "✓" if self.FITNESS_SPEED else "✗")
        # table.add_row("Store: Novelty", "✓" if self.STORE_NOVELTY else "✗")
        # table.add_row("Store: Speed", "✓" if self.STORE_SPEED else "✗")
        return self.large_description()

    def large_description(self) -> Table:
        """Full table showing all config values."""
        table = Table(title="EA Configuration (Full)", show_header=True)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        table.add_column("Type", style="dim")

        # Exclude non-serializable derived fields
        exclude = {"RNG", "EA_SETTINGS", "REVDE", "NDE", "SIM_CONFIGS"}

        for field_name, field_info in self.model_fields.items():
            if field_name in exclude:
                continue

            value = getattr(self, field_name)

            # Format value for display
            if isinstance(value, bool):
                value_str = "✓" if value else "✗"
            elif isinstance(value, Path):
                try:
                    value_str = str(value.relative_to(Path.cwd()))
                except ValueError:
                    value_str = str(value)
            else:
                value_str = str(value)

            # Get type annotation
            type_str = str(field_info.annotation).replace("typing.", "").replace("<class '", "").replace("'>", "")

            table.add_row(field_name, value_str, type_str)

        return table

    @classmethod
    def load(cls, folder: Path | str) -> Config:
        """Read config from a JSON file and return a new instance."""
        folder = Path(folder)
        path = folder / "config.json"
        data = json.loads(path.read_text())

        # Build instance without triggering setup, then initialize with existing folder
        instance = cls.model_construct(**data)
        instance._skip_setup = True
        instance._initialize(folder)

        return instance

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
    @click.option("-p", "--POPULATION_SIZE", type=int, help="Population size (default: 300)")
    @click.option("-g", "--NUM_GENERATIONS", type=int, help="Number of generations (default: 100)")
    @click.option("-r", "--RUN_NAME", type=str, help="Name for output folder (default: ea_run)")
    @click.option("-o", "--output-dir", type=click.Path(path_type=Path), help="Output directory (overwrites if exists, bypasses auto-increment)")
    @click.option("-T", "--no-TOURNAMENT", "tournament", is_flag=True, default=None, flag_value=False, help="Disable tournament survival selection for deterministic elitism")
    @click.option("-P", "--PENALTY", is_flag=True, default=None, help="Add short limb fitness penalty")
    @click.option("-q", "--QUIET", is_flag=True, default=None, help="Suppress output")
    @click.option("-N", "--no-FITNESS_NOVELTY", "fitness_novelty", is_flag=True, flag_value=False, default=None, help="Disable novelty in fitness")
    @click.option("-S", "--no-FITNESS_SPEED", "fitness_speed", is_flag=True, flag_value=False, default=None, help="Disable speed in fitness")
    @click.option("-n", "--STORE_NOVELTY", is_flag=True, default=None, help="Store novelty")
    @click.option("-s", "--STORE_SPEED", is_flag=True, default=None, help="Store speed")
    @click.option(
        "-m",
        "--NOVELTY_METHOD",
        type=click.Choice(["ctk", "fuda"], case_sensitive=False),
        help="Select novelty algorithm: 'ctk' (graph-based) or 'fuda' (morphological)"
    )
    def wrapper(**cli_args):
        global config
        # Extract output_dir separately (it's a private attr, not a field)
        output_dir = cli_args.pop("output_dir", None)
        # Click lowercases option names, but Config expects uppercase
        overrides = {k.upper(): v for k, v in cli_args.items() if v is not None}

        if output_dir is not None:
            # Warn and delete if folder already exists
            if output_dir.exists():
                console.log(f"⚠️  Deleting folder: {output_dir}", style="yellow")
                shutil.rmtree(output_dir)

            # Build without auto-setup, then initialize with output_dir
            instance = Config.model_construct(**overrides)
            instance._skip_setup = True
            # Apply validation that model_construct skips
            if instance.STORE_NOVELTY is None:
                object.__setattr__(instance, "STORE_NOVELTY", instance.FITNESS_NOVELTY)
            if instance.STORE_SPEED is None:
                object.__setattr__(instance, "STORE_SPEED", instance.FITNESS_SPEED)
            instance._initialize(output_dir=output_dir)
            config = instance
        else:
            config = Config(**overrides)

        # Display config table
        if not config.QUIET:
            console.print(config)

        # Append to run history
        history_path = Path.cwd() / "run_history.csv"
        file_exists = history_path.exists()
        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["datetime", "cli_args", "output_folder"])
            cli_str = " ".join(f"--{k}={v}" for k, v in overrides.items())
            try:
                output_path = str(config.OUTPUT_FOLDER.relative_to(Path.cwd()))
            except ValueError:
                output_path = str(config.OUTPUT_FOLDER)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                cli_str,
                output_path,
            ])

        return func()
    return wrapper


if __name__ == "__main__":
    config = Config.load('__data__/run_0001')

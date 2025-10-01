"""ARIEL."""

# Standard library
from pathlib import Path

# Third-party libraries
import numpy as np
from rich.console import Console
from rich.traceback import install

# --- DATA SETUP --- #
CWD = Path.cwd()
DATA = Path(CWD / "__data__")

# --- TERMINAL OUTPUT SETUP --- #
install()
console = Console()

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

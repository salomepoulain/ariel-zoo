"""Compile all terrains to XML files."""

# Standard library
import inspect

# Third-party libraries
from rich.console import Console
from rich.traceback import install

# Local libraries
import ariel.simulation.environments as envs

# Global constants
# Global functions
# Warning Control
# Type Checking
# Type Aliases

# --- TERMINAL OUTPUT SETUP --- #
install(show_locals=False)
console = Console()


def main() -> None:
    """Entry point."""
    msg = "This script is deprecated. Use the new terrain system."
    raise NotImplementedError(msg)

    for name, cls in inspect.getmembers(envs, inspect.isclass):
        world = cls(load_precompiled=False)
        world.compile_to_xml()
        console.print(f"Compiled {name} to XML.")
        # TODO: add load_precompiled to all worlds


if __name__ == "__main__":
    main()

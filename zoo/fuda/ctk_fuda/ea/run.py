import sys
from pathlib import Path

# Allow running directly: add parent folder to path for relative imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ea.config import cli_options


from rich.console import Console

from ariel.ec.a004 import EA, EAStep

console = Console()


@cli_options
def main():
    """Run the evolutionary algorithm."""
    
    from ea.operations import (
        initial_pop,
        evaluate_pop,
        revde,
        survivor_selection,
    )
    
    
    population_list = initial_pop()
    population_list = evaluate_pop(population_list)

    # Define EA operations
    ops = [
        EAStep("mutation", revde),
        EAStep("evaluation", evaluate_pop),
        EAStep("survivor_selection", survivor_selection),
    ]

    # Run EA
    ea = EA(population_list, operations=ops)
    ea.run()

    # Log results
    console.log("Best:", ea.get_solution("best", only_alive=False))
    console.log("Median:", ea.get_solution("median", only_alive=False))
    console.log("Worst:", ea.get_solution("worst", only_alive=False))

    return ea


if __name__ == "__main__":
    main()

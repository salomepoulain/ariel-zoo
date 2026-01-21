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
    from ea.config import config
    from ea.operations import (
        initial_pop,
        evaluate_pop,
        revde,
        store_img,
        store_string,
        save_features,
        survivor_selection,
    )
    
    # print(config.STORE_IMG)
    ops = [
        EAStep("children", revde),
        EAStep("store_string", store_string),
        EAStep("store_img", store_img) if config.STORE_IMG else None,
        EAStep("save_features", save_features),
        EAStep("evaluation", evaluate_pop),
        EAStep("survivor_selection", survivor_selection),
    ]
    ops = [op for op in ops if op is not None]
    
    print(config.NOVELTY_METHOD)

    population_list = initial_pop()
    population_list = store_string(population_list) 
    population_list = save_features(population_list)
    if config.STORE_IMG:
        population_list = store_img(population_list)
    population_list = evaluate_pop(population_list)

    ea = EA(population_list, operations=ops)
    ea.run()

    console.log("Best:", ea.get_solution("best", only_alive=False).fitness, ea.get_solution("best", only_alive=False).tags['ctk_string'])
    console.log("Median:", ea.get_solution("median", only_alive=False).fitness, ea.get_solution("median", only_alive=False).tags['ctk_string'])
    console.log("Worst:", ea.get_solution("worst", only_alive=False).fitness, ea.get_solution("worst", only_alive=False).tags['ctk_string'])

    return ea


if __name__ == "__main__":
    main()

# Standard library
import contextlib
from gc import collect
from pathlib import Path
from typing import Any, cast

import numpy as np

# Third-party libraries
from joblib import Parallel, delayed
from joblib.parallel import cpu_count
from networkx import DiGraph
from rich.console import Console
from rich.progress import track

# Local library
from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)

# Global constants
# SCRIPT_NAME = __file__.split("/")[-1][:-3]
# CWD = Path.cwd()
# DATA = Path(CWD / "__data__" / SCRIPT_NAME)
# DATA.mkdir(exist_ok=True)
SEED = 42

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)


def initialize_random_graph(
    num_modules: int = 20,
    seed: int | None = None,
) -> DiGraph:
    """
    Generate a random modular individual as a directed graph.
    
    Parameters
    ----------
    num_modules : int, default 20
        Number of modules to include in the generated individual.
    seed : int, default SEED
        Random seed for reproducible generation.
    
    Returns
    -------
    DiGraph
        A directed graph representing the randomly generated modular 
        individual with modules, connections, and rotations.
    
    Notes
    -----
    - Uses three probability spaces: module types, connections between 
      faces, and rotations
    - Probability matrices have shapes (num_modules, NUM_OF_TYPES_OF_MODULES),
      (num_modules, num_modules, NUM_OF_FACES), and 
      (num_modules, NUM_OF_ROTATIONS) respectively
    - HighProbabilityDecoder converts probability matrices to graph structure
    """
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = RNG
    type_probability_space = rng.random(
        size=(num_modules, NUM_OF_TYPES_OF_MODULES),
        dtype=np.float32,
    )

    # "Connection" probability space
    conn_probability_space = rng.random(
        size=(num_modules, num_modules, NUM_OF_FACES),
        dtype=np.float32,
    )

    # "Rotation" probability space
    rotation_probability_space = rng.random(
        size=(num_modules, NUM_OF_ROTATIONS),
        dtype=np.float32,
    )

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(num_modules)
    return hpd.probability_matrices_to_graph(
        type_probability_space,
        conn_probability_space,
        rotation_probability_space,
    )


def intitialize_random_graph_population(
    population_size: int,
    num_modules: int = 20,
    *,
    n_jobs: int = -1,
    track_console: bool = True,
) -> list[DiGraph]:
    """
    Generate a population of random directed graphs using parallel processing.
    
    Parameters
    ----------
    population_size : int
        Number of random individuals to generate in the population.
    num_modules : int, default 20
        Number of modules/nodes in each generated graph.
    n_jobs : int, default -1
        Number of parallel jobs to use. If -1, uses CPU count minus 25%.
    track_console : bool, default True
        Whether to display progress tracking in the console.
    
    Returns
    -------
    list[DiGraph]
        List of randomly generated directed graphs representing the population.
    
    Raises
    ------
    BaseException
        Re-raises any exception that occurs during parallel execution after
        attempting to clean up the parallel pool.
    
    Notes
    -----
    - Uses joblib.Parallel for multiprocessing with automatic batch sizing
    - Seeds are generated using numpy RNG for reproducible randomness
    - Parallel pool is properly terminated and cleaned up on exceptions
    - Memory is explicitly collected after parallel execution completes
    - Progress tracking uses rich.progress.track when enabled
    """
    if n_jobs == -1:
        n_jobs = max(1, (cpu_count() or 1) - (cpu_count() // 4))

    # produce per-individual seeds in parent so they're unique (and reproducible)
    seeds = RNG.integers(0, 2**31 - 1, size=population_size, dtype=np.int64)

    population: list[DiGraph[Any]] = []
    with Parallel(n_jobs=n_jobs, batch_size="auto") as parallel_pool:
        try:
            iterator = (
                track(
                    range(population_size),
                    description=f"Generating population (n_jobs={n_jobs})",
                )
                if track_console
                else range(population_size)
            )
            results: Any = parallel_pool(
                delayed(initialize_random_graph)(num_modules, int(seeds[i]))
                for i in iterator
            )
            cast("list[DiGraph]", results)
            population.extend(results)
        except BaseException:
            with contextlib.suppress(Exception):
                parallel_pool._terminate_and_reset()
            raise
        finally:
            with contextlib.suppress(Exception):
                parallel_pool._terminate_and_reset()
            del parallel_pool
            collect()
    return population


if __name__ == "__main__":
    from time import perf_counter

    from rich.progress import track

    population = intitialize_random_graph_population(20, n_jobs=1)
    console.print(population)

    # Benchmark single individual
    console.print("[bold cyan]Generating a single individual...[/bold cyan]")
    start_one = perf_counter()
    graph = initialize_random_graph()
    end_one = perf_counter()
    single_time = end_one - start_one
    console.print("graph:", graph)
    console.log(f"Single individual generation took {single_time:.4f} seconds.")

    # Benchmark 10,000 individuals (serial, with progress)
    n_serial = 10_000

    console.print(
        f"[bold yellow]Generating {n_serial} individuals serially...[/bold yellow]",
    )
    start_serial = perf_counter()
    serial_population = [
        initialize_random_graph()
        for _ in track(range(n_serial), description="Serial generation")
    ]
    end_serial = perf_counter()
    serial_time = end_serial - start_serial
    serial_time_per_individual = serial_time / n_serial
    console.log(
        f"Serial generation of {n_serial} took {serial_time:.2f} seconds.",
    )
    console.log(
        f"Serial time per individual: {serial_time_per_individual:.6f} seconds.",
    )

    # Benchmark 500,000 individuals (parallel, with progress)
    n_parallel = 500_000
    console.print(
        f"[bold green]Generating {n_parallel} individuals in parallel (with progress)...[/bold green]",
    )
    start_parallel = perf_counter()
    parallel_population = intitialize_random_graph_population(
        n_parallel,
        num_modules=20,
        n_jobs=-1,
    )

    end_parallel = perf_counter()
    parallel_time = end_parallel - start_parallel
    parallel_time_per_individual = parallel_time / n_parallel
    console.log(
        f"Parallel generation of {n_parallel} took {parallel_time:.2f} seconds.",
    )
    console.log(
        f"Parallel time per individual: {parallel_time_per_individual:.6f} seconds.",
    )

    # Speedup calculation
    speedup = (
        serial_time_per_individual / parallel_time_per_individual
        if parallel_time_per_individual > 0
        else float("inf")
    )
    console.print(f"[yellow]Speedup (per individual): {speedup:.2f}x[/yellow]")

    # Benchmark 500,000 individuals (parallel, without progress)
    console.print(
        f"[bold green]Generating {n_parallel} individuals in parallel (no progress bar)...[/bold green]",
    )
    start_parallel_np = perf_counter()
    parallel_population_np = intitialize_random_graph_population(
        n_parallel,
        num_modules=20,
        n_jobs=-1,
        track_console=False,
    )
    end_parallel_np = perf_counter()
    parallel_time_np = end_parallel_np - start_parallel_np
    parallel_time_per_individual_np = parallel_time_np / n_parallel
    console.log(
        f"Parallel (no progress) generation of {n_parallel} took {parallel_time_np:.2f} seconds.",
    )
    console.log(
        f"Parallel (no progress) time per individual: {parallel_time_per_individual_np:.6f} seconds.",
    )

    # Speedup calculation (no progress)
    speedup_np = (
        serial_time_per_individual / parallel_time_per_individual_np
        if parallel_time_per_individual_np > 0
        else float("inf")
    )
    console.print(
        f"[yellow]Speedup (per individual, no progress): {speedup_np:.2f}x[/yellow]",
    )

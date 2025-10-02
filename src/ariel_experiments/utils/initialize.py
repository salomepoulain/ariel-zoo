# Standard library
from os import cpu_count
from pathlib import Path
from typing import cast

import numpy as np

# Third-party libraries
from joblib import Parallel, delayed
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
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
SEED = 42

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)


def generate_random_individual(num_modules: int = 20) -> DiGraph:
    """
    Generate a random modular body structure as a directed nx graph.

    Parameters
    ----------
    num_modules : int, default 20
        Number of modules to include in the generated body structure.

    Returns
    -------
    DiGraph
        A directed graph representing the randomly generated modular body
        with nodes as modules and edges as connections.
    """
    # "Type" probability space
    type_probability_space = RNG.random(
        size=(num_modules, NUM_OF_TYPES_OF_MODULES),
        dtype=np.float32,
    )

    # "Connection" probability space
    conn_probability_space = RNG.random(
        size=(num_modules, num_modules, NUM_OF_FACES),
        dtype=np.float32,
    )

    # "Rotation" probability space
    rotation_probability_space = RNG.random(
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


def generate_random_population_parallel(
    population_size: int,
    num_modules: int = 20,
    *,
    n_jobs: int = -1,
    track_console: bool = True,
) -> list[DiGraph]:
    """
    Generate a random population of modular robot graphs in parallel.

    Parameters
    ----------
    population_size : int
        Number of individuals to generate in the population.
    num_modules : int, default=20
        Number of modules per individual robot.
    n_jobs : int, default=-1
        Number of parallel jobs to use. If -1, uses CPU count minus 2.
    track_console : bool, default=True
        Whether to display progress tracking in console.

    Returns
    -------
    list[DiGraph]
        List of directed graphs representing the generated robot population.
    """
    if n_jobs == -1:
        n_jobs = max(1, (cpu_count() or 1) - 2)

    # Step 1: Batch-generate random probability spaces (vectorized, very fast in NumPy)
    with console.status(f"Generating probability spaces for {population_size} individuals..."):
        type_space = RNG.random(
            size=(population_size, num_modules, NUM_OF_TYPES_OF_MODULES),
            dtype=np.float32,
        )
        conn_space = RNG.random(
            size=(population_size, num_modules, num_modules, NUM_OF_FACES),
            dtype=np.float32,
        )
        rot_space = RNG.random(
            size=(population_size, num_modules, NUM_OF_ROTATIONS),
            dtype=np.float32,
        )
        hpd = HighProbabilityDecoder(num_modules)

    # Step 2: Parallel decoding
    iterator = (
        track(range(population_size), 
        description=f"Decoding population with {n_jobs} cpu's")
        if track_console else range(population_size)
    )
    population = Parallel(n_jobs=n_jobs, batch_size="auto")(
        delayed(hpd.probability_matrices_to_graph)(
            type_space[i], conn_space[i], rot_space[i],
        )
        for i in iterator
    )
    return population #type: ignore


if __name__ == "__main__":
    from time import perf_counter
    from rich.progress import track

    # Benchmark single individual
    console.print("[bold cyan]Generating a single individual...[/bold cyan]")
    start_one = perf_counter()
    graph = generate_random_individual()
    end_one = perf_counter()
    single_time = end_one - start_one
    console.print("graph:", graph)
    console.log(f"Single individual generation took {single_time:.4f} seconds.")

    # Benchmark 10,000 individuals (serial, with progress)
    n_serial = 10_000
    console.print(f"[bold yellow]Generating {n_serial} individuals serially...[/bold yellow]")
    start_serial = perf_counter()
    serial_population = [generate_random_individual() for _ in track(range(n_serial), description="Serial generation")]
    end_serial = perf_counter()
    serial_time = end_serial - start_serial
    serial_time_per_individual = serial_time / n_serial
    console.log(f"Serial generation of {n_serial} took {serial_time:.2f} seconds.")
    console.log(f"Serial time per individual: {serial_time_per_individual:.6f} seconds.")

    # Benchmark 500,000 individuals (parallel, with progress)
    n_parallel = 500_000
    console.print(f"[bold green]Generating {n_parallel} individuals in parallel (with progress)...[/bold green]")
    start_parallel = perf_counter()
    parallel_population = generate_random_population_parallel(n_parallel, num_modules=20, n_jobs=-1)

    end_parallel = perf_counter()
    parallel_time = end_parallel - start_parallel
    parallel_time_per_individual = parallel_time / n_parallel
    console.log(f"Parallel generation of {n_parallel} took {parallel_time:.2f} seconds.")
    console.log(f"Parallel time per individual: {parallel_time_per_individual:.6f} seconds.")

    # Speedup calculation
    speedup = serial_time_per_individual / parallel_time_per_individual if parallel_time_per_individual > 0 else float("inf")
    console.print(f"[yellow]Speedup (per individual): {speedup:.2f}x[/yellow]")

    # Benchmark 500,000 individuals (parallel, without progress)
    console.print(f"[bold green]Generating {n_parallel} individuals in parallel (no progress bar)...[/bold green]")
    start_parallel_np = perf_counter()
    parallel_population_np = generate_random_population_parallel(n_parallel, num_modules=20, n_jobs=-1, track_console=False)
    end_parallel_np = perf_counter()
    parallel_time_np = end_parallel_np - start_parallel_np
    parallel_time_per_individual_np = parallel_time_np / n_parallel
    console.log(f"Parallel (no progress) generation of {n_parallel} took {parallel_time_np:.2f} seconds.")
    console.log(f"Parallel (no progress) time per individual: {parallel_time_per_individual_np:.6f} seconds.")

    # Speedup calculation (no progress)
    speedup_np = serial_time_per_individual / parallel_time_per_individual_np if parallel_time_per_individual_np > 0 else float("inf")
    console.print(f"[yellow]Speedup (per individual, no progress): {speedup_np:.2f}x[/yellow]")

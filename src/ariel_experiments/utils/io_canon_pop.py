from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rich.console import Console
from sqlalchemy import create_engine, text

from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.ec.a001 import Individual
from ariel.ec.genotypes.nde.nde import NeuralDevelopmentalEncoding
from ariel_experiments.characterize.canonical.core.toolkit import (
    CanonicalToolKit as ctk,
)

if TYPE_CHECKING:
    from typing import Any

    from networkx import DiGraph

# Global constants
SCRIPT_NAME = "string_populations"
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)

console = Console()


def save_population(
    population: (
        list[DiGraph[Any]] | list[Individual] | list[list[list[float]]]
    ),
    title: str | None = None,
    save_dir: Path | str | None = None,
    num_modules: int = 20,
) -> Path:
    """
    Save a population of robots as canonical strings to CSV.

    Accepts DiGraph objects, Individual objects, or raw genotypes (NDE format).

    Parameters
    ----------
    population : list[DiGraph[Any]] | list[Individual] | list[list[list[float]]]
        List of robot graphs, Individual objects with genotypes,
        or raw genotypes (3D list for NDE: [type, conn, rotation])
    title : str | None
        Optional filename for the CSV (without extension).
        If None, generates timestamp-based filename.
    save_dir : Path | str | None
        Directory to save the file. If None, uses DATA directory.
    num_modules : int
        Number of modules for NDE/HPD decoding (only used if
        population contains Individuals or genotypes)

    Returns
    -------
    Path
        Path to the saved CSV file
    """
    # Determine population type and convert to DiGraphs if needed
    if len(population) == 0:
        msg = "Cannot save empty population"
        raise ValueError(msg)

    first_item = population[0]

    # Check if it's raw genotypes (list of lists)
    if isinstance(first_item, list):
        console.log(f"Converting {len(population)} genotypes to DiGraphs...")
        graph_population = []
        nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
        hpd = HighProbabilityDecoder(num_modules=num_modules)

        for genotype in population:
            matrixes = nde.forward(np.array(genotype))
            ind_graph = hpd.probability_matrices_to_graph(
                matrixes[0],
                matrixes[1],
                matrixes[2],
            )
            graph_population.append(ind_graph)
    # Check if it's Individual objects
    elif isinstance(first_item, Individual):
        console.log(f"Converting {len(population)} Individuals to DiGraphs...")
        graph_population = []
        nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
        hpd = HighProbabilityDecoder(num_modules=num_modules)

        for ind in population:
            matrixes = nde.forward(np.array(ind.genotype))
            ind_graph = hpd.probability_matrices_to_graph(
                matrixes[0],
                matrixes[1],
                matrixes[2],
            )
            graph_population.append(ind_graph)
    else:
        # Assume it's already DiGraphs
        graph_population = population

    # Convert graphs to canonical strings
    console.log(
        f"Converting {len(graph_population)} graphs to canonical strings..."
    )
    canon_string_population = [
        ctk.from_graph(individual).canonicalize().to_string()
        for individual in graph_population
    ]

    # Create DataFrame
    df = pd.DataFrame(
        {
            "canonical_string": canon_string_population,
            "index": range(len(canon_string_population)),
        }
    )

    # Determine save directory
    if save_dir is None:
        save_dir = DATA
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename
    if title is None:
        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        title = f"population_{timestamp}"

    # Ensure .csv extension
    if not title.endswith(".csv"):
        title = f"{title}.csv"

    # Save to CSV
    save_path = save_dir / title
    df.to_csv(save_path, index=False)

    console.log(f"Saved {len(population)} individuals to {save_path}")

    return save_path


def load_population(
    file_path: Path | str,
    *,
    load_dir: Path | str | None = None,
) -> list[DiGraph[Any]]:
    """
    Load a population of robots from canonical strings CSV.

    Parameters
    ----------
    file_path : Path | str
        Path to the CSV file or filename (if using load_dir)
    load_dir : Path | str | None
        Directory to load from. If None, uses DATA directory.

    Returns
    -------
    list[DiGraph[Any]]
        List of robot graphs
    """
    # Determine load path
    if load_dir is not None:
        load_dir = Path(load_dir)
        full_path = load_dir / file_path
    else:
        full_path = Path(file_path)
        if not full_path.is_absolute():
            full_path = DATA / file_path

    # Check if file exists
    if not full_path.exists():
        msg = f"File not found: {full_path}"
        raise FileNotFoundError(msg)

    # Load CSV
    console.log(f"Loading population from {full_path}...")
    df = pd.read_csv(full_path)

    if "canonical_string" not in df.columns:
        msg = (
            "CSV must have 'canonical_string' column. "
            f"Found columns: {list(df.columns)}"
        )
        raise ValueError(msg)

    # Convert canonical strings to graphs
    console.log(
        f"Converting {len(df)} canonical strings to DiGraphs..."
    )
    graph_population = []
    for canonical_string in df["canonical_string"]:
        # Parse string and convert to graph
        node = ctk.from_string(canonical_string)
        graph = ctk.to_graph(node)
        graph_population.append(graph)

    console.log(f"Loaded {len(graph_population)} individuals")

    return graph_population


def load_genotypes_from_database(
    db_path: Path | str | None = None,
    *,
    only_alive: bool = True,
    limit: int | None = None,
    num_modules: int = 20,
) -> list[DiGraph[Any]]:
    """
    Load genotypes from an EA database and convert to DiGraphs.

    Parameters
    ----------
    db_path : Path | str | None
        Path to the SQLite database file. If None, uses default
        __data__/database.db in current working directory.
    only_alive : bool
        If True, only load alive individuals
    limit : int | None
        Maximum number of genotypes to load. If None, loads all.
    num_modules : int
        Number of modules for NDE/HPD decoding

    Returns
    -------
    list[DiGraph[Any]]
        List of robot graphs

    Raises
    ------
    FileNotFoundError
        If the database file does not exist
    """
    # Use default database path if none provided
    if db_path is None:
        db_path = Path.cwd() / "__data__" / "database.db"
    else:
        db_path = Path(db_path)

    if not db_path.exists():
        msg = f"Database not found: {db_path}"
        raise FileNotFoundError(msg)

    console.log(f"Loading genotypes from database: {db_path}")
    engine = create_engine(f"sqlite:///{db_path}")

    # Build query
    query = "SELECT genotype_ FROM individual"
    if only_alive:
        query += " WHERE alive = 1"
    if limit is not None:
        query += f" LIMIT {limit}"

    genotypes = []
    with engine.connect() as conn:
        result = conn.execute(text(query))
        for row in result:
            genotype_str = row[0]
            genotype = (
                json.loads(genotype_str)
                if isinstance(genotype_str, str)
                else genotype_str
            )
            genotypes.append(genotype)

    console.log(f"Loaded {len(genotypes)} genotypes from database")

    # Convert to graphs
    console.log(
        f"Converting {len(genotypes)} genotypes to DiGraphs...",
    )
    nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
    hpd = HighProbabilityDecoder(num_modules=num_modules)

    graphs = []
    for genotype in genotypes:
        matrixes = nde.forward(np.array(genotype))
        ind_graph = hpd.probability_matrices_to_graph(
            matrixes[0],
            matrixes[1],
            matrixes[2],
        )
        graphs.append(ind_graph)

    console.log(f"Converted to {len(graphs)} DiGraphs")
    return graphs

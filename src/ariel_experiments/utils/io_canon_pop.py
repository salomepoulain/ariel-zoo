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
# from ariel_experiments.characterize.canonical_toolkit.tests.old.toolkit import (
#     CanonicalToolKit as ctk,
# )

import canonical_toolkit as ctk

if TYPE_CHECKING:
    from typing import Any

    from networkx import DiGraph

# Global constants
# SCRIPT_NAME = "string_populations"
# CWD = Path.cwd()
# DATA = Path(CWD / "__data__" / SCRIPT_NAME)
# DATA.mkdir(exist_ok=True)

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
        # console.log(f"Converting {len(population)} genotypes to DiGraphs...")
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
        # console.log(f"Converting {len(population)} Individuals to DiGraphs...")
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

    # # Convert graphs to canonical strings
    # console.log(
    #     f"Converting {len(graph_population)} graphs to canonical strings..."
    # )
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

    # console.log(f"Saved {len(population)} individuals to {save_path}")

    return save_path


def load_population(
    file_path: Path | str,
    *,
    load_dir: Path | str | None = None,
) -> list[ctk.Node]:
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
    # console.log(f"Loading population from {full_path}...")
    df = pd.read_csv(full_path)

    if "ctk_string" not in df.columns:
        msg = (
            "CSV must have 'ctk_string' column. "
            f"Found columns: {list(df.columns)}"
        )
        raise ValueError(msg)

    # Convert canonical strings to graphs
    # console.log(
    #     f"Converting {len(df)} canonical strings to DiGraphs..."
    # )
    nodes = []
    for canonical_string in df["ctk_string"]:
        nodes.append(ctk.from_string(canonical_string))

    return nodes


# def load_genotypes_from_database(
#     db_path: Path | str | None = None,
#     *,
#     only_alive: bool = True,
#     limit: int | None = None,
#     num_modules: int = 20,
# ) -> list[DiGraph[Any]]:
#     """
#     Load genotypes from an EA database and convert to DiGraphs.

#     Parameters
#     ----------
#     db_path : Path | str | None
#         Path to the SQLite database file. If None, uses default
#         __data__/database.db in current working directory.
#     only_alive : bool
#         If True, only load alive individuals
#     limit : int | None
#         Maximum number of genotypes to load. If None, loads all.
#     num_modules : int
#         Number of modules for NDE/HPD decoding

#     Returns
#     -------
#     list[DiGraph[Any]]
#         List of robot graphs

#     Raises
#     ------
#     FileNotFoundError
#         If the database file does not exist
#     """
#     # Use default database path if none provided
#     if db_path is None:
#         db_path = Path.cwd() / "__data__" / "database.db"
#     else:
#         db_path = Path(db_path)

#     if not db_path.exists():
#         msg = f"Database not found: {db_path}"
#         raise FileNotFoundError(msg)

#     # console.log(f"Loading genotypes from database: {db_path}")
#     engine = create_engine(f"sqlite:///{db_path}")

#     # Build query
#     query = "SELECT genotype_ FROM individual"
#     if only_alive:
#         query += " WHERE alive = 1"
#     if limit is not None:
#         query += f" LIMIT {limit}"

#     strings = []
#     with engine.connect() as conn:
#         result = conn.execute(text(query))
#         for row in result:
#             genotype_str = row[0]
#             genotype = (
#                 json.loads(genotype_str)
#                 if isinstance(genotype_str, str)
#                 else genotype_str
#             )
#             strings.append(genotype)

#     # console.log(f"Loaded {len(genotypes)} genotypes from database")

#     # Convert to graphs
#     # console.log(
#     #     f"Converting {len(genotypes)} genotypes to DiGraphs...",
#     # )

#     # graphs = []
#     # for genotype in genotypes:
#     #     nde = NeuralDevelopmentalEncoding(number_of_modules=num_modules)
#     #     hpd = HighProbabilityDecoder(num_modules=num_modules)
#     #     matrixes = nde.forward(np.array(genotype))
#     #     ind_graph = hpd.probability_matrices_to_graph(
#     #         matrixes[0],
#     #         matrixes[1],
#     #         matrixes[2],
#     #     )
#     #     graphs.append(ind_graph)

#     # console.log(f"Converted to {len(graphs)} DiGraphs")
#     return graphs


from pathlib import Path
from typing import Any
import json
from sqlalchemy import create_engine, text

def load_ctk_strings_from_database(
    db_path: Path | str | None = None,
    *,
    only_alive: bool = True,
    limit: int | None = None,
) -> list[str]:
    """
    Loads ONLY the 'ctk_string' tag from the database for each individual.

    Returns
    -------
    list[str]
        A list of ctk_strings (phenotype representations).
    """
    if db_path is None:
        db_path = Path.cwd() / "__data__" / "database.db"
    else:
        db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    engine = create_engine(f"sqlite:///{db_path}")

    # --- CHANGE 1: Select ONLY the 'tags' column ---
    query = "SELECT tags_ FROM individual"

    if only_alive:
        query += " WHERE alive = 1"
    if limit is not None:
        query += f" LIMIT {limit}"

    ctk_strings = []

    with engine.connect() as conn:
        result = conn.execute(text(query))

        for row in result:
            tags_str = row[0]

            if tags_str:
                # Parse the JSON tag string
                try:
                    # Handle case where it might already be a dict or a string
                    tags = json.loads(tags_str) if isinstance(tags_str, str) else tags_str

                    # --- CHANGE 2: Extract specific key ---
                    # Get the 'ctk_string', defaulting to None or empty string if missing
                    val = tags.get("ctk_string")
                    if val:
                        ctk_strings.append(val)

                except (json.JSONDecodeError, AttributeError):
                    continue

    return ctk_strings


import csv

def export_sorted_ctk_to_csv(
    db_path: Path | str | None = None,
    output_csv_path: str = "sorted_population.csv",
    *,
    only_alive: bool = True,
    limit: int | None = None,
) -> None:
    """
    Loads 'ctk_string' and 'fitness' from the database, sorts them by fitness
    (descending), and writes the result to a CSV file.

    Parameters
    ----------
    db_path : Path | str | None
        Path to the database. Defaults to ./__data__/database.db
    output_csv_path : str
        Filename for the output CSV.
    only_alive : bool
        If True, filters for alive=1.
    limit : int | None
        Limit the number of rows queried (applied before sorting in Python,
        so use with caution if database isn't sorted).
    """
    # 1. Setup Database Path
    if db_path is None:
        db_path =Path.cwd() / "__data__" / "database.db"
    else:
        db_path = Path(db_path)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    engine = create_engine(f"sqlite:///{db_path}")

    # 2. Construct Query
    # We select tags_ to get the string, and fitness to sort by.
    # We order by fitness in SQL for efficiency if limit is used.
    query = "SELECT tags_, fitness_ FROM individual"

    conditions = []
    if only_alive:
        conditions.append("alive = 1")

    if conditions:
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY fitness_ DESC"

    if limit is not None:
        query += f" LIMIT {limit}"

    # 3. Fetch and Process Data
    rows_to_export = []

    with engine.connect() as conn:
        result = conn.execute(text(query))

        for row in result:
            tags_str = row[0]
            fitness_val = row[1]

            # Default fitness to -inf if None (to handle failed evaluations safely)
            if fitness_val is None:
                fitness_val = float('-inf')

            ctk_string = None

            if tags_str:
                try:
                    # Handle both string JSON and direct dict (if middleware converted it)
                    tags = json.loads(tags_str) if isinstance(tags_str, str) else tags_str

                    # Extract the phenotype string
                    ctk_string = tags.get("ctk_string")
                except (json.JSONDecodeError, AttributeError):
                    # Skip rows with corrupted tags
                    continue

            # Only add if we successfully found the string
            if ctk_string:
                rows_to_export.append({
                    "fitness_": float(fitness_val),
                    "ctk_string": ctk_string
                })

    # 4. Sort (Redundant if SQL ORDER BY is used, but good safety for Python logic)
    # Sort by fitness descending
    rows_to_export.sort(key=lambda x: x["fitness_"], reverse=True)

    # 5. Write to CSV
    try:
        with open(output_csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
            fieldnames = ['rank', 'fitness_', 'ctk_string']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()

            for rank, data in enumerate(rows_to_export, start=1):
                writer.writerow({
                    "rank": rank,
                    "fitness_": data["fitness_"],
                    "ctk_string": data["ctk_string"]
                })

        print(f"Successfully exported {len(rows_to_export)} individuals to {output_csv_path}")

    except IOError as e:
        print(f"Error writing to CSV file: {e}")


export_sorted_ctk_to_csv()

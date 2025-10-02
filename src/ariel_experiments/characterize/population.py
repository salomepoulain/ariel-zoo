from __future__ import annotations

"""
This module is an aid in categorizing and analyzing individuals in an population in an ordered matter.
By following the typing closely, functions are easily implemntable in ....

The final results will be to create this type of dictionary

all_stats; CompletePopulationStats = {
    "degree": {
        "raw": [3, 2, 3, 5, 7],
        "derived": {
            "uniques": {
                "3": {
                    "count": 2,
                    "idx": [0, 2]
                },
                "2": {
                    "count": 1,
                    "idx": [1]
                }
                "5": {
                    "count": 1,
                    "idx": [3]
                },
                ...,
            }
            "stats": {
                "count": 5,
                "uniques": 4,
                "mean": 4.25,
                ...,
                "Q3": 6.0,
                "num_outliers": 1,
                "outlier_indexes": [3],
            }
        }
    },
    "density": {
        "raw": [0.3, 0.25, 0.4, 0.5]
        # no derived yet
    }
}

"""


# Standard library
from collections import defaultdict
from collections.abc import Callable, Sequence
from os import cpu_count
from pathlib import Path
from typing import Any, TypedDict, cast

# Third-party libraries
import numpy as np
from joblib import Parallel, delayed
from networkx import DiGraph
from rich.console import Console
from rich.progress import track

# Global constants
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
SEED = 42

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)

# Typing
# A population is a sequence (list/tuple) of directed graphs (individuals)
type Population = Sequence[DiGraph]

# Index of a graph in the population
type GraphIndex = int

# List of indexes to find individuals in a population
type IndexMappings = list[GraphIndex]

# Name of a property being analyzed (e.g., "mass", "degree")
type GraphPropertyName = str

# Numeric property types (for stats)
type NumericProperty = int | float

# Non-numeric property types (for categorical or other data)
type NonNumericProperty = str | bool | list | tuple | dict | set

# Any property type (numeric or non-numeric)
type GraphPropertyType = NumericProperty | NonNumericProperty

# Result for one or more properties of a single individual graph
type GraphPropertyResult = dict[GraphPropertyName, GraphPropertyType]

# A function that analyzes a property from one individual graph
type PropertyAnalyzer = Callable[[DiGraph], GraphPropertyResult]

# List of one property value for all individuals in the population
type EnsemblePropertyResults = list[GraphPropertyType]

type DerivedPopulationProperties = dict[
    DerivedPropertyName, DerivedPropertyResults,
]

type PopulationProperties = dict[
    EnsemblePropertyResults, DerivedPopulationProperties | None
]
# Aggregated results for all individual graphs, grouped by property
# (key: property name, value: PopulationProperties TypedDict)
type CompletePopulationStats = dict[
    GraphPropertyName, PopulationProperties | Optional[DerivedPopulationProperties],
]


# Name for a derived property (e.g., "mean", "uniques")
type DerivedPropertyName = str

# Results for derived properties (numeric or categorical stats)
type DerivedPropertyResults = NumericPropertyStats | CategoricalStats

# Categorical stats: mapping from property value to Uniques TypedDict
type CategoricalStats = dict[GraphPropertyType, Uniques]



# Mapping from derived property name to its results



# class PopulationProperties(TypedDict):
#     """Everything known about a single property across the population."""
#     raw: EnsemblePropertyResults
#     derived: DerivedPopulationProperties | None

class NumericPropertyStats(TypedDict):
    count: int
    uniques: int
    mean: float
    std: float
    median: float
    Q1: float
    Q3: float
    num_outliers: int
    outlier_indexes: IndexMappings

class Uniques(TypedDict):
    count: int
    indexes: IndexMappings


def get_population_properties(
    population: list[DiGraph],
    analyzers: list[PropertyAnalyzer],
    *,
    n_jobs: int = -1,
) -> PopulationProperties:
    """
    Analyze all individuals in a population using multiple property analyzers.

    Parameters
    ----------
    population : list[DiGraph]
        List of directed graphs representing individuals to analyze.
    analyzers : list[PropertyAnalyzer]
        List of analyzer objects to apply to each individual.
    n_jobs : int, default -1
        Number of parallel jobs to run. -1 uses all available processors.

    Returns
    -------
    PopulationProperties
        Aggregated analysis results for the entire population.

    Notes
    -----
    - Uses parallel processing when n_jobs != 1 for improved performance
    - Each analyzer is applied to every individual in the population
    - Results are automatically aggregated across all individuals and
      analyzers
    """
    if n_jobs == -1:
        cpus = cpu_count()
        n_jobs = cpus - 1 if cpus else 1

    console.log(f"Using {n_jobs} cpu(s)")
    console.log(f"Starting analysis of {len(population)} individuals...")

    def _analyze(individual: DiGraph) -> PopulationProperties:
        result: PopulationProperties = {}
        for analyzer in analyzers:
            analyzer_result = analyzer(individual)
            for key, value in analyzer_result.items():
                result.setdefault(key, []).append(value)
        return result

    if n_jobs == 1:
        results: list[PopulationProperties] = [
            _analyze(ind)
            for ind in track(population, description="Analyzing population")
        ]
    else:
        parallel_results: Any = Parallel(n_jobs=n_jobs, batch_size="auto")(
            delayed(_analyze)(ind)
            for ind in track(population, description="Analyzing population")
        )
        results = cast("list[PopulationProperties]", parallel_results)

    aggregated: PopulationProperties = defaultdict(list)
    for result in results:
        for key, value in result.items():
            aggregated[key].append(value)

    console.log("Analysis complete.")
    return aggregated


def analyze_population_metrics() -> None:
    pass


# def statistical_df_from_dict(
#     properties_dict: dict[str, Any],
#     keys: list[str] | None = None,
#     save_file: Path | str | None = None,
# ) -> pd.DataFrame:
#     """
#     For each key in properties_dict:
#       - If numeric: mean, std, median, count, Q1, Q3, outlier indexes, num_outliers, uniques.
#       - If non-numeric: key, count, uniques.
#     Returns a DataFrame with correct dtypes.
#     """
#     if keys is None:
#         keys = list(properties_dict)

#     rows = []
#     for key in keys:
#         values = properties_dict[key]
#         if not values:
#             continue
#         uniques = len(set(values))
#         if isinstance(values[0], (float, int)):
#             arr = np.asarray(values, dtype=float)
#             mean = arr.mean()
#             std = arr.std(ddof=1)
#             median = np.median(arr)
#             q1 = np.percentile(arr, 25)
#             q3 = np.percentile(arr, 75)
#             iqr = q3 - q1
#             lower = q1 - 1.5 * iqr
#             upper = q3 + 1.5 * iqr
#             outlier_indexes = np.where((arr < lower) | (arr > upper))[
#                 0
#             ].tolist()
#             row = {
#                 "key": key,
#                 "count": len(arr),
#                 "uniques": int(uniques),
#                 "mean": float(mean),
#                 "std": float(std),
#                 "median": float(median),
#                 "Q1": float(q1),
#                 "Q3": float(q3),
#                 "num_outliers": len(outlier_indexes),
#                 "outlier_indexes": outlier_indexes,
#             }
#         else:
#             row = {
#                 "key": key,
#                 "count": len(values),
#                 "uniques": int(uniques),
#                 "mean": None,
#                 "std": None,
#                 "median": None,
#                 "Q1": None,
#                 "Q3": None,
#                 "num_outliers": None,
#                 "outlier_indexes": None,
#             }
#         rows.append(row)

#     df = pd.DataFrame.from_records(rows)
#     df = df.set_index("key")

#     # Set dtypes explicitly
#     df["count"] = df["count"].astype("Int64")
#     df["uniques"] = df["uniques"].astype("Int64")
#     for col in ["mean", "std", "median", "Q1", "Q3"]:
#         df[col] = pd.to_numeric(df[col], errors="coerce")
#     df["num_outliers"] = df["num_outliers"].astype("Int64")
#     df["outlier_indexes"] = df["outlier_indexes"].astype(object)

#     if save_file:
#         df.to_csv(save_file)

#     return df


def count_df_from_list(value_list: list[str]) -> pd.DataFrame:
    """
    TODO: add info_dict, and key, appends.

    Returns a DataFrame with:
      - index: hash value
      - count: number of times the hash appears
      - indexes: list of indexes in the original list where the hash appears.
    """
    index_dict = defaultdict(list)
    for idx, h in enumerate(value_list):
        index_dict[h].append(idx)

    results = []
    for h, idxs in index_dict.items():
        results.append({
            "value": h,
            "count": len(idxs),
            "indexes": idxs,
        })

    df = pd.DataFrame(results)
    return df.set_index("value")


def summarize_numeric_properties(
    properties_dict: dict[PropertyName, list[NumericProperty]],
    properties: list[PropertyName] | None = None,
    *,
    outlier_bound: float = 1.5,
) -> dict[str, NumericPropertyStats]:
    summary: dict[str, NumericPropertyStats] = {}

    # Filter keys if requested and ensure values are numeric
    filtered_dict: dict[str, list[float]] = {}
    for k, v in properties_dict.items():
        if properties is not None and k not in properties:
            continue
        if not v:
            continue
        # Only include numeric lists
        if isinstance(v[0], (int, float)):
            filtered_dict[k] = [
                float(x) for x in v
            ]  # convert to float for consistency

    for key, values in filtered_dict.items():
        arr = np.array(values, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=1))
        median = float(np.median(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        lower = q1 - outlier_bound * iqr
        upper = q3 + outlier_bound * iqr
        outlier_indexes: IndexMappings = np.where(
            (arr < lower) | (arr > upper),
        )[0].tolist()

        summary[key] = NumericPropertyStats(
            count=len(arr),
            uniques=len(set(values)),
            mean=mean,
            std=std,
            median=median,
            Q1=q1,
            Q3=q3,
            num_outliers=len(outlier_indexes),
            outlier_indexes=outlier_indexes,
        )

    return summary


def summarize_non_numeric_properties(
    properties: dict[str, List[NonNumericProperty]],
    keys: list[str] | None = None,
) -> None:
    """
    Summarize non-numeric properties: count, uniques.
    Returns a dict[PropertyName, dict[stats]].
    """


if __name__ == "__main__":
    from ariel_experiments.characterize.individual import (
        analyze_json_hash,
        analyze_mass,
        analyze_module_counts,
    )
    from ariel_experiments.utils.initialize import generate_random_individual

    population_size = 5
    population = [generate_random_individual() for i in range(population_size)]
    individual_analyzers = [
        analyze_module_counts,
        analyze_mass,
        analyze_json_hash,
    ]
    analyis_results = analyze_all_individuals(
        population,
        individual_analyzers,
        n_jobs=6,
    )
    console.print(analyis_results)

    console.rule("test speed")

    big_population = [
        generate_random_individual()
        for _ in track(range(100_000), description="Generating big population")
    ]
    console.print("population initialized, running results now:")
    _ = analyze_population(big_population, analyzers)

    # all_individual_dict
    # population_results_dict

    """
    individual_data: len(population_size), data per individual

    population_data: can be called on individual data.
                    options:
                        - sorted_data [all idx. sorted],
                        - normalized_data [all idx. sorted],

                        - counts [outlier indexes],
                        - statistical_analysis [outlier indexes],
                    uses:
                        - counts and statistical_analysis can show index of outliers
    """

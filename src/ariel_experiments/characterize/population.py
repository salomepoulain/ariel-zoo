from __future__ import annotations

"""
This module is an aid in categorizing and analyzing individuals in an population in an ordered matter.
By following the typing closely, functions are easily implemntable in ....

The final results will be to create this type of dictionary

all_stats; CompletePopulationStats = {
    "degree": {
        "raw": [3, 2, 3, 5, 7],
        "derived": {                # DerivedProperty
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


propertyAnalyzers work on 1 individual to give NamedProperty(s) per individual

propertyDerivers work on 1 (or more) NamedProperty(s) to give DerivedProperty

"""


# Standard library
from collections import defaultdict
from collections.abc import Callable, Sequence
from os import cpu_count
from pathlib import Path
from typing import Any, TypedDict, cast, TypeVar, Protocol

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

# Type Aliases and TypeVars
T = TypeVar("T")

# Concrete value aliases (don't use TypeVar for concrete aliases)
NumericProperty = int | float
NonNumProperty = str | bool | list | tuple | dict | set
GraphProperty = NumericProperty | NonNumProperty

# Index of a graph in the population
type GraphIndex = int
type IndexMappings = list[GraphIndex]

# Derived property can be a value or list of indexes
DerivedProperty = GraphProperty | IndexMappings

# name aliases
type GraphPropertyName = str
type DerivedPropertyName = str

# Generic mapping when values are homogeneous (use NamedGraphPropertiesT[T])
NamedGraphPropertiesT = dict[GraphPropertyName, T]
# Backwards-compatible mixed container
type NamedGraphProperties = dict[GraphPropertyName, GraphProperty]
type NamedDerivedProperties = dict[DerivedPropertyName, DerivedProperty]

# Raw / derived population shapes
type RawPopulationProperties = dict[GraphPropertyName, list[GraphProperty]]
type DerivedPopulationProperties = dict[GraphPropertyName, NamedDerivedProperties]

# end goal structure!
class AnalyzedPopulation(TypedDict):
    raw: RawPopulationProperties
    derived: DerivedPopulationProperties
    

# Generic analyzer Protocol: callable that returns dict[str, T]
class PropertyAnalyzer(Protocol[T]):
    def __call__(self, individual: DiGraph) -> NamedGraphPropertiesT[T]:
        ...

# Deriver (non-generic here) — returns NamedDerivedProperties
class PropertyDeriver(Protocol):
    def __call__(
        self,
        named_props: NamedGraphProperties,
        keys: GraphPropertyName | list[GraphPropertyName],
    ) -> NamedDerivedProperties:
        ...



# ----- property types: individual, population, derived -----



# # NumericProperty = TypeVar("NumericProperty", int, float)
# # NonNumProperty = TypeVar("NonNumProperty", str, bool, list, tuple, dict, set)
# # GraphProperty = NumericProperty | NonNumProperty

# # Index of a graph in the population



# # ----- important structures -----

# # end goal structure!
# class AnalyzedPopulation(TypedDict):
#     raw: RawPopulationProperties
#     derived: DerivedPopulationProperties

# type GraphIndex = int
# type IndexMappings = list[GraphIndex]
# type GraphPropertyName = str
# type DerivedPropertyName = GraphPropertyName | str

# GraphProperty = TypeVar("GraphProperty")
# DerivedProperty = TypeVar("DerivedProperty")

# # Derived properties across 1 individual
# NamedGraphProperties = dict[
#     GraphPropertyName, 
#     GraphProperty
# ]
# # Derived properties calculated across 1 or more NamedGraphProperties.
# NamedDerivedProperties = dict[
#     DerivedPropertyName,
#     DerivedProperty
# ]

# # Raw property values across the entire population from the analyzers,
# RawPopulationProperties = dict[
#     GraphPropertyName,
#     list[GraphProperty],
# ]
# # DerivedProperties are not limited to 1 graph property, thus custom keys are allowed
# DerivedPopulationProperties = dict[
#     GraphPropertyName | str,
#     NamedDerivedProperties
# ]

# # ----- functions -----

# # A function that analyzes a property from one individual graph
# # type PropertyAnalyzer = Callable[
# #     [DiGraph], 
# #     NamedGraphProperties
# # ]
# # # A function that analyzes 1 or more graph properties, to derive new ones
# # type PropertyDeriver = Callable[
# #     [NamedGraphProperties, GraphPropertyName | list[GraphPropertyName]], 
# #     NamedDerivedProperties
# # ]

# # Generic analyzer: a callable that takes a DiGraph and returns a mapping
# class PropertyAnalyzer(Protocol[GraphProperty]):
#     def __call__(
#         self, 
#         individual: DiGraph
#     ) -> NamedGraphProperties[GraphProperty]:
#         ...

# # Deriver (non-generic here for simplicity) — returns NamedDerivedProperties
# class PropertyDeriver(Protocol[DerivedProperty]):
#     def __call__(
#         self,
#         named_props: NamedGraphProperties,
#         keys: GraphPropertyName | list[GraphPropertyName],
#     ) -> NamedDerivedProperties:
#         ...


# Most common Derived Properties -------

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

class UniqueEntry(TypedDict):
    count: int
    indexes: IndexMappings
Uniques = dict[GraphProperty, UniqueEntry]

class MinFirstIdx(TypedDict):
    indexes: IndexMappings

class MaxFirstIdx(TypedDict):
    indexes: IndexMappings
    
class BasicDerivedProperties(TypedDict):
    numeric_stats: NumericPropertyStats | None
    uniques: Uniques
    min_first_idx: MinFirstIdx
    max_first_idx: MaxFirstIdx
    
# ----------------------------------------


# region PropertyDerivers

def derive_numeric_summary(
    value_list: list[GraphProperty],
    *,
    outlier_bound: float = 1.5,
) -> NumericPropertyStats:
    
    
    if not value_list:
        raise ValueError("Input list is empty.")
    first_type = type(value_list[0])
    if not issubclass(first_type, (int, float)):
        raise TypeError("All values must be numeric (int or float).")
    
    arr = np.array(value_list, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    median = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    lower = q1 - outlier_bound * iqr
    upper = q3 + outlier_bound * iqr
    outlier_indexes: IndexMappings = np.where((arr < lower) | (arr > upper))[0].tolist()

    return NumericPropertyStats(
        count=len(arr),
        uniques=len(set(arr)),
        mean=mean,
        std=std,
        median=median,
        Q1=q1,
        Q3=q3,
        num_outliers=len(outlier_indexes),
        outlier_indexes=outlier_indexes,
    )

def derive_uniques(value_list: list[GraphProperty]) -> Uniques:
    index_dict = defaultdict(list)
    for idx, h in enumerate(value_list):
        try:
            key = h
            hash(key)
        except Exception:
            key = repr(h)
        index_dict[key].append(idx)

    uniques: Uniques = {}
    for value, idxs in index_dict.items():
        uniques[value] = UniqueEntry(count=len(idxs), indexes=idxs)

    return uniques

def derive_min_first_idx(value_list: list[GraphProperty]) -> MinFirstIdx:
    """
    Return indexes of the values sorted ascending (indexes in the order of increasing value).
    Example:
      values = [0,5,2,4] -> sorted values [0,2,4,5] -> return [0,2,3,1]
    """
    if not value_list:
        return MinFirstIdx(indexes=[])

    try:
        sorted_indexes = sorted(range(len(value_list)), key=lambda i: (value_list[i], i))
    except TypeError:
        # Fallback for unorderable/mixed types: sort by stable string representation
        sorted_indexes = sorted(range(len(value_list)), key=lambda i: (repr(value_list[i]), i))

    return MinFirstIdx(indexes=sorted_indexes)


def derive_max_first_idx(value_list: list[GraphProperty]) -> MaxFirstIdx:
    """
    Return indexes of the values sorted descending (indexes in the order of decreasing value).
    """
    if not value_list:
        return MaxFirstIdx(indexes=[])

    try:
        sorted_indexes = sorted(range(len(value_list)), key=lambda i: (value_list[i], i), reverse=True)
    except TypeError:
        # Fallback for unorderable/mixed types: sort by stable string representation (reverse)
        sorted_indexes = sorted(range(len(value_list)), key=lambda i: (repr(value_list[i]), i), reverse=True)

    return MaxFirstIdx(indexes=sorted_indexes)

def derive_basic_derived_properties(value_list: list[GraphProperty]) -> BasicDerivedProperties:
    # uniques and index-based derived props always available
    uniques = derive_uniques(value_list)
    min_first_idx = derive_min_first_idx(value_list)
    max_first_idx = derive_max_first_idx(value_list)

    # numeric summary when possible
    try:
        numeric_stats = derive_numeric_summary(value_list)
    except Exception:
        numeric_stats = None

    BasicDerivedProperties(
        numeric_stats=numeric_stats,
        uniques=uniques,
        min_first_idx=min_first_idx,
        max_first_idx=max_first_idx,
    )


# EndRegion


def get_raw_population_properties(
    population: list[DiGraph],
    analyzers: list[PropertyAnalyzer],
    *,
    n_jobs: int = -1,
) -> RawPopulationProperties:
    """
    Extract raw properties from a population of graphs using analyzers.
    
    Parameters
    ----------
    population : list[DiGraph]
        List of directed graphs to analyze for properties.
    analyzers : list[PropertyAnalyzer]
        List of analyzer objects to apply to each graph.
    n_jobs : int, default -1
        Number of parallel jobs for processing. -1 uses all processors.
    
    Returns
    -------
    RawPopulationProperties
        Dictionary with extracted properties from all graphs and analyzers.
    
    Notes
    -----
    - Processing is parallelized across graphs when n_jobs != 1
    - Each analyzer is applied to every graph in the population
    - Results maintain correspondence between graphs and their properties
    - Resuts per individual remain ordered as original population
    """
    if n_jobs == -1:
        n_jobs = max(1, (cpu_count() or 1) - (cpu_count() // 4))

    def _analyze(individual: DiGraph) -> RawPopulationProperties:
        result: RawPopulationProperties = {}
        for analyzer in analyzers:
            analyzer_result = analyzer(individual)
            for key, value in analyzer_result.items():
                result.setdefault(key, []).append(value)
        return result

    parallel_results: Any = Parallel(
        n_jobs=n_jobs, batch_size="auto"
    )(
        delayed(_analyze)(ind)
        for ind in track(population, description=f"Analyzing population (n_jobs={n_jobs})")
    )
    results = cast(list[RawPopulationProperties], parallel_results)

    aggregated: RawPopulationProperties = defaultdict(list)
    for result in results:
        for key, value_list in result.items():
            aggregated[key].extend(value_list)

    return dict(aggregated)




# todo: just for each thing the derived thing
# todo: might already be that get basic thingy...?
# todo: its not.
def get_derived_population_properties(analyzers):
    pass


# todo: apply specified 



def convert_to_df() -> None:
    pass


def save_df() -> None:
    pass



if __name__ == "__main__":
    from ariel_experiments.characterize.individual import (
        analyze_json_hash,
        analyze_mass,
        analyze_module_counts,
    )
    from ariel_experiments.utils.initialize import generate_random_population_parallel
    from time import perf_counter

    # ----------------------------------------------

    population_size = 5
    population = generate_random_population_parallel(population_size, n_jobs=1)

    individual_analyzers = [
        analyze_module_counts,
        analyze_mass,
        analyze_json_hash,
    ]
    raw_properties = get_raw_population_properties(
        population,
        individual_analyzers,
    )
    
    console.rule("raw properties:")
    console.print(raw_properties)
    
    # ----------------------------------------------

    console.rule("test analyze speed:")

    population_size = 100_000
    population = generate_random_population_parallel(population_size)
    
    start = perf_counter()
    raw_properties = get_raw_population_properties(
        population,
        individual_analyzers,
    )
    end = perf_counter()
    console.print(f"population of size {population_size} took {end-start:.4f} seconds")

    # ----------------------------------------------


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

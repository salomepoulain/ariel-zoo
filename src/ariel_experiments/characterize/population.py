from __future__ import annotations

import itertools

# Standard library imports
from collections import defaultdict
from os import cpu_count
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TypedDict, TypeVar, cast
import numpy as np
from tqdm.auto import tqdm
# Third-party imports
import numpy as np
import pandas as pd
from joblib import Parallel, delayed  # type: ignore[import]
from rich.console import Console
from rich.progress import track
from rich.tree import Tree

from ariel_experiments.characterize.canonical.core.toolkit import (
    CanonicalToolKit as ctk,
)

if TYPE_CHECKING:
    from networkx import DiGraph

# Global constants
# SCRIPT_NAME = Path(__file__).stem
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(parents=True, exist_ok=True)
# console.log(f"DATA dir = {DATA}")
SEED = 42
import pickle
from multiprocessing import shared_memory

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)

# Type Aliases and TypeVars
T = TypeVar("T")
# Concrete value aliases (don't use TypeVar for concrete aliases)
NumericProperty = int | float
NonNumProperty = str | bool | list[Any] | tuple[Any, ...] | dict[Any, Any] | set[Any]
GraphProperty = NumericProperty | NonNumProperty

# Index of a graph in the population
type GraphIndex = int
type IndexMappings = list[GraphIndex]

# Derived property can be a value or list of indexes
DerivedProperty = GraphProperty | IndexMappings

# name aliases
type GraphPropertyName = str
type DerivedPropertyName = GraphPropertyName | str

# Generic mapping when values are homogeneous
NamedGraphPropertiesT = dict[GraphPropertyName, T]
NamedDerivedPropertiesT = dict[DerivedPropertyName, T]

# Backwards-compatible mixed container
type NamedGraphProperties = dict[GraphPropertyName, GraphProperty]
type NamedDerivedProperties = dict[DerivedPropertyName, DerivedProperty]

# Raw / derived population shapes
type RawPopulationProperties = dict[GraphPropertyName, list[GraphProperty]]
type DerivedPopulationProperties = dict[GraphPropertyName, NamedDerivedProperties]

# Used to doctate if derived population properties map on 1:1 on graph properties, or different
B = TypeVar("B", bound=str)
DerivedPopulationPropertiesT = dict[B, NamedDerivedProperties]


# CustomDerivedPopulationProperties = dict[str, NamedDerivedProperties]

# Generic analyzer Protocol: callable that returns dict[str, T]
class PropertyAnalyzer(Protocol[T]):
    def __call__(self, individual: DiGraph[Any]) -> NamedGraphPropertiesT[T]:
        ...


# Deriver (non-generic here) — returns NamedDerivedProperties
class PropertyDeriver(Protocol[T]):
    def __call__(
        self,
        named_props: NamedGraphProperties,
        key: GraphPropertyName | list[GraphPropertyName],
    ) -> NamedDerivedPropertiesT[T]:
        ...

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
    outlier_idxs: IndexMappings


class UniqueEntry(TypedDict):
    count: int
    idxs: IndexMappings


Uniques = dict[GraphProperty, UniqueEntry]


class MinFirstIdx(TypedDict):
    idxs: IndexMappings
    values: list[GraphProperty]


class MaxFirstIdx(TypedDict):
    idxs: IndexMappings
    values: list[GraphProperty]


class SimilarityMatrix(TypedDict, total=False):
    full: list[list[float]]  # or np array
    matrix_stats: dict[int, NumericPropertyStats]
    matrix_min_first_idx: dict[int, MinFirstIdx]
    matrix_max_first_idx: dict[int, MaxFirstIdx]


# ----------------------------------------

# region helpers

def _find_property_list[T](
    named_props: NamedGraphPropertiesT[T],
    key: GraphPropertyName,
) -> list[T]:
    """
    Find and validate a property list from named graph properties.

    Parameters
    ----------
    named_props : NamedGraphPropertiesT[T]
        Dictionary-like container of named graph properties
    key : GraphPropertyName
        The property name to look up in named_props

    Returns
    -------
    list[T]
        List of property values for the specified key

    Raises
    ------
    KeyError
        If the key is not found in named_props
    TypeError
        If the property value is not a sequence (list, tuple, or ndarray)
    ValueError
        If the property sequence is empty

    Notes
    -----
    - Converts tuples and numpy arrays to lists for consistent return type
    - Empty sequences are considered invalid and raise ValueError
    """
    if key not in named_props:
        msg = f"Property {key!r} not found in named_props"
        raise KeyError(msg)

    value_list = named_props[key]
    if not isinstance(value_list, (list, tuple, np.ndarray)):
        msg = f"Expected a sequence for '{key}', got {type(value_list).__name__}"
        raise TypeError(msg)

    seq = list(value_list)
    if len(seq) == 0:
        msg = f"No values for property '{key}'"
        raise ValueError(msg)

    return seq

# endregion

# region PropertyDerivers

def matrix_derive_neighbourhood[T](
    named_props: NamedGraphPropertiesT[T],
    key: GraphPropertyName = "neighbourhood",
    *,
    config: ctk.SimilarityConfig,
    symmetric: bool = True,
    n_jobs: int = -1,
    batch_size: int = 5000,
) -> NamedDerivedPropertiesT[SimilarityMatrix]:
    if key not in named_props:
        msg = f"Property '{key}' not found in raw properties"
        raise KeyError(msg)

    if n_jobs == -1:
        cpus = cpu_count()
        if cpus is None:
            cpus = 1
        n_jobs = max(1, cpus - (cpus // 4))

    property_values = named_props[key]
    n = len(property_values)
    matrix = np.full((n, n), 1.0, dtype=np.float32)

    # Calculate total pairs without storing them
    n_pairs = n * (n - 1) // 2 if symmetric else n * (n - 1)
    n_batches = (n_pairs + batch_size - 1) // batch_size

    # Create shared memory for property_values
    data_bytes = pickle.dumps(property_values)
    shm = shared_memory.SharedMemory(create=True, size=len(data_bytes))
    shm.buf[:len(data_bytes)] = data_bytes
    shm_name = shm.name
    shm_size = len(data_bytes)

    try:
        # Generator for pair batches - no memory accumulation
        def batch_generator():
            if symmetric:
                pair_gen = itertools.combinations(range(n), 2)
            else:
                pair_gen = itertools.permutations(range(n), 2)

            batch = []
            for pair in pair_gen:
                batch.append(pair)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        def compute_batch_shared(batch_pairs):
            """Compute using shared memory."""
            shm = shared_memory.SharedMemory(name=shm_name)
            prop_vals = pickle.loads(bytes(shm.buf[:shm_size]))

            return [
                float(ctk.calculate_similarity_from_dicts(
                    prop_vals[i], prop_vals[j], config,
                ))
                for i, j in batch_pairs
            ]

        # Process batches with streaming
        with tqdm(total=n_batches, desc=f"Computing {key}", unit="batch") as pbar:
            for batch_pairs, batch_result in zip(
                batch_generator(),
                Parallel(
                    n_jobs=n_jobs,
                    backend="loky",
                    return_as="generator",
                )(
                    delayed(compute_batch_shared)(batch)
                    for batch in batch_generator()
                ), strict=False,
            ):
                # Fill matrix immediately
                for (i, j), value in zip(batch_pairs, batch_result, strict=False):
                    matrix[i, j] = value
                    if symmetric:
                        matrix[j, i] = value
                pbar.update(1)

    finally:
        # Clean up shared memory
        shm.close()
        shm.unlink()

    return {"similarity_matrix": SimilarityMatrix(full=matrix)}


def derive_numeric_summary(
    named_props: NamedGraphPropertiesT[int | float],
    key: GraphPropertyName,
    *,
    outlier_bound: float = 1.5,
) -> NamedDerivedPropertiesT[NumericPropertyStats]:
    """
    Compute comprehensive statistical summary for numeric graph properties.

    Parameters
    ----------
    named_props : NamedGraphPropertiesT[int | float]
        Dictionary containing graph properties with numeric values.
    key : GraphPropertyName
        The specific property key to analyze from named_props.
    outlier_bound : float, optional
        Multiplier for IQR to determine outlier boundaries (default 1.5).

    Returns
    -------
    NamedDerivedPropertiesT[NumericPropertyStats]
        Dictionary with 'numeric_stats' key containing statistical measures
        including count, mean, median, quartiles, and outlier information.

    Notes
    -----
    - Uses sample standard deviation (ddof=1) for populations > 1
    - Outliers defined as values beyond Q1 - 1.5*IQR or Q3 + 1.5*IQR
    - Returns std=0.0 for single-value datasets
    """
    value_list = _find_property_list(named_props, key)

    arr = np.array(value_list, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    median = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    lower = q1 - outlier_bound * iqr
    upper = q3 + outlier_bound * iqr
    outlier_idxs: IndexMappings = np.where((arr < lower) | (arr > upper))[0].tolist()

    stats: NumericPropertyStats = {
        "count": int(arr.size),
        "uniques": len(set(arr.tolist())),
        "mean": mean,
        "std": std,
        "median": median,
        "Q1": q1,
        "Q3": q3,
        "num_outliers": len(outlier_idxs),
        "outlier_idxs": outlier_idxs,
    }
    return {"numeric_stats": stats}


def derive_uniques(
    named_props: NamedGraphProperties,
    key: GraphPropertyName,
) -> NamedDerivedPropertiesT[Uniques]:
    """
    Derive unique values and their occurrence information from a graph property.

    Parameters
    ----------
    named_props : NamedGraphProperties
        Collection of named graph properties to analyze.
    key : GraphPropertyName
        Name of the property to extract unique values from.

    Returns
    -------
    NamedDerivedPropertiesT[Uniques]
        Dictionary containing "uniques" key mapping to unique value analysis
        with counts and index positions for each unique value.

    Notes
    -----
    - Uses hash() for hashable values, falls back to repr() for unhashable
    - Each unique value maps to dict with "count" and "indexes" keys
    - Indexes track all positions where each unique value appears
    """
    value_list = _find_property_list(named_props, key)

    index_dict = defaultdict(list)
    for idx, val in enumerate(value_list):
        try:
            hash(val)
            dict_key = val
        except Exception:
            dict_key = repr(val)
        index_dict[dict_key].append(idx)

    uniques: Uniques = {}
    for value_key, idxs in index_dict.items():
        uniques[value_key] = {"count": len(idxs), "idxs": idxs}

    return {"uniques": uniques}


def derive_min_first_idx[T](
    named_props: NamedGraphPropertiesT[T],
    key: GraphPropertyName,
) -> NamedDerivedPropertiesT[MinFirstIdx]:
    """
    Derive minimum-first indices by sorting property values in ascending order.

    Parameters
    ----------
    named_props : NamedGraphPropertiesT[T]
        Named graph properties containing the property to sort
    key : GraphPropertyName
        Name of the property to use for sorting

    Returns
    -------
    NamedDerivedPropertiesT[MinFirstIdx]
        Dictionary with "min_first_idx" key containing sorted indices

    Raises
    ------
    TypeError
        When property values are not directly comparable, falls back to
        string representation comparison

    Notes
    -----
    - Stable sort preserves original order for equal values using index
      as tiebreaker
    - For non-comparable types (e.g., mixed types), uses repr() for
      comparison
    - Returns indices that would sort the original values, not the sorted
      values themselves
    """
    value_list = _find_property_list(named_props, key)

    try:
        sorted_indexes = sorted(range(len(value_list)), key=lambda i: (value_list[i], i))
    except TypeError:
        sorted_indexes = sorted(range(len(value_list)), key=lambda i: (repr(value_list[i]), i))

    sorted_values = [value_list[i] for i in sorted_indexes]

    return {"min_first_idx": MinFirstIdx(idxs=sorted_indexes, values=sorted_values)}


def derive_max_first_idx[T](
    named_props: NamedGraphPropertiesT[T],
    key: GraphPropertyName,
) -> NamedDerivedPropertiesT[MinFirstIdx]:
    """
    Derive the maximum-first index ordering for a graph property.

    Parameters
    ----------
    named_props : NamedGraphPropertiesT[T]
        Named graph properties containing the property to sort.
    key : GraphPropertyName
        The name of the property to derive max-first ordering from.

    Returns
    -------
    NamedDerivedPropertiesT[MinFirstIdx]
        Dictionary with "max_first_idx" key containing MaxFirstIdx with
        sorted indexes in descending order.

    Raises
    ------
    TypeError
        When property values are not directly comparable, falls back to
        string representation comparison.

    Notes
    -----
    - Sorts indexes by property values in descending order (largest first)
    - Uses stable sort with index as tiebreaker for consistent ordering
    - Falls back to repr() comparison for non-comparable types like mixed
      numeric/string values
    - Return type annotation uses MinFirstIdx but actually returns
      MaxFirstIdx due to descending sort order
    """
    value_list = _find_property_list(named_props, key)

    try:
        sorted_indexes = sorted(range(len(value_list)), key=lambda i: (value_list[i], i), reverse=True)
    except TypeError:
        sorted_indexes = sorted(range(len(value_list)), key=lambda i: (repr(value_list[i]), i), reverse=True)

    sorted_values = [value_list[i] for i in sorted_indexes]

    return {"max_first_idx": MaxFirstIdx(idxs=sorted_indexes, values=sorted_values)}


def derive_matrix_numerics(
    named_props: NamedGraphPropertiesT[Any],
    key: GraphPropertyName,
    *,
    row_idx: int | None = None,
) -> NamedDerivedPropertiesT[SimilarityMatrix]:
    """
    Compute numeric stats and sorted indices for similarity matrix rows.
    Adds the stats directly to the SimilarityMatrix structure.

    If row_idx is specified, computes stats for that row only.
    Otherwise computes stats for all rows.
    """
    if key not in named_props:
        msg = f"Property '{key}' not found"
        raise KeyError(msg)

    matrix_data = named_props[key].get("similarity_matrix", {})
    matrix = matrix_data.get("full")

    if matrix is None:
        msg = f"No similarity matrix found for '{key}'"
        raise ValueError(msg)

    if isinstance(matrix, list):
        matrix = np.array(matrix)

    def _stats_for_row(row: np.ndarray) -> tuple[NumericPropertyStats, MinFirstIdx, MaxFirstIdx]:
        temp = {"row": row.tolist()}
        stats = derive_numeric_summary(temp, "row")["numeric_stats"]
        min_idx = derive_min_first_idx(temp, "row")["min_first_idx"]
        max_idx = derive_max_first_idx(temp, "row")["max_first_idx"]
        return stats, min_idx, max_idx

    if row_idx is not None:
        stats, min_idx, max_idx = _stats_for_row(matrix[row_idx, :])
        matrix_data["matrix_stats"] = {row_idx: stats}
        matrix_data["matrix_min_first_idx"] = {row_idx: min_idx}
        matrix_data["matrix_max_first_idx"] = {row_idx: max_idx}
    else:
        all_stats = {}
        all_min = {}
        all_max = {}
        for i in range(matrix.shape[0]):
            stats, min_idx, max_idx = _stats_for_row(matrix[i, :])
            all_stats[i] = stats
            all_min[i] = min_idx
            all_max[i] = max_idx

        matrix_data["matrix_stats"] = all_stats
        matrix_data["matrix_min_first_idx"] = all_min
        matrix_data["matrix_max_first_idx"] = all_max

    return {"similarity_matrix": matrix_data}

# endregion

# region Main appliers


def get_raw_population_properties(
    population: list[DiGraph[Any]],
    analyzers: list[PropertyAnalyzer[Any]],
    *,
    n_jobs: int = -1,
) -> RawPopulationProperties:
    """
    Analyze a population of directed graphs using multiple property analyzers.

    Parameters
    ----------
    population : list[DiGraph[Any]]
        List of directed graphs to analyze.
    analyzers : list[PropertyAnalyzer[Any]]
        List of analyzer functions to apply to each graph.
    n_jobs : int, default -1
        Number of parallel jobs. If -1, uses CPU count minus 25% for
        overhead.

    Returns
    -------
    RawPopulationProperties
        Dictionary mapping property names to lists of values collected
        across the population.

    Notes
    -----
    - Uses joblib for parallel processing with automatic batch sizing
    - When n_jobs=-1, reserves 25% of CPUs to prevent system overload
    - Progress tracking shows current n_jobs value during execution
    - Results are aggregated by extending lists for each property key
    """
    if n_jobs == -1:
        cpus = cpu_count()
        if cpus is None:
            cpus = 1
        n_jobs = max(1, cpus - (cpus // 4))

    def _analyze(individual: DiGraph[Any]) -> RawPopulationProperties:
        result: RawPopulationProperties = {}
        for analyzer in analyzers:
            analyzer_result = analyzer(individual)
            for key, value in analyzer_result.items():
                result.setdefault(key, []).append(value)
        return result

    parallel_results: Any = Parallel(
        n_jobs=n_jobs, batch_size="auto",
    )(
        delayed(_analyze)(ind)
        for ind in track(population, description=f"Analyzing population (n_jobs={n_jobs})")
    )
    results = cast("list[RawPopulationProperties]", parallel_results)

    aggregated: RawPopulationProperties = defaultdict(list)
    for result in results:
        for key, value_list in result.items():
            aggregated[key].extend(value_list)

    return dict(aggregated)


def get_derived_population_properties(
    raw_ppulation_properties: RawPopulationProperties,
    derivers: list[PropertyDeriver[Any]],
    *,
    specific_keys: list[GraphPropertyName] | None = None,
    n_jobs: int = -1,
) -> DerivedPopulationPropertiesT[GraphPropertyName]:
    """
    Derive population properties using multiple derivers in parallel.

    Parameters
    ----------
    raw_ppulation_properties : RawPopulationProperties
        Raw population properties to derive from.
    derivers : list[PropertyDeriver[Any]]
        List of property derivers to apply to each property.
    specific_keys : list[GraphPropertyName] | None, optional
        Specific property keys to process. If None, processes all keys from
        raw population properties.
    n_jobs : int, default -1
        Number of parallel jobs. If -1, uses CPU count minus 25% for
        optimal performance.

    Returns
    -------
    DerivedPopulationPropertiesT[GraphPropertyName]
        Dictionary mapping property names to their derived properties.

    Notes
    -----
    - Automatically determines optimal job count when n_jobs=-1 by using
      75% of available CPUs
    - Silently skips derivers that raise exceptions for specific properties
    - Progress tracking shows total number of keys being processed
    """
    if n_jobs == -1:
        cpus = cpu_count() or 1
        n_jobs = max(1, cpus - (cpus // 4))

    keys: list[GraphPropertyName] = list(specific_keys or raw_ppulation_properties.keys())

    def _derive_all_for_property(
        pop: RawPopulationProperties,
        key: GraphPropertyName,
    ) -> tuple[GraphPropertyName, NamedDerivedProperties]:
        derived: NamedDerivedProperties = {}
        for der in derivers:
            try:
                res = der(pop, key)
            except Exception:
                continue
            derived.update(res)
        return key, derived

    parallel_results: Any = Parallel(n_jobs=n_jobs, batch_size="auto")(
        delayed(_derive_all_for_property)(raw_ppulation_properties, k) for k in track(keys, description=f"Deriving properties (n_keys={len(keys)})")
    )

    derived_pop_props: DerivedPopulationProperties = {}
    for property_name, named_deriv_props in parallel_results:
        derived_pop_props[property_name] = named_deriv_props

    return derived_pop_props


# EUCLIDEAN DISTANCE SETTING & FUNCTION
from ariel_experiments.characterize.individual import (
    analyze_branching,
    analyze_coverage,
    analyze_joints,
    analyze_length_of_limbs,
    analyze_mass,
    analyze_number_of_limbs,
    analyze_proportion_spatial,
    analyze_size,
    analyze_symmetry,
)

# Defining the coordinates/vectors that describe individuals
ANALYZERS: dict[str, PropertyAnalyzer[float]] = {
    "branching": analyze_branching,
    "number_of_limbs": analyze_number_of_limbs,
    "length_of_limbs": analyze_length_of_limbs,
    "coverage": analyze_coverage,
    "proportion_spatial": analyze_proportion_spatial,
    "symmetry": analyze_symmetry,
    "size": analyze_size,
    "mass": analyze_mass,
    "joints": analyze_joints,
}


def get_morphological_vector(individual: DiGraph) -> np.ndarray:
    vector = []
    for func in ANALYZERS.values():
        result = func(individual)
        value = next(iter(result.values()))
        vector.append(float(value))
    return np.array(vector, dtype=float)


# Defining the Euclidean distance between two individuals
def euclidean_distance(ind1: DiGraph, ind2: DiGraph) -> float:
    v1 = get_morphological_vector(ind1)
    v2 = get_morphological_vector(ind2)
    return float(np.linalg.norm(v1 - v2))


# TODO: add a way to apply custom derivations that dont map 1:1 on property
def get_custom_derived_population_properties(
    raw_ppulation_properties: RawPopulationProperties,
    analyzer: list[PropertyAnalyzer[Any]],
    *,
    specific_keys: list[GraphPropertyName] | None = None,
    n_jobs: int = -1,
) -> DerivedPopulationPropertiesT[DerivedPropertyName]:
    raise NotImplementedError


def get_full_analyzed_population(
    population: list[DiGraph[Any]],
    analyzers: list[PropertyAnalyzer[Any]],
    derivers: list[PropertyDeriver[Any]],
    *,
    n_jobs: int = -1,
) -> AnalyzedPopulation:
    """
    Analyze a population of graphs with both raw and derived properties.

    Parameters
    ----------
    population : list[DiGraph[Any]]
        List of directed graphs to analyze.
    analyzers : list[PropertyAnalyzer[Any]]
        List of property analyzers to apply to each graph.
    derivers : list[PropertyDeriver[Any]]
        List of property derivers to compute derived properties.
    n_jobs : int, default -1
        Number of parallel jobs for computation. -1 uses all processors.

    Returns
    -------
    AnalyzedPopulation
        Container with both raw and derived population properties.
    """
    raw = get_raw_population_properties(population, analyzers, n_jobs=n_jobs)
    derived = get_derived_population_properties(raw, derivers, n_jobs=n_jobs)
    return AnalyzedPopulation(raw=raw, derived=derived)


class AnalyzedPopulation:
    """
    Runtime container for analyzed population data.

    Attributes
    ----------
    - raw: RawPopulationProperties
    - derived: DerivedPopulationProperties

    Convenience methods mirror the former module helpers.
    """

    def __init__(
        self,
        raw: RawPopulationProperties | None = None,
        derived: DerivedPopulationProperties | None = None,
    ) -> None:
        self.raw: RawPopulationProperties = raw or {}
        self.derived: DerivedPopulationProperties = derived or {}

    # @classmethod
    # def from_typed(cls, ap: dict) -> AnalyzedPopulation:
    #     return cls(raw=ap.get("raw", {}), derived=ap.get("derived", {}))

    def to_dict(self) -> dict:
        return {"raw": self.raw, "derived": self.derived}

    def df_from_derived(
        self,
        derived_property_name: DerivedPropertyName,
        *,
        keys: list[NamedGraphProperties] | None = None,
        decimals: int = 2,
        exclude_idxs: bool = True,
        max_map_keys: int = 20,
        sort_columns: bool = False,
        save_file: Path | str | None = None,
    ) -> pd.DataFrame:
        """
        Build a DataFrame where index is the property name and columns are the
        keys from the derived sub-dictionary (e.g. numeric_stats -> count, mean, ...).

        Only include properties that actually contain the requested derived entry.
        Non-scalar values (lists, dicts) are kept as-is (cell contains the object).

        If `exclude_idxs` is True, any derived dict keys containing "idxs"
        will be omitted from the resulting DataFrame columns.

        Floats in scalar or nested structures will be rounded to `decimals`.

        If `sort_columns` is True, top-level dict keys (and nested subkeys when
        flattened) are inserted in numeric order when possible (otherwise lexicographic).
        """
        rows: list[dict[str, Any]] = []

        # helper for deterministic sorting of keys (try numeric first)
        def _sort_key(x: Any):
            # numeric types sort by value
            if isinstance(x, (int, float)):
                return (0, float(x))
            # strings: try parse leading token (before a dot) as int/float
            if isinstance(x, str):
                leading = x.split(".", 1)[0]
                for conv in (int, float):
                    try:
                        return (0, float(conv(leading)))
                    except Exception:
                        pass
                # fallback to case-insensitive string sort
                return (1, x.lower())
            # fallback to string representation
            return (2, str(x))

        candidate_props = keys if keys is not None else list(self.derived.keys())

        for prop in candidate_props:
            derived_dict = self.derived.get(prop)
            if not derived_dict:
                continue
            deriv_prop = derived_dict.get(derived_property_name, None)
            if deriv_prop is None:
                continue

            if isinstance(deriv_prop, dict):
                # if top-level dict has too many keys (e.g. uniques mapping), avoid exploding into many columns
                if len(deriv_prop) > max_map_keys:
                    console.print(
                        f"[yellow]Warning:[/] derived '{derived_property_name}' for property '{prop}' "
                        f"contains {len(deriv_prop)} top-level keys and would produce many columns. "
                        f"Skipping detailed flattening for this property. "
                        f"Consider excluding it from `keys` if you don't want it in the table.",
                    )
                    rows.append({"property": prop, f"{derived_property_name}_map_len": len(deriv_prop)})
                    continue

                row: dict[str, Any] = {"property": prop}
                items_iter = sorted(deriv_prop.items(), key=lambda kv: _sort_key(kv[0])) if sort_columns else deriv_prop.items()
                for k, v in items_iter:
                    # only apply substring check when key is a string
                    if isinstance(k, str) and exclude_idxs and "idxs" in k:
                        continue
                    # if value is a dict (e.g. uniques -> {val: {...}}), flatten one level
                    if isinstance(v, dict):
                        sub_items = sorted(v.items(), key=lambda kv: _sort_key(kv[0])) if sort_columns else v.items()
                        for subk, subv in sub_items:
                            col_name = f"{k}.{subk}"
                            if exclude_idxs and "idxs" in col_name:
                                continue
                            row[col_name] = subv
                    else:
                        row[k] = v
            else:
                row = {"property": prop, derived_property_name: deriv_prop}
            rows.append(row)

        if not rows:
            return pd.DataFrame(columns=["property"]).set_index("property")

        df = pd.DataFrame(rows).set_index("property")

        # round floats in every cell (handles nested lists/tuples/dicts)
        def _round_obj(obj: Any) -> Any:
            # numpy types
            if isinstance(obj, (float, np.floating)):
                return round(float(obj), decimals)
            if isinstance(obj, list):
                return [_round_obj(x) for x in obj]
            if isinstance(obj, tuple):
                return tuple(_round_obj(x) for x in obj)
            if isinstance(obj, dict):
                out: dict[Any, Any] = {}
                for kk, vv in obj.items():
                    # preserve keys, but optionally skip idx keys (should already be excluded)
                    if exclude_idxs and isinstance(kk, str) and "idxs" in kk:
                        continue
                    out[kk] = _round_obj(vv)
                return out
            return obj

        df = df.apply(lambda col: col.map(_round_obj))

        # ensure final column ordering respects numeric-aware sorting when requested
        if sort_columns:
            try:
                sorted_cols = sorted(df.columns, key=_sort_key)
                df = df.reindex(columns=sorted_cols)
            except Exception:
                # fallback: leave whatever pandas produced
                pass

        if save_file:
            path = DATA / Path(save_file)
            console.log(f"saving file to {path}")
            df.to_csv(path)

        return df

    def update_derived_with_deriver(
        self,
        deriver: PropertyDeriver[Any],
        key: GraphPropertyName,
    ) -> AnalyzedPopulation:
        """Run a single deriver for `key` and merge results (in-place)."""
        res = deriver(self.raw, key)
        existing = self.derived.get(key, {})
        existing.update(res)
        self.derived[key] = existing
        return self

    def recompute_derived_for_key(
        self,
        derivers: list[PropertyDeriver[Any]],
        key: GraphPropertyName,
    ) -> AnalyzedPopulation:
        """Recompute provided derivers for `key` and replace that derived mapping."""
        merged: NamedDerivedProperties = {}
        for d in derivers:
            merged.update(d(self.raw, key))
        self.derived[key] = merged
        return self

    def show_tree(self) -> None:
        """Compact tree: raw keys with length/type and derived keys as children."""
        raw = self.raw
        derived = self.derived

        type_colors = {
            "numeric": "bright_blue",
            "str": "bright_magenta",
            "mixed": "red",
            "empty": "grey50",
        }
        derive_colors = {
            "min_first_idx": "cyan",
            "max_first_idx": "magenta",
            "numeric_stats": "green",
            "uniques": "yellow",
        }
        root = Tree(f"AnalyzedPopulation — props={len(raw)} derived_keys={len(derived)}")

        raw_node = root.add("raw")
        for key in sorted(raw.keys()):
            vals = raw[key]
            try:
                length = len(vals)
            except Exception:
                length = 0

            if length == 0:
                typ_label = "empty"
            else:
                first_type = type(vals[0]).__name__
                typ_label = first_type if all(type(v).__name__ == first_type for v in vals) else "mixed"
                if typ_label in {"int", "float"}:
                    typ_label = "numeric"

            style = type_colors.get(typ_label, "white")
            raw_node.add(f"{key}: len={length}, type={typ_label}", style=style)

        derived_node = root.add("derived")
        for key in sorted(raw.keys()):
            named = derived.get(key, {})
            prop_node = derived_node.add(f"{key}")
            if not named:
                prop_node.add("(no derived)")
                continue
            for dname in sorted(named.keys()):
                dstyle = derive_colors.get(dname, "white")
                prop_node.add(str(dname), style=dstyle)

        console.print(root)

# endregion

# MARK entry point


if __name__ == "__main__":
    from time import perf_counter

    from ariel_experiments.characterize.individual import (
        analyze_json_hash,
        analyze_mass,
        analyze_module_counts,
    )
    from ariel_experiments.utils.initialize import (
        generate_random_population_parallel,
    )

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

    console.rule("test derivation of properties")

    derivation_analyzers = [
        derive_numeric_summary,
        derive_uniques,
        derive_min_first_idx,
        derive_max_first_idx,
    ]

    deriv_pop_props = get_derived_population_properties(
        raw_properties,
        derivation_analyzers,
    )

    console.rule("derived properties:")
    console.print(deriv_pop_props)

     # ----------------------------------------------

    console.rule("test full creation")

    population_size = 5
    population = generate_random_population_parallel(population_size)
    analyzed_population = get_full_analyzed_population(population, individual_analyzers, derivation_analyzers)
    analyzed_population.show_tree()

    df1 = analyzed_population.df_from_derived("numeric_stats")
    df2 = analyzed_population.df_from_derived(
        "uniques",
        keys=["core", "brick", "hinge", "none"],
        sort_columns=True,
        save_file="test.csv",
    )
    console.print(df1)
    console.print(df2)

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
    console.print(f"population of size {population_size} took {end - start:.4f} seconds")

    # ----------------------------------------------

    console.rule("test derive speed:")

    start = perf_counter()
    deriv_pop_props = get_derived_population_properties(
        raw_properties,
        derivation_analyzers,
    )
    end = perf_counter()
    console.print(f"population of size {population_size} took {end - start:.4f} seconds")

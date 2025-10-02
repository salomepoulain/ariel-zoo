# Standard library
import json
import os
from collections import Counter, defaultdict
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import Any, Never
import collections.abc

import matplotlib.pyplot as plt

# Third-party libraries
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from matplotlib.ticker import PercentFormatter
from networkx import DiGraph
from rich.console import Console
from rich.progress import track

from ariel.body_phenotypes.robogen_lite.config import (
    NUM_OF_FACES,
    NUM_OF_ROTATIONS,
    NUM_OF_TYPES_OF_MODULES,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
)
from ariel.body_phenotypes.robogen_lite.decoders.visualize_tree import (
    visualize_tree_from_graph,
)

# Local libraries
from ariel.body_phenotypes.robogen_lite.modules.brick import BRICK_MASS
from ariel.body_phenotypes.robogen_lite.modules.core import CORE_MASS
from ariel.body_phenotypes.robogen_lite.modules.hinge import (
    ROTOR_MASS,
    STATOR_MASS,
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




def create_boxplot_from_dict(
    properties_dict: dict[str, Any],
    keys: list[str] | None = None,
) -> None:
    # If no keys provided, plot all except 'population'
    if keys is None:
        keys = list(properties_dict.keys())
    # Prepare data for seaborn
    data = []
    labels = []
    for key in keys:
        if key in properties_dict:
            data.append(properties_dict[key])
            labels.append(key)
    # Create boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data)
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.ylabel("Value")
    plt.title("Boxplot of Robot Properties")
    plt.show()
    
    
def create_histogram_from_dict(
    properties_dict: dict[str, Any],
    keys: list[str] | None = None,
    bins: int = 30,
) -> None:
    """
    Draw a grouped histogram of the selected keys.
    For string-valued keys (e.g., hashes), plot the distribution of their frequencies (separately).
    For numeric keys, plot all together in one histogram.
    Title is the key names if there are fewer than 3, otherwise "Histogram of Robot Properties".
    """
    if keys is None:
        keys = list(properties_dict.keys())

    valid_keys = [k for k in keys if k in properties_dict]

    # Check if all selected keys are numeric
    if all(isinstance(properties_dict[k][0], (int, float)) for k in valid_keys):
        # Plot all numeric keys together
        data = [properties_dict[k] for k in valid_keys]
        plt.figure(figsize=(10, 6))
        n, bins_, _patches = plt.hist(
            data,
            bins=bins,
            label=valid_keys,
            edgecolor="black",
            linewidth=1.2,
            alpha=0.8,
            histtype="bar",
            rwidth=0.9,
        )
        total = n.sum()
        plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=total))
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.legend()
        plt.title(
            ", ".join(valid_keys)
            if len(valid_keys) <= 3
            else "Histogram of Robot Properties",
        )
        plt.show()
    else:
        # Plot string-valued keys separately
        for key in valid_keys:
            values = properties_dict[key]
            if not values:
                continue
            plt.figure(figsize=(10, 6))
            if isinstance(values[0], str):
                counts = Counter(values)
                freqs = list(counts.values())
                n, bins_, _patches = plt.hist(
                    freqs,
                    bins=bins,
                    edgecolor="black",
                    linewidth=1.2,
                    alpha=0.8,
                    rwidth=0.9,
                    label=[key],
                    histtype="bar",
                )
                total = n.sum()
                plt.gca().yaxis.set_major_formatter(
                    PercentFormatter(xmax=total),
                )
                plt.xlabel("Genotype frequency")
                plt.ylabel("Number of unique genotypes")
                plt.title(key)
                plt.legend()
                plt.show()

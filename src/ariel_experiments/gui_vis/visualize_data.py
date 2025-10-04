# Standard library
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

# Third-party libraries
import numpy as np
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from rich.console import Console

# Local libraries

# Global constants
# SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = Path(CWD / "__data__")
DATA.mkdir(exist_ok=True)
SEED = 42
DPI=300

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)


def create_boxplot_from_raw(
    properties_dict: dict[str, Any],
    keys: list[str] | None = None,
    *,
    title: str | None = None,
    save_file: Path | str | None = None,
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
    plt.figure(figsize=(10, 6), dpi=DPI)
    sns.boxplot(data=data)
    plt.xticks(ticks=range(len(labels)), labels=labels)
    plt.ylabel("Value")
    
    if title:
        plt.title(title)
    else:
        plt.title("Boxplot of Robot Properties")

    if save_file:
        path = DATA / Path(save_file)
        console.log(f"saving file to {path}")
        plt.savefig(path)    

    plt.show()


def create_histogram_from_raw(
    properties_dict: dict[str, Any],
    keys: list[str] | None = None,
    *,
    bins: int = 30,
    title: str | None = None,
    save_file: Path | str | None = None,
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
        plt.figure(figsize=(10, 6), dpi=DPI)
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

        if title:
            plt.title(title)
        else:
            plt.title(
                ", ".join(valid_keys)
                if len(valid_keys) <= 3
                else "Histogram of Robot Properties",
            )

        if save_file:
            path = DATA / Path(save_file)
            console.log(f"saving file to {path}")
            plt.savefig(path)
        
        plt.show()
    else:
        # Plot string-valued keys separately
        for key in valid_keys:
            values = properties_dict[key]
            if not values:
                continue
            plt.figure(figsize=(10, 6), dpi=DPI)
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
                
                if title:
                    plt.title(title)
                else:
                    plt.title(key)
                
                plt.legend()
                
                if save_file:
                    path = DATA / Path(save_file)
                    console.log(f"saving file to {path}")
                    plt.savefig(path)
                
                plt.show()

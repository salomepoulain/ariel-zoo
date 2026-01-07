"""Grid configuration dataclasses for visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class EmbeddingGridConfig:
    """Embedding grid subplot - works for both bokeh interactive and matplotlib static."""

    title: str
    embeddings: np.ndarray  # (n_points, 2)
    idxs: list[int]  # Population indices

    # Optional hover data (only used if interactive=True)
    hover_data: list[str] | None = None

    # Styling
    default_dot_size: int = 8
    highlight_dot_size: int = 8
    axis_label_fontsize: str = "8pt"
    tick_fontsize: str = "6pt"
    title_fontsize: str = "10pt"
    hover_fontsize: str = "10px"


@dataclass
class HeatmapGridConfig:
    """Heatmap grid subplot."""

    title: str
    heatmap_data: np.ndarray  # (n_robots, n_robots)

    # Styling
    colormap: str = "viridis"
    vmin: float | None = None
    vmax: float | None = None
    show_labels: bool = True
    title_fontsize: str = "10pt"


@dataclass
class RobotGridConfig:
    """Robot image grid - standalone, not matrix-based."""

    title: str
    under_title: str
    idxs: list[int]

    img_under_title: list[str] | None = None
    img_under_title_fontsize: int = 10
    title_fontsize: int | str = 10
    under_title_fontsize: int | str = 20
    title_fontweight: str = "normal"
    under_title_fontweight: str = "normal"

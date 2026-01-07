"""Embedding grid plotting - supports both interactive (Bokeh) and static (Matplotlib) modes."""

from __future__ import annotations

import base64
import logging
import warnings
from io import BytesIO

# Suppress Bokeh warnings about missing renderers
warnings.filterwarnings("ignore", message=".*MISSING_RENDERERS.*")
warnings.filterwarnings("ignore", category=UserWarning, module="bokeh")

# Suppress Bokeh logger warnings
logging.getLogger("bokeh").setLevel(logging.ERROR)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.layouts import column, gridplot
from bokeh.models import ColumnDataSource, Div, HoverTool
from bokeh.plotting import figure, show
from matplotlib.colors import to_hex
from PIL import Image
from rich.progress import track

from .grid_config import EmbeddingGridConfig


def plot_embedding_grid(
    sub_plots: list[list[EmbeddingGridConfig]],
    interactive: bool = True,
    population_thumbnails: list[str] | None = None,
    follow_idx_list: list[int] | None = None,
    plot_width: int | None = None,
    plot_height: int = 200,
    max_full_width: int = 800,
    super_title: str | None = None,
    global_axis: bool = True,
) -> None:
    """
    Plot embedding grid - either interactive (bokeh) or static (matplotlib).

    Args:
        sub_plots: 2D list of EmbeddingGridConfig objects
        interactive: If True, use bokeh with hover. If False, use matplotlib.
        population_thumbnails: Required if interactive=True (base64 encoded thumbnails)
        follow_idx_list: Indices to highlight (black border, opaque). Others are faded.
        plot_width: Width of each subplot. If None, auto-calculated from max_full_width.
        plot_height: Height of each subplot
        max_full_width: Maximum total width of the grid (used when plot_width=None)
        super_title: Overall title for the grid
        global_axis: If True, all subplots share the same axis range (default: True)
    """
    if interactive:
        if population_thumbnails is None:
            msg = "population_thumbnails required for interactive mode"
            raise ValueError(msg)
        _plot_interactive_bokeh(
            sub_plots,
            population_thumbnails,
            follow_idx_list,
            plot_width,
            plot_height,
            max_full_width,
            super_title,
            global_axis,
        )
    else:
        _plot_static_matplotlib(
            sub_plots,
            follow_idx_list,
            plot_width,
            plot_height,
            max_full_width,
            super_title,
            global_axis,
        )


def _plot_interactive_bokeh(
    sub_plots: list[list[EmbeddingGridConfig]],
    population_thumbnails: list[str],
    follow_idx_list: list[int] | None,
    plot_width: int | None,
    plot_height: int,
    max_full_width: int,
    super_title: str | None,
    global_axis: bool,
) -> None:
    """Create interactive Bokeh grid plot."""
    n_robots = len(population_thumbnails)
    n_rows = len(sub_plots)
    n_cols = len(sub_plots[0]) if n_rows > 0 else 0

    # Auto-calculate plot_width if not specified
    if plot_width is None:
        spacing_per_plot = 20  # Approximate spacing between plots
        available_width = max_full_width - (spacing_per_plot * (n_cols - 1))
        plot_width = max(150, available_width // n_cols)  # Min 150px per plot

    # Setup colors (rainbow with transparency)
    rgba_colors = plt.cm.rainbow(np.linspace(0, 1, n_robots))
    rgba_colors[:, 3] = 0.4  # Base transparency

    # Calculate global axis ranges if requested
    global_x_range = None
    global_y_range = None

    if global_axis:
        x_min, x_max = float("inf"), float("-inf")
        y_min, y_max = float("inf"), float("-inf")

        for row_subplots in sub_plots:
            for subplot in row_subplots:
                emb = subplot.embeddings
                if (
                    emb.size > 0
                    and emb.ndim == 2
                    and emb.shape[1] == 2
                    and emb.shape[0] > 0
                ):
                    x_min = min(x_min, emb[:, 0].min())
                    x_max = max(x_max, emb[:, 0].max())
                    y_min = min(y_min, emb[:, 1].min())
                    y_max = max(y_max, emb[:, 1].max())

        # Add padding (10% on each side)
        if x_min != float("inf"):
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1
            global_x_range = (x_min - x_padding, x_max + x_padding)
            global_y_range = (y_min - y_padding, y_max + y_padding)

    # Build grid
    grid_layout = []

    for row_subplots in sub_plots:
        row_plots = []

        for subplot in row_subplots:
            emb = subplot.embeddings

            # Validation - create clean white placeholder for empty/invalid data
            if (
                emb.size == 0
                or emb.ndim != 2
                or emb.shape[1] != 2
                or emb.shape[0] == 0
            ):
                p = figure(
                    title=subplot.title,
                    width=plot_width,
                    height=plot_height,
                    toolbar_location=None,
                    x_range=global_x_range,
                    y_range=global_y_range,
                )
                # Remove grid lines and axes for clean white appearance
                p.xgrid.visible = False
                p.ygrid.visible = False
                p.xaxis.visible = False
                p.yaxis.visible = False
                p.outline_line_color = None
                p.background_fill_color = "white"
                p.border_fill_color = "white"

                # Apply font size settings for title consistency
                p.title.text_font_size = subplot.title_fontsize

                row_plots.append(p)
                continue

            n_points = len(subplot.idxs)

            # Setup sizes and line colors per-subplot based on settings
            sizes = [subplot.default_dot_size] * n_robots
            line_colors = [None] * n_robots

            # Apply follow_idx highlighting
            if follow_idx_list:
                follow_set = set(follow_idx_list)
                sizes = [
                    subplot.highlight_dot_size if i in follow_set else subplot.default_dot_size
                    for i in range(n_robots)
                ]
                line_colors = [
                    "black" if i in follow_set else None for i in range(n_robots)
                ]

                # Update colors for highlighting
                for i in range(n_robots):
                    if i in follow_set:
                        rgba_colors[i, 3] = 1.0  # Fully opaque
                    else:
                        rgba_colors[i] = [0.0, 0.0, 0.0, 0.2]  # Faded grey

            hex_colors = [to_hex(c, keep_alpha=True) for c in rgba_colors]

            # Build DataFrame with data for this subplot
            robots_df = pd.DataFrame(
                {
                    "x": emb[:, 0],
                    "y": emb[:, 1],
                    "digit": [str(idx) for idx in subplot.idxs],
                    "image": [population_thumbnails[idx] for idx in subplot.idxs],
                    "color": [hex_colors[idx] for idx in subplot.idxs],
                    "size": [sizes[idx] for idx in subplot.idxs],
                    "line_color": [line_colors[idx] for idx in subplot.idxs],
                    "hover_info": subplot.hover_data or [""] * n_points,
                }
            )

            # Sort so highlighted points render on top
            if follow_idx_list:
                robots_df["sort_order"] = [
                    1 if idx in set(follow_idx_list) else 0 for idx in subplot.idxs
                ]
                robots_df = robots_df.sort_values("sort_order", ascending=True)

            source = ColumnDataSource(robots_df)

            # Create figure with global axis ranges if enabled
            p = figure(
                title=subplot.title,
                width=plot_width,
                height=plot_height,
                tools="pan,wheel_zoom,reset,save",
                toolbar_location="above",
                x_range=global_x_range,
                y_range=global_y_range,
            )

            p.scatter(
                "x",
                "y",
                source=source,
                color="color",
                line_alpha=1,
                line_color="line_color",
                line_width=1,
                size="size",
            )

            # Apply font size settings from subplot
            p.title.text_font_size = subplot.title_fontsize
            p.xaxis.axis_label_text_font_size = subplot.axis_label_fontsize
            p.yaxis.axis_label_text_font_size = subplot.axis_label_fontsize
            p.xaxis.major_label_text_font_size = subplot.tick_fontsize
            p.yaxis.major_label_text_font_size = subplot.tick_fontsize

            # Hover tooltip
            hover = HoverTool(
                tooltips=f"""
                  <div>
                      <img src='@image' style='float:left; margin:5px; width:auto; height:auto;'/>
                  </div>
                  <div style="font-size:{subplot.hover_fontsize}; font-weight: bold;">
                      <span style='color:#224499'>ID: @digit</span><br>
                      <span style='color:#333'>@hover_info</span>
                  </div>
              """
            )
            p.add_tools(hover)
            row_plots.append(p)

        grid_layout.append(row_plots)

    grid = gridplot(grid_layout)
    # Show with optional super title
    if super_title:
        title_div = Div(
            text=f"<h2 style='text-align: center; font-weight: bold; color: #224499; margin-bottom:10px;'>{super_title}</h2>",
            width=max_full_width,
            height=40,
        )
        show(column(title_div, grid))
    else:
        show(grid)


def _plot_static_matplotlib(
    sub_plots: list[list[EmbeddingGridConfig]],
    follow_idx_list: list[int] | None,
    plot_width: int | None,
    plot_height: int,
    max_full_width: int,
    super_title: str | None,
    global_axis: bool,
) -> None:
    """Create static Matplotlib grid plot (no thumbnails)."""
    n_rows = len(sub_plots)
    n_cols = len(sub_plots[0]) if n_rows > 0 else 0

    # Calculate figure size
    fig_width = max_full_width / 100  # Convert pixels to inches (approx)
    fig_height = (plot_height / 100) * n_rows

    # Calculate global axis ranges if requested
    global_x_range = None
    global_y_range = None

    if global_axis:
        x_min, x_max = float("inf"), float("-inf")
        y_min, y_max = float("inf"), float("-inf")

        for row_subplots in sub_plots:
            for subplot in row_subplots:
                emb = subplot.embeddings
                if (
                    emb.size > 0
                    and emb.ndim == 2
                    and emb.shape[1] == 2
                    and emb.shape[0] > 0
                ):
                    x_min = min(x_min, emb[:, 0].min())
                    x_max = max(x_max, emb[:, 0].max())
                    y_min = min(y_min, emb[:, 1].min())
                    y_max = max(y_max, emb[:, 1].max())

        # Add padding (10% on each side)
        if x_min != float("inf"):
            x_padding = (x_max - x_min) * 0.1
            y_padding = (y_max - y_min) * 0.1
            global_x_range = (x_min - x_padding, x_max + x_padding)
            global_y_range = (y_min - y_padding, y_max + y_padding)

    # Create figure
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )

    if super_title:
        fig.suptitle(super_title, fontsize=14, fontweight="bold")

    # Determine n_robots from first valid subplot
    n_robots = 0
    for row_subplots in sub_plots:
        for subplot in row_subplots:
            if len(subplot.idxs) > 0:
                n_robots = max(subplot.idxs) + 1
                break
        if n_robots > 0:
            break

    # Setup colors
    rgba_colors = plt.cm.rainbow(np.linspace(0, 1, n_robots))
    rgba_colors[:, 3] = 0.6  # Base transparency for static plots

    # Plot each subplot
    for row_idx, row_subplots in enumerate(sub_plots):
        for col_idx, subplot in enumerate(row_subplots):
            ax = axes[row_idx, col_idx]
            emb = subplot.embeddings

            # Skip empty subplots
            if (
                emb.size == 0
                or emb.ndim != 2
                or emb.shape[1] != 2
                or emb.shape[0] == 0
            ):
                ax.set_title(subplot.title)
                ax.axis("off")
                continue

            # Determine colors and sizes
            colors = [rgba_colors[idx] for idx in subplot.idxs]
            sizes = [subplot.default_dot_size * 5 for _ in subplot.idxs]  # Matplotlib uses larger sizes

            if follow_idx_list:
                follow_set = set(follow_idx_list)
                colors = []
                sizes = []
                for idx in subplot.idxs:
                    if idx in follow_set:
                        color = rgba_colors[idx].copy()
                        color[3] = 1.0  # Fully opaque
                        colors.append(color)
                        sizes.append(subplot.highlight_dot_size * 5)
                    else:
                        colors.append([0.0, 0.0, 0.0, 0.2])  # Faded grey
                        sizes.append(subplot.default_dot_size * 5)

            # Plot scatter
            ax.scatter(
                emb[:, 0],
                emb[:, 1],
                c=colors,
                s=sizes,
                edgecolors="black" if follow_idx_list else "none",
                linewidths=0.5 if follow_idx_list else 0,
            )

            # Apply global axis if enabled
            if global_x_range and global_y_range:
                ax.set_xlim(global_x_range)
                ax.set_ylim(global_y_range)

            ax.set_title(subplot.title)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

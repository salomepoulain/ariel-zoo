import logging
import warnings

# Suppress Bokeh warnings about missing renderers
warnings.filterwarnings("ignore", message=".*MISSING_RENDERERS.*")
warnings.filterwarnings("ignore", category=UserWarning, module="bokeh")

# Suppress Bokeh logger warnings
logging.getLogger("bokeh").setLevel(logging.ERROR)

import base64
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bokeh.layouts import column, gridplot
from bokeh.models import (
    ColumnDataSource,
    Div,
    HoverTool,
)
from bokeh.plotting import figure, show
from matplotlib.colors import to_hex
from PIL import Image
from rich.progress import track


@dataclass
class EmbedSubplot:
    title: str
    embeddings: list[Any]  # idk what umap returns actually
    idxs: list[int]  # idxs that match the embeddings
    hover_data: list[Any]  # additional data i want to show for my tooltip

    # Dot size settings
    default_dot_size: int = 8  # Size for regular (non-highlighted) dots
    highlight_dot_size: int = 8  # Size for highlighted/followed dots
    # Font size settings
    axis_label_fontsize: str = "8pt"  # Font size for axis labels
    tick_fontsize: str = "6pt"  # Font size for tick labels
    title_fontsize: str = "10pt"  # Font size for subplot title
    hover_fontsize: str = (
        "10px"  # Font size for hover tooltip (use px for HTML)
    )


def plot_interactive_embed_grid(
    sub_plots: list[list[EmbedSubplot]],
    population_thumbnails: list[str],
    follow_idx_list: list[int] | None = None,
    plot_width: int | None = None,
    plot_height: int = 200,
    max_full_width: int = 800,
    super_title: str | None = None,
    upscale: int | None = None,
    global_axis: bool = True,
) -> None:
    """
    Creates a Grid of Interactive UMAP Scatter plots.

    Args:
        sub_plots: 2D list of UmapSubplot objects defining the grid
        population_thumbnails: Pre-generated base64 thumbnail strings
        follow_idx_list: Indices to highlight (black border, opaque). Others are faded.
        plot_width: Width of each subplot. If None, auto-calculated from max_full_width.
        plot_height: Height of each subplot
        max_full_width: Maximum total width of the grid (used when plot_width=None)
        super_title: Overall title for the grid
        upscale: If provided, upscales tooltip images by this factor (e.g., 2 = 2x larger)
        global_axis: If True, all subplots share the same axis range (default: True)
    """
    n_robots = len(population_thumbnails)
    n_rows = len(sub_plots)
    n_cols = len(sub_plots[0]) if n_rows > 0 else 0

    # Auto-calculate plot_width if not specified
    if plot_width is None:
        spacing_per_plot = 20  # Approximate spacing between plots
        available_width = max_full_width - (spacing_per_plot * (n_cols - 1))
        plot_width = max(150, available_width // n_cols)  # Min 150px per plot

    # Upscale thumbnails if requested
    if upscale is not None and upscale != 1:

        def upscale_thumbnail(b64_str: str, scale: float) -> str:
            """Upscale a base64 thumbnail by a factor."""
            _header, encoded = b64_str.split(",", 1)
            data = base64.b64decode(encoded)
            img = Image.open(BytesIO(data))

            new_size = (int(img.width * scale), int(img.height * scale))
            img_scaled = img.resize(new_size, Image.Resampling.BICUBIC)

            buffer = BytesIO()
            img_scaled.save(
                buffer, format="PNG", optimize=False, compress_level=1
            )
            return (
                "data:image/png;base64,"
                + base64.b64encode(buffer.getvalue()).decode()
            )

        display_thumbnails = [
            upscale_thumbnail(thumb, upscale)
            for thumb in track(
                population_thumbnails,
                description=f"Upscaling thumbnails {upscale}x...",
                disable=True,
            )
        ]
    else:
        display_thumbnails = population_thumbnails

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
                emb = np.array(subplot.embeddings)
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
            emb = np.array(subplot.embeddings)

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
                    x_range=global_x_range or None,
                    y_range=global_y_range or None,
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
                    subplot.highlight_dot_size
                    if i in follow_set
                    else subplot.default_dot_size
                    for i in range(n_robots)
                ]
                line_colors = [
                    "black" if i in follow_set else None
                    for i in range(n_robots)
                ]

                # Update colors for highlighting
                for i in range(n_robots):
                    if i in follow_set:
                        rgba_colors[i, 3] = 1.0  # Fully opaque
                    else:
                        rgba_colors[i] = [0.0, 0.0, 0.0, 0.2]  # Faded grey

            hex_colors = [to_hex(c, keep_alpha=True) for c in rgba_colors]

            # Build DataFrame with data for this subplot
            robots_df = pd.DataFrame({
                "x": emb[:, 0],
                "y": emb[:, 1],
                "digit": [str(idx) for idx in subplot.idxs],
                "image": [display_thumbnails[idx] for idx in subplot.idxs],
                "color": [hex_colors[idx] for idx in subplot.idxs],
                "size": [sizes[idx] for idx in subplot.idxs],
                "line_color": [line_colors[idx] for idx in subplot.idxs],
                "hover_info": subplot.hover_data or [""] * n_points,
            })

            # Sort so highlighted points render on top
            if follow_idx_list:
                robots_df["sort_order"] = [
                    1 if idx in set(follow_idx_list) else 0
                    for idx in subplot.idxs
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
                x_range=global_x_range or None,
                y_range=global_y_range or None,
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

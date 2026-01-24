from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, cast
import re
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:

    class AxesProxy(Axes):
        def __getitem__(self, key: str) -> "AxesProxy": ...

else:
    AxesProxy = Any


__all__ = [
    "Alignment",
    "GridPlotter",
    "PlotterConfig",
]


class Alignment(Enum):
    NW, N, NE = (0.0, 0.0), (0.0, 0.5), (0.0, 1.0)
    W, C, E = (0.5, 0.0), (0.5, 0.5), (0.5, 1.0)
    SW, S, SE = (1.0, 0.0), (1.0, 0.5), (1.0, 1.0)


class CellType(Enum):
    EMPTY = "empty"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    LINE = "line"
    IMAGE = "image"


@dataclass
class CellData:
    """Stores all data for a single grid cell."""

    ax: Axes | None = None
    raw_data: np.ndarray | None = None
    artist: Any = None
    cell_type: CellType = CellType.EMPTY


class PlotterConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # Layout Geometry
    n_rows: int = 1
    n_cols: int = 1
    cell_ratio: float = 1.0  # width/height
    fig_ratio: float | None = None
    dpi: int = 250
    fig_height: float = 6.0
    margin: tuple[float, float, float, float] = (0.05, 0.05, 0.05, 0.05)
    row_space: float = 0.05
    col_space: float = 0.05

    # Image Scaling & Behavior
    keep_global_scale: bool = True  # Maintain relative pixel sizes
    fit_fig_to_content: bool = True  # Change cell_ratio to match largest image
    img_alignment: Alignment = Alignment.C

    # Theme Defaults (Applied in background)
    title_size: float = 12.0
    label_size: float = 8.0
    tick_size: float = 5.0
    spine_color: str = "black"
    spine_width: float = 0.8
    grid_on: bool = False

    _on_change_callback: Callable[[], None] | None = None

    def __setattr__(self, name, value) -> None:
        super().__setattr__(name, value)
        if (
            not name.startswith("_")
            and hasattr(self, "_on_change_callback")
            and self._on_change_callback
        ):
            self._on_change_callback()


# --- Vectorized Recursive Proxy ---
class SubplotGroup:
    def __init__(self, items: Iterable[Any]) -> None:
        self._items = (
            items.flatten() if isinstance(items, np.ndarray) else list(items)
        )

    def __getattr__(self, name: str) -> "SubplotGroup" | Callable[..., Any]:
        attrs = [getattr(obj, name) for obj in self._items if obj is not None]
        if attrs and callable(attrs[0]):

            def wrapper(*args, **kwargs):
                return [func(*args, **kwargs) for func in attrs]

            return wrapper
        return SubplotGroup(attrs)

    def __getitem__(self, key: Any) -> "SubplotGroup":
        items = [item[key] for item in self._items if item is not None]
        return SubplotGroup(items)


# --- Core Engine ---
class GridPlotter:
    """
    A flexible grid-based plotter for images, scatter, lines, and heatmaps.

    Supports vectorized axis access (plotter[0, :].set_xlabel(...)), automatic
    data type detection, pixel-perfect image rendering, and grid reshaping.

    Future Enhancement - Nested Grids:
        To support sub-grids (a GridPlotter inside a cell of another GridPlotter),
        refactor to use matplotlib's GridSpecFromSubplotSpec:

        1. Refactor __init__ to optionally skip figure creation (accept parent_ax)
        2. Add method: add_subplot_grid(row, col, n_rows, n_cols) -> GridPlotter
        3. Child uses GridSpecFromSubplotSpec attached to parent's cell
        4. Child shares parent's _fig but has its own _gs

        This preserves full interactivity (unlike render-to-image approach).
    """

    def __init__(
        self,
        n_rows=None,
        n_cols=None,
        cell_ratio=None,
        fig_ratio=None,
        config=None,
    ) -> None:
        self.config = config or PlotterConfig()
        self._n_data_rows = n_rows or self.config.n_rows
        self._n_data_cols = n_cols or self.config.n_cols
        self._cell_ratio = cell_ratio or self.config.cell_ratio
        self._fig_ratio = fig_ratio or self.config.fig_ratio

        self._init_storage()
        self._fig = plt.figure(dpi=self.config.dpi)
        self.config._on_change_callback = self._set_refresh_flag
        self._refresh()

    def _init_storage(self) -> None:
        # Create grid of CellData objects
        self._cells = np.array(
            [
                [CellData() for _ in range(self._n_data_cols)]
                for _ in range(self._n_data_rows)
            ],
            dtype=object,
        )
        self._margin_axs, self._padding_axs = [], []
        self._needs_refresh = True
        self._image_pixel_dims = None  # (max_h, max_w) for pixel-perfect mode

    def __getitem__(self, key: Any) -> AxesProxy:
        self._refresh()
        selection = self._cells[key]
        if isinstance(selection, np.ndarray):
            # Extract axes from CellData objects
            axes = [cell.ax for cell in selection.flatten()]
            return cast("AxesProxy", SubplotGroup(axes))
        # Single cell
        if selection.ax is None:
            r, c = key if isinstance(key, tuple) else (key, 0)
            self.add_subplot(r, c)
            selection = self._cells[key]
        return cast("AxesProxy", selection.ax)

    def _set_refresh_flag(self) -> None:
        self._needs_refresh = True

    def _refresh(self) -> None:
        if not self._needs_refresh:
            return
        fw, fh = self._calculate_figure_size()
        self._fig.set_size_inches(fw, fh)
        for ax in self._margin_axs + self._padding_axs:
            if ax:
                ax.remove()
        self._margin_axs, self._padding_axs = [], []

        nr, nc = 2 * self._n_data_rows + 1, 2 * self._n_data_cols + 1
        self._gs = self._fig.add_gridspec(nr, nc, wspace=0, hspace=0)
        h_ratios, w_ratios = (
            self._calculate_ratios("h"),
            self._calculate_ratios("w"),
        )
        self._gs.set_height_ratios(h_ratios)
        self._gs.set_width_ratios(w_ratios)

        def add_v(r, c, store) -> None:
            if h_ratios[r] > 0 and w_ratios[c] > 0:
                ax = self._fig.add_subplot(self._gs[r, c])
                ax.axis("off")
                store.append(ax)

        for c in range(nc):
            (add_v(0, c, self._margin_axs), add_v(nr - 1, c, self._margin_axs))
        for r in range(1, nr - 1):
            (add_v(r, 0, self._margin_axs), add_v(r, nc - 1, self._margin_axs))
        for ri in range(self._n_data_rows - 1):
            for c in range(1, nc - 1):
                add_v(2 + ri * 2, c, self._padding_axs)
        for ci in range(self._n_data_cols - 1):
            for r in range(1, nr - 1, 2):
                add_v(r, 2 + ci * 2, self._padding_axs)

        for r in range(self._n_data_rows):
            for c in range(self._n_data_cols):
                if self._cells[r, c].ax is not None:
                    self._cells[r, c].ax.set_subplotspec(
                        self._gs[1 + r * 2, 1 + c * 2]
                    )
        self._needs_refresh = False

    def _scan_content_scales(self, data_list):
        """Scans data to find max dimensions and handle figure fitting."""
        max_h, max_w = 0, 0
        for item in data_list:
            # Check for PIL Image first (has .size as a tuple)
            if hasattr(item, "size") and isinstance(item.size, tuple):
                w, h = item.size
            # Then check for NumPy Array (has .shape)
            elif isinstance(item, np.ndarray) and item.ndim >= 2:
                h, w = item.shape[:2]
            else:
                continue

            max_h, max_w = max(max_h, h), max(max_w, w)

        if self.config.fit_fig_to_content and max_h > 0:
            self._cell_ratio = max_w / max_h
            self._set_refresh_flag()
            self._refresh()

        return max_h, max_w

    def _get_img_extent(self, img_h, img_w, max_h, max_w):
        """Calculates normalized [left, right, bottom, top] for alignment."""
        cell_aspect = self._cell_ratio  # Current width/height of the Axis box

        # 1. Calculate the 'Master Aspect' (the ratio of the largest image)
        master_aspect = max_w / max_h

        # 2. Determine how the 'Master Box' fits into the 'Cell Box'
        if master_aspect > cell_aspect:
            master_w_in_cell = 1.0
            master_h_in_cell = cell_aspect / master_aspect
        else:
            master_h_in_cell = 1.0
            master_w_in_cell = master_aspect / cell_aspect

        # 3. Apply Global Scaling
        if self.config.keep_global_scale:
            w_factor = (img_w / max_w) * master_w_in_cell
            h_factor = (img_h / max_h) * master_h_in_cell
        else:
            img_aspect = img_w / img_h
            if img_aspect > cell_aspect:
                w_factor = 1.0
                h_factor = cell_aspect / img_aspect
            else:
                h_factor = 1.0
                w_factor = img_aspect / cell_aspect

        # 4. Alignment math (using the Alignment enum values)
        va, ha = self.config.img_alignment.value
        left = ha * (1.0 - w_factor)
        top = 1.0 - (va * (1.0 - h_factor))

        return [left, left + w_factor, top - h_factor, top]

    def _get_pixel_extent(self, img_h, img_w, max_h, max_w):
        """Calculates pixel-based extent [left, right, bottom, top] for alignment."""
        va, ha = self.config.img_alignment.value

        if self.config.keep_global_scale:
            # Image keeps its actual pixel size, aligned within the cell
            left = ha * (max_w - img_w)
            top = va * (max_h - img_h)
            return [left, left + img_w, top + img_h, top]
        # Scale image to fill the cell while maintaining aspect
        img_aspect = img_w / img_h
        cell_aspect = max_w / max_h
        if img_aspect > cell_aspect:
            # Image is wider: fit to width
            scaled_w = max_w
            scaled_h = max_w / img_aspect
        else:
            # Image is taller: fit to height
            scaled_h = max_h
            scaled_w = max_h * img_aspect
        left = ha * (max_w - scaled_w)
        top = va * (max_h - scaled_h)
        return [left, left + scaled_w, top + scaled_h, top]

    # def add_data(self, data_list: list, shape: tuple[int, int] = None,
    #              titles: list = None, force_type: str = None, **kwargs):
    #     if shape:
    #         self._n_data_rows, self._n_data_cols = shape
    #         self._init_storage()
    #         self._refresh()

    #     max_h, max_w = self._scan_content_scales(data_list)

    #     for i, data in enumerate(data_list):
    #         if i >= self._n_data_rows * self._n_data_cols:
    #             break
    #         r, c = divmod(i, self._n_data_cols)
    #         ax = self.add_subplot(r, c)
    #         cell = self._cells[r, c]

    #         if titles and i < len(titles):
    #             ax.set_title(titles[i], fontsize=self.config.title_size)

    #         # --- NORMALIZE DATA ---
    #         if hasattr(data, 'size') and isinstance(data.size, tuple):
    #             plot_data = np.array(data)
    #         else:
    #             plot_data = data

    #         # --- PLOT LOGIC HIERARCHY ---
    #         is_scatter_shape = (np.ndim(plot_data) == 2 and np.shape(plot_data)[1] == 2)
    #         is_square = (np.ndim(plot_data) == 2 and np.shape(plot_data)[0] == np.shape(plot_data)[1])

    #         if (is_scatter_shape and not is_square) or force_type == 'scatter':
    #             pts = np.array(plot_data)
    #             ax.axis('on')
    #             cell.artist = ax.scatter(pts[:, 0], pts[:, 1], **kwargs)
    #             cell.raw_data = pts
    #             cell.cell_type = CellType.SCATTER

    #         elif isinstance(plot_data, (list, np.ndarray)) and len(plot_data) == 2 and np.ndim(plot_data[0]) == 1:
    #             ax.axis('on')
    #             cell.artist = ax.show(plot_data[0], plot_data[1], **kwargs)[0]
    #             cell.raw_data = np.array(plot_data)
    #             cell.cell_type = CellType.LINE

    #         elif isinstance(plot_data, np.ndarray) and plot_data.ndim >= 2:
    #             h, w = plot_data.shape[:2]
    #             ext = self._get_img_extent(h, w, max_h, max_w)
    #             cell.artist = ax.imshow(plot_data, extent=ext, aspect='auto', **kwargs)
    #             cell.raw_data = plot_data
    #             cell.cell_type = CellType.IMAGE
    #             ax.set_xlim(0, 1)
    #             ax.set_ylim(0, 1)

    #     return self

    def add_subplot(
        self, row: int, col: int, data: np.ndarray | None = None, **kwargs
    ):
        self._refresh()
        cell = self._cells[row, col]
        if cell.ax is None:
            cell.ax = self._fig.add_subplot(self._gs[1 + row * 2, 1 + col * 2])
        cell.ax.axis("off")
        if data is not None:
            cell.raw_data = data
            h, w = data.shape[:2]
            ext = self._get_img_extent(h, w, h, w)
            cell.artist = cell.ax.imshow(
                data, extent=ext, aspect="auto", **kwargs
            )
            cell.ax.set_xlim(0, 1)
            cell.ax.set_ylim(0, 1)
        return cell.ax

    def transpose(self):
        self._n_data_rows, self._n_data_cols = (
            self._n_data_cols,
            self._n_data_rows,
        )
        self._cells = self._cells.T
        if self._fig_ratio:
            self._fig_ratio = 1.0 / self._fig_ratio
        self._cell_ratio = 1.0 / self._cell_ratio
        self._set_refresh_flag()
        self._refresh()
        return self

    def reshape(self, n_rows: int, n_cols: int):
        """Reshape the grid to new dimensions. New size must be >= current size."""
        old_size = self._n_data_rows * self._n_data_cols
        new_size = n_rows * n_cols

        if new_size < old_size:
            msg = (
                f"New shape ({n_rows}x{n_cols}={new_size}) must be >= "
                f"current ({self._n_data_rows}x{self._n_data_cols}={old_size})"
            )
            raise ValueError(
                msg,
            )

        # Flatten existing cells in row-major order
        flat_cells = self._cells.flatten().tolist()

        # Create new grid with empty CellData, then fill with existing
        new_cells = np.array(
            [[CellData() for _ in range(n_cols)] for _ in range(n_rows)],
            dtype=object,
        )

        for i, cell in enumerate(flat_cells):
            if i >= new_size:
                break
            r, c = divmod(i, n_cols)
            new_cells[r, c] = cell

        # Update state
        self._n_data_rows, self._n_data_cols = n_rows, n_cols
        self._cells = new_cells

        self._set_refresh_flag()
        self._refresh()
        return self

    def _apply_theme(self) -> None:
        cfg = self.config
        group = self[:, :]
        group.tick_params(axis="both", which="major", labelsize=cfg.tick_size)
        for edge in ["top", "right", "bottom", "left"]:
            group.spines[edge].set_color(cfg.spine_color)
            group.spines[edge].set_linewidth(cfg.spine_width)
        if cfg.grid_on:
            group.grid(True, linestyle="--", alpha=0.5)

    def show(
        self,
        highlight_margin=False,
        highlight_padding=False,
        show_grid_numbers=False,
    ) -> None:
        self._refresh()
        self._apply_theme()
        for ax in self._margin_axs:
            ax.patch.set_visible(highlight_margin)
            if highlight_margin:
                ax.set_facecolor("pink")
        for ax in self._padding_axs:
            ax.patch.set_visible(highlight_padding)
            if highlight_padding:
                ax.set_facecolor("lightblue")

        if show_grid_numbers:
            for r in range(self._n_data_rows):
                for c in range(self._n_data_cols):
                    cell = self._cells[r, c]
                    ax = (
                        cell.ax
                        if cell.ax is not None
                        else self.add_subplot(r, c)
                    )
                    ax.text(
                        0.5,
                        0.5,
                        f"({r},{c})",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color="red",
                        weight="bold",
                        bbox={
                            "facecolor": "white",
                            "alpha": 0.8,
                            "boxstyle": "round",
                        },
                    )

        self._fig.subplots_adjust(top=1, left=0, right=1, bottom=0)
        plt.show()

    def _calculate_figure_size(self):
        m = self.config.margin
        rs, cs = self.config.row_space, self.config.col_space

        # NOTE: Pixel-perfect mode commented out for testing.
        # Uncomment to enable 1:1 pixel rendering (figure size varies with image resolution)
        # if self._image_pixel_dims:
        #     # Pixel-perfect: base cell size from image dimensions
        #     max_h, max_w = self._image_pixel_dims
        #     cell_h = max_h / self.config.dpi
        #     cell_w = max_w / self.config.dpi
        # else:

        # Always use fig_height as base (cell_ratio derived from images in _scan_content_scales)
        h_units = m[0] + self._n_data_rows + (self._n_data_rows - 1) * rs + m[2]
        cell_h = self.config.fig_height / h_units
        cell_w = cell_h * self._cell_ratio

        # Common spacing logic
        data_w = self._n_data_cols * cell_w
        data_h = self._n_data_rows * cell_h
        pad_w = (self._n_data_cols - 1) * cs * cell_w
        pad_h = (self._n_data_rows - 1) * rs * cell_h
        margin_w = (m[1] + m[3]) * cell_w
        margin_h = (m[0] + m[2]) * cell_h

        return data_w + pad_w + margin_w, data_h + pad_h + margin_h

    def _calculate_ratios(self, axis):
        m, s, n = (
            self.config.margin,
            (self.config.row_space if axis == "h" else self.config.col_space),
            (self._n_data_rows if axis == "h" else self._n_data_cols),
        )
        start, end = (m[0], m[2]) if axis == "h" else (m[3], m[1])
        ratios = [start]
        for i in range(n):
            ratios.append(1.0)
            if i < n - 1:
                ratios.append(s)
        ratios.append(end)
        return ratios

    def suptitle(self, t, font_size=None, y=0.98, **kwargs) -> None:
        fs = font_size or (self.config.title_size)
        self._fig.suptitle(t, fontsize=fs, y=y, **kwargs)

    def save(self, filepath, **kwargs) -> None:
        self._refresh()
        self._apply_theme()
        save_params = {
            "dpi": self.config.dpi,
            "bbox_inches": None,
            "pad_inches": 0,
        }
        save_params.update(kwargs)
        self._fig.savefig(filepath, **save_params)

    def add_numeric_data(
        self,
        data_list: list,
        shape: tuple[int, int] | None = None,
        titles: list | None = None,
        **kwargs,
    ):
        """Handles Scatter (Nx2), Lines ([x,y]), and Heatmaps (MxN). Supports None."""
        if shape:
            self._n_data_rows, self._n_data_cols = shape
            self._init_storage()
            self._refresh()

        for i, data in enumerate(data_list):
            if i >= self._n_data_rows * self._n_data_cols:
                break
            if data is None:
                continue

            r, c = divmod(i, self._n_data_cols)
            ax = self.add_subplot(r, c)
            cell = self._cells[r, c]

            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=self.config.title_size)

            is_scatter_shape = np.ndim(data) == 2 and np.shape(data)[1] == 2
            is_square = (
                np.ndim(data) == 2 and np.shape(data)[0] == np.shape(data)[1]
            )

            if is_scatter_shape and not is_square:
                ax.axis("on")
                cell.artist = ax.scatter(data[:, 0], data[:, 1], **kwargs)
                cell.raw_data = np.array(data)
                cell.cell_type = CellType.SCATTER

            elif (
                isinstance(data, (list, np.ndarray))
                and len(data) == 2
                and np.ndim(data[0]) == 1
            ):
                ax.axis("on")
                cell.artist = ax.show(data[0], data[1], **kwargs)[0]
                cell.raw_data = np.array(data)
                cell.cell_type = CellType.LINE

            elif isinstance(data, np.ndarray) and data.ndim == 2:
                h, w = data.shape[:2]
                ext = self._get_img_extent(h, w, h, w)
                cell.artist = ax.imshow(
                    data, extent=ext, aspect="auto", **kwargs
                )
                cell.raw_data = data
                cell.cell_type = CellType.HEATMAP
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)

        return self

    def add_image_data(
        self,
        data_list: list | None = None,
        data_folder: str | None = None,
        shape: tuple[int, int] | None = None,
        titles: list | None = None,
        auto_title: bool = False,
        **kwargs,
    ):
        """Pixel-perfect image rendering. Supports data_folder path and None entries."""
        if shape:
            self._n_data_rows, self._n_data_cols = shape
            self._init_storage()
            self._refresh()

        assert data_list is not None or data_folder is not None, (
            "Provide either data_list or data_folder."
        )

        def natural_sort_key(s):
            return [int(text) if text.isdigit() else text.lower() 
                    for text in re.split('([0-9]+)', str(s))]
        # 1. Handle folder loading
        if data_folder:
            from pathlib import Path
            from PIL import Image

            path = Path(data_folder)
            extensions = (".png", ".jpg", ".jpeg", ".bmp", ".gif")
            img_files = sorted(
                [f for f in path.iterdir() if f.suffix.lower() in extensions],
                key=natural_sort_key
            )

            data_list = []
            # Only populate folder_titles if auto_title is True
            folder_titles = [] if auto_title else None 
            
            for f in img_files:
                try:
                    img = Image.open(f)
                    img.load()
                    data_list.append(img)
                    if auto_title:
                        folder_titles.append(f.stem)
                except Exception:
                    pass
            
            titles = folder_titles
        # 2. Global Scale Scan (Filter out None for scanning)
        valid_data = [d for d in data_list if d is not None]
        max_h, max_w = self._scan_content_scales(valid_data)

        # 3. Enable pixel-perfect mode (refresh happens lazily in add_subplot)
        if max_h > 0 and max_w > 0:
            self._image_pixel_dims = (max_h, max_w)
            self._set_refresh_flag()

        # 4. Plotting Loop with pixel-perfect rendering
        for i, data in enumerate(data_list):
            if i >= self._n_data_rows * self._n_data_cols:
                break
            if data is None:
                continue

            r, c = divmod(i, self._n_data_cols)
            ax = self.add_subplot(r, c)
            cell = self._cells[r, c]

            if titles and i < len(titles):
                ax.set_title(titles[i], fontsize=self.config.title_size)

            # Convert PIL to Numpy
            plot_data = np.array(data) if hasattr(data, "size") else data

            if isinstance(plot_data, np.ndarray) and plot_data.ndim >= 2:
                h, w = plot_data.shape[:2]
                ext = self._get_pixel_extent(h, w, max_h, max_w)
                # Pixel-perfect: use actual pixel coordinates + aspect='equal'
                ax.set_xlim(0, max_w)
                ax.set_ylim(max_h, 0)  # Y inverted for image coordinates
                ax.set_anchor("NW")  # Anchor to top-left like images.py
                cell.artist = ax.imshow(
                    plot_data,
                    extent=ext,
                    aspect="equal",
                    interpolation="antialiased",
                    **kwargs,
                )
                cell.raw_data = plot_data
                cell.cell_type = CellType.IMAGE
        return self

    def add_global_colorbar(self, location: str = "right"):
        """
        Add a shared colorbar for all heatmaps, normalizing them to the same scale.

        Scans all cells with CellType.HEATMAP, computes global min/max from raw_data,
        applies set_clim() to normalize all heatmaps, then adds a single colorbar.

        Args:
            location: Where to place colorbar - 'right', 'left', 'top', 'bottom'
        """
        # 1. Find all heatmap cells
        heatmap_cells = [
            cell
            for cell in self._cells.flatten()
            if cell.cell_type == CellType.HEATMAP and cell.raw_data is not None
        ]

        if not heatmap_cells:
            return self

        # 2. Compute global min/max
        global_min = min(cell.raw_data.min() for cell in heatmap_cells)
        global_max = max(cell.raw_data.max() for cell in heatmap_cells)

        # 3. Normalize all heatmaps
        for cell in heatmap_cells:
            cell.artist.set_clim(global_min, global_max)

        # 4. Add colorbar using one of the heatmap artists
        # Use the margin area for the colorbar
        if location == "right":
            cbar_ax = self._fig.add_axes([1, 0.15, 0.01, 0.7])
        elif location == "left":
            cbar_ax = self._fig.add_axes([0.01, 0.15, 0.01, 0.7])
        elif location == "top":
            cbar_ax = self._fig.add_axes([0.15, 1, 0.7, 0.01])
        elif location == "bottom":
            cbar_ax = self._fig.add_axes([0.15, 0.01, 0.7, 0.01])
        else:
            msg = f"Invalid location: {location}. Use 'right', 'left', 'top', 'bottom'"
            raise ValueError(msg)

        orientation = (
            "horizontal" if location in {"top", "bottom"} else "vertical"
        )
        self._fig.colorbar(
            heatmap_cells[0].artist, cax=cbar_ax, orientation=orientation
        )

        return self

    def add_2D_image_data(
        self,
        data_2d: list[list],
        titles_2d: list[list[str]] | None = None,
        **kwargs,
    ):
        """Like add_image_data, but accepts 2D array where data_2d[i][j] maps to axes[i,j]."""
        n_rows = len(data_2d)
        n_cols = max(len(row) for row in data_2d) if data_2d else 0

        # Flatten row-major, padding ragged rows with None
        flat_data = [
            data_2d[i][j] if j < len(data_2d[i]) else None
            for i in range(n_rows)
            for j in range(n_cols)
        ]

        # Flatten titles similarly
        flat_titles = None
        if titles_2d:
            flat_titles = [
                titles_2d[i][j]
                if i < len(titles_2d) and j < len(titles_2d[i])
                else None
                for i in range(n_rows)
                for j in range(n_cols)
            ]

        return self.add_image_data(
            data_list=flat_data,
            shape=(n_rows, n_cols),
            titles=flat_titles,
            **kwargs,
        )



    def add_2D_numeric_data(
        self,
        data_2d: list[list],
        titles_2d: list[list[str]] | None = None,
        **kwargs,
    ):
        """Like add_numeric_data, but accepts 2D array where data_2d[i][j] maps to axes[i,j]."""
        n_rows = len(data_2d)
        n_cols = max(len(row) for row in data_2d) if data_2d else 0

        # Flatten row-major, padding ragged rows with None
        flat_data = [
            data_2d[i][j] if j < len(data_2d[i]) else None
            for i in range(n_rows)
            for j in range(n_cols)
        ]

        # Flatten titles similarly
        flat_titles = None
        if titles_2d:
            flat_titles = [
                titles_2d[i][j]
                if i < len(titles_2d) and j < len(titles_2d[i])
                else None
                for i in range(n_rows)
                for j in range(n_cols)
            ]

        return self.add_numeric_data(
            data_list=flat_data,
            shape=(n_rows, n_cols),
            titles=flat_titles,
            **kwargs,
        )

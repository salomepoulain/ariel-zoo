# from __future__ import annotations

# from collections.abc import Iterable
# from enum import Enum

# import numpy as np
# import pandas as pd
# from bokeh.layouts import column, gridplot
# from bokeh.models import (
#     ColumnDataSource,
#     CustomJS,
#     Div,
#     HoverTool,
#     LinearColorMapper,
#     Range1d,
# )
# from bokeh.plotting import figure, show
# from pydantic import BaseModel, ConfigDict
# from bokeh.palettes import Blues256

# import logging

# # Silence specific Bokeh validation warnings
# logging.getLogger("bokeh.core.validation.check").setLevel(logging.ERROR)

# __all__ = [
#     "BokehConfig",
#     "BokehGridPlotter",
# ]


# class CellType(Enum):
#     EMPTY = "empty"
#     HEATMAP = "heatmap"
#     SCATTER = "scatter"
#     LINE = "line"


# class BokehConfig(BaseModel):
#     model_config = ConfigDict(validate_assignment=True)
#     n_rows: int = 1
#     n_cols: int = 1
#     plot_width: int = 400
#     plot_height: int = 400
#     global_axis: bool = True
#     hover_enabled: bool = True
#     sidebar_enabled: bool = True
#     hover_img_height: int = 64
#     default_color: str = "#1f77b4"
#     selection_color: str = "#ff7f0e"
#     nonselection_alpha: float = 0.2
#     title_font_size: str = "13pt"


# class CellData:
#     def __init__(self) -> None:
#         self.df = pd.DataFrame()
#         self.raw_heatmap = None
#         self.heatmap_ids = None
#         self.heatmap_images = None
#         self.title = ""
#         self.cell_type = CellType.EMPTY
#         self.style_overrides = {"color": None}


# class SubplotGroup:
#     def __init__(self, cells: Iterable[CellData], default_color: str) -> None:
#         self._cells = [c for c in cells if c is not None]
#         self._default_color = default_color

#     def set_color(self, color: str, global_ids: np.ndarray | list = None):
#         """Vectorized color update for specific global IDs across selected cells."""
#         for cell in self._cells:
#             if cell.df.empty:
#                 continue
#             if global_ids is None:
#                 cell.df["color"] = color
#             else:
#                 mask = cell.df["global_id"].isin(global_ids)
#                 cell.df.loc[mask, "color"] = color
#         return self


# class BokehGridPlotter:
#     def __init__(
#         self,
#         n_rows: int | None = None,
#         n_cols: int | None = None,
#         config: BokehConfig | None = None,
#     ) -> None:
#         self.config = config or BokehConfig()
#         self.n_rows = n_rows or self.config.n_rows
#         self.n_cols = n_cols or self.config.n_cols
#         self.thumb_cache = {}
#         # Single ID -> single image (for scatter)
#         self._v_get_html_img = np.vectorize(
#             lambda gid: f'<img src="{self.thumb_cache.get(gid, "")}" style="height:{self.config.hover_img_height}px;">'
#             if gid in self.thumb_cache
#             else "",
#             otypes=[str],
#         )
#         self._init_storage()

#     def transpose(self) -> BokehGridPlotter:
#         self.n_rows, self.n_cols = self.n_cols, self.n_rows
#         self._cells = self._cells.T
#         return self

#     def _get_html_img_pair(self, pair) -> str:
#         try:
#             if len(pair) == 2:
#                 id1, id2 = pair[0], pair[1]
#                 img1 = self.thumb_cache.get(id1, "")
#                 img2 = self.thumb_cache.get(id2, "")
#                 if img1 and img2:
#                     h = self.config.hover_img_height
#                     return f'<img src="{img1}" style="height:{h}px;"><img src="{img2}" style="height:{h}px;">'
#         except (TypeError, IndexError):
#             pass
#         return ""

#     def _v_get_html_img_pair(self, arr):
#         flat = arr.flatten()
#         result = np.array([self._get_html_img_pair(p) for p in flat], dtype=str)
#         return result.reshape(arr.shape)

#     def _init_storage(self) -> None:
#         self._cells = np.array(
#             [
#                 [CellData() for _ in range(self.n_cols)]
#                 for _ in range(self.n_rows)
#             ],
#             dtype=object,
#         )

#     def add_thumbnails(self, ids: Iterable, b64s: Iterable) -> None:
#         self.thumb_cache.update(dict(zip(ids, b64s, strict=False)))

#     def __getitem__(self, key) -> SubplotGroup:
#         selection = self._cells[key]
#         cells = (
#             selection.flatten()
#             if isinstance(selection, np.ndarray)
#             else [selection]
#         )
#         return SubplotGroup(cells, self.config.default_color)

#     def add_numeric_data(
#         self,
#         data_list: list[np.ndarray],
#         global_ids_list: list[list[int]] | None = None,
#         shape: tuple[int, int] | None = None,
#         titles: list[str] | None = None,
#     ):
#         if shape:
#             self.n_rows, self.n_cols = shape
#             self._init_storage()

#         for i, data in enumerate(data_list):
#             if i >= self.n_rows * self.n_cols or data is None:
#                 continue
#             r, c = divmod(i, self.n_cols)
#             cell = self._cells[r, c]
#             if titles and i < len(titles):
#                 cell.title = titles[i]

#             ids = global_ids_list[i] if global_ids_list is not None else None

#             if data.ndim == 2 and data.shape[1] == 2:
#                 cell.cell_type = CellType.SCATTER
#                 cell.df = pd.DataFrame(data, columns=["x", "y"])
#                 current_ids = ids if ids is not None else np.arange(len(data))
#                 cell.df["global_id"] = current_ids[: len(data)]
#                 cell.df["img_html"] = self._v_get_html_img(cell.df["global_id"])
#                 cell.df["color"] = self.config.default_color

#             elif data.ndim == 2:
#                 cell.cell_type = CellType.HEATMAP
#                 cell.raw_heatmap = data

#                 if ids is not None:
#                     ids_arr = np.asarray(ids)
#                     if ids_arr.shape == data.shape:
#                         cell.heatmap_ids = np.array(ids, dtype=object)
#                     elif ids_arr.shape == (*data.shape, 2):
#                         cell.heatmap_ids = np.empty(data.shape, dtype=object)
#                         for i_idx in range(data.shape[0]):
#                             for j_idx in range(data.shape[1]):
#                                 cell.heatmap_ids[i_idx, j_idx] = (
#                                     ids_arr[i_idx, j_idx, 0],
#                                     ids_arr[i_idx, j_idx, 1],
#                                 )

#                     if cell.heatmap_ids is not None:
#                         cell.heatmap_images = self._v_get_html_img_pair(
#                             cell.heatmap_ids
#                         )
#         return self

#     def add_2D_numeric_data(self, data_2d, global_ids_2d=None, titles_2d=None):
#         nr = len(data_2d)
#         nc = max(len(row) for row in data_2d)
#         flat_data = [
#             data_2d[i][j] if j < len(data_2d[i]) else None
#             for i in range(nr)
#             for j in range(nc)
#         ]
#         flat_ids = [
#             global_ids_2d[i][j]
#             if global_ids_2d and j < len(global_ids_2d[i])
#             else None
#             for i in range(nr)
#             for j in range(nc)
#         ]
#         flat_titles = [
#             titles_2d[i][j] if titles_2d and j < len(titles_2d[i]) else None
#             for i in range(nr)
#             for j in range(nc)
#         ]
#         return self.add_numeric_data(
#             flat_data,
#             flat_ids,
#             shape=(nr, nc),
#             titles=flat_titles,
#         )

#     def _compile_figure(self, cell: CellData, bounds: tuple | None):
#         """Compiles a single cell into a Bokeh figure with independent ranges."""
#         is_heatmap = cell.cell_type == CellType.HEATMAP
#         use_global = self.config.global_axis and not is_heatmap

#         kwargs = {
#             "title": cell.title,
#             "width": self.config.plot_width,
#             "height": self.config.plot_height,
#             "tools": "pan,wheel_zoom,box_select,lasso_select,reset,save",
#         }

#         # Independent Axis Logic: Create NEW Range1d objects for every figure
#         if use_global and bounds:
#             x_min, x_max, y_min, y_max = bounds
#             kwargs["x_range"] = Range1d(x_min, x_max)
#             kwargs["y_range"] = Range1d(y_min, y_max)

#         p = figure(**kwargs)

#         if is_heatmap:
#             h, w = cell.raw_heatmap.shape
#             mapper = LinearColorMapper(
#                 palette=Blues256,
#                 low=float(cell.raw_heatmap.max()),
#                 high=float(cell.raw_heatmap.min()),
#             )

#             xx, yy = np.meshgrid(np.arange(w), np.arange(h))
#             ds_data = {
#                 "x": xx.flatten() + 0.5,
#                 "y": yy.flatten() + 0.5,
#                 "value": cell.raw_heatmap.flatten(),
#                 "alpha": np.ones(h * w),
#             }

#             if cell.heatmap_ids is not None:
#                 ds_data["ids"] = [f"({int(p_val[0])}, {int(p_val[1])})" for p_val in cell.heatmap_ids.flatten()]
#                 ds_data["imgs"] = cell.heatmap_images.flatten()
#                 flat_ids = cell.heatmap_ids.flatten()
#                 ds_data["id1"] = [int(p_val[0]) for p_val in flat_ids]
#                 ds_data["id2"] = [int(p_val[1]) for p_val in flat_ids]

#             source = ColumnDataSource(data=ds_data)
#             p.rect(
#                 x="x", y="y", width=1, height=1, source=source,
#                 fill_color={"field": "value", "transform": mapper},
#                 fill_alpha="alpha", line_color=None,
#             )

#             tt = """
#                 <div style="padding:4px;">
#                     <div style="display:flex; gap:4px; align-items:center;">
#                         @imgs{safe}
#                     </div>
#                     <div style="margin-top:4px;">
#                         <b>IDs: @ids</b><br>
#                         <span style="color:#666;">Value: @value{0.000}</span>
#                     </div>
#                 </div>
#             """
#             if self.config.hover_enabled:
#                 p.add_tools(HoverTool(tooltips=tt))

#             p.x_range = Range1d(0, w)
#             p.y_range = Range1d(0, h)
#             return p, ("heatmap", source)

#         if cell.cell_type == CellType.EMPTY:
#             return p, None

#         src = ColumnDataSource(cell.df)
#         p.scatter(
#             "x", "y", source=src, color="color", size=8, fill_alpha=0.6,
#             selection_color=self.config.selection_color,
#             nonselection_alpha=self.config.nonselection_alpha,
#         )

#         if self.config.hover_enabled:
#             p.add_tools(HoverTool(tooltips='<div style="padding:0px;">@img_html{safe}<br><b>ID: @global_id</b></div>'))

#         return p, ("scatter", src)

#     def show(self, super_title: str | None = None) -> None:
#         """Render the layout to the notebook with independent zoom/pan performance."""
#         # 1. Calculate Global Bounds as raw numbers, not Range objects
#         all_dfs = [c.df for c in self._cells.flatten() if not c.df.empty]
#         bounds = None
#         if self.config.global_axis and all_dfs:
#             f = pd.concat(all_dfs)
#             x_min, x_max, y_min, y_max = f.x.min(), f.x.max(), f.y.min(), f.y.max()
#             xr = x_max - x_min if x_max != x_min else 1.0
#             yr = y_max - y_min if y_max != y_min else 1.0
#             bounds = (x_min - 0.1 * xr, x_max + 0.1 * xr,
#                       y_min - 0.1 * yr, y_max + 0.1 * yr)

#         # 2. Build Grid
#         grid_rows = []
#         scatter_sources = []
#         heatmap_sources = []
#         for r in range(self.n_rows):
#             row_figs = []
#             for c in range(self.n_cols):
#                 # Pass bounds tuple; _compile_figure creates independent Range1d objects
#                 fig, src_info = self._compile_figure(self._cells[r, c], bounds)
#                 row_figs.append(fig)
#                 if src_info:
#                     src_type, src = src_info
#                     if src_type == "scatter":
#                         scatter_sources.append(src)
#                     elif src_type == "heatmap":
#                         heatmap_sources.append(src)
#             grid_rows.append(row_figs)

#         # 3. Selection Sync between scatter plots
#         if len(scatter_sources) > 1:
#             for i in range(1, len(scatter_sources)):
#                 scatter_sources[0].selected.js_link("indices", scatter_sources[i].selected, "indices")
#                 scatter_sources[i].selected.js_link("indices", scatter_sources[0].selected, "indices")

#         # 4. Link scatter selection to heatmap highlighting
#         if scatter_sources and heatmap_sources:
#             scatter_to_heatmap_js = CustomJS(
#                 args={
#                     "scatter_src": scatter_sources[0],
#                     "heatmap_sources": heatmap_sources,
#                     "default_alpha": 1.0,
#                     "dim_alpha": 0.15,
#                 },
#                 code="""
#                     const sel_indices = scatter_src.selected.indices;
#                     const scatter_data = scatter_src.data;
#                     const selected_ids = new Set();
#                     for (const idx of sel_indices) {
#                         selected_ids.add(scatter_data['global_id'][idx]);
#                     }

#                     for (const hm_src of heatmap_sources) {
#                         const hm_data = hm_src.data;
#                         if (!hm_data['id1'] || !hm_data['id2']) continue;
#                         const new_alpha = [];
#                         for (let i = 0; i < hm_data['id1'].length; i++) {
#                             const id1 = hm_data['id1'][i];
#                             const id2 = hm_data['id2'][i];
#                             if (sel_indices.length === 0 || (selected_ids.has(id1) && selected_ids.has(id2))) {
#                                 new_alpha.push(default_alpha);
#                             } else {
#                                 new_alpha.push(dim_alpha);
#                             }
#                         }
#                         hm_data['alpha'] = new_alpha;
#                         hm_src.change.emit();
#                     }
#                 """,
#             )
#             scatter_sources[0].selected.js_on_change("indices", scatter_to_heatmap_js)

#         # 5. Bottom Gallery
#         total_plot_width = self.config.plot_width * self.n_cols
#         gallery = Div(
#             text="<b>Inspector</b><br>Lasso points to inspect...",
#             width=total_plot_width,
#             styles={
#                 "overflow-x": "auto", "white-space": "nowrap",
#                 "border-top": "1px solid #ccc", "padding": "10px",
#                 "width": f"{total_plot_width}px",
#             },
#         )

#         if self.config.sidebar_enabled and scatter_sources:
#             scatter_sources[0].selected.js_on_change(
#                 "indices",
#                 CustomJS(
#                     args={"s": scatter_sources[0], "g": gallery},
#                     code="""
#                         const idxs = s.selected.indices;
#                         if (idxs.length === 0) {
#                             g.text = "<b>Inspector</b><br>Lasso points to inspect...";
#                             return;
#                         }
#                         let h = `<div style='display: flex; gap: 15px;'>`;
#                         for (let i of idxs) {
#                             h += `<div style='flex: 0 0 auto; text-align: center;'>
#                                     ${s.data['img_html'][i]}<br>
#                                     <small>ID: ${s.data['global_id'][i]}</small>
#                                 </div>`;
#                         }
#                         h += `</div>`;
#                         g.text = `<b>Selected (${idxs.length})</b><hr>` + h;
#                     """,
#                 ),
#             )

#         plot_grid = gridplot(grid_rows)
#         final_layout = column(plot_grid, gallery) if self.config.sidebar_enabled else plot_grid

#         if super_title:
#             title_div = Div(text=f"<h1 style='text-align:center; font-family: sans-serif;'>{super_title}</h1>")
#             show(column(title_div, final_layout))
#         else:
#             show(final_layout)

from __future__ import annotations

from collections.abc import Iterable
from enum import Enum

import numpy as np
import pandas as pd
from bokeh.layouts import column, gridplot
from bokeh.models import (
    ColumnDataSource,
    CustomJS,
    Div,
    HoverTool,
    LinearColorMapper,
    Range1d,
)
from bokeh.palettes import Viridis256
from bokeh.plotting import figure, show
from pydantic import BaseModel, ConfigDict
from bokeh.palettes import linear_palette, Blues256

import logging

# Silence specific Bokeh validation warnings (like MISSING_RENDERERS)
logging.getLogger("bokeh.core.validation.check").setLevel(logging.ERROR)

__all__ = [
    "BokehConfig",
    "BokehGridPlotter",
]


class CellType(Enum):
    EMPTY = "empty"
    HEATMAP = "heatmap"
    SCATTER = "scatter"
    LINE = "line"


class BokehConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    n_rows: int = 1
    n_cols: int = 1
    plot_width: int = 400
    plot_height: int = 400
    global_axis: bool = True
    hover_enabled: bool = True
    sidebar_enabled: bool = True
    # sidebar_width: int = 250
    hover_img_height: int = 64
    default_color: str = "#1f77b4"
    default_size: float = 8.0
    default_alpha: float = 0.6
    selection_color: str = "#ff7f0e"
    nonselection_alpha: float = 0.2
    title_font_size: str = "13pt"


class CellData:
    def __init__(self) -> None:
        self.df = pd.DataFrame()
        self.raw_heatmap = None
        self.heatmap_ids = None
        self.heatmap_images = None
        self.title = ""
        self.cell_type = CellType.EMPTY
        self.style_overrides = {"color": None}


class SubplotGroup:
    def __init__(self, cells: Iterable[CellData], default_color: str) -> None:
        self._cells = [c for c in cells if c is not None]
        self._default_color = default_color

    def set_color(self, color: str, global_ids: np.ndarray | list = None):
        """Vectorized color update for specific global IDs across selected cells."""
        for cell in self._cells:
            if cell.df.empty:
                continue
            if global_ids is None:
                cell.df["color"] = color
            else:
                mask = cell.df["global_id"].isin(global_ids)
                cell.df.loc[mask, "color"] = color
        return self


class BokehGridPlotter:
    def __init__(
        self,
        n_rows: int | None = None,
        n_cols: int | None = None,
        config: BokehConfig | None = None,
    ) -> None:
        self.config = config or BokehConfig()
        self.n_rows = n_rows or self.config.n_rows
        self.n_cols = n_cols or self.config.n_cols
        self.thumb_cache = {}
        self.style_cache = {}  # {id: {"color": ..., "size": ..., "alpha": ...}}
        self.tooltip_cache = {}  # {id: "tooltip text"}
        # Single ID -> single image (for scatter)
        self._v_get_html_img = np.vectorize(
            lambda gid: f'<img src="{self.thumb_cache.get(gid, "")}" style="height:{self.config.hover_img_height}px;">'
            if gid in self.thumb_cache
            else "",
            otypes=[str],
        )
        self._init_storage()

    def transpose(self) -> BokehGridPlotter:
        """Transpose the subplot grid (rows become columns)."""
        # Swap dimensions
        self.n_rows, self.n_cols = self.n_cols, self.n_rows
        # Transpose the numpy array holding the cell data
        self._cells = self._cells.T
        return self

    def _get_html_img_pair(self, pair) -> str:
        """Convert a pair (id1, id2) to two side-by-side img tags."""
        try:
            if len(pair) == 2:
                id1, id2 = pair[0], pair[1]
                img1 = self.thumb_cache.get(id1, "")
                img2 = self.thumb_cache.get(id2, "")
                if img1 and img2:
                    h = self.config.hover_img_height
                    return f'<img src="{img1}" style="height:{h}px;"><img src="{img2}" style="height:{h}px;">'
        except (TypeError, IndexError):
            pass
        return ""

    def _v_get_html_img_pair(self, arr):
        """Apply _get_html_img_pair over a 2D array of tuples."""
        flat = arr.flatten()
        result = np.array([self._get_html_img_pair(p) for p in flat], dtype=str)
        return result.reshape(arr.shape)

    def _init_storage(self) -> None:
        self._cells = np.array(
            [
                [CellData() for _ in range(self.n_cols)]
                for _ in range(self.n_rows)
            ],
            dtype=object,
        )

    def add_id_thumbnails(self, ids: Iterable, b64s: Iterable) -> None:
        self.thumb_cache.update(dict(zip(ids, b64s, strict=False)))

    def add_id_styling(
        self,
        ids: Iterable,
        colors: Iterable | None = None,
        sizes: Iterable | None = None,
        alphas: Iterable | None = None,
    ) -> None:
        """Update style_cache with per-ID styling (color, size, alpha)."""
        ids = list(ids)
        colors = list(colors) if colors is not None else None
        sizes = list(sizes) if sizes is not None else None
        alphas = list(alphas) if alphas is not None else None

        for i, id_ in enumerate(ids):
            style = self.style_cache.get(id_, {})
            if colors is not None:
                style["color"] = colors[i]
            if sizes is not None:
                style["size"] = sizes[i]
            if alphas is not None:
                style["alpha"] = alphas[i]
            self.style_cache[id_] = style

    def add_tooltip_text(self, ids: Iterable, text: Iterable) -> None:
        """Update tooltip_cache with per-ID tooltip text."""
        self.tooltip_cache.update(dict(zip(ids, text, strict=False)))

    def _apply_style_cache(self, df: pd.DataFrame) -> None:
        """Apply style_cache and tooltip_cache overrides to DataFrame columns."""
        if df.empty:
            return
        for idx, gid in df["global_id"].items():
            if gid in self.style_cache:
                style = self.style_cache[gid]
                if "color" in style:
                    df.at[idx, "color"] = style["color"]
                if "size" in style:
                    df.at[idx, "size"] = style["size"]
                if "alpha" in style:
                    df.at[idx, "alpha"] = style["alpha"]
            if gid in self.tooltip_cache:
                df.at[idx, "tooltip_text"] = self.tooltip_cache[gid]

    def __getitem__(self, key) -> SubplotGroup:
        selection = self._cells[key]
        cells = (
            selection.flatten()
            if isinstance(selection, np.ndarray)
            else [selection]
        )
        return SubplotGroup(cells, self.config.default_color)

    def add_numeric_data(
        self,
        data_list: list[np.ndarray],
        global_ids_list: list[int] | None = None,
        shape: tuple[int, int] | None = None,
        titles: list[str] | None = None,
    ):
        if shape:
            self.n_rows, self.n_cols = shape
            self._init_storage()

        for i, data in enumerate(data_list):
            if i >= self.n_rows * self.n_cols or data is None:
                continue
            r, c = divmod(i, self.n_cols)
            cell = self._cells[r, c]
            if titles and i < len(titles):
                cell.title = titles[i]

            ids = global_ids_list[i] if global_ids_list is not None else None

            if data.ndim == 2 and data.shape[1] == 2:
                cell.cell_type = CellType.SCATTER
                cell.df = pd.DataFrame(data, columns=["x", "y"])
                current_ids = ids if ids is not None else np.arange(len(data))
                cell.df["global_id"] = current_ids[: len(data)]
                cell.df["img_html"] = self._v_get_html_img(cell.df["global_id"])
                cell.df["color"] = self.config.default_color
                cell.df["size"] = self.config.default_size
                cell.df["alpha"] = self.config.default_alpha
                cell.df["tooltip_text"] = ""

            elif data.ndim == 2:
                cell.cell_type = CellType.HEATMAP
                cell.raw_heatmap = data

                if ids is not None:
                    ids_arr = np.asarray(ids)
                    # Handle both (n,n) of tuples and (n,n,2) shapes
                    if ids_arr.shape == data.shape:
                        # Shape (n, n) with tuple/array objects
                        cell.heatmap_ids = np.array(ids, dtype=object)
                    elif ids_arr.shape == (*data.shape, 2):
                        # Shape (n, n, 2) - convert to (n, n) of tuples
                        cell.heatmap_ids = np.empty(data.shape, dtype=object)
                        for i in range(data.shape[0]):
                            for j in range(data.shape[1]):
                                cell.heatmap_ids[i, j] = (
                                    ids_arr[i, j, 0],
                                    ids_arr[i, j, 1],
                                )

                    if cell.heatmap_ids is not None and cell.raw_heatmap.size <= 100:
                        # Only pre-generate images for small matrices (10x10 or less)
                        cell.heatmap_images = self._v_get_html_img_pair(
                            cell.heatmap_ids
                        )
        return self

    def add_2D_numeric_data(self, data_2d, global_ids_2d=None, titles_2d=None):
        """Maps a 2D list-of-lists directly to the subplot grid."""
        nr = len(data_2d)
        nc = max(len(row) for row in data_2d)
        flat_data = [
            data_2d[i][j] if j < len(data_2d[i]) else None
            for i in range(nr)
            for j in range(nc)
        ]
        flat_ids = [
            global_ids_2d[i][j]
            if global_ids_2d and j < len(global_ids_2d[i])
            else None
            for i in range(nr)
            for j in range(nc)
        ]
        flat_titles = [
            titles_2d[i][j] if titles_2d and j < len(titles_2d[i]) else None
            for i in range(nr)
            for j in range(nc)
        ]
        return self.add_numeric_data(
            flat_data,
            flat_ids,
            shape=(nr, nc),
            titles=flat_titles,
        )

    def _compile_figure(self, cell: CellData, x_r, y_r, heatmap_bounds=None):
        is_heatmap = cell.cell_type == CellType.HEATMAP
        use_global = self.config.global_axis and not is_heatmap

        kwargs = {
            "title": cell.title,
            "width": self.config.plot_width,
            "height": self.config.plot_height,
            "tools": "pan,wheel_zoom,box_select,lasso_select,reset,save",
        }

        if use_global and x_r:
            kwargs["x_range"], kwargs["y_range"] = x_r, y_r

        p = figure(**kwargs)

        if is_heatmap:
            h, w = cell.raw_heatmap.shape
            # Use global heatmap bounds if provided, otherwise use cell-specific bounds
            if heatmap_bounds is not None:
                hm_low, hm_high = heatmap_bounds
            else:
                hm_low = float(cell.raw_heatmap.min())
                hm_high = float(cell.raw_heatmap.max())
            mapper = LinearColorMapper(
                palette=Blues256,
                low=hm_high,  # swapped for Blues256 (darker = higher)
                high=hm_low,
            )

            # Flatten heatmap to rect glyph data (Bokeh standard for interactive heatmaps)
            xx, yy = np.meshgrid(np.arange(w), np.arange(h))
            ds_data = {
                "x": xx.flatten() + 0.5,  # center of each cell
                "y": yy.flatten() + 0.5,
                "value": cell.raw_heatmap.flatten(),
                "alpha": np.ones(h * w),  # for selection highlighting
            }

            # Add IDs and images if available
            has_images = False
            if cell.heatmap_ids is not None:
                flat_ids = cell.heatmap_ids.flatten()
                ds_data["ids"] = [f"({int(p[0])}, {int(p[1])})" for p in flat_ids]
                ds_data["id1"] = [int(p[0]) for p in flat_ids]
                ds_data["id2"] = [int(p[1]) for p in flat_ids]

                if cell.heatmap_images is not None:
                    ds_data["imgs"] = cell.heatmap_images.flatten()
                    has_images = True

            source = ColumnDataSource(data=ds_data)

            # Use rect glyphs - each cell is a rectangle
            p.rect(
                x="x",
                y="y",
                width=1,
                height=1,
                source=source,
                fill_color={"field": "value", "transform": mapper},
                fill_alpha="alpha",
                line_color=None,
            )

            # Tooltip - with or without images
            if has_images:
                tt = """
                    <div style="padding:4px;">
                        <div style="display:flex; gap:4px; align-items:center;">
                            @imgs{safe}
                        </div>
                        <div style="margin-top:4px;">
                            <b>IDs: @ids</b><br>
                            <span style="color:#666;">Value: @value{0.000}</span>
                        </div>
                    </div>
                """
            else:
                tt = """
                    <div style="padding:4px;">
                        <b>IDs: (@id1, @id2)</b><br>
                        <span style="color:#666;">Value: @value{0.000}</span>
                    </div>
                """

            if self.config.hover_enabled:
                p.add_tools(HoverTool(tooltips=tt))

            p.x_range = Range1d(0, w)
            p.y_range = Range1d(0, h)
            # Return source with 'heatmap' marker for linked selection
            return p, ("heatmap", source)

        if cell.cell_type == CellType.EMPTY:
            return p, None

        # Apply style_cache overrides to the DataFrame
        self._apply_style_cache(cell.df)

        src = ColumnDataSource(cell.df)
        p.scatter(
            "x",
            "y",
            source=src,
            color="color",
            size="size",
            fill_alpha="alpha",
            selection_color=self.config.selection_color,
            nonselection_alpha=self.config.nonselection_alpha,
        )

        if self.config.hover_enabled:
            p.add_tools(
                HoverTool(
                    tooltips='<div style="padding:0px;">@img_html{safe}<br>'
                             '<b>ID: @global_id</b><br>@tooltip_text</div>',
                ),
            )
        return p, ("scatter", src)

    def show(self, super_title: str | None = None) -> None:
        """Render the layout to the notebook."""
        # 1. Global Ranges
        all_dfs = [c.df for c in self._cells.flatten() if not c.df.empty]
        x_r = y_r = None
        if self.config.global_axis and all_dfs:
            f = pd.concat(all_dfs)
            x_min, x_max, y_min, y_max = (
                f.x.min(),
                f.x.max(),
                f.y.min(),
                f.y.max(),
            )
            xr = x_max - x_min if x_max != x_min else 1.0
            yr = y_max - y_min if y_max != y_min else 1.0
            x_r, y_r = (
                Range1d(x_min - 0.1 * xr, x_max + 0.1 * xr),
                Range1d(y_min - 0.1 * yr, y_max + 0.1 * yr),
            )

        # 2. Compute global heatmap color bounds
        heatmap_cells = [
            c for c in self._cells.flatten()
            if c.cell_type == CellType.HEATMAP and c.raw_heatmap is not None
        ]
        heatmap_bounds = None
        if heatmap_cells:
            all_vals = np.concatenate([c.raw_heatmap.flatten() for c in heatmap_cells])
            heatmap_bounds = (float(all_vals.min()), float(all_vals.max()))

        # 3. Build Grid
        grid_rows = []
        scatter_sources = []
        heatmap_sources = []
        for r in range(self.n_rows):
            row_figs = []
            for c in range(self.n_cols):
                fig, src_info = self._compile_figure(
                    self._cells[r, c], x_r, y_r, heatmap_bounds
                )
                row_figs.append(fig)
                if src_info:
                    src_type, src = src_info
                    if src_type == "scatter":
                        scatter_sources.append(src)
                    elif src_type == "heatmap":
                        heatmap_sources.append(src)
            grid_rows.append(row_figs)

        # 3. Selection Sync between scatter plots (by global_id, not row index)
        if len(scatter_sources) > 1:
            for i, src in enumerate(scatter_sources):
                other_sources = [s for j, s in enumerate(scatter_sources) if j != i]
                sync_callback = CustomJS(
                    args={"source": src, "targets": other_sources},
                    code="""
                        const sel_indices = source.selected.indices;
                        const selected_ids = new Set();
                        for (let idx of sel_indices) {
                            selected_ids.add(source.data['global_id'][idx]);
                        }
                        for (let target of targets) {
                            const target_indices = [];
                            for (let j = 0; j < target.data['global_id'].length; j++) {
                                if (selected_ids.has(target.data['global_id'][j])) {
                                    target_indices.push(j);
                                }
                            }
                            target.selected.indices = target_indices;
                        }
                    """,
                )
                src.selected.js_on_change("indices", sync_callback)

        # 4. Link scatter selection to heatmap highlighting
        if scatter_sources and heatmap_sources:
            # JS callback: when scatter selection changes, highlight heatmap cells
            scatter_to_heatmap_js = CustomJS(
                args={
                    "scatter_src": scatter_sources[0],
                    "heatmap_sources": heatmap_sources,
                    "default_alpha": 1.0,
                    "dim_alpha": 0.15,
                },
                code="""
                    const sel_indices = scatter_src.selected.indices;
                    const scatter_data = scatter_src.data;

                    // Get selected global_ids from scatter
                    const selected_ids = new Set();
                    for (const idx of sel_indices) {
                        selected_ids.add(scatter_data['global_id'][idx]);
                    }

                    // Update each heatmap source
                    for (const hm_src of heatmap_sources) {
                        const hm_data = hm_src.data;
                        if (!hm_data['id1'] || !hm_data['id2']) continue;

                        const new_alpha = [];
                        for (let i = 0; i < hm_data['id1'].length; i++) {
                            const id1 = hm_data['id1'][i];
                            const id2 = hm_data['id2'][i];
                            // Highlight if BOTH ids are in selection (or no selection)
                            if (sel_indices.length === 0 || (selected_ids.has(id1) && selected_ids.has(id2))) {
                                new_alpha.push(default_alpha);
                            } else {
                                new_alpha.push(dim_alpha);
                            }
                        }
                        hm_data['alpha'] = new_alpha;
                        hm_src.change.emit();
                    }
                """,
            )
            scatter_sources[0].selected.js_on_change(
                "indices", scatter_to_heatmap_js
            )

        # Combine for gallery (use scatter sources)
        sources = scatter_sources

        # 5. Bottom Gallery
        total_plot_width = self.config.plot_width * self.n_cols
        gallery = Div(
            text="<b>Inspector</b><br>Lasso points to inspect...",
            width=total_plot_width,
            # Use flexbox to make images stay in a horizontal row
            styles={
                "overflow-x": "auto",
                "white-space": "nowrap",
                "border-top": "1px solid #ccc",
                "padding": "10px",
                "width": f"{total_plot_width}px",
            },
        )

        if self.config.sidebar_enabled and scatter_sources:
            # Callback that aggregates selections from all scatter sources
            inspector_callback = CustomJS(
                args={"sources": scatter_sources, "g": gallery},
                code="""
                    // Collect unique global_ids and their data from all sources
                    const selected_data = new Map();  // global_id -> {img_html, tooltip}
                    for (let src of sources) {
                        const idxs = src.selected.indices;
                        for (let i of idxs) {
                            const gid = src.data['global_id'][i];
                            if (!selected_data.has(gid)) {
                                selected_data.set(gid, {
                                    img_html: src.data['img_html'][i],
                                    tooltip: src.data['tooltip_text'][i] || ''
                                });
                            }
                        }
                    }
                    if (selected_data.size === 0) {
                        g.text = "<b>Inspector</b><br>Lasso points to inspect...";
                        return;
                    }
                    let h = `<div style='display: flex; gap: 15px; flex-wrap: wrap;'>`;
                    for (let [gid, data] of selected_data) {
                        h += `<div style='flex: 0 0 auto; text-align: center;'>
                                ${data.img_html}<br>
                                <small>ID: ${gid}</small>
                                ${data.tooltip ? '<br><small>' + data.tooltip + '</small>' : ''}
                            </div>`;
                    }
                    h += `</div>`;
                    g.text = `<b>Selected (${selected_data.size})</b><hr>` + h;
                """,
            )
            for src in scatter_sources:
                src.selected.js_on_change("indices", inspector_callback)

        # # 4. Sidebar Gallery
        # gallery = Div(text="<b>Inspector</b><br>Lasso points to inspect...",
        #               width=self.config.sidebar_width,
        #               styles={"overflow-y": "auto", "height": f"{self.config.plot_height*self.n_rows}px"})

        # if self.config.sidebar_enabled and sources:
        #     sources[0].selected.js_on_change('indices', CustomJS(args=dict(s=sources[0], g=gallery), code="""
        #         const idxs = s.selected.indices;
        #         if (idxs.length === 0) {
        #             g.text = "<b>Inspector</b><br>Lasso points to inspect...";
        #             return;
        #         }
        #         let h = `<b>Selected (${idxs.length})</b><hr>`;
        #         for (let i of idxs) {
        #             h += `<div style='margin-bottom:10px;'>${s.data['img_html'][i]}<br>ID: ${s.data['global_id'][i]}</div>`;
        #         }
        #         g.text = h;
        #     """))

        plot_grid = gridplot(grid_rows)
        # final_layout = row(plot_grid, gallery) if self.config.sidebar_enabled else plot_grid
        final_layout = (
            column(plot_grid, gallery)
            if self.config.sidebar_enabled
            else plot_grid
        )

        if super_title:
            title_div = Div(
                text=f"<h1 style='text-align:center; font-family: sans-serif;'>{super_title}</h1>",
            )
            show(column(title_div, final_layout))
        else:
            show(final_layout)

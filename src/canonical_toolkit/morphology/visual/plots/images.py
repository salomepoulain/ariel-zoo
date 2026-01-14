from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Union, cast

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

__all__ = ["SnapshotGridPlotter"]

def _natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r"([0-9]+)", str(s))]

class SnapshotSubPlotStyle(BaseModel):
    title: str = ""
    sub_title: str = ""
    title_fontsize: int = 6
    sub_title_fontsize: int = 6
    title_fontweight: str = "bold"
    sub_title_fontweight: str = "normal"

    class Config:
        validate_assignment = True
        arbitrary_types_allowed = True

class SnapshotGridGlobalStyle(BaseModel):
    title: str = ""
    subtitle: str = ""
    dpi: int = 300
    figsize: tuple[int, int] | None = None
    wspace: float = 0.05
    hspace: float = 0.3
    # CSS: (Top, Right, Bottom, Left) in %
    padding: tuple[float, float, float, float] = (10.0, 0.0, 0.0, 0.0)

    class Config:
        validate_assignment = True

# Internal Proxy Logic

class _BatchStyleModifier:
    def __init__(self, items: list[SnapshotSubPlot]) -> None:
        object.__setattr__(self, "_items", items)

    def __setattr__(self, name: str, value: Any) -> None:
        for sp in self._items:
            setattr(sp.style, name, value)

    def __getattr__(self, name: str) -> Any:
        if not self._items: raise AttributeError("Empty selection.")
        return getattr(self._items[0].style, name)

# Core Components

class SnapshotSubPlot(BaseModel):
    snapshots: list[Image.Image] = Field(default_factory=list)
    style: SnapshotSubPlotStyle = Field(default_factory=SnapshotSubPlotStyle)

    class Config:
        arbitrary_types_allowed = True

    def render(self, scaling_factor: float = 1.0) -> np.ndarray:
        if not self.snapshots:
            return np.full((100, 100, 3), 255, dtype=np.uint8)
        
        gap_px = int(15 * scaling_factor)
        total_w = sum(img.width for img in self.snapshots) + (gap_px * (len(self.snapshots)-1))
        max_h = max(img.height for img in self.snapshots)
        
        canvas = Image.new("RGBA", (total_w, max_h), (255, 255, 255, 255))
        x_off = 0
        for img in self.snapshots:
            img_rgba = img.convert("RGBA")
            canvas.paste(img_rgba, (x_off, (max_h - img_rgba.height)//2), mask=img_rgba)
            x_off += img_rgba.width + gap_px
            
        return np.array(canvas.convert("RGB"))

class SnapshotGridPlotter:
    """A NumPy-backed grid plotter that supports recursive slicing and batch styling."""

    def __init__(
        self, 
        data_2d=None, 
        titles_2d=None, 
        configs_2d=None, 
        global_style=None
    ) -> None:
        self._global_style = global_style or SnapshotGridGlobalStyle()
        self._is_view = False 
        
        if data_2d:
            r, c = len(data_2d), max(len(row) for row in data_2d)
            grid = [[SnapshotSubPlot(snapshots=data_2d[i][j] if j < len(data_2d[i]) else []) 
                     for j in range(c)] for i in range(r)]
            self._grid = np.array(grid, dtype=object)
            
            # Initialization titles/configs
            for i in range(r):
                for j in range(c):
                    if titles_2d and i < len(titles_2d) and j < len(titles_2d[i]):
                        self._grid[i, j].style.title = titles_2d[i][j]
        else:
            self._grid = np.array([[SnapshotSubPlot()]], dtype=object)


    @classmethod
    def from_snapshot_folder(
        cls,
        snapshot_folder: Path | str,
        *,
        grid_shape: tuple[int, int] | None = None,
        snapshots_per_subplot: int = 5,
        snapshot_names_2d: list[list[list[str]]] | None = None,
        titles_2d: list[list[str]] | None = None,
        auto_title_from_snapshots: bool = True,
    ) -> SnapshotGridPlotter:
        """Factory method to build a grid from a directory of images."""
        from ..snapshots import snapshot_loader
        folder = Path(snapshot_folder)

        if snapshot_names_2d:
            # Map specific filenames to 2D grid structure
            all_names = {n for r in snapshot_names_2d for c in r for n in c}
            loaded = snapshot_loader(snapshot_folder=folder, snapshot_names=list(all_names))
            snapshot_map = dict(zip(all_names, loaded))
            data_2d = [
                [[snapshot_map[n] for n in cell if n in snapshot_map] for cell in row]
                for row in snapshot_names_2d
            ]
        else:
            # Distribute all images automatically
            if not grid_shape:
                raise ValueError("grid_shape required for auto-distribution.")
            all_snapshots = snapshot_loader(snapshot_folder=folder)
            all_snapshots.sort(key=lambda x: _natural_sort_key(getattr(x, "filename", str(x))))
            
            data_2d = []
            idx = 0
            for _ in range(grid_shape[0]):
                row = []
                for _ in range(grid_shape[1]):
                    cell = all_snapshots[idx : idx + snapshots_per_subplot]
                    row.append(cell)
                    idx += len(cell)
                data_2d.append(row)

        return cls(data_2d=data_2d, titles_2d=titles_2d)



    def __getitem__(self, key: Any) -> SnapshotGridPlotter:
        """Slicing returns a NEW SnapshotGridPlotter instance acting as a VIEW."""
        selection = self._grid[key]
        
        # Normalize to 2D
        if not isinstance(selection, np.ndarray):
            selection = np.array([[selection]], dtype=object)
        elif selection.ndim == 1:
            # Handle row vs col slicing to maintain 2D grid shape
            is_col_slice = isinstance(key, tuple) and isinstance(key[1], int)
            selection = selection[:, np.newaxis] if is_col_slice else selection[np.newaxis, :]
            
        view = SnapshotGridPlotter.__new__(SnapshotGridPlotter)
        view._grid = selection
        view._global_style = self._global_style
        view._is_view = True 
        return view

    @property
    def style(self) -> Union[SnapshotGridGlobalStyle, SnapshotSubPlotStyle]:
        """
        Smart Property: 
        - If plotter is whole: returns Global Settings.
        - If plotter is a slice: returns a Batch Proxy for subplots.
        """
        if self._is_view:
            return cast(SnapshotSubPlotStyle, _BatchStyleModifier(self._grid.flatten().tolist()))
        return self._global_style

    @property
    def shape(self) -> tuple[int, int]:
        """Cropped grid dimensions based on cells containing data."""
        rows, cols = self._grid.shape
        active = [(i, j) for i in range(rows) for j in range(cols) if self._grid[i, j].snapshots]
        if not active: return (0, 0)
        return (max(i for i, j in active) + 1, max(j for i, j in active) + 1)

    def plot(self) -> None:
        """Render and display the current grid view."""
        rows, cols = self._grid.shape
        rendered_images = []
        max_h, max_w = 0, 0
        
        for i in range(rows):
            row_imgs = []
            for j in range(cols):
                img = self._grid[i, j].render()
                row_imgs.append(img)
                max_h, max_w = max(max_h, img.shape[0]), max(max_w, img.shape[1])
            rendered_images.append(row_imgs)

        # Margins & Figure Setup
        p_top, p_right, p_bottom, p_left = self._global_style.padding
        dpi = self._global_style.dpi
        
        fig, axes = plt.subplots(
            rows, cols, figsize=((cols*max_w)/dpi, (rows*max_h)/dpi), dpi=dpi, squeeze=False
        )

        plt.subplots_adjust(
            left=p_left/100, right=1-(p_right/100), bottom=p_bottom/100, top=1-(p_top/100),
            wspace=self._global_style.wspace, hspace=self._global_style.hspace
        )

        for i in range(rows):
            for j in range(cols):
                ax, img = axes[i, j], rendered_images[i][j]
                h, w = img.shape[:2]
                ax.set_xlim(0, max_w); ax.set_ylim(max_h, 0); ax.set_anchor("NW")
                ax.imshow(img, aspect="equal", extent=(0, w, h, 0))
                ax.axis("off")
                
                subplot = self._grid[i, j]
                if subplot.style.title:
                    ax.text(w/2, -5, subplot.style.title, ha="center", va="bottom",
                            fontsize=subplot.style.title_fontsize, fontweight=subplot.style.title_fontweight,
                            transform=ax.transData)
        plt.show()



    def reshape(x, y, groupsize):
        pass

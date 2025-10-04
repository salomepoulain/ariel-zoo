import collections.abc

import collections.abc
from pathlib import Path
from IPython.display import clear_output

import ipyvuetify as v

# Third-party libraries
import ipywidgets as widgets
import numpy as np
from IPython.display import display
from rich.console import Console

# Local libraries
from ariel_experiments.gui_vis.visualize_tree import (
    visualize_tree_from_graph,
)

import ipyvuetify as v


class SimpleNavigator:
    """
    Navigate any nested dictionary structure and display values.
    Simplified version that doesn't filter - just shows everything.
    """

    def __init__(
        self,
        data_dict: dict,
        console: Console | None = None,
        gui: bool = True,
    ) -> None:
        self._dict = data_dict  # No filtering - use raw dict
        self._console = console or Console(
            force_terminal=False, force_interactive=False, legacy_windows=False
        )
        self._output_area = None
        self._viz_output_area = None

        self._path: list[str] = []
        self._index = 0
        self._current_page = 0
        self._total_pages = 0
        self._MAX_ITEMS = 10

        if gui:
            self.gui()

        self.display_current_level()

    def _print_status(self, message: str, clear: bool = True) -> None:
        try:
            from IPython.display import clear_output

            if self._output_area:
                with self._output_area:
                    if clear:
                        clear_output(wait=True)
                    self._console.print(message)
            else:
                self._console.print(message)
        except ImportError:
            self._console.print(message)

    def _get_current_level_data(self):
        """Get data at current path level."""
        current_data = self._dict
        for key in self._path:
            current_data = current_data.get(key, {})
        return current_data

    def _get_current_keys(self) -> list:
        """Get keys at current level."""
        data = self._get_current_level_data()
        if isinstance(data, collections.abc.Mapping):
            return list(data.keys())
        return []

    def _is_dict(self, value) -> bool:
        """Check if value is a dictionary (nested level)."""
        return isinstance(value, collections.abc.Mapping)

    def up(self) -> None:
        """Move up in current page."""
        current_keys = self._get_current_keys()
        if not current_keys:
            return

        total = len(current_keys)
        start = self._current_page * self._MAX_ITEMS
        end = min(start + self._MAX_ITEMS, total)
        page_len = max(0, end - start)

        if page_len == 0:
            return

        local = self._index - start
        local = (local - 1) % page_len
        self._index = start + local
        self.display_current_level()

    def down(self) -> None:
        """Move down in current page."""
        current_keys = self._get_current_keys()
        if not current_keys:
            return

        total = len(current_keys)
        start = self._current_page * self._MAX_ITEMS
        end = min(start + self._MAX_ITEMS, total)
        page_len = max(0, end - start)

        if page_len == 0:
            return

        local = self._index - start
        local = (local + 1) % page_len
        self._index = start + local
        self.display_current_level()

    def select_current(self) -> None:
        """Enter selected item if it's a dict, otherwise show value."""
        current_keys = self._get_current_keys()
        if not current_keys:
            return

        self._index = max(0, min(self._index, len(current_keys) - 1))
        selected_key = current_keys[self._index]
        value = self._get_current_level_data()[selected_key]

        if self._is_dict(value):
            # Enter nested dict
            self._path.append(selected_key)
            self._index = 0
            self._current_page = 0
            self.display_current_level()
        else:
            # Display value in viz area
            self._display_value(selected_key, value)

    def _display_value(self, key, value):
        """Display the selected value in the visualization area."""
        if not self._viz_output_area:
            return

        from IPython.display import clear_output
        from rich.console import Console
        import ipywidgets as widgets
        from IPython.display import display
        from networkx import DiGraph
        from ariel_experiments.gui_vis.visualize_tree import visualize_tree_from_graph

        with self._viz_output_area:
            clear_output(wait=True)
            self._console.print(
                f"[yellow]Type:[/yellow] {type(value).__name__}"
            )

            # Format value display based on type
            if isinstance(value, (list, tuple, set)):
                self._console.print(f"[yellow]Length:[/yellow] {len(value)}")
                self._console.print(f"[green]Value:[/green]")
                if len(value) <= 20:
                    self._console.print(value)
                else:
                    self._console.print(f"{list(value)}")

            if isinstance(value, DiGraph):
                visualize_tree_from_graph(value, title=self._path)

            else:
                self._console.print(f"[green]Value:[/green]")
                value_str = str(value)
                if len(value_str) > 500:
                    self._console.print(value_str[:500] + "... (truncated)")
                else:
                    self._console.print(value)

    def backward(self) -> None:
        """Go to previous page or up one level."""
        if self._current_page > 0:
            self._current_page -= 1
            current_keys = self._get_current_keys()
            self._index = (
                min(len(current_keys) - 1, self._current_page * self._MAX_ITEMS)
                if current_keys
                else 0
            )
            self.display_current_level()
            return

        # Go up one level
        if self._path:
            self._path.pop()
            self._index = 0
            self._current_page = 0
            self.display_current_level()

    def forward(self) -> None:
        """Go to next page."""
        current_keys = self._get_current_keys()
        total = len(current_keys)
        if total == 0:
            return

        total_pages = max(1, (total + self._MAX_ITEMS - 1) // self._MAX_ITEMS)
        if self._current_page < total_pages - 1:
            self._current_page += 1
            self._index = min(
                len(current_keys) - 1, self._current_page * self._MAX_ITEMS
            )
            self.display_current_level()

    def _get_path(self) -> str:
        """Get current path string."""
        return (
            "ROOT"
            if not self._path
            else "ROOT/" + "/".join(map(str, self._path))
        )

    def display_current_level(self) -> None:
        """Display current dictionary level."""
        current_keys = self._get_current_keys()
        if not current_keys:
            self._print_status(
                "[red]Empty dictionary at this level.[/red]", clear=True
            )
            return

        # Pagination
        total = len(current_keys)
        self._total_pages = max(
            1, (total + self._MAX_ITEMS - 1) // self._MAX_ITEMS
        )
        self._current_page = max(
            0, min(self._current_page, self._total_pages - 1)
        )
        self._index = max(0, min(self._index, total - 1))

        start = self._current_page * self._MAX_ITEMS
        end = min(start + self._MAX_ITEMS, total)
        display_slice = current_keys[start:end]

        # Format display
        current_data = self._get_current_level_data()
        lines = []

        for local_i, key in enumerate(display_slice):
            global_i = start + local_i
            is_selected = global_i == self._index
            value = current_data[key]

            # Format key
            key_display = f"{str(key):<20.20}"
            selected_key_display = f"{(str(key) + ' ●'):<20.20}"

            # Type indicator
            if self._is_dict(value):
                type_str = (
                    f"[dim][cyan]    (dict, {len(value)} keys)[/cyan][/dim]"
                )
            elif isinstance(value, (list, tuple, set)):
                type_str = f"[dim][yellow]    ({type(value).__name__}, {len(value)} items)[/yellow][/dim]"
            else:
                type_str = f"[dim][magenta]    ({type(value).__name__})[/magenta][/dim]"

            # Highlight selected
            if is_selected:
                line = (
                    f"[bold cyan]{selected_key_display}[/bold cyan]{type_str}"
                )
            else:
                line = f"{key_display}{type_str}"

            lines.append(f"[cyan dim]{global_i + 1:>4}   [/cyan dim]{line}")

        # Pad to fixed height
        displayed = len(display_slice)
        if displayed < self._MAX_ITEMS:
            for _ in range(displayed, self._MAX_ITEMS):
                lines.append("")

        # Build output
        path_str = self._get_path()
        full_output = (
            f"\n[bold cyan underline]{path_str}[/bold cyan underline]\n\n"
        )
        full_output += f"[dim]total [cyan]{total}:[/cyan][/dim]\n\n"
        full_output += "\n".join(lines)
        full_output += (
            f"\n\n[dim]page {self._current_page + 1}/{self._total_pages}[/dim]"
        )
        full_output += f"\n\n[dim]Press ⬤ to select / ◀ to go back[/dim]"

        self._print_status(full_output, clear=True)

    def gui(self) -> None:
        """Create GUI with directional pad."""
        v.theme.dark = False

        self._output_area = widgets.Output(
            layout=widgets.Layout(
                border="1px solid #ddd",
                background_color="#ffffff",
                height="330px",
                width="450px",
                overflow_y="auto",
                padding="0 0 0 20px",
            )
        )

        self._viz_output_area = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                height="250px",
                overflow_y="auto",
            )
        )

        button_size = 60
        btn_up = v.Btn(
            children=["▲"],
            color="cyan lighten-4",
            rounded=True,
            class_="ma-1",
            width=button_size,
            height=button_size,
            style_="color: #008080;",
        )
        btn_down = v.Btn(
            children=["▼"],
            color="cyan lighten-4",
            rounded=True,
            class_="ma-1",
            width=button_size,
            height=button_size,
            style_="color: #008080;",
        )
        btn_left = v.Btn(
            children=["◀"],
            color="cyan lighten-4",
            rounded=True,
            class_="ma-1",
            width=button_size,
            height=button_size,
            style_="color: #008080;",
        )
        btn_right = v.Btn(
            children=["▶"],
            color="cyan lighten-4",
            rounded=True,
            class_="ma-1",
            width=button_size,
            height=button_size,
            style_="color: #008080;",
        )
        btn_center = v.Btn(
            children=["🔘"],
            color="grey lighten-2",
            rounded=True,
            class_="ma-1",
            width=button_size,
            height=button_size,
        )

        btn_up.on_event("click", lambda w, e, d: self.up())
        btn_down.on_event("click", lambda w, e, d: self.down())
        btn_left.on_event("click", lambda w, e, d: self.backward())
        btn_right.on_event("click", lambda w, e, d: self.forward())
        btn_center.on_event("click", lambda w, e, d: self.select_current())

        pad = v.Container(
            class_="pa-6 d-flex flex-column align-center justify-center",
            style_="width: 250px; height: 250px;",
            children=[
                v.Row(
                    class_="justify-center",
                    style_="min-height:70px; align-items:center;",
                    children=[btn_up],
                ),
                v.Row(
                    class_="justify-center",
                    style_="min-height:70px; align-items:center;",
                    children=[btn_left, btn_center, btn_right],
                ),
                v.Row(
                    class_="justify-center",
                    style_="min-height:70px; align-items:center;",
                    children=[btn_down],
                ),
            ],
        )

        top = widgets.HBox(
            [
                pad,
                widgets.Box(
                    [self._output_area],
                    layout=widgets.Layout(margin="15px 0 0 30px"),
                ),
            ],
            layout=widgets.Layout(
                align_items="flex-start", justify_content="center"
            ),
        )

        layout = widgets.VBox(
            [top, self._viz_output_area],
            layout=widgets.Layout(
                align_items="stretch", justify_content="flex-start"
            ),
        )

        display(layout)

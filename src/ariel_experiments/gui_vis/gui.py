# ...existing code...
# Standard library
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
from ariel_experiments.gui_vis.view_mujoco import view

# Global constants
# SCRIPT_NAME = __file__.split("/")[-1][:-3]
SCRIPT_NAME = "hi"
CWD = Path.cwd()
DATA = Path(CWD / "__data__" / SCRIPT_NAME)
DATA.mkdir(exist_ok=True)
SEED = 42

# Global functions
console = Console()
RNG = np.random.default_rng(SEED)

from collections.abc import Callable

from ariel_experiments.characterize.canonical_toolkit.tests.old.toolkit import CanonicalToolKit as ctk

class IndividualVisualizer:
    """
    Navigate a nested index_dict and visualize selected individuals from a population.
    Clean, single implementation with consistent pagination and backward behavior.
    """

    def __init__(
        self,
        index_dict: dict,
        population: list,
        console: Console | None = None,
        gui: bool = True,
        visualize_fn: Callable | None = None,
    ) -> None:
        self._visualize_fn = visualize_fn
        self._population = population
        self._console = console or Console()
        self._output_area = None
        self._viz_output_area = None
        self._viewing_individuals = False
        self._visualized_individuals: list[int] = []
        self._pinned_individuals: list[
            int
        ] = []  # NEW: track pinned individuals

        self._dict = self._filter_nested_index_dict(index_dict)

        self._path: list[str] = []
        self._index = 0
        self._individual_index = 0

        self._current_page = 0
        self._total_pages = 0

        self._MAX_ITEMS = 10

        if gui:
            self.gui()

        self.display_current_menu()

        self._console = console or Console(
            force_terminal=False,
            force_interactive=False,
            legacy_windows=False,
        )

    def _print_status(self, message: str, clear: bool = True) -> None:
        try:

            if self._output_area:
                with self._output_area:
                    if clear:
                        clear_output(wait=True)
                    self._console.print(message)
            else:
                self._console.print(message)
        except ImportError:
            self._console.print(message)

    def _get_current_level_data(self) -> dict:
        current_data = self._dict
        for key in self._path:
            current_data = current_data.get(key, {})
        return current_data

    def _get_current_keys(self) -> list:
        data = self._get_current_level_data()
        return (
            list(data.keys())
            if isinstance(data, collections.abc.Mapping)
            else []
        )

    def _is_current_selection_a_menu(
        self,
        current_keys: list,
        index: int,
    ) -> bool:
        if not current_keys or index >= len(current_keys):
            return False
        selected_key = current_keys[index]
        return isinstance(
            self._get_current_level_data().get(selected_key),
            collections.abc.Mapping,
        )

    def _get_index_list(self) -> list:
        current_keys = self._get_current_keys()
        if not current_keys:
            return []
        # clamp index in case keys changed
        self._index = max(0, min(self._index, len(current_keys) - 1))
        selected_key = current_keys[self._index]
        value = self._get_current_level_data().get(selected_key)
        return list(value) if isinstance(value, (list, tuple, set)) else []

    def up(self) -> None:
        # Move up within the current page (menu or individuals) — do not wrap across whole list
        if self._viewing_individuals:
            indexes = self._get_index_list()
            if not indexes:
                return
            total = len(indexes)
            start = (getattr(self, "_current_page", 0)) * self._MAX_ITEMS
            end = min(start + self._MAX_ITEMS, total)
            page_len = max(0, end - start)
            if page_len == 0:
                return
            # local position within page, wrap inside page
            local = self._individual_index - start
            local = (local - 1) % page_len
            self._individual_index = start + local

            # Auto-visualize the new selection (removes unpinned visualizations)

            # self.display_individual_list()
            self._auto_visualize_current()
            return

        current_keys = self._get_current_keys()
        if not current_keys:
            self._print_status("[red]No items to navigate at this level.[/red]")
            return
        total = len(current_keys)
        start = (getattr(self, "_current_page", 0)) * self._MAX_ITEMS
        end = min(start + self._MAX_ITEMS, total)
        page_len = max(0, end - start)
        if page_len == 0:
            return
        local = self._index - start
        local = (local - 1) % page_len
        self._index = start + local
        self._individual_index = 0

        self.display_current_menu()

    def down(self) -> None:
        # Move down within the current page (menu or individuals) — do not wrap across whole list
        if self._viewing_individuals:
            indexes = self._get_index_list()
            if not indexes:
                return
            total = len(indexes)
            start = (getattr(self, "_current_page", 0)) * self._MAX_ITEMS
            end = min(start + self._MAX_ITEMS, total)
            page_len = max(0, end - start)
            if page_len == 0:
                return
            local = self._individual_index - start
            local = (local + 1) % page_len
            self._individual_index = start + local

            # Auto-visualize the new selection (removes unpinned visualizations)
            # self.display_individual_list()
            self._auto_visualize_current()
            return

        current_keys = self._get_current_keys()
        if not current_keys:
            self._print_status("[red]No items to navigate at this level.[/red]")
            return
        total = len(current_keys)
        start = (getattr(self, "_current_page", 0)) * self._MAX_ITEMS
        end = min(start + self._MAX_ITEMS, total)
        page_len = max(0, end - start)
        if page_len == 0:
            return
        local = self._index - start
        local = (local + 1) % page_len
        self._index = start + local
        self._individual_index = 0

        self.display_current_menu()

    def select_current(self) -> None:
        if getattr(self, "_viewing_individuals", False):
            # In individuals view: middle button pins/unpins the current individual
            idx = self._get_current_individual_value()
            if idx is None:
                return
            if idx in self._pinned_individuals:
                self._pinned_individuals.remove(idx)
            else:
                self._pinned_individuals.append(idx)

            # Re-render visualizations with new pin state
            self._render_all_visualizations()
            self.display_individual_list()
            return

        current_keys = self._get_current_keys()
        if not current_keys:
            return
        # clamp index in case structure changed
        self._index = max(0, min(self._index, len(current_keys) - 1))
        if self._is_current_selection_a_menu(current_keys, self._index):
            selected_key = current_keys[self._index]
            self._path.append(selected_key)
            self._index = 0
            self._individual_index = 0
            self._current_page = 0
            self.display_current_menu()
        elif self._get_index_list():
            self._viewing_individuals = True
            self._individual_index = 0
            self._current_page = 0
            self._pinned_individuals = []  # Clear pins when entering new individual list

            # Auto-visualize the first individual immediately
            self._auto_visualize_current()

    def _auto_visualize_current(self) -> None:
        """Auto-visualize the currently selected individual, removing unpinned ones."""
        idx = self._get_current_individual_value()
        if idx is None:
            return

        # Remove all unpinned visualizations
        self._visualized_individuals = [
            i
            for i in self._visualized_individuals
            if i in self._pinned_individuals
        ]

        # Add current selection if not already visualized
        if idx not in self._visualized_individuals:
            self._visualized_individuals.insert(0, idx)  # Insert at top
        else:
            # Move to top if already in list
            self._visualized_individuals.remove(idx)
            self._visualized_individuals.insert(0, idx)

        self.display_individual_list()
        self._render_all_visualizations()

    def _render_all_visualizations(
        self,

    ) -> None:
        """Render all visualized individuals in order (most recent first)."""
        if not self._viz_output_area or not self._visualized_individuals:
            return

        with self._viz_output_area:
            clear_output(wait=True)
            # Render in order (first in list = most recent = displayed at top)
            for idx in self._visualized_individuals:
                if idx < len(self._population):
                    pin_marker = (
                        " | PINNED" if idx in self._pinned_individuals else ""
                    )
                    title = (
                        self._get_path() + f" | Individual {idx}{pin_marker}"
                    )
                    if self._visualize_fn is visualize_tree_from_graph:
                        node = ctk.from_graph(self._population[idx])
                        node.canonicalize()
                        self._visualize_fn(node.to_graph(), title=node.to_string())
                    else:
                        node = ctk.from_graph(self._population[idx])
                        node.canonicalize()
                        view(node.to_graph())


    def backward(self) -> None:
        # when viewing individuals
        if getattr(self, "_viewing_individuals", False):
            # if on a non-zero individuals page -> go to previous page
            if getattr(self, "_current_page", 0) > 0:
                self._current_page -= 1
                indexes = self._get_index_list()
                total = len(indexes)
                if total:
                    self._individual_index = min(
                        total - 1,
                        self._current_page * self._MAX_ITEMS,
                    )

                # Auto-visualize on page change
                self._auto_visualize_current()
                # self.display_individual_list()
                return

            # otherwise exit individuals view back to menu
            self._viewing_individuals = False
            self._individual_index = 0
            self._current_page = 0
            self._visualized_individuals = []
            self._pinned_individuals = []
            self._clear_visualizations()
            self.display_current_menu()
            return

        # not viewing individuals: handle menu paging / go up
        if getattr(self, "_current_page", 0) > 0:
            self._current_page -= 1
            current_keys = self._get_current_keys()
            self._index = (
                min(len(current_keys) - 1, self._current_page * self._MAX_ITEMS)
                if current_keys
                else 0
            )
            self.display_current_menu()
            return

        # page == 0: go up one level in the tree (if possible)
        if self._path:
            self._path.pop()
            self._index = 0
            self._individual_index = 0
            self._current_page = 0
            self.display_current_menu()

    def forward(self) -> None:
        if self._viewing_individuals:
            indexes = self._get_index_list()
            total = len(indexes)
            if total == 0:
                return
            total_pages = max(
                1,
                (total + self._MAX_ITEMS - 1) // self._MAX_ITEMS,
            )
            if self._current_page < total_pages - 1:
                self._current_page += 1
                # move individual index to first item of new page
                self._individual_index = min(
                    total - 1,
                    self._current_page * self._MAX_ITEMS,
                )

                # Auto-visualize on page change
                # self.display_individual_list()
                self._auto_visualize_current()

            return

        current_keys = self._get_current_keys()
        total = len(current_keys)
        if total == 0:
            return
        total_pages = max(1, (total + self._MAX_ITEMS - 1) // self._MAX_ITEMS)
        if self._current_page < total_pages - 1:
            self._current_page += 1
            # move selection to first item on new page (global index)
            self._index = min(
                len(current_keys) - 1,
                self._current_page * self._MAX_ITEMS,
            )
            self.display_current_menu()

    def _get_path(self) -> str:
        return (
            "ROOT/"
            if not self._path
            else "ROOT/" + "/".join(map(str, self._path))
        )

    def display_individual_list(self) -> None:
        indexes = self._get_index_list()
        if not indexes:
            self._print_status(
                "[red]No individuals to display.[/red]",
                clear=True,
            )
            return

        # pagination for individuals (same _MAX_ITEMS as menu)
        total = len(indexes)
        total_pages = max(1, (total + self._MAX_ITEMS - 1) // self._MAX_ITEMS)
        self._total_pages = total_pages
        # clamp current page and individual index
        self._current_page = max(
            0,
            min(getattr(self, "_current_page", 0), total_pages - 1),
        )
        self._individual_index = max(0, min(self._individual_index, total - 1))

        start = self._current_page * self._MAX_ITEMS
        end = min(start + self._MAX_ITEMS, total)
        display_slice = indexes[start:end]

        current_path_str = self._get_path()
        individual_lines = []
        for local_i, idx in enumerate(display_slice):
            global_i = start + local_i
            is_selected = global_i == self._individual_index
            is_visualized = idx in self._visualized_individuals

            marker = ""
            if is_selected:
                marker = "◘" if is_visualized else "o"
            elif is_visualized:
                marker = "[bold green]●[/bold green]"

            # marker = "●" if is_visualized else ""
            if is_selected:
                line = f"Individual [green bold]{idx}[/green bold] {marker}"
            else:
                line = f"Individual [yellow]{idx}[/yellow] {marker}"
            if is_selected:
                if is_visualized:
                    line = f"[bold green]{line}[/bold green]"
                else:
                    line = f"[bold green]{line}[/bold green]"
            # show the global position in the list (keeps numbering consistent across pages)
            individual_lines.append(
                f"[yellow dim]{global_i:>3}[/yellow dim]  {line}",
            )

        full_output = (
            f"\n[bold underline]{current_path_str}[/bold underline]\n\n"
        )
        full_output += f"[yellow]Viewing {total} individuals:[/yellow]\n\n"
        full_output += "\n".join(individual_lines)

        # page indicator for individuals when paged
        if total_pages > 1:
            full_output += (
                f"\n\n[dim]page {self._current_page + 1}/{total_pages}[/dim]"
            )

        if self._visualized_individuals:
            full_output += (
                "\n\n[dim]Press ● again to clear visualized individual[/dim]"
            )
        else:
            full_output += "\n\n[dim]Press ◀ to go back to menu[/dim]"

        self._print_status(full_output, clear=True)

    def _clear_visualizations(self) -> None:
        if self._viz_output_area:

            with self._viz_output_area:
                clear_output(wait=False)

    def visualize_individual(
        self,
        cycle: bool = False,
    ) -> None:
        idx = self._get_current_individual_value()
        if idx is None or not self._population or idx >= len(self._population):
            return
        if idx not in self._visualized_individuals:
            self._visualized_individuals.append(idx)
        title = self._get_path() + " | Population individual index: " + str(idx)
        if self._viz_output_area:
            with self._viz_output_area:
                self._visualize_fn(self._population[idx], title=title)
        self.display_individual_list()

    def _get_current_individual_value(self) -> int | None:
        indexes = self._get_index_list()
        return (
            indexes[self._individual_index % len(indexes)] if indexes else None
        )

    def _show_current_selection(self, clear: bool = False) -> None:
        current_keys = self._get_current_keys()
        if not current_keys:
            self._print_status("[red]No keys found.[/red]", clear=clear)
            return
        key = current_keys[self._index]
        indexes = self._get_index_list()
        total = len(indexes)
        path_str = " > ".join([*self._path, key])
        if indexes:
            idx = self._individual_index % total
            individual_value = indexes[idx]
            progress_str = f"[{idx + 1}/{total}]"
            status_message = (
                f"\n[bold green]Path:[/bold green] {path_str} | "
                f"[bold green]Individual Index:[/bold green] {idx} {progress_str} | "
                f"[magenta]Value:[/magenta] {individual_value}"
            )
        else:
            status_message = f"\n[bold green]Path:[/bold green] {path_str} | [red]No individuals (Menu item or empty node)[/red]"
        self._print_status(status_message, clear=clear)

    def _filter_nested_index_dict(self, index_dict: dict) -> dict:
        if not isinstance(index_dict, collections.abc.Mapping):
            return {}
        filtered_dict = {}
        for k, v in index_dict.items():
            if isinstance(v, collections.abc.Mapping):
                filtered_v = self._filter_nested_index_dict(v)
                if filtered_v:
                    filtered_dict[k] = filtered_v
            else:
                is_valid_list = (
                    isinstance(v, (list, tuple, set))
                    and v
                    and all(isinstance(item, int) for item in v)
                )
                if is_valid_list:
                    filtered_dict[k] = v
        return filtered_dict

    def remove_visualization(self, idx: int) -> None:
        if idx in self._visualized_individuals:
            self._visualized_individuals.remove(idx)
            self._clear_visualizations()
            if self._visualized_individuals:
                for viz_idx in self._visualized_individuals:
                    if (
                        viz_idx < len(self._population)
                        and self._viz_output_area
                    ):
                        with self._viz_output_area:
                            visualize_tree_from_graph(self._population[viz_idx])
            self.display_individual_list()

    def _paginate_and_display(
        self,
        items: list,
        current_index: int,
        format_item_func,
        header: str,
        footer_hints: list[str],
    ) -> None:
        """Common pagination and display logic for both menus and individual lists."""
        if not items:
            self._print_status("[red]No items to display.[/red]", clear=True)
            return None

        # Compute pagination
        total = len(items)
        self._total_pages = max(
            1,
            (total + self._MAX_ITEMS - 1) // self._MAX_ITEMS,
        )

        # Clamp current page and index
        current_index = max(0, min(current_index, total - 1))
        self._current_page = max(
            0,
            min(self._current_page, self._total_pages - 1),
        )

        # Calculate slice
        start = self._current_page * self._MAX_ITEMS
        end = min(start + self._MAX_ITEMS, total)
        display_slice = items[start:end]

        # Format items
        index_prefix_fmt = "[grey][dim]{:>4}   [/dim][/grey]"
        lines = []
        for local_i, item in enumerate(display_slice):
            global_i = start + local_i
            is_selected = global_i == current_index
            formatted_line = format_item_func(item, global_i, is_selected)
            lines.append(
                f"{index_prefix_fmt.format(global_i + 1)}{formatted_line}",
            )

        # Pad to fixed page height
        displayed = len(display_slice)
        if displayed < self._MAX_ITEMS:
            for pad_i in range(displayed, self._MAX_ITEMS):
                global_i = start + pad_i
                index_prefix_fmt.format(global_i + 1)
                # lines.append(f"{prefix}")
                lines.append("")

        # Build output
        current_path_str = self._get_path()
        full_output = f"\n[bold cyan underline]{current_path_str}[/bold cyan underline]\n\n"
        full_output += header + "\n".join(lines)

        # Footer
        footer_parts = [
            f"page {self._current_page + 1}/{self._total_pages}",
            *footer_hints,
        ]
        full_output += "\n\n" + "   ".join(
            f"[dim]{p}[/dim]" for p in footer_parts
        )

        self._print_status(full_output, clear=True)
        return current_index

    def display_current_menu(self) -> None:
        """Display the current menu level."""
        current_keys = self._get_current_keys()

        def format_menu_item(key, global_i, is_selected) -> str:
            is_menu = self._is_current_selection_a_menu(current_keys, global_i)
            value = self._get_current_level_data().get(key)

            # Fixed width key column
            key_col = 20
            key_display = f"{key!s:<{key_col}.{key_col}}"
            key_display_selected = f"{(str(key) + ' ●'):<{key_col}.{key_col}}"

            # Suffix based on type
            if is_menu:
                suffix = f"[dim][cyan]    ({len(value.keys())} sub-menus)[/cyan][/dim]"
            else:
                count = (
                    len(value) if isinstance(value, (list, tuple, set)) else 0
                )
                suffix = (
                    f"[dim][yellow]    ({count} individuals)[/yellow][/dim]"
                )

            # Highlight selected line
            if is_selected:
                return f"[bold cyan]{key_display_selected}[/bold cyan]{suffix}"
            return f"{key_display}{suffix}"

        # indexes = self._get_index_list()
        total = len(current_keys)
        header = f"[dim]total [cyan]{total}:[/cyan][/dim]\n\n"

        footer_hints = ["\n\nPress ⬤ to enter / ◀ to move back"]
        self._index = self._paginate_and_display(
            items=current_keys,
            current_index=self._index,
            format_item_func=format_menu_item,
            header=header,
            footer_hints=footer_hints,
        )

    def display_individual_list(self) -> None:
        """Display the list of individuals."""

        def format_individual_item(idx, global_i, is_selected) -> str:
            is_visualized = idx in self._visualized_individuals
            is_pinned = idx in self._pinned_individuals

            # Markers
            marker = ""
            if is_selected:
                marker = "◘" if is_pinned else "o"
            elif is_visualized:
                marker = "[cyan bold]●[/cyan bold]"

            # Line body

            # Highlight selected
            if is_selected:
                return f"[cyan bold]Individual {idx} {marker}[/cyan bold] "  # return f"[bold cyan]{line_body}[/bold cyan]"
            return f"[yellow]Individual {idx}[/yellow] {marker}"

        indexes = self._get_index_list()
        total = len(indexes)
        header = f"[dim]total [yellow]{total}:[/yellow][/dim]\n\n"

        # Dynamic footer based on state
        footer_hints = []
        if self._pinned_individuals:
            footer_hints.append(
                "\n\nPress ⬤ to unpin individual / ◀ to go back",
            )
        else:
            footer_hints.append("\n\nPress ⬤ to pin individual / ◀ to go back")
        # footer_hints.append("")

        self._individual_index = self._paginate_and_display(
            items=indexes,
            current_index=self._individual_index,
            format_item_func=format_individual_item,
            header=header,
            footer_hints=footer_hints,
        )

    def gui(self) -> None:
        v.theme.dark = False

        self._output_area = widgets.Output(
            layout=widgets.Layout(
                border="1px solid #ddd",
                background_color="#ffffff",
                height="330px",
                width="450px",
                overflow_y="auto",
                padding="0 0 0 20px",
            ),
        )

        self._viz_output_area = widgets.Output(
            layout=widgets.Layout(
                # margin="50px 0 0 0",
                width="100%",
                height="450px",
                overflow_y="auto",
                # padding="8px",
            ),
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

        # callbacks
        def on_up_click(widget, event, data) -> None:
            self.up()

        def on_down_click(widget, event, data) -> None:
            self.down()

        # left/right now control pages (prev/next)
        def on_left_click(widget, event, data) -> None:
            # if viewing individuals, page individuals; otherwise page menu
            self.backward()

        def on_right_click(widget, event, data) -> None:
            self.forward()

        def on_middle_click(widget, event, data) -> None:
            # when viewing individuals, middle toggles visualization for current individual
            self.select_current()

        btn_up.on_event("click", on_up_click)
        btn_down.on_event("click", on_down_click)
        btn_left.on_event("click", on_left_click)
        btn_right.on_event("click", on_right_click)
        btn_center.on_event("click", on_middle_click)

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

        # top area: controls on the left and textual output on the right
        top = widgets.HBox(
            [
                pad,
                widgets.Box(
                    [self._output_area],
                    layout=widgets.Layout(margin="15px 0 0 30px"),
                ),
            ],
            layout=widgets.Layout(
                align_items="flex-start",
                justify_content="center",
            ),
        )

        # full layout: stack the top area above the visualization output area
        layout = widgets.VBox(
            [top, self._viz_output_area],
            layout=widgets.Layout(
                align_items="stretch",
                justify_content="flex-start",
            ),
        )

        display(layout)

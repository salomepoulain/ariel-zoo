# Standard library
import collections.abc

# Third-party libraries
from pathlib import Path
import numpy as np
from rich.console import Console

# Local libraries
from experiments.gui_vis.visualize_tree import (
    visualize_tree_from_graph,
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


class IndividualVisualizer:
    """
    A stateful class to navigate a nested dictionary structure (index_dict)
    and select individuals from a population list for visualization.
    """

    def __init__(
        self,
        index_dict: dict,
        population: list,
        console: Console | None = None,
        gui: bool = True,
    ) -> None:
        """
        index_dict: dict mapping keys to lists of individual indexes (may be empty).
        population: list of individuals (e.g., DiGraph objects).
        console: Console instance for printing (optional).
        """
        # 1. Initialize State
        self._population = population
        self._console = console or Console()
        self._direction = 1
        self._output_area = None
        self._viz_output_area = None  # NEW: Dedicated visualization area
        self._viewing_individuals = False
        self._visualized_individuals = []  # NEW: Track visualized individuals

        # 2. Process Dictionary
        self._dict = self._filter_nested_index_dict(index_dict)

        # 3. Initialize Menu Navigation State
        self._path = []
        self._index = 0
        self._individual_index = 0

        if gui:
            self.gui()

        # Display the initial state
        self.display_current_menu()

    def _print_status(self, message: str, clear: bool = True) -> None:
        """Prints a message to the dedicated output area, optionally clearing it first."""
        try:
            from IPython.display import clear_output

            if self._output_area:
                with self._output_area:
                    if clear:
                        clear_output(wait=False)
                    # Use plain print instead of Rich for speed
                    self._console.print(message)
            else:
                # Fallback uses Rich for terminal
                self._console.print(message)
        except ImportError:
            self._console.print(message)

    def _get_current_level_data(self) -> dict:
        """Helper to get the dictionary (or leaf value) at the current path."""
        current_data = self._dict
        for key in self._path:
            current_data = current_data.get(key, {})
        return current_data

    def _get_current_keys(self) -> list:
        """Helper to get the list of keys (menu items) at the current level."""
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
        """Checks if the item at the current index is a nested dictionary (a sub-menu)."""
        current_data = self._get_current_level_data()

        if not current_keys or index >= len(current_keys):
            return False

        selected_key = current_keys[index]
        return isinstance(
            current_data.get(selected_key),
            collections.abc.Mapping,
        )

    def _get_index_list(self) -> list:
        """Retrieves the list of individual indices from the currently selected leaf node."""
        current_keys = self._get_current_keys()
        if not current_keys:
            return []

        selected_key = current_keys[self._index]
        current_data = self._get_current_level_data()
        value = current_data.get(selected_key)

        if isinstance(value, (list, tuple, set)):
            return list(value)

        return []

    def up(self) -> None:
        """Moves the selection index up (decreases index) with modulus wrapping."""
        if self._viewing_individuals:
            indexes = self._get_index_list()
            if indexes:
                N = len(indexes)
                self._individual_index = (self._individual_index - 1 + N) % N
                self.display_individual_list()
                if hasattr(self, "_update_middle_button"):
                    self._update_middle_button()
            return

        current_keys = self._get_current_keys()
        if not current_keys:
            self._print_status("[red]No items to navigate at this level.[/red]")
            return

        N = len(current_keys)
        self._index = (self._index - 1 + N) % N
        self._individual_index = 0
        self.display_current_menu()

    def down(self) -> None:
        """Moves the selection index down (increases index) with modulus wrapping."""
        if self._viewing_individuals:
            indexes = self._get_index_list()
            if indexes:
                N = len(indexes)
                self._individual_index = (self._individual_index + 1) % N
                self.display_individual_list()
                if hasattr(self, "_update_middle_button"):
                    self._update_middle_button()
            return

        current_keys = self._get_current_keys()
        if not current_keys:
            self._print_status("[red]No items to navigate at this level.[/red]")
            return

        N = len(current_keys)
        self._index = (self._index + 1) % N
        self._individual_index = 0
        self.display_current_menu()

    def forward(self) -> None:
        """Moves into a nested dictionary (increases depth) or enters individual view mode."""
        current_keys = self._get_current_keys()

        if self._is_current_selection_a_menu(current_keys, self._index):
            # It's a submenu, navigate into it
            selected_key = current_keys[self._index]
            self._path.append(selected_key)
            self._index = 0
            self._individual_index = 0
            self.display_current_menu()
        else:
            # It's a leaf node with individuals, enter viewing mode
            if self._get_index_list():
                self._viewing_individuals = True
                self._individual_index = 0
                self.display_individual_list()

    def backward(self) -> None:
        """Moves out of a nested dictionary or exits individual view mode."""
        if self._viewing_individuals:
            # If there are visualizations, clear them first
            if self._visualized_individuals:
                self._visualized_individuals = []
                self._clear_visualizations()
                self.display_individual_list()  # Refresh the list
                return

            # Exit individual view mode
            self._viewing_individuals = False
            self._individual_index = 0
            self.display_current_menu()
            return

        if self._path:
            self._path.pop()
            self._index = 0
            self._individual_index = 0
            self.display_current_menu()

    def _get_path(self) -> str:
        current_path_str = ""
        current_keys = self._get_current_keys()
        selected_key = current_keys[self._index]
        if not self._path:
            current_path_str = f"ROOT/{selected_key}"
        else:
            current_path_str = (
                "ROOT/" + "/".join(self._path) + f"/{selected_key}"
            )
        return current_path_str

    def display_individual_list(self) -> None:
        """Displays a vertical list of individuals when in viewing mode."""
        indexes = self._get_index_list()

        if not indexes:
            self._print_status(
                "[red]No individuals to display.[/red]",
                clear=True,
            )
            return

        current_path_str = self._get_path()

        # Build individual list
        individual_lines = []
        for i, idx in enumerate(indexes):
            is_selected = i == self._individual_index
            is_visualized = idx in self._visualized_individuals

            # Use ⦿ for visualized, ○ for not visualized
            marker = "⦿" if is_visualized else ""
            line = f"Individual {idx} {marker}"

            if is_selected:
                if is_visualized:
                    line = f"[bold green]{line} [/bold green]"
                else:
                    line = f"[bold green]{line}◯[/bold green]"

            individual_lines.append(f"{i:>3}  {line}")

        # Build output
        full_output = (
            f"\n[bold underline]{current_path_str}[/bold underline]\n\n"
        )
        full_output += (
            f"[yellow]Viewing {len(indexes)} individuals:[/yellow]\n\n"
        )
        full_output += "\n".join(individual_lines)

        if self._visualized_individuals:
            full_output += "\n\n[dim]Press ↶ once to clear visualizations, twice to go back to menu[/dim]"
        else:
            full_output += "\n\n[dim]Press ↶ to go back to menu[/dim]"

        self._print_status(full_output, clear=True)

    def _clear_visualizations(self) -> None:
        """Clears all visualizations from the visualization area."""
        if self._viz_output_area:
            from IPython.display import clear_output

            with self._viz_output_area:
                clear_output(wait=False)

    def visualize_individual(
        self,
        cycle=False,
        visualize_fn=visualize_tree_from_graph,
    ) -> None:
        """Visualizes the current individual and stacks it in the visualization area."""
        idx = self._get_current_individual_value()

        if idx is None or not self._population or idx >= len(self._population):
            return

        # Add to visualized list if not already there
        if idx not in self._visualized_individuals:
            self._visualized_individuals.append(idx)

        title = self._get_path() + " | Population individual index: " + str(idx)
        # Visualize in the dedicated area
        if self._viz_output_area:
            with self._viz_output_area:
                # Don't clear - stack visualizations
                visualize_fn(self._population[idx][0], title=title)

        # Refresh the individual list to update markers
        self.display_individual_list()

    def display_current_menu(self) -> None:
        """Prints the current menu state."""
        current_keys = self._get_current_keys()

        if not current_keys:
            self._print_status(
                "[red]No keys or valid individuals found at this level.[/red]",
                clear=True,
            )
            return

        menu_lines = []
        for i, key in enumerate(current_keys):
            is_selected = i == self._index
            is_menu = self._is_current_selection_a_menu(current_keys, i)
            value = self._get_current_level_data().get(key)

            prefix = f"{i:>3} "
            suffix = ""
            key_display = key

            if is_menu:
                suffix = f" [cyan]({len(value.keys())} sub-menus)[/cyan]"
                key_display = f"{key} (MENU)"
            else:
                count = (
                    len(value) if isinstance(value, (list, tuple, set)) else 0
                )
                if is_selected:
                    suffix = f" [green]({count} individuals)[/green]"
                else:
                    suffix = f" [yellow]({count} individuals)[/yellow]"

            suffix_marker = " >" if is_selected else ""

            full_line = f"{key_display}{suffix}{suffix_marker}"
            if is_selected:
                full_line = f"[bold green]{full_line}[/bold green]"

            menu_lines.append(f"{prefix}{full_line}")

        # Build path like a file system: ROOT/parent/child
        if not self._path:
            current_path_str = "ROOT"
        else:
            current_path_str = "ROOT/" + "/".join(self._path)

        # BUILD EVERYTHING AS ONE STRING
        full_output = (
            f"\n[bold underline]{current_path_str}[/bold underline]\n\n"
        )
        full_output += "\n".join(menu_lines)

        # SINGLE PRINT CALL
        self._print_status(full_output, clear=True)

    def _get_current_individual_value(self) -> int | None:
        """Returns the index of the currently selected individual within the population."""
        indexes = self._get_index_list()
        if indexes:
            return indexes[self._individual_index % len(indexes)]
        return None

    def _show_current_selection(self, clear: bool = False) -> None:
        """Displays the key path and the individual index currently selected."""
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
        """
        Recursively filters a dictionary, keeping only valid leaf nodes (lists of ints)
        or nested dictionaries that contain valid leaf nodes.
        """
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

    def remove_visualization(self, idx) -> None:
        """Removes a specific individual's visualization."""
        if idx in self._visualized_individuals:
            self._visualized_individuals.remove(idx)

            # Re-render all remaining visualizations
            self._clear_visualizations()

            if self._visualized_individuals:
                for viz_idx in self._visualized_individuals:
                    if viz_idx < len(self._population):
                        if self._viz_output_area:
                            with self._viz_output_area:
                                visualize_tree_from_graph(
                                    self._population[viz_idx][0],
                                )

            # Refresh the individual list
            self.display_individual_list()

    def gui(self) -> None:
        try:
            import ipywidgets as widgets
            from IPython.display import display
        except ImportError:
            self._console.print(
                "ERROR: ipywidgets or IPython.display not found. Cannot create GUI.",
            )
            self._console.print(
                "Please ensure you are running this code in a Jupyter Notebook.",
            )
            return

        from ipywidgets import (
            Box,
            Button,
            ButtonStyle,
            GridBox,
            HBox,
            Layout,
            Output,
            VBox,
        )

        # Create dedicated output widgets
        self._output_area = Output(layout=Layout(margin="0 0 0 20px"))
        self._viz_output_area = Output(layout=Layout(margin="20px 0 0 0"))

        # Create buttons first so we can reference them in callbacks
        button5 = Button(
            description="",
            layout=Layout(width="auto", height="auto", border_radius="50%"),
            style=ButtonStyle(button_color="khaki"),
        )

        # Define callback functions
        def on_up_click(b) -> None:
            self.up()

        def on_down_click(b) -> None:
            self.down()

        def on_left_click(b) -> None:
            self.backward()
            update_middle_button()

        def on_right_click(b) -> None:
            self.forward()
            update_middle_button()

        def on_middle_click(b) -> None:
            if not self._viewing_individuals:
                return

            idx = self._get_current_individual_value()
            if idx is None:
                return

            if idx in self._visualized_individuals:
                # Remove visualization
                self.remove_visualization(idx)
            else:
                # Add visualization
                self.visualize_individual()

            update_middle_button()

        def update_middle_button() -> None:
            """Updates the middle button icon based on current state."""
            if self._viewing_individuals:
                idx = self._get_current_individual_value()
                if idx is not None and idx in self._visualized_individuals:
                    button5.description = "✕"  # '◯' #'✕'  # X for remove
                else:
                    button5.description = "⦿"  # '⦿'  # Circle for visualize
            else:
                button5.description = " "

        # Create other buttons
        button1 = Button(
            description="↑",
            layout=Layout(width="auto", height="auto"),
            style=ButtonStyle(button_color="lightblue"),
        )
        button2 = Button(
            description="↶",
            layout=Layout(width="auto", height="auto"),
            style=ButtonStyle(button_color="salmon"),
        )
        button3 = Button(
            description=">",
            layout=Layout(width="auto", height="auto"),
            style=ButtonStyle(button_color="lightgreen"),
        )
        button4 = Button(
            description="↓",
            layout=Layout(width="auto", height="auto"),
            style=ButtonStyle(button_color="lightblue"),
        )

        # Connect buttons to callbacks
        button1.on_click(on_up_click)
        button2.on_click(on_left_click)
        button3.on_click(on_right_click)
        button4.on_click(on_down_click)
        button5.on_click(on_middle_click)

        # Store button reference for updates
        self._middle_button = button5
        self._update_middle_button = update_middle_button

        # Create the grid
        grid = GridBox(
            children=[
                Box(),
                button1,
                Box(),
                button2,
                button5,
                button3,
                Box(),
                button4,
                Box(),
            ],
            layout=Layout(
                grid_template_columns="50px 50px 50px",
                grid_template_rows="50px 50px 50px",
                grid_gap="0px",
            ),
        )

        # Layout: grid on left, menu output on right, visualizations below
        top_row = HBox([grid, self._output_area])
        full_layout = VBox([top_row, self._viz_output_area])

        display(full_layout)

        # Show initial menu in the output area
        self.display_current_menu()

import sys
import time
import importlib
from rich.console import Console
from rich.table import Table

console = Console()

def time_imports(module_name: str):
    # 1. Capture current modules to see what's new
    initial_modules = set(sys.modules.keys())
    
    start_time = time.perf_counter()
    try:
        # 2. Perform the import
        module = importlib.import_module(module_name)
    except ImportError as e:
        console.print(f"[bold red]Error:[/bold red] Could not import {module_name}: {e}")
        return
    end_time = time.perf_counter()
    
    # 3. Identify newly loaded modules
    new_modules = set(sys.modules.keys()) - initial_modules
    duration = end_time - start_time
    
    # 4. Print Summary
    console.rule(f"[bold green]Import Profile: {module_name}")
    console.print(f"Total Import Time: [bold cyan]{duration:.4f} seconds[/bold cyan]")
    console.print(f"Total New Modules Loaded: [bold cyan]{len(new_modules)}[/bold cyan]\n")
    
    # 5. Build Table of New Imports
    table = Table(title="Newly Imported Modules", box=None)
    table.add_column("Module Name", style="magenta")
    table.add_column("Type", style="dim")
    
    # Sort for readability
    for m in sorted(new_modules):
        m_type = "Built-in/Stdlib" if m.split('.')[0] in sys.builtin_module_names else "External/Package"
        table.add_row(m, m_type)
        
    console.print(table)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        console.print("[yellow]Usage: python timer_script.py <canonical_toolkit>[/yellow]")
    else:
        time_imports(sys.argv[1])

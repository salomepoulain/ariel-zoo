
"""Image saving, loading, and thumbnail generation utilities."""

from __future__ import annotations

import base64
import traceback
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import TypeVar

from PIL import Image
from rich.console import Console
from rich.progress import track

__all__ = [
    "snapshot_saver",
    "snapshot_loader",
    "snapshot_thumbnail_converter"
]

console = Console()

T = TypeVar('T')



def _default_snapshot_converter(data: T) -> Image.Image:
    """Default converter using quick_view from viewer."""
    from .viewer import quick_view
    return quick_view(data, return_img=True, tilted=True, remove_background=True, white_background=True)


def _save_single_image_worker(args):
    """Worker function for parallel image generation."""
    idx, data, converter, save_path = args
    try:
        snapshot = converter(data)
        if not isinstance(snapshot, Image.Image):
            snapshot = Image.fromarray(snapshot)
        snapshot.save(save_path) 
        return idx
    except Exception as e:
        console.print(f"[red]Error saving image {idx}: {e}[/red]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return None


def snapshot_saver(
    raw_data: list[T],
    save_folder: Path | str | None = None,
    *,
    snapshot_converter: Callable[[T], Image.Image] | None = None,
    snapshot_naming: list[str] | None = None,
    n_jobs: int = 1,
) -> None:
    """
    Save snapshots from raw data.

    Args:
        raw_data: List of data to convert to snapshots
        save_folder: Directory to save snapshots (default: __data__/snapshots)
        snapshot_converter: Function to convert data → PIL.Image (default: quick_view)
        snapshot_naming: Custom filenames (default: "0.png", "1.png", ...)
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = serial)
    """
    # Set defaults
    if save_folder is None:
        save_folder = Path.cwd() / "__data__" / "snapshots"
    save_folder = Path(save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)

    if snapshot_converter is None:
        snapshot_converter = _default_snapshot_converter

    if snapshot_naming is None:
        snapshot_naming = [f"{i}.png" for i in range(len(raw_data))]

    if len(snapshot_naming) != len(raw_data):
        raise ValueError(f"snapshot_naming length ({len(snapshot_naming)}) must match raw_data length ({len(raw_data)})")

    # Determine parallelism
    n_workers = cpu_count() if n_jobs == -1 else max(1, n_jobs)
    use_parallel = n_workers > 1

    # Prepare arguments
    args = [
        (i, raw_data[i], snapshot_converter, save_folder / snapshot_naming[i])
        for i in range(len(raw_data))
    ]

    if use_parallel:
        raise DeprecationWarning
        try:
            with Pool(processes=n_workers) as pool:
                results = list(
                    track(
                        pool.imap(_save_single_image_worker, args),
                        total=len(raw_data),
                        description=f"Saving snapshots (parallel, {n_workers} workers)...",
                    )
                )
            successful = [r for r in results if r is not None]
            console.print(f"[green]Saved {len(successful)}/{len(raw_data)} snapshots to {save_folder}[/green]")
        except Exception as e:
            console.print(f"[yellow]Multiprocessing failed ({e}), falling back to serial...[/yellow]")
            use_parallel = False

    if not use_parallel:
        successful = 0
        for arg in track(args, description="Saving snapshots (serial)..."):
            result = _save_single_image_worker(arg)
            if result is not None:
                successful += 1
        console.print(f"[green]Saved {successful}/{len(raw_data)} snapshots to {save_folder}[/green]")


def snapshot_loader(
    snapshot_folder: Path | str | None = None,
    snapshot_names: list[str] | None = None,
) -> list[Image.Image]:
    """
    Load snapshots from disk.

    Args:
        snapshot_folder: Directory containing snapshots (default: __data__/snapshots)
        snapshot_names: List of filenames to load (default: all snapshots in folder)

    Returns:
        List of PIL snapshots in same order as snapshot_names
    """
    # Set defaults
    if snapshot_folder is None:
        snapshot_folder = Path.cwd() / "__data__" / "snapshots"
    snapshot_folder = Path(snapshot_folder)

    if not snapshot_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {snapshot_folder}")

    # If no names specified, load all snapshots
    if snapshot_names is None:
        snapshot_names = sorted([f.name for f in snapshot_folder.glob("*.png")])
        if not snapshot_names:
            console.print(f"[yellow]No .png files found in {snapshot_folder}[/yellow]")
            return []

    # Load snapshots in order
    snapshots = []
    for name in track(snapshot_names, description=f"Loading {len(snapshot_names)} snapshots..."):
        snapshot_path = snapshot_folder / name
        if not snapshot_path.exists():
            console.print(f"[yellow]Warning: {name} not found, skipping[/yellow]")
            continue
        snapshots.append(Image.open(snapshot_path))

    return snapshots


def _load_and_encode_thumbnail(args):
    """Worker function for parallel thumbnail conversion."""
    snapshot_name, snapshot_folder, scale = args
    snapshot_path = snapshot_folder / snapshot_name

    if not snapshot_path.exists():
        return ""  # Empty placeholder

    try:
        snapshot = Image.open(snapshot_path)
        new_size = (int(snapshot.width * scale), int(snapshot.height * scale))
        snapshot_small = snapshot.resize(new_size, Image.Resampling.BICUBIC)
        buffer = BytesIO()
        snapshot_small.save(buffer, format="PNG", optimize=False, compress_level=1)
        return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    except Exception as e:
        console.print(f"[red]Error converting {snapshot_name}: {e}[/red]")
        return ""


def snapshot_thumbnail_converter(
    snapshot_folder: Path | str | None = None,
    snapshot_names: list[str] | None = None,
    *,
    scale: float = 0.3,
    n_jobs: int = -1,
) -> list[str]:
    """
    Convert saved snapshots to base64 thumbnails.

    Args:
        snapshot_folder: Directory containing snapshots (default: __data__/snapshots)
        snapshot_names: List of filenames to convert (default: all snapshots in folder)
        scale: Scaling factor (0.3 = 30% of original size)
        n_jobs: Number of parallel jobs (-1 = all cores, 1 = serial)

    Returns:
        List of base64-encoded PNG data URLs in same order as snapshot_names
    """
    # Set defaults
    if snapshot_folder is None:
        snapshot_folder = Path.cwd() / "__data__" / "snapshots"
    snapshot_folder = Path(snapshot_folder)

    if not snapshot_folder.exists():
        raise FileNotFoundError(f"Image folder not found: {snapshot_folder}")

    # If no names specified, convert all snapshots
    if snapshot_names is None:
        snapshot_names = sorted([f.name for f in snapshot_folder.glob("*.png")])
        if not snapshot_names:
            console.print(f"[yellow]No .png files found in {snapshot_folder}[/yellow]")
            return []

    # Determine parallelism
    n_workers = cpu_count() if n_jobs == -1 else max(1, n_jobs)
    use_parallel = n_workers > 1

    if use_parallel:
        # Use ThreadPoolExecutor (I/O bound task)
        thumbnails = []
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            args = [(name, snapshot_folder, scale) for name in snapshot_names]
            futures = [executor.submit(_load_and_encode_thumbnail, arg) for arg in args]
            for future in track(
                futures,
                total=len(snapshot_names),
                description=f"Converting {len(snapshot_names)} thumbnails at {scale * 100:.0f}% scale..."
            ):
                thumbnails.append(future.result())
    else:
        # Serial processing
        thumbnails = []
        for name in track(
            snapshot_names,
            description=f"Converting {len(snapshot_names)} thumbnails at {scale * 100:.0f}% scale..."
        ):
            thumbnails.append(_load_and_encode_thumbnail((name, snapshot_folder, scale)))

    return thumbnails

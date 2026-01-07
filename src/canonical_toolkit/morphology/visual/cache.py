"""Image caching and thumbnail generation utilities."""

from __future__ import annotations

import base64
import traceback
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from multiprocessing import Pool
from pathlib import Path

from PIL import Image
from rich.console import Console
from rich.progress import track

from .viewer import view

console = Console()

# Default cache directory (follows same pattern as matrix frame)
CWD = Path.cwd()
DATA = CWD / "__data__" / "img"
DATA.mkdir(parents=True, exist_ok=True)


def _generate_single_image_worker(args):
    """
    Worker function for parallel image generation.
    Must be at module level for multiprocessing to work.

    Args:
        args: Tuple of (i, robot, cache_dir)

    Returns:
        int: robot_id if successful, None on error
    """
    i, robot, cache_dir = args
    try:
        # Generate image using view function
        img = view(robot, return_img=True, tilted=True, remove_background=True, with_viewer=False)

        # If it's already a PIL Image, use it directly; otherwise convert from array
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        img.save(f"{cache_dir}/robot_{i:04d}.png", optimize=True, compress_level=6)
        return i
    except Exception as e:
        console.print(f"[red]Error generating robot {i}: {e}[/red]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")
        return None


def generate_image_cache(
    population: list,
    cache_dir: str | Path | None = None,
    parallel: bool = False,
    max_workers: int = 4,
) -> None:
    """
    Generate robot images to cache directory.

    Args:
        population: List of robot objects (e.g., NetworkX graphs)
        cache_dir: Directory to save images (default: DATA = __data__/img)
        parallel: Use multiprocessing for speed (requires macOS/Linux setup)
        max_workers: Number of parallel workers

    Returns:
        None - images are saved to cache_dir
    """
    if cache_dir is None:
        cache_dir = DATA
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if parallel:
        try:
            args = [(i, population[i], cache_dir) for i in range(len(population))]

            with Pool(processes=max_workers) as pool:
                results = list(
                    track(
                        pool.imap(_generate_single_image_worker, args),
                        total=len(population),
                        description=f"Generating images (parallel, {max_workers} workers)...",
                    )
                )
            successful = [r for r in results if r is not None]
            console.print(f"[green]Generated {len(successful)}/{len(population)} images[/green]")
        except Exception as e:
            console.print(f"[yellow]Multiprocessing failed ({e}), falling back to serial...[/yellow]")
            parallel = False

    if not parallel:
        # Serial generation (slower but reliable in notebooks)
        successful = 0
        for i in track(range(len(population)), description="Generating images (serial)..."):
            result = _generate_single_image_worker((i, population[i], cache_dir))
            if result is not None:
                successful += 1
        console.print(f"[green]Generated {successful}/{len(population)} images[/green]")


def _load_single_thumbnail(i: int, cache_dir: Path, scale: float) -> str:
    """
    Load and scale a single robot image from cache.

    Args:
        i: Robot index
        cache_dir: Directory containing cached images
        scale: Scaling factor (0.25 = 25% size)

    Returns:
        Base64 encoded PNG data URL string, or empty string if not found
    """
    img_path = cache_dir / f"robot_{i:04d}.png"

    if not img_path.exists():
        return ""  # Empty placeholder

    img = Image.open(img_path)
    new_size = (int(img.width * scale), int(img.height * scale))
    img_small = img.resize(new_size, Image.Resampling.BICUBIC)
    buffer = BytesIO()
    img_small.save(buffer, format="PNG", optimize=False, compress_level=1)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def load_thumbnails(
    n_robots: int,
    cache_dir: str | Path | None = None,
    scale: float = 0.3,
    max_workers: int = 8,
) -> list[str]:
    """
    Load all thumbnails from cache (parallel with threading).

    Args:
        n_robots: Number of robots to load
        cache_dir: Directory containing cached images (default: DATA = __data__/img)
        scale: Scaling factor (0.3 = 30% size, ~7MB for 1000 robots)
        max_workers: Number of parallel threads

    Returns:
        List of base64 encoded PNG data URL strings
    """
    if cache_dir is None:
        cache_dir = DATA
    cache_dir = Path(cache_dir)

    thumbnails = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_single_thumbnail, i, cache_dir, scale) for i in range(n_robots)]
        for future in track(futures, total=n_robots, description=f"Loading {n_robots} thumbnails at {scale * 100:.0f}% scale..."):
            thumbnails.append(future.result())

    return thumbnails


def cache_exists(n_robots: int, cache_dir: str | Path | None = None) -> bool:
    """
    Check if image cache has all required images.

    Args:
        n_robots: Number of robots expected
        cache_dir: Directory to check (default: DATA = __data__/img)

    Returns:
        bool: True if cache exists and is valid
    """
    if cache_dir is None:
        cache_dir = DATA
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return False

    # Check if all images exist
    return all((cache_dir / f"robot_{i:04d}.png").exists() for i in range(n_robots))


def load_or_generate_cache(
    population: list,
    cache_dir: str | Path | None = None,
    scale: float = 0.3,
    parallel: bool = False,
    max_workers: int = 4,
) -> list[str]:
    """
    Convenience function: Load cache if it exists, otherwise generate it.

    Args:
        population: List of robot objects (only needed if generating)
        cache_dir: Directory for cache (default: DATA = __data__/img)
        scale: Thumbnail scale for loading
        parallel: Use multiprocessing for generation
        max_workers: Number of workers for generation

    Returns:
        List of base64 encoded thumbnail strings
    """
    if cache_dir is None:
        cache_dir = DATA
    cache_dir = Path(cache_dir)

    if cache_exists(len(population), cache_dir=cache_dir):
        console.print(f"[cyan]Cache found, loading {len(population)} thumbnails...[/cyan]")
        thumbnails = load_thumbnails(len(population), cache_dir=cache_dir, scale=scale)
    else:
        console.print(f"[yellow]Cache not found, generating {len(population)} images...[/yellow]")
        generate_image_cache(
            population,
            cache_dir=cache_dir,
            parallel=parallel,
            max_workers=max_workers,
        )
        thumbnails = load_thumbnails(len(population), cache_dir=cache_dir, scale=scale)

    return thumbnails

"""
is slow asf
"""

import base64
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO

from ea.config import logger

import canonical_toolkit as ctk
from ariel.ec.a004 import Population
from canonical_toolkit.morphology.visual.utils import center_on_canvas

# --- CONFIGURATION ---
# 192 workers is too many for EGL rendering; it will crash the GPU driver.
# We cap the rendering workers to a stable number (typically 16-32 on HPC nodes).
TOTAL_CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
NUM_RENDER_WORKERS = TOTAL_CPUS  # min(TOTAL_CPUS, 32)

# logger.info(f"Visuals: System has {TOTAL_CPUS} CPUs. Using {NUM_RENDER_WORKERS} workers for rendering to maintain EGL stability.")

# Use 'spawn' to avoid issues with forked processes and GPU libraries (MuJoCo/EGL)
# _MP_CONTEXT = mp.get_context("spawn")


def _worker_init():
    """Initialize worker process before any imports."""
    os.environ["_EA_WORKER"] = "1"  # Suppress spinner in config.py
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"


def _render_single(ctk_string: str) -> str:
    """
    Worker function for parallel rendering.
    Includes stability measures for EGL/MuJoCo initialization.
    """

    # 2. STAGGERED START:
    # Adding a random delay prevents 'Thundering Herd' where 100+ processes
    # try to hit the GPU driver at the exact same millisecond.
    time.sleep(random.uniform(0, 3.0))

    # Build the graph
    graph = ctk.node_from_string(ctk_string).to_graph()

    # Render
    img = ctk.quick_view(
        graph,
        return_img=True,
        white_background=True,
        remove_background=True,
        width=140,
        height=140,
        tilted=True,
    )

    img = center_on_canvas(img)

    # Convert to Base64
    buffer = BytesIO()
    img.save(buffer, format="WEBP", quality=80)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/webp;base64,{b64}"


def store_img(population: Population) -> Population:
    """
    Render robot images and store as base64 WebP in individual tags.
    Requires ctk_string to be pre-computed in ind.tags["ctk_string"].
    """
    to_render = [ind for ind in population if ind.requires_eval and "ctk_string" in ind.tags]

    if not to_render:
        return population

    render_start = time.perf_counter()
    ctk_strings = [ind.tags["ctk_string"] for ind in to_render]

    # Use the throttled NUM_RENDER_WORKERS count here
    try:
        with ProcessPoolExecutor(max_workers=NUM_RENDER_WORKERS, initializer=_worker_init) as executor:
            results = list(executor.map(_render_single, ctk_strings))
    except Exception as e:
        logger.error(f"Visuals: Process pool crashed: {e}")
        # Return population as-is to prevent the entire EA run from failing
        return population

    # Assign results back to individuals
    for ind, b64_img in zip(to_render, results, strict=False):
        if b64_img.startswith("ERROR"):
            logger.warning(f"Visuals: Rendering failed for an individual: {b64_img}")
            ind.tags["image"] = None
        else:
            ind.tags["image"] = b64_img

    render_dur = time.perf_counter() - render_start
    n = len(to_render)

    # Write to file instead of console
    from ea.config import config
    from datetime import datetime
    log_file = config.OUTPUT_FOLDER / "time_logs.txt"
    with open(log_file, "a") as f:
        timestamp = datetime.now().strftime("%H:%M:%S")
        f.write(f"[{timestamp}] Rendered {n} images in {render_dur:.2f}s ({render_dur / n:.3f}s/ind using {NUM_RENDER_WORKERS} workers)\n")

    return population

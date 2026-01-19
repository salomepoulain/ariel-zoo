import base64
import os
import time
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO

import canonical_toolkit as ctk
from ea.config import logger

from ariel.ec.a004 import Population
from canonical_toolkit.morphology.visual.utils import (
    center_on_canvas,
)
from canonical_toolkit.morphology.visual.viewer import quick_view

NUM_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
logger.info(f"Visuals: Using {NUM_WORKERS} workers for rendering")


def _render_single(ctk_string: str) -> str:
    """Worker function for parallel rendering - uses pre-computed ctk_string."""
    graph = ctk.node_from_string(ctk_string).to_graph()

    img = quick_view(
        graph,
        return_img=True,
        white_background=True,
        remove_background=True,
        width=140,
        height=140,
        tilted=True,
    )

    img = center_on_canvas(img)
    buffer = BytesIO()
    img.save(buffer, format="WEBP", quality=80)
    b64 = base64.b64encode(buffer.getvalue()).decode("ascii")

    return f"data:image/webp;base64,{b64}"


def save_img(population: Population) -> Population:
    """Render robot images and store as base64 WebP in individual tags.

    Requires ctk_string to be pre-computed in ind.tags["ctk_string"].
    """
    to_render = [ind for ind in population if ind.requires_eval and "ctk_string" in ind.tags]

    if not to_render:
        return population

    render_start = time.perf_counter()
    ctk_strings = [ind.tags["ctk_string"] for ind in to_render]

    if NUM_WORKERS > 1 and len(to_render) > 1:
        with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            results = list(executor.map(_render_single, ctk_strings))
    else:
        results = [_render_single(s) for s in ctk_strings]

    # Assign results back to individuals
    for ind, b64_img in zip(to_render, results, strict=False):
        ind.tags["image"] = b64_img

    render_dur = time.perf_counter() - render_start
    n = len(to_render)
    logger.info(
        f"Visuals:Rendered {n} images in {render_dur:.2f}s "
        f"({render_dur / n:.3f}s/ind)",
    )

    return population

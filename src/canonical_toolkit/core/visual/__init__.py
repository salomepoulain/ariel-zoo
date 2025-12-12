"""
Visual toolkit for robot visualization and analysis.

Provides tools for:
- Robot grid plotting with customizable layouts
- Interactive embedding visualizations
- MuJoCo-based robot viewing and rendering
- Image caching and thumbnail generation
"""

# ===== Robot Grid Visualization =====
from canonical_toolkit.core.visual.robot_grid import (
    RobotSubplot,
    plot_robot_grid,
)

# ===== Interactive Embedding Plots =====
from canonical_toolkit.core.visual.embeddings_grid import (
    EmbedSubplot,
    plot_interactive_embed_grid,
)

# ===== Robot Viewer & Rendering =====
from canonical_toolkit.core.visual.viewer import (
    view,
    remove_black_background_and_crop,
    look_at,
    get_camera_params,
)

# ===== Image Caching & Thumbnails =====
from canonical_toolkit.core.visual.viewer import (
    generate_single_image_worker,
    generate_image_cache_with_index,
    load_single_thumbnail,
    load_all_thumbnails,
    load_index,
    cache_exists,
    load_or_generate_cache,
)

__all__ = [
    # Robot grid
    "RobotSubplot",
    "plot_robot_grid",

    # Embedding plots
    "EmbedSubplot",
    "plot_interactive_embed_grid",

    # Viewer
    "view",
    "remove_black_background_and_crop",
    "look_at",
    "get_camera_params",

    # Caching
    "generate_single_image_worker",
    "generate_image_cache_with_index",
    "load_single_thumbnail",
    "load_all_thumbnails",
    "load_index",
    "cache_exists",
    "load_or_generate_cache",
]

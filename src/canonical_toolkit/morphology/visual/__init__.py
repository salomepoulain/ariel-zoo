"""
Visual toolkit for robot visualization and analysis.

Provides tools for:
- Robot grid plotting with customizable layouts
- Interactive embedding visualizations
- MuJoCo-based robot viewing and rendering
- Image caching and thumbnail generation
"""

# # ===== Grid Configuration Dataclasses =====
# from .grid_config import (
#     EmbeddingGridConfig,
#     HeatmapGridConfig,
#     RobotGridConfig,
# )

# # ===== Robot Grid Visualization =====
# from .robot_grid import (
#     plot_robot_grid,
# )

# # ===== Interactive Embedding Plots =====
# from .embeddings_grid import (
#     plot_embedding_grid,
# )

# ===== Robot Viewer & Rendering =====
from .viewer import (
    quick_view,
    RobotViewer,
)

from .snapshots import *

# __all__ = [
#     # Grid configs
#     "EmbeddingGridConfig",
#     "HeatmapGridConfig",
#     "RobotGridConfig",

#     # Robot grid
#     "plot_robot_grid",

#     # Embedding plots
#     "plot_embedding_grid",

#     # Viewer
#     "view",
#     "RobotViewer",
#     # "remove_black_background_and_crop",
#     # "look_at",

#     # # Caching
#     # "generate_image_cache",
#     # "load_thumbnails",
#     # "cache_exists",
#     # "load_or_generate_cache",
# ]

import logging
import warnings

# Suppress Bokeh warnings about missing renderers
warnings.filterwarnings("ignore", message=".*MISSING_RENDERERS.*")
warnings.filterwarnings("ignore", category=UserWarning, module="bokeh")

# Suppress Bokeh logger warnings
logging.getLogger("bokeh").setLevel(logging.ERROR)

from dataclasses import dataclass
from pathlib import Path

# Import the functions from view_mujoco
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console

console = Console()


@dataclass
class RobotSubplot:
    title: str
    under_title: str  # smaller title which is <br> under it
    idxs: list[int]  # idxs to plot in this order

    img_under_title: list[str] | None = None
    img_under_title_fontsize: int = 10

    # Font sizes (matplotlib uses plain numbers, not "10pt")
    axis_label_fontsize: int | str = 8  # int or named size like 'small'
    tick_fontsize: int | str = 6
    title_fontsize: int | str = 10
    under_title_fontsize: int | str = 20

    title_fontweight: str = "normal"  # 'normal' not 'regular'
    under_title_fontweight: str = "normal"


def _stitch_images_with_fixed_spacing(
    images: list[np.ndarray],
    target_height: int,
    gap_px: int = 15,
    img_under_titles: list[str] | None = None,
    text_height_px: int = 25,
    text_fontsize: int = 14,
) -> np.ndarray:
    """
    Stitch images horizontally with FIXED white gaps.
    Images keep their ORIGINAL widths, only HEIGHT is padded to target_height.
    Optionally adds text under each image (aligned at the bottom of tallest image).
    Handles transparency by compositing onto white background.
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Calculate new target height if we're adding text
    actual_target_height = (
        target_height + text_height_px if img_under_titles else target_height
    )

    # Normalize all images to RGB uint8
    normalized_imgs = []

    for i, img in enumerate(images):
        # Handle float 0-1
        if np.issubdtype(img.dtype, np.floating):
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

        # Handle Alpha channel - composite onto white background
        if len(img.shape) == 3 and img.shape[2] == 4:
            rgb = img[:, :, :3]
            alpha = img[:, :, 3:4] / 255.0
            white_bg = np.full_like(rgb, 255, dtype=np.uint8)
            img = (rgb * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        elif len(img.shape) == 3 and img.shape[2] >= 3:
            img = img[:, :, :3]

        normalized_imgs.append(img)

    # Process images and add text AFTER padding to target_height
    processed_imgs = []

    for i, img in enumerate(normalized_imgs):
        h, w = img.shape[:2]

        # First, pad the image to target_height (all images same height)
        if h < target_height:
            pad = np.full((target_height - h, w, 3), 255, dtype=np.uint8)
            img = np.vstack((img, pad))

        # Convert to PIL
        img_pil = Image.fromarray(img)

        # Now add text under the padded image (so all text aligns at same height)
        if img_under_titles and i < len(img_under_titles):
            # Create canvas with extra space for text
            canvas = Image.new(
                "RGB", (w, target_height + text_height_px), (255, 255, 255)
            )
            canvas.paste(img_pil, (0, 0))

            # Draw text at the bottom
            draw = ImageDraw.Draw(canvas)
            text = img_under_titles[i]

            # Try to use a default font with custom size, fallback to PIL default
            try:
                font = ImageFont.truetype(
                    "/System/Library/Fonts/Helvetica.ttc", text_fontsize
                )
            except:
                font = ImageFont.load_default()

            # Get text bounding box for centering
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_x = (w - text_w) // 2

            # Place text right after target_height
            draw.text(
                (text_x, target_height + 5), text, fill=(0, 0, 0), font=font
            )
            img = np.array(canvas)
        else:
            img = np.array(img_pil)

        processed_imgs.append(img)

    # Create white gap column with actual target height
    white_gap = np.full((actual_target_height, gap_px, 3), 255, dtype=np.uint8)

    # Stitch with FIXED gaps
    stitched = None
    for img in processed_imgs:
        if stitched is None:
            stitched = img
        else:
            stitched = np.hstack((stitched, white_gap, img))

    return stitched


def plot_robot_grid(
    sub_plots: list[list[RobotSubplot]],
    cache_dir: str | Path = "__data__/img",
    max_full_width: int = 16,
    subplot_height: int = 2.5,
    main_title: str | None = None,
    robot_gap_px: int = 20,
    dpi: int = 300,
) -> None:
    """uses the cache dir to find the images. if it cant find it, it raises filenotfound error"""
    cache_path = Path(cache_dir)

    # Step 1: Collect all unique robot indices
    all_robot_idxs = set()
    for row_subplots in sub_plots:
        for subplot in row_subplots:
            all_robot_idxs.update(subplot.idxs)

    # Step 2: Load images
    robot_images = {}
    global_max_h = 0

    for robot_idx in all_robot_idxs:
        try:
            img = Image.open(cache_path / f"robot_{robot_idx:04d}.png")
            img_array = np.array(img)
            robot_images[robot_idx] = img_array
            global_max_h = max(global_max_h, img_array.shape[0])
        except FileNotFoundError:
            # console.print(f"[yellow]Warning: robot_{robot_idx:04d}.png not found[/yellow]")
            robot_images[robot_idx] = np.zeros((100, 100, 4), dtype=np.uint8)
            global_max_h = max(global_max_h, 100)

    # Step 3: Prepare stitched images
    n_rows = len(sub_plots)
    n_cols = len(sub_plots[0]) if n_rows > 0 else 0

    stitched_images = []
    global_max_stitched_w = 0

    for row_subplots in sub_plots:
        row_stitched = []
        for subplot in row_subplots:
            images = [robot_images[robot_idx] for robot_idx in subplot.idxs]

            if images:
                stitched = _stitch_images_with_fixed_spacing(
                    images,
                    global_max_h,
                    robot_gap_px,
                    img_under_titles=subplot.img_under_title,
                    text_fontsize=subplot.img_under_title_fontsize,
                )
                row_stitched.append(stitched)
                global_max_stitched_w = max(
                    global_max_stitched_w, stitched.shape[1]
                )
            else:
                row_stitched.append(None)
        stitched_images.append(row_stitched)

    # Step 4: Create grid
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(max_full_width, subplot_height * n_rows),
        squeeze=False,
        facecolor="white",
        dpi=dpi,
    )

    if main_title:
        fig.suptitle(
            main_title, fontsize=16, fontweight="bold", y=0.9, color="black"
        )

    # Step 5: Plot
    for row_idx, row_subplots in enumerate(sub_plots):
        for col_idx, subplot in enumerate(row_subplots):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("white")

            stitched = stitched_images[row_idx][col_idx]

            if stitched is not None:
                actual_width = stitched.shape[1]
                x_offset = (global_max_stitched_w - actual_width) / 2

                ax.imshow(
                    stitched,
                    extent=[
                        x_offset,
                        x_offset + actual_width,
                        global_max_h,
                        0,
                    ],
                )

            ax.set_xlim(0, global_max_stitched_w)
            ax.set_ylim(global_max_h, 0)
            ax.set_aspect("equal")
            ax.axis("off")

            # Add title and under_title
            title_text = f"{subplot.title}\n{subplot.under_title}"
            ax.set_title(
                title_text,
                fontsize=subplot.title_fontsize,
                fontweight=subplot.title_fontweight,
                pad=10,  # <--- INCREASED from 10 to 25 (Space between title and image)
                color="black",
            )

    plt.subplots_adjust(
        wspace=0.1,
        hspace=0.4,
        # CHANGE 2: top=0.85 (was 0.90). Pushes the actual charts down to make room for the lower title.
        top=0.87 if main_title else 0.94,
        bottom=0.04,
        left=0.1,
        right=0.9,
    )
    plt.show()

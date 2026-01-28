from __future__ import annotations

import operator
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self, cast

import numpy as np
import scipy.sparse as sp
from rich.console import Console

from ..options import MatrixDomain
from .s1_matrix import SimilarityMatrix, SimilarityMatrixTags
from .s2_series import SimilaritySeries
from .s3_frame import SimilarityFrame

if TYPE_CHECKING:
    from collections.abc import Callable

    from .transformers import FitTransformer, TransformerGrid

__all__ = ["SimilarityArchive"]


SEED = 42
RNG = np.random.default_rng(SEED)
console = Console()


type ID_MAPPER = dict[int, dict[int, int]]
"""
maps key = generation
local_idx to GLOBAL_ID
"""

type ALIVE_MAPPER = dict[int, dict[int, bool]]
"""
maps key = generation
to global individual being alive or dead at that gen (True == Alive)
"""


class SimilarityArchive:
    """
    Archive of SimilarityFrames across EA generations.

    Supports 3D indexing: [gen, space, radius]
    - gen: generation index/slice
    - space: space label(s) like "FRONT", "BACK", etc.
    - radius: radius index/slice

    Implements "Smart View" slicing:
    - Slicing returns a NEW SimilarityArchive with physically subsetted data.
    - Preserves metadata about the original shape and generation indices.
    """

    def __init__(
        self,
        frames: list[SimilarityFrame],
        id_mapper: ID_MAPPER,
        alive_mapper: ALIVE_MAPPER,
        *,
        # Metadata / State
        gen_indices: list[int] | None = None,
        original_shape: tuple[int, int, int] | None = None,
    ) -> None:
        self._frames = frames
        self._id_mapper = id_mapper
        self._alive_mapper = alive_mapper

        n_radii, n_series = frames[0].shape

        for frame in frames:
            if frame.shape != (n_radii, n_series):
                msg = f"Shape mismatch: {frame.shape} != {(n_series, n_radii)}"
                raise ValueError(
                    msg,
                )

        # Track which original generation each frame corresponds to
        if gen_indices is None:
            self._gen_indices = list(range(len(frames)))
        else:
            assert len(gen_indices) == len(frames), (
                "gen_indices must match frames length"
            )
            self._gen_indices = gen_indices

        # Remember the full original shape for context
        if original_shape is None:
            n_gens = len(frames)
            n_spaces = len(frames[0].labels) if frames else 0
            n_radii = (
                len(frames[0].series[0].radii)
                if frames and frames[0].series
                else 0
            )
            self._original_shape = (n_radii, n_spaces, n_gens)
        else:
            self._original_shape = original_shape

    @property
    def frames(self) -> list[SimilarityFrame]:
        return self._frames.copy()

    @property
    def series(self) -> list[list[SimilaritySeries]]:
        return [frame.series for frame in self.frames]

    @property
    def matrices(self) -> list[list[list[SimilarityMatrix]]]:
        return [frame.matrices for frame in self.frames]

    def grab_matrix(self) -> SimilarityMatrix:
        """Grab single matrix. Shape must be (1, 1, 1)."""
        r, s, g = self.shape
        if (g, s, r) != (1, 1, 1):
            msg = f"Shape must be (1,1,1), got {self.shape}"
            raise ValueError(msg)
        return self._frames[0].matrices[0][0]

    def grab_series(self) -> SimilaritySeries:
        """Grab single series. Shape must be (1, 1, *)."""
        r, s, g = self.shape
        if g != 1 or s != 1:
            msg = f"Shape must be (1,1,*), got {self.shape}"
            raise ValueError(msg)
        return self._frames[0].series[0]

    def grab_frame(self) -> SimilarityFrame:
        """Grab single frame. Shape must be (1, *, *)."""
        if self.shape[2] != 1:
            msg = f"Shape must be (1,*,*), got {self.shape}"
            raise ValueError(msg)
        return self._frames[0]

    @property
    def gens(self) -> list[int]:
        """Indices of the current generations relative to the ORIGINAL archive."""
        return self._gen_indices

    @property
    def spaces(self) -> list[str]:
        """Currently available space labels."""
        if not self._frames:
            return []
        return self._frames[0].labels

    @property
    def radii(self) -> list[int]:
        """Currently available radius indices."""
        if not self._frames:
            return []
        return self._frames[0].series[0].radii

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape of current selection: (n_radii, n_spaces, n_gens)."""
        return (len(self.radii), len(self.spaces), len(self._frames))

    @property
    def original_shape(self) -> tuple[int, int, int]:
        """The shape of the original archive this view was derived from."""
        return self._original_shape

    @property
    def is_2d(self) -> bool:
        """True if exactly one dimension is singleton (size 1)."""
        return self.shape.count(1) >= 1

    @classmethod
    def load(
        cls,
        stem_frame_name: str = "gen_*",
        frame_folder_path: str | Path | None = "__data__/feature_frames/",
        id_mapper: ID_MAPPER | None = None,
        alive_mapper: ALIVE_MAPPER | None = None,
        db_file_path: str | Path | None = "__data__/database.db",
    ) -> SimilarityArchive:
        """
        Loads the archive and uses survivor logic (death_date > current_gen)
        to determine the alive status of individuals.
        """
        import sqlite3

        import pandas as pd

        folder_path = (
            Path(frame_folder_path) if frame_folder_path else Path()
        )
        frame_folders = sorted(folder_path.glob(stem_frame_name))
        frame_folders.pop(-1)
        all_feature_frames = [SimilarityFrame.load(f) for f in frame_folders]
        loaded_gen_indices = [int(f.stem.split("_")[-1]) for f in frame_folders]

        if id_mapper is None or alive_mapper is None:
            import json

            # 1. Database Connection
            conn = sqlite3.connect(str(db_file_path))
            query = (
                "SELECT id, time_of_birth, time_of_death, tags_ FROM individual"
            )
            data = pd.read_sql(query, conn)
            conn.close()

            # 2. Extract ctk_string from tags_ JSON column
            tags_expanded = (
                data["tags_"]
                .apply(lambda x: json.loads(x) if isinstance(x, str) else x)
                .apply(pd.Series)
            )
            data = pd.concat([data, tags_expanded], axis=1)

            # 3. Reconstruct Gen History
            data["gen"] = data.apply(
                lambda row: list(
                    range(
                        int(row["time_of_birth"]), int(row["time_of_death"]) + 1,
                    ),
                ),
                axis=1,
            )

            # THE ESSENTIAL SORT
            gen_df = data.explode("gen").sort_values(
                ["gen", "ctk_string"], ascending=[True, True],
            )
            gen_df["rank"] = gen_df.groupby("gen").cumcount()

            # 4. Build ID Mapper
            if id_mapper is None:
                id_mapper = {}
                for g_idx in loaded_gen_indices:
                    gen_subset = gen_df[gen_df["gen"] == g_idx]
                    id_mapper[g_idx] = {
                        int(rank): int(gid)
                        for rank, gid in zip(
                            gen_subset["rank"], gen_subset["id"], strict=False,
                        )
                    }

            # 5. Build Alive Mapper (Survivor Logic)
            # alive_mapper[gen][global_id] = True if time_of_death > gen
            if alive_mapper is None:
                alive_mapper = {}
                for g_idx in loaded_gen_indices:
                    gen_subset = gen_df[gen_df["gen"] == g_idx]
                    alive_mapper[g_idx] = {
                        int(row["id"]): row["time_of_death"] > g_idx
                        for _, row in gen_subset.iterrows()
                    }

        return SimilarityArchive(
            frames=all_feature_frames,
            id_mapper=id_mapper,
            alive_mapper=alive_mapper,
            gen_indices=loaded_gen_indices,
        )

    # @classmethod
    # def load(
    #     cls,
    #     stem_frame_name: str =  "gen_*",
    #     frame_folder_path: str | Path | None = "__data__/feature_frames/",
    #     id_mapper: ID_MAPPER | None = None,
    #     alive_mapper: ALIVE_MAPPER | None = None,
    #     db_file_path: str | Path | None = "__data__/database.db"
    # ) -> SimilarityArchive:
    #     frame_folders = sorted((frame_folder_path).glob(stem_frame_name))
    #     all_feature_frames = [SimilarityFrame.load(f) for f in frame_folders]

    #     if not id_mapper:
    #         # try to find the database from the db file path

    #         # load id mapper from databse with pandas
    #         # load only alive with pandas

    #     return SimilarityArchive(
    #         frames=all_feature_frames,
    #         id_mapper=
    #     )

    # --- Dimension Info ---

    # --- Selection Methods ---

    def alive_only(self, inplace: bool = True) -> Self:
        """Filter to keep only alive individuals in each generation."""
        new_id_mapper = {k: v.copy() for k, v in self._id_mapper.items()}
        new_frames = list(self._frames)

        for frame_idx, gen_idx in enumerate(self.gens):
            local_to_global = new_id_mapper[gen_idx]
            gen_alive_map = self._alive_mapper.get(gen_idx, {})

            # Sort by local_idx (the keys) to ensure we match NumPy's row order
            ordered_locals = sorted(local_to_global.items())

            alive_globals = [
                g_id
                for l_idx, g_id in ordered_locals
                if gen_alive_map.get(g_id, True)
            ]

            dead_idxs = [
                l_idx
                for l_idx, g_id in ordered_locals
                if not gen_alive_map.get(g_id, True)
            ]

            if dead_idxs:
                # Physically remove rows (this shifts indices!)
                frame = self._frames[frame_idx]
                new_frames[frame_idx] = frame.map(
                    "remove_idxs", idxs=dead_idxs, inplace=inplace,
                )

                # Sync the mapper: Assign new sequential indices (0, 1, 2...)
                # to the global IDs that we kept, in the same order they were in.
                new_id_mapper[gen_idx] = dict(enumerate(alive_globals))

        # new_frames.pop(-1)
        if inplace:
            self._frames = new_frames
            self._id_mapper = new_id_mapper
            return self

        return self.replace(frames=new_frames, id_mapper=new_id_mapper)

    def new_only(self, inplace: bool = True) -> Self:
        """
        Filter to keep individuals only in the first generation they appear in this archive.
        Useful for deduplicating data before fitting transformers.
        """
        new_id_mapper = {k: v.copy() for k, v in self._id_mapper.items()}
        new_frames = list(self._frames)
        seen_global_ids = set()

        for frame_idx, gen_idx in enumerate(self.gens):
            local_to_global = new_id_mapper[gen_idx]

            # Sort by local_idx to match matrix rows
            ordered_locals = sorted(local_to_global.items())

            # Determine which rows to keep (new IDs) and remove (seen IDs)
            unique_globals = []
            dead_idxs = []

            for l_idx, g_id in ordered_locals:
                if g_id not in seen_global_ids:
                    seen_global_ids.add(g_id)
                    unique_globals.append(g_id)
                else:
                    dead_idxs.append(l_idx)

            if dead_idxs:
                frame = self._frames[frame_idx]
                new_frames[frame_idx] = frame.map(
                    "remove_idxs", idxs=dead_idxs, inplace=inplace,
                )

                # Sync the mapper
                new_id_mapper[gen_idx] = dict(enumerate(unique_globals))

        if inplace:
            self._frames = new_frames
            self._id_mapper = new_id_mapper
            return self

        return self.replace(frames=new_frames, id_mapper=new_id_mapper)

    def downsample(self, amount: int = 100, inplace: bool = True) -> Self:
        """
        Randomly selects a maximum of 'amount' individuals from every
        generation and removes the rest.
        """
        # Initialize containers for the new state
        new_id_mapper = {k: v.copy() for k, v in self._id_mapper.items()}
        new_frames = list(self._frames)  # shallow copy of list

        for frame_idx, gen_idx in enumerate(self.gens):
            local_to_global = new_id_mapper[gen_idx]

            # 1. Get current individuals in their correct row order
            items = sorted(local_to_global.items())
            total_count = len(items)

            # If the generation is already small enough, skip it
            if total_count <= amount:
                continue

            # 2. Randomly pick which ones to KEEP
            # We sample from the list of items to preserve the (idx, id) relationship
            keep_sample = RNG.choice(len(items), size=amount, replace=False)
            keep_items = [items[i] for i in keep_sample]

            # Important: Re-sort the kept items by their original local_idx
            # so they stay in the same relative order as the original matrix
            keep_items.sort(key=operator.itemgetter(0))

            # 3. Identify which indices to REMOVE
            keep_local_idxs = {item[0] for item in keep_items}
            dead_idxs = [
                l_idx for l_idx, g_id in items if l_idx not in keep_local_idxs
            ]

            # 4. Perform the physical removal on a copy or inplace
            frame = self._frames[frame_idx]
            new_frames[frame_idx] = frame.map(
                "remove_idxs", idxs=dead_idxs, inplace=inplace,
            )

            # 5. Rebuild the mapper for the new sequential rows
            new_id_mapper[gen_idx] = {
                new_idx: item[1] for new_idx, item in enumerate(keep_items)
            }

        if inplace:
            self._frames = new_frames
            self._id_mapper = new_id_mapper
            return self

        return self.replace(frames=new_frames, id_mapper=new_id_mapper)

    def _normalize_gen(
        self,
        key: int | slice | list[int] | None,
    ) -> list[int] | None:
        """Normalize generation selector to list of LOCAL indices."""
        # Note: This returns indices into self._frames (0..N), not original gen IDs
        n_frames = len(self._frames)
        all_indices = list(range(n_frames))

        if key is None:
            return None

        if isinstance(key, int):
            if key < 0:
                key = n_frames + key
            if key < 0 or key >= n_frames:
                msg = f"Generation index {key} out of bounds"
                raise IndexError(msg)
            return [key]

        if isinstance(key, slice):
            return all_indices[key]

        if isinstance(key, list):
            result = []
            for k in key:
                if k < 0:
                    k = n_frames + k
                if k < 0 or k >= n_frames:
                    msg = f"Generation index {k} out of bounds"
                    raise IndexError(msg)
                result.append(k)
            return result

        msg = f"Invalid gen selector type: {type(key)}"
        raise TypeError(msg)

    # def select(
    #     self,
    #     gen: int | slice | list[int] | None = None,
    #     space: str | list[str] | None = None,
    #     radius: int | slice | list[int] | None = None,
    # ) -> SimilarityArchive:
    #     """
    #     Select a subset of the archive. Returns a NEW SimilarityArchive object.
    #     Data is physically subsetted.
    #     """
    #     # 1. Determine Generation Indices to Keep (Local Indices)
    #     # -----------------------------------------------------
    #     if gen is not None:
    #         local_gen_indices = self._normalize_gen(gen)
    #     else:
    #         local_gen_indices = list(range(len(self._frames)))

    #     # Subset frames list and tracking indices
    #     # We assume local_gen_indices is a list of ints at this point
    #     if local_gen_indices is None: # Should be caught above, but for typing
    #             local_gen_indices = list(range(len(self._frames)))

    #     current_frames = [self._frames[i] for i in local_gen_indices]
    #     current_gen_indices = [self._gen_indices[i] for i in local_gen_indices]

    #     # 2. Slice Space and Radius (Deep Slice)
    #     # --------------------------------------
    #     # We delegate this to SimilarityFrame.__getitem__
    #     # frame[radius_selector, space_selector]

    #     if space is not None or radius is not None:
    #         final_frames = []

    #         # Prepare selectors for Frame
    #         # To ensure we always get a FRAME back (maintaining 3D structure),
    #         # we must ensure selectors are slices or lists, never single int/str.

    #         # Radius (row)
    #         if radius is None:
    #             r_sel = slice(None)
    #         elif isinstance(radius, int):
    #             r_sel = [radius] # Force list to keep dimension
    #         else:
    #             r_sel = radius

    #         # Space (col)
    #         if space is None:
    #             s_sel = slice(None)
    #         elif isinstance(space, str):
    #             s_sel = [space] # Force list to keep dimension
    #         else:
    #             s_sel = space

    #         for f in current_frames:
    #             # Frame.__getitem__ supports tuple (row, col)
    #             subset_frame = f[r_sel, s_sel] # type: ignore
    #             final_frames.append(subset_frame)

    #         current_frames = final_frames

    #     # 3. Return New Archive
    #     # ---------------------
    #     return SimilarityArchive(
    #         frames=current_frames,
    #         id_mapper=self._id_mapper,
    #         alive_mapper=self._alive_mapper,
    #         gen_indices=current_gen_indices,
    #         original_shape=self._original_shape,
    #         ftransformer=self._ftransformer,
    #         alive_only=self._alive_only,
    #     )

    # TODO; in_place
    def select(
        self,
        radius: int | slice | list[int] | None = None,
        space: str
        | list[str]
        | None
        | int
        | slice = None,  # TODO: fix space to also be able to use slice or int
        gen: int | slice | list[int] | None = None,
    ) -> Self:
        """
        Select a subset of the archive. Returns a NEW SimilarityArchive object.
        Physically subsets frames and the parallel transformer grid.
        """
        # --- 1. Determine Generation Indices ---
        if gen is not None:
            local_gen_indices = self._normalize_gen(gen)
        else:
            local_gen_indices = list(range(len(self._frames)))

        # Deep copy frames to avoid shared references
        current_frames = [deepcopy(self._frames[i]) for i in local_gen_indices]
        current_gen_indices = [self._gen_indices[i] for i in local_gen_indices]

        # --- 2. Determine Space and Radius Integer Indices (for Grid Slicing) ---
        # We need these to slice the self._transformers list-of-lists
        n_radii, n_series, gen = self.shape
        keep_s_indices = list(range(n_series))
        keep_r_indices = list(range(n_radii))

        # Resolve Space Names to Indices
        if space is not None:
            if isinstance(space, int):
                keep_s_indices = [space]
            elif isinstance(space, slice):
                keep_s_indices = list(range(n_series)[space])
            elif isinstance(space, str):
                all_s_names = [s.label for s in self._frames[0].series]
                keep_s_indices = [all_s_names.index(space)]
            else:
                # Assume list of strings
                all_s_names = [s.label for s in self._frames[0].series]
                keep_s_indices = [all_s_names.index(s) for s in space]

        # Resolve Radius to Indices
        if radius is not None:
            if isinstance(radius, int):
                keep_r_indices = [radius]
            elif isinstance(radius, slice):
                keep_r_indices = list(range(n_radii)[radius])
            else:
                keep_r_indices = radius

        # --- 3. Slice the Frames ---
        if space is not None or radius is not None:
            final_frames = []

            # Prepare selectors for SimilarityFrame.__getitem__
            # r_sel and s_sel must be lists or slices to preserve 2D structure
            keep_r_indices if not isinstance(radius, slice) else radius
            (
                keep_s_indices if not isinstance(space, (str, list)) else space
            )

            # If space/radius were passed as raw strings/ints,
            # we already converted them to lists in keep_indices.
            for f in current_frames:
                # subset_frame = f[rows, cols]
                # Deep copy the sliced frame to avoid shared references
                subset_frame = deepcopy(f[keep_r_indices, keep_s_indices])
                final_frames.append(subset_frame)

            current_frames = final_frames

        # --- 4. Filter id_mapper to only include selected generations ---
        # Deep copy to avoid shared references
        new_id_mapper = {
            gen_idx: deepcopy(self._id_mapper[gen_idx])
            for gen_idx in current_gen_indices
            if gen_idx in self._id_mapper
        }

        # --- 5. Return the New Archive ---
        return self.replace(
            frames=current_frames,
            gen_indices=current_gen_indices,
            id_mapper=new_id_mapper,
        )

    def __getitem__(self, key: Any) -> SimilarityArchive:
        """3D indexing: archive[radius, space, gen]."""
        if isinstance(key, tuple):
            return self.select(*key)

        return self.select(gen=key)

        # if not isinstance(key, tuple):
        #     return self.select(gen=key)

        # n = len(key)
        # gen = key[0] if n > 0 and key[0] is not None else None
        # space = key[1] if n > 1 and key[1] is not None else None
        # radius = key[2] if n > 2 and key[2] is not None else None

        # # Handle : (slice(None)) as None
        # if isinstance(gen, slice) and gen == slice(None): gen = None
        # if isinstance(space, slice) and space == slice(None): space = None
        # if isinstance(radius, slice) and radius == slice(None): radius = None

        # return self.select(gen=gen, space=space, radius=radius)

    # --- Representation ---

    def _get_representative_frame(self) -> SimilarityFrame | None:
        """Get a frame representing current selection for display."""
        if not self._frames:
            return None
        # Since we physically sliced, just return the first frame!
        return self._frames[0]

    def _add_shadow_layer(
        self,
        lines: list[str],
        plain_lines: list[str],
        h_spacing: int = 2,
        v_spacing: int = 1,
    ) -> tuple[list[str], list[str]]:
        """Add one shadow layer (top edge + right side) to lines."""
        # Extract the rightmost character from each plain line
        right_chars = [pl[-1] for pl in plain_lines]

        # Extract the top edge (first line, excluding corner char)
        top_edge = plain_lines[0][:-1] if plain_lines else ""

        result = []
        result_plain = []

        # Add top edge shifted to the right + corner char
        shadow_top = (
            " " * h_spacing + top_edge + right_chars[0] if right_chars else ""
        )
        # console.print('shadow_top:\n', shadow_top)
        result.append(shadow_top)
        result_plain.append(shadow_top)

        # Each line: original + v_spacing + right char (shifted up by 1)
        for i, line in enumerate(lines):
            right_char = right_chars[i + 1] if i + 1 < len(right_chars) else " "
            new_line = line + " " * v_spacing + right_char
            new_plain = plain_lines[i] + " " * v_spacing + right_char
            result.append(new_line)
            result_plain.append(new_plain)

        return result, result_plain

    def _get_nice_layers(
        self, min_gen: int, max_gen: int,
    ) -> tuple[int, int, list[tuple[int, int]]]:
        """Find nice batch size and number of layers for a generation range.

        Returns (batch_size, n_layers, ranges) where:
        - batch_size is a "nice" number (1, 2, 5, 10, 20, 25, 50...)
        - n_layers is between 2-5 for good visuals
        - ranges is list of (start, end) tuples for each layer
        """
        total_span = max_gen - min_gen + 1

        if total_span <= 1:
            return (1, 1, [(min_gen, max_gen)])

        # Nice divisors to try (prefer rounder numbers)
        nice_numbers = [1, 2, 5, 10, 20, 25, 50, 100]

        # Preferred layer counts
        preferred_layers = [4, 5, 3, 2]

        best_batch = None
        best_n_layers = 4

        for n_layers in preferred_layers:
            target_batch = total_span / n_layers
            # Find closest nice number
            nice_batch = min(nice_numbers, key=lambda n: abs(target_batch - n))
            actual_layers = max(1, total_span // nice_batch)

            if 2 <= actual_layers <= 5:
                best_batch = nice_batch
                best_n_layers = actual_layers
                break

        # Fallback
        if best_batch is None:
            best_batch = max(1, total_span // 4)
            best_n_layers = min(4, total_span)

        # Generate ranges starting from min_gen
        ranges = []
        for i in range(best_n_layers):
            start = min_gen + i * best_batch
            end = min(min_gen + (i + 1) * best_batch - 1, max_gen)
            ranges.append((start, end))

        # Make sure last range goes to max_gen
        if ranges:
            ranges[-1] = (ranges[-1][0], max_gen)

        return (best_batch, best_n_layers, ranges)

    def _add_3d_depth(
        self,
        table_str: str,
        gen_ranges: list[tuple[int, int]] | None = None,
        layers: int = 1,
        h_spacing: int = 2,
        v_spacing: int = 1,
    ) -> str:
        """Wrap a table string with 3D depth effect."""
        from rich.text import Text

        lines = table_str.split("\n")
        lines = [line for line in lines if line.strip()]

        if not lines:
            return table_str

        # Remove title line(s) - lines before the first box character (┏ or ┌)
        while lines and not any(c in lines[0] for c in "┏┌╭"):
            lines.pop(0)

        if not lines:
            return table_str

        # Get plain text versions to find edge characters
        plain_lines = [Text.from_ansi(line).plain for line in lines]

        # Apply shadow layers recursively
        for _ in range(layers):
            lines, plain_lines = self._add_shadow_layer(
                lines, plain_lines, h_spacing, v_spacing,
            )

        # Add generation range labels to each shadow layer
        if gen_ranges and layers > 0:
            # Labels go on the right side, one per range
            for i, (start, end) in enumerate(gen_ranges):
                if i + 1 > len(lines):
                    break
                if i == 0:
                    # Note: These are now typically arbitrary ranges of indices if the list is filtered
                    # We might want to use self.gens[start] to show real Gen ID
                    # For simplicity, we show indices within current view
                    # Or better: show actual gen IDs!

                    # Map loop index 'start' (which is view index) to real gen ID
                    # But gen_ranges is (start_idx, end_idx) in VIEW space
                    real_start = (
                        self.gens[start] if start < len(self.gens) else start
                    )

                    label = f"← gen {real_start}"
                    offset = i * h_spacing
                    lines[-(i + 1)] = lines[-(i + 1)] + " " * offset + label
                if i == len(gen_ranges) - 1:
                    real_end = self.gens[end] if end < len(self.gens) else end
                    label = f"← gen {real_end} | {self.original_shape[0] - 1}"
                    offset = i * h_spacing
                    lines[-(i + 1)] = lines[-(i + 1)] + " " * offset + label
        else:
            if self.gens:
                label = f"← gen {self.gens[0]} | {self.original_shape[0] - 1}"
            else:
                label = "← (empty)"
            lines[-1] = lines[-1] + " " + label

        return "\n".join(lines)

    def _format_filter_summary(self) -> str:
        """Format the VIEW and FULL filter summary with aligned columns."""
        # FULL archive info (from metadata)
        full_gens = self.original_shape[0]
        # We don't have full space/radii list stored in metadata, but we can infer counts
        # Or just show current vs total count

        # Current VIEW info
        view_gens = self.gens
        view_spaces = self.spaces
        view_radii = self.radii

        # Format gens
        str(full_gens)
        if len(view_gens) == 1:
            view_gen_str = str(view_gens[0])
        elif view_gens:
            view_gen_str = f"{len(view_gens)} selected"  # Simplified
        else:
            view_gen_str = "none"

        # Format spaces
        if len(view_spaces) <= 3:
            view_space_str = ", ".join(view_spaces)
        else:
            view_space_str = f"{len(view_spaces)} spaces"

        # Format radii as range
        if len(view_radii) == 1:
            view_radii_str = str(view_radii[0])
        elif view_radii:
            view_radii_str = f"{min(view_radii)}-{max(view_radii)}"
        else:
            view_radii_str = "none"

        return f"Selection: {view_gen_str} gens | {view_space_str} | radii {view_radii_str}"

    def __repr__(self) -> str:
        """Show 3D structure with current selection."""
        # Get the representative frame's repr
        frame = self._get_representative_frame()
        if frame is None:
            return (
                f"SimilarityArchive [Empty] (Original: {self.original_shape})"
            )

        frame_repr = repr(frame)

        # print(repr(frame_repr[:100]))

        # Add 3D depth effect
        # We construct ranges based on current view length
        n_view_gens = len(self._frames)
        if n_view_gens > 1:
            # Just mock ranges 0..N
            _, n_ranges, gen_ranges = self._get_nice_layers(0, n_view_gens - 1)
            # Shadow layers = n_ranges - 1
            result = self._add_3d_depth(
                frame_repr, gen_ranges, layers=n_ranges - 1,
            )
        else:
            result = self._add_3d_depth(frame_repr, gen_ranges=None, layers=0)

        # Add header and footer info
        lines = [result]

        # print(lines)

        lines.append(f"{self.shape} | {self.original_shape} (Original)")

        # lines.append('test')
        return "\n".join(lines)

    # --- Transformation & Helpers ---

    def replace(
        self,
        frames: list[SimilarityFrame] | None = None,
        id_mapper: ID_MAPPER | None = None,
        alive_mapper: ALIVE_MAPPER | None = None,
        *,
        # Metadata / State
        gen_indices: list[int] | None = None,
        original_shape: tuple[int, int, int] | None = None,
    ) -> Self:
        """Helper method to create a new instance based on self instance."""
        new_frames = self.frames if frames is None else frames
        new_id_mapper = (
            deepcopy(self._id_mapper) if id_mapper is None else id_mapper
        )
        new_alive_mapper = (
            deepcopy(self._alive_mapper)
            if alive_mapper is None
            else alive_mapper
        )

        new_gen_indices = (
            self._gen_indices.copy() if gen_indices is None else gen_indices
        )
        new_original_shape = (
            self._original_shape if original_shape is None else original_shape
        )

        return self.__class__(
            frames=new_frames,
            id_mapper=new_id_mapper,
            alive_mapper=new_alive_mapper,
            gen_indices=new_gen_indices,
            original_shape=new_original_shape,
        )

    # --- Fit / Transform ---

    def fit(
        self,
        transformer: FitTransformer,
    ) -> FitTransformer:
        """
        Fit a transformer on stacked data from all frames.

        The archive should be sliced to a single space/radius before calling.
        Stacks matrices across all generations and fits the transformer.

        Parameters
        ----------
        transformer : FitTransformer
            An unfitted sklearn-style transformer (UMAP, PCA, etc.)

        Returns
        -------
        FitTransformer
            The same transformer, now fitted to the stacked data.

        Example
        -------
        >>> subset = archive[:, "FRONT", 0].alive_only().new_only()
        >>> from umap import UMAP
        >>> fitted = subset.fit_transformer(UMAP(metric="cosine"))
        """
        n_radii, n_series, _ = self.shape

        if n_series != 1 or n_radii != 1:
            msg = (
                f"Archive must be sliced to single space/radius before fitting. "
                f"Current shape: {self.shape} (need [*, 1, 1])"
            )
            raise ValueError(
                msg,
            )

        # Stack matrices from all generations
        blocks = [f.matrices[0][0].matrix for f in self._frames]

        if any(sp.issparse(b) for b in blocks):
            X = sp.vstack(blocks, format="csr")
        else:
            X = np.vstack(blocks)

        transformer.fit(X)
        return transformer

    def fit_grid(
        self, transformer_grid: TransformerGrid,
    ) -> TransformerGrid:
        """In place modification/updating for the grid of transformers."""
        n_radii, n_spaces, _ = self.shape

        if transformer_grid.shape != (n_radii, n_spaces):
            msg = (
                f"TransformerGrid {transformer_grid.shape} must be "
                f"(n_radii, n_spaces) = ({n_radii}, {n_spaces})"
            )
            raise IndexError(msg)

        if not transformer_grid.is_filled:
            msg = "TransformerGrid is not fully filled with transformers"
            raise ValueError(msg)

        for s_idx in range(n_spaces):
            for r_idx in range(n_radii):
                blocks = [
                    frame.matrices[s_idx][r_idx].matrix
                    for frame in self._frames
                ]

                if any(sp.issparse(b) for b in blocks):
                    X = sp.vstack(blocks, format="csr")
                else:
                    X = np.vstack(blocks)

                transformer = transformer_grid[r_idx, s_idx]
                cast("FitTransformer", transformer)
                transformer.fit(X)

        return transformer_grid

    def transform_grid(
        self,
        transformer_grid: TransformerGrid,
        unique_only: bool = True,
        inplace: bool = True,
    ) -> Self:
        """Apply fitted transformer grid to all (radius, space) cells.

        Parameters
        ----------
        transformer_grid : TransformerGrid
            Grid of fitted transformers with shape (n_radii, n_spaces).
        unique_only : bool, optional
            If True (default), transforms unique individuals once and maps
            coordinates back to all generations. Ensures consistent coordinates.
            If False, transforms each generation independently.
        inplace : bool, optional
            If True, modifies frames in place. Default is False.

        Returns
        -------
        SimilarityArchive
            Archive with transformed (embedding) matrices.
        """
        n_radii, n_spaces, _ = self.shape

        if transformer_grid.shape != (n_radii, n_spaces):
            msg = (
                f"TransformerGrid {transformer_grid.shape} must be "
                f"(n_radii, n_spaces) = ({n_radii}, {n_spaces})"
            )
            raise IndexError(msg)

        # Build new frames structure (deep copy if not inplace)
        if inplace:
            new_frames = self._frames
        else:
            new_frames = [deepcopy(f) for f in self._frames]

        if not unique_only:
            # Simple path: transform each generation independently
            for s_idx in range(n_spaces):
                for r_idx in range(n_radii):
                    transformer = transformer_grid[r_idx, s_idx]

                    for frame_idx, frame in enumerate(new_frames):
                        original_matrix = self._frames[frame_idx].matrices[s_idx][r_idx]
                        data = original_matrix.matrix

                        embeddings = transformer.transform(data)
                        if embeddings.ndim == 1:
                            embeddings = embeddings.reshape(-1, 1)

                        tags = SimilarityMatrixTags(
                            domain=MatrixDomain.EMBEDDING,
                            radius=original_matrix.radius,
                            is_gap=False,
                        )
                        new_matrix = SimilarityMatrix(
                            embeddings, original_matrix.label, tags,
                        )
                        frame._series[s_idx]._matrices[r_idx] = new_matrix

            if inplace:
                return self
            return self.replace(frames=new_frames)

        # Unique path: transform once, map back to all generations
        first_occurrence: dict[int, tuple[int, int]] = {}
        for frame_idx, gen_idx in enumerate(self._gen_indices):
            mapper = self._id_mapper[gen_idx]
            for local_idx, global_id in mapper.items():
                if global_id not in first_occurrence:
                    first_occurrence[global_id] = (frame_idx, local_idx)

        unique_ids = list(first_occurrence.keys())
        id_to_row = {gid: i for i, gid in enumerate(unique_ids)}

        for s_idx in range(n_spaces):
            for r_idx in range(n_radii):
                transformer = transformer_grid[r_idx, s_idx]

                # Extract unique rows for this cell
                rows = []
                for global_id in unique_ids:
                    frame_idx, local_idx = first_occurrence[global_id]
                    matrix = self._frames[frame_idx].matrices[s_idx][r_idx].matrix
                    rows.append(matrix[local_idx])

                # Stack
                if rows and sp.issparse(rows[0]):
                    unique_data = sp.vstack(rows, format="csr")
                else:
                    unique_data = np.vstack(rows)

                # Transform ONCE
                embeddings = transformer.transform(unique_data)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(-1, 1)

                # Reconstruct per-generation matrices
                for frame_idx, (frame, gen_idx) in enumerate(
                    zip(new_frames, self._gen_indices, strict=False)
                ):
                    mapper = self._id_mapper[gen_idx]
                    n_individuals = len(mapper)

                    embed_rows = [id_to_row[mapper[i]] for i in range(n_individuals)]
                    gen_embedding = embeddings[embed_rows]

                    original_matrix = self._frames[frame_idx].matrices[s_idx][r_idx]
                    tags = SimilarityMatrixTags(
                        domain=MatrixDomain.EMBEDDING,
                        radius=original_matrix.radius,
                        is_gap=False,
                    )
                    new_matrix = SimilarityMatrix(
                        gen_embedding, original_matrix.label, tags,
                    )
                    frame._series[s_idx]._matrices[r_idx] = new_matrix

        if inplace:
            return self

        return self.replace(frames=new_frames)

    def transform(
        self,
        transformer: FitTransformer,
        unique_only: bool = True,
        inplace: bool = True,
    ) -> Self:
        """
        Transform unique individuals once, map coordinates back to all generations.

        This solves UMAP's stochastic transform issue by ensuring each individual
        gets exactly one coordinate, regardless of how many generations it appears in.

        The archive should be sliced to a single space/radius before calling.

        Parameters
        ----------
        transformer : FitTransformer
            A fitted sklearn-style transformer.
        inplace : bool, optional
            If True, modifies frames in place. Default is False.

        Returns
        -------
        SimilarityArchive
            Archive with transformed (embedding) matrices where each individual
            has consistent coordinates across all generations.

        Example
        -------
        >>> fitted = subset.fit_transformer(UMAP(metric="cosine"))
        >>> embedded = archive[:, "FRONT", 0].transform_unique(fitted)
        """
        n_radii, n_series, g = self.shape

        if n_series != 1 or n_radii != 1:
            msg = (
                f"Archive must be sliced to single space/radius before transforming. "
                f"Current shape: {self.shape} (need [*, 1, 1])"
            )
            raise ValueError(
                msg,
            )

        if not unique_only:
            return self.map(
                "transform_embed", transformer=transformer, inplace=inplace,
            )

        # 1. Find first occurrence of each unique individual: global_id → (frame_idx, local_idx)
        first_occurrence: dict[int, tuple[int, int]] = {}
        for frame_idx, gen_idx in enumerate(self._gen_indices):
            mapper = self._id_mapper[gen_idx]
            for local_idx, global_id in mapper.items():
                if global_id not in first_occurrence:
                    first_occurrence[global_id] = (frame_idx, local_idx)

        unique_ids = list(first_occurrence.keys())

        # 2. Extract rows in order (no toarray - keep sparse if sparse)
        rows = []
        for global_id in unique_ids:
            frame_idx, local_idx = first_occurrence[global_id]
            matrix = self._frames[frame_idx].matrices[0][0].matrix
            rows.append(matrix[local_idx])

        # 3. Stack rows (sparse vstack if sparse, else numpy vstack)
        if rows and sp.issparse(rows[0]):
            unique_data = sp.vstack(rows, format="csr")
        else:
            unique_data = np.vstack(rows)

        # 4. Transform ONCE
        embeddings = transformer.transform(unique_data)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)

        # 5. Build lookup: global_id → row index in embeddings
        id_to_row = {gid: i for i, gid in enumerate(unique_ids)}

        # 6. Reconstruct per-generation matrices using fancy indexing
        embeddings.shape[1]
        new_frames = []

        for frame, gen_idx in zip(self._frames, self._gen_indices, strict=False):
            mapper = self._id_mapper[gen_idx]
            n_individuals = len(mapper)

            # Map local indices to embedding rows via global_id
            embed_rows = [id_to_row[mapper[i]] for i in range(n_individuals)]
            gen_embedding = embeddings[embed_rows]

            # Wrap in SimilarityMatrix with EMBEDDING domain
            original_matrix = frame.matrices[0][0]
            tags = SimilarityMatrixTags(
                domain=MatrixDomain.EMBEDDING,
                radius=original_matrix.radius,
                is_gap=False,
            )
            new_matrix = SimilarityMatrix(
                gen_embedding, original_matrix.label, tags,
            )
            new_series = SimilaritySeries([new_matrix], frame.series[0].label)
            new_frames.append(SimilarityFrame([new_series]))

        if inplace:
            self._frames = new_frames
            return self

        return self.replace(frames=new_frames)

    def map(self, func: Callable | str, **kwargs: Any) -> Self:
        new_data = []
        for frame in self._frames:
            if len(frame.series) == 1:
                if (
                    isinstance(func, str)
                    and hasattr(frame.series[0], func)
                    and callable(getattr(frame.series[0], func))
                ):
                    result = getattr(frame.series[0], func)(**kwargs)
                else:
                    result = frame.series[0].map(func, **kwargs)
            elif (
                isinstance(func, str)
                and hasattr(frame, func)
                and callable(getattr(frame, func))
            ):
                result = getattr(frame, func)(**kwargs)
            else:
                result = frame.map(func, **kwargs)

            # If result is a Matrix (e.g., from aggregate on single series), wrap it
            if isinstance(result, SimilarityMatrix):
                result = SimilaritySeries(matrices=[result])

            # If result is a Series (e.g., from aggregate), wrap it back into a Frame
            if isinstance(result, SimilaritySeries):
                result = SimilarityFrame(series=[result])

            new_data.append(result)

        if kwargs.get("inplace"):
            self._frames = new_data
            return self
        return self.replace(frames=new_data)

    def get_2d_data(self) -> list[list[np.ndarray | sp.spmatrix]]:
        """
        Extracts a 2D grid of raw matrices from a 3D archive where
        one dimension has been reduced to size 1.
        """
        r, s, g = self.shape

        # 1. Check which dimension is 'flat' (size 1)
        dims = {"gens": g, "spaces": s, "radii": r}
        ones = [k for k, v in dims.items() if v == 1]

        if len(ones) == 0:
            msg = f"Archive must be subsetted to a 2D slice first. Current shape: {g}x{s}x{r}"
            raise KeyError(
                msg,
            )

        # We take the first dimension that is 1 as our "collapsed" axis
        collapsed = ones[0]

        result = []

        # 2. Logic for Gen-Space grid (Radius is 1)
        if collapsed == "radii":
            # Rows: Generations | Cols: Spaces
            for f_idx in range(g):
                # radius is fixed at index 0
                row = [self._frames[f_idx].matrices[s_idx][0].matrix for s_idx in range(s)]
                result.append(row)

        # 3. Logic for Gen-Radius grid (Space is 1)
        elif collapsed == "spaces":
            # Rows: Generations | Cols: Radii
            for f_idx in range(g):
                row = []
                for r_idx in range(r):
                    # space is fixed at index 0
                    row.append(self._frames[f_idx].matrices[0][r_idx].matrix)
                result.append(row)

        # 4. Logic for Space-Radius grid (Gen is 1)
        elif collapsed == "gens":
            # Rows: Spaces | Cols: Radii
            # We only have one frame (index 0)
            frame = self._frames[0]
            for s_idx in range(s):
                row = []
                for r_idx in range(r):
                    row.append(frame.matrices[s_idx][r_idx].matrix)
                result.append(row)

        return result

    def get_2d_ids(self) -> list[list[list[int] | list[tuple[int, int]]]]:
        """
        Extracts a 2D grid of Global IDs corresponding to the matrix indices.
        Maps local row/col indices to the global population IDs using self._id_mapper.
        """
        r, s, g = self.shape
        dims = {"gens": g, "spaces": s, "radii": r}
        ones = [k for k, v in dims.items() if v == 1]

        if not ones:
            msg = f"Archive must be subsetted to a 2D slice first. Shape: {g}x{s}x{r}"
            raise KeyError(
                msg,
            )

        collapsed = ones[0]
        result = []

        # Helper to map a matrix to Global IDs based on its domain
        def map_matrix_to_global(frame_idx: int, s_idx: int, r_idx: int):
            mat_obj = self._frames[frame_idx].matrices[s_idx][r_idx]
            # Get the original gen index to look up the correct mapper
            orig_gen = self._gen_indices[frame_idx]
            mapper = self._id_mapper[orig_gen]

            # Use your logic: 1D for Features/Embedding, 2D for Similarity
            if mat_obj.domain in {
                MatrixDomain.FEATURES.name,
                MatrixDomain.EMBEDDING.name,
            }:
                # Map row index i -> Global ID
                return [mapper[i] for i in range(mat_obj.shape[0])]

            if mat_obj.domain == MatrixDomain.SIMILARITY.name:
                # Map (row, col) -> (Global ID A, Global ID B)
                N = mat_obj.shape[0]
                return [
                    [(mapper[r], mapper[c]) for c in range(N)] for r in range(N)
                ]

            msg = f"Unknown domain: {mat_obj.domain}"
            raise ValueError(msg)

        # --- Grid Extraction Logic ---

        if collapsed == "radii":
            for f_idx in range(g):
                row = [
                    map_matrix_to_global(f_idx, s_idx, 0) for s_idx in range(s)
                ]
                result.append(row)

        elif collapsed == "spaces":
            for f_idx in range(g):
                row = [
                    map_matrix_to_global(f_idx, 0, r_idx) for r_idx in range(r)
                ]
                result.append(row)

        elif collapsed == "gens":
            for s_idx in range(s):
                row = [
                    map_matrix_to_global(0, s_idx, r_idx) for r_idx in range(r)
                ]
                result.append(row)

        return result

    def get_2d_titles(self) -> list[list[str]]:
        """
        Extract a 2D grid of descriptive titles for a subsetted archive.

        One dimension must have been reduced to size 1 (collapsed) to produce a 2D grid.

        Returns
        -------
        list[list[str]]
            A 2D grid of titles in 'r:{radius}, s:{space}, gen:{gen}' format.

        Raises
        ------
        KeyError
            If the archive has not been subsetted to a 2D slice first.
        """
        r, s, g = self.shape
        dims = {"gens": g, "spaces": s, "radii": r}
        ones = [k for k, v in dims.items() if v == 1]

        if not ones:
            msg = f"Archive must be subsetted to a 2D slice first. Shape: {g}x{s}x{r}"
            raise KeyError(msg)

        collapsed = ones[0]
        result = []

        if collapsed == "radii":
            # Rows: Generations | Cols: Spaces
            radius_val = self.radii[0]
            for f_idx in range(g):
                gen_val = self.gens[f_idx]
                row = [
                    f"r{radius_val}, {space_val}, gen:{gen_val}"
                    for space_val in self.spaces
                ]
                result.append(row)

        elif collapsed == "spaces":
            # Rows: Generations | Cols: Radii
            space_val = self.spaces[0]
            for f_idx in range(g):
                gen_val = self.gens[f_idx]
                row = [
                    f"r{radius_val}, {space_val}, gen:{gen_val}"
                    for radius_val in self.radii
                ]
                result.append(row)

        elif collapsed == "gens":
            # Rows: Spaces | Cols: Radii
            gen_val = self.gens[0]
            for space_val in self.spaces:
                row = [
                    f"r{radius_val}, {space_val}, gen:{gen_val}"
                    for radius_val in self.radii
                ]
                result.append(row)

        return result


if __name__ == "__main__":
    import scipy.sparse as sp

    from .s1_matrix import SimilarityMatrix, SimilarityMatrixTags
    from .s2_series import SimilaritySeries
    from .s3_frame import SimilarityFrame

    # --- Mock Data Factory ---
    def make_mock_matrix(
        space: str, radius: int, n_individuals: int = 10,
    ) -> SimilarityMatrix:
        """Create a mock feature matrix."""
        n_features = (radius + 1) * 5  # More features at higher radii
        matrix = sp.random(n_individuals, n_features, density=0.3, format="csr")
        tags = SimilarityMatrixTags(
            domain=MatrixDomain.FEATURES, radius=radius, is_gap=False,
        )
        return SimilarityMatrix(matrix=matrix, label=space, tags=tags)

    def make_mock_series(
        space: str, max_radius: int = 5, n_individuals: int = 10,
    ) -> SimilaritySeries:
        """Create a mock series with matrices at each radius."""
        matrices = [
            make_mock_matrix(space, r, n_individuals)
            for r in range(max_radius + 1)
        ]
        return SimilaritySeries(matrices=matrices, label=space)

    def make_mock_frame(
        spaces: list[str], max_radius: int = 5, n_individuals: int = 10,
    ) -> SimilarityFrame:
        """Create a mock frame with series for each space."""
        series_list = [
            make_mock_series(s, max_radius, n_individuals) for s in spaces
        ]
        return SimilarityFrame(series=series_list)

    def make_mock_archive(
        n_generations: int = 20,
        spaces: list[str] | None = None,
        max_radius: int = 5,
        pop_size: int = 10,
    ) -> SimilarityArchive:
        """Create a mock archive with multiple generations."""
        if spaces is None:
            spaces = ["LEFT", "RIGHT", "TOP", "BOTTOM", "FRONT", "BACK"]

        frames = [
            make_mock_frame(spaces, max_radius, pop_size)
            for _ in range(n_generations)
        ]

        # Mock ID mapper: gen -> {matrix_row_idx -> global_id}
        id_mapper: ID_MAPPER = {}
        global_id = 0
        for gen in range(n_generations):
            id_mapper[gen] = {}
            for row_idx in range(pop_size):
                id_mapper[gen][row_idx] = global_id
                global_id += 1

        # Mock alive mapper: global_id -> is_alive (all alive for simplicity)
        alive_mapper: ALIVE_MAPPER = dict.fromkeys(range(global_id), True)

        return SimilarityArchive(
            frames=frames, id_mapper=id_mapper, alive_mapper=alive_mapper,
        )

    # --- Test ---
    archive = make_mock_archive(n_generations=20, pop_size=15)

    sliced = archive[5]

    sliced = archive[0:10]

    sliced = archive[:, "FRONT"]

    sliced = archive[5, "FRONT", 2]

    sliced = archive[:, ["FRONT", "BACK"], 0:3]

    sliced = archive[0:10].select(space="FRONT").select(radius=2)

    sliced = archive[-1]

    # 1. Collapsed Radii (Gen x Space)
    sliced_gs = archive[:, :, 0]
    titles_gs = sliced_gs.get_2d_titles()
    for _row in titles_gs[:2]:  # Show first 2 gens
        pass  # Show first 2 spaces

    # 2. Collapsed Spaces (Gen x Radius)
    sliced_gr = archive[:, "FRONT", :]
    titles_gr = sliced_gr.get_2d_titles()
    for _row in titles_gr[:2]:  # Show first 2 gens
        pass  # Show first 2 radii

    # 3. Collapsed Gens (Space x Radius)
    sliced_sr = archive[0, :, :]
    titles_sr = sliced_sr.get_2d_titles()
    for _row in titles_sr[:2]:  # Show first 2 spaces
        pass  # Show first 2 radii

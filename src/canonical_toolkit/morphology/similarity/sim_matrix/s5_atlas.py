"""
Only implemented not tested
"""

from __future__ import annotations

import operator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import scipy.sparse as sp

from ..options import MatrixDomain
from .s1_matrix import SimilarityMatrix, SimilarityMatrixTags
from .s2_series import SimilaritySeries
from .s3_frame import SimilarityFrame
from .s4_archive import SimilarityArchive
from .transformers import FitTransformer, TransformerGrid

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

__all__ = ["SimilarityAtlas"]


type ArchiveKey = str | int | slice | list[str] | list[int]
type DimSelector = int | slice | list[int] | None
type SpaceSelector = str | list[str] | int | slice | None


@dataclass
class SimilarityAtlas:
    """Collection of named SimilarityArchives with unified operations.

    Usefull for mainting Multiple Runs

    Supports 4D indexing: [radius, space, gen, archive_id]
    - Non-tuple indexing selects archives (by name, index, slice, or list)
    - Tuple indexing slices all archives: (radius, space, gen, archive_id)

    Example
    -------
    >>> atlas = SimilarityAtlas({"run_a": archive_a, "run_b": archive_b})
    >>> atlas["run_a"]  # Single archive by name
    >>> atlas[0]  # Single archive by index
    >>> atlas[0:2]  # First 2 archives → Atlas
    >>> atlas[0, "FRONT"]  # radius=0, space=FRONT, all archives → Atlas
    >>> atlas[0, "FRONT", :, "run_a"]  # Sliced single archive → Archive
    """

    _archives: dict[str, SimilarityArchive] = field(default_factory=dict)

    # --- Container interface ---

    def __len__(self) -> int:
        return len(self._archives)

    def __iter__(self) -> Iterator[str]:
        return iter(self._archives)

    def __contains__(self, key: str) -> bool:
        return key in self._archives

    def keys(self):
        return self._archives.keys()

    def values(self):
        return self._archives.values()

    def items(self):
        return self._archives.items()

    # --- Slicing ---

    def _select_archives(self, key: ArchiveKey) -> dict[str, SimilarityArchive]:
        """Select subset of archives by key."""
        names = list(self._archives.keys())

        if isinstance(key, slice):
            selected_names = names[key]
        elif isinstance(key, int):
            selected_names = [names[key]]
        elif isinstance(key, list):
            if key and isinstance(key[0], int):
                selected_names = [names[i] for i in key]
            else:
                selected_names = key  # list of str
        else:
            selected_names = names

        return {n: self._archives[n] for n in selected_names}

    def __getitem__(
        self,
        key: ArchiveKey | tuple,
    ) -> SimilarityArchive | Self:
        # Non-tuple: archive selector
        if not isinstance(key, tuple):
            if isinstance(key, str):
                return self._archives[key]
            if isinstance(key, int):
                name = list(self._archives.keys())[key]
                return self._archives[name]
            # slice or list → subset of archives
            return self.replace(archives=self._select_archives(key))

        # Tuple: (radius, space, gen, archive_id)
        selectors = list(key) + [None] * (4 - len(key))
        radius, space, gen, arch_key = selectors

        # Select archives
        if arch_key is not None:
            if isinstance(arch_key, str):
                # Single archive by name → return Archive
                archive = self._archives[arch_key]
                return archive.select(radius=radius, space=space, gen=gen)
            selected = self._select_archives(arch_key)
        else:
            selected = dict(self._archives)

        # Apply (radius, space, gen) to each archive
        if radius is not None or space is not None or gen is not None:
            selected = {
                name: arch.select(radius=radius, space=space, gen=gen)
                for name, arch in selected.items()
            }

        return self.replace(archives=selected)

    def __setitem__(self, key: str, archive: SimilarityArchive) -> None:
        self._archives[key] = archive

    # --- Map ---

    def map(self, func: str | Callable[..., Any], **kwargs: Any) -> Self:
        """Apply function/method to all archives.

        Parameters
        ----------
        func : str | Callable
            If str, calls the method with that name on each archive.
            If Callable, calls func(archive, **kwargs) for each.
        **kwargs
            Arguments passed to the function/method.

        Returns
        -------
        SimilarityAtlas
            New atlas with transformed archives.

        Example
        -------
        >>> atlas.map("new_only")
        >>> atlas.map("select", space="FRONT", radius=0)
        >>> atlas.map("alive_only")
        """
        new_archives = {}
        for name, archive in self._archives.items():
            if isinstance(func, str):
                method = getattr(archive, func)
                new_archives[name] = method(**kwargs)
            else:
                new_archives[name] = func(archive, **kwargs)
        return self.replace(archives=new_archives)

    # --- Fit / Transform (Single Transformer) ---

    def fit_combined(
        self,
        transformer: FitTransformer,
    ) -> FitTransformer:
        """Fit transformer on combined unique data from all archives.

        Each archive is deduplicated internally (new_only), then stacked.
        Archives must be sliced to single space/radius before calling.

        Parameters
        ----------
        transformer : FitTransformer
            Unfitted transformer (UMAP, PCA, etc.)

        Returns
        -------
        FitTransformer
            The same transformer, now fitted.

        Example
        -------
        >>> sliced = atlas[0, "FRONT"]  # radius=0, space=FRONT
        >>> fitted = sliced.fit_combined(UMAP(metric="cosine"))
        >>> embedded = sliced.transform_unique(fitted)
        """
        all_blocks = []

        for name, archive in self._archives.items():
            _, n_series, n_radii = archive.shape
            if n_series != 1 or n_radii != 1:
                msg = (
                    f"Archive '{name}' must be sliced to single space/radius. "
                    f"Shape: {archive.shape}"
                )
                raise ValueError(
                    msg,
                )

            # Deduplicate within this archive
            unique = archive.new_only(inplace=False)

            # Stack matrices from all generations
            blocks = [f.matrices[0][0].matrix for f in unique.frames]
            if blocks:
                if sp.issparse(blocks[0]):
                    all_blocks.append(sp.vstack(blocks, format="csr"))
                else:
                    all_blocks.append(np.vstack(blocks))

        if not all_blocks:
            msg = "No data to fit on (empty archives)"
            raise ValueError(msg)

        # Combine all archives
        if sp.issparse(all_blocks[0]):
            combined = sp.vstack(all_blocks, format="csr")
        else:
            combined = np.vstack(all_blocks)

        transformer.fit(combined)
        return transformer

    def transform_unique(
        self,
        transformer: FitTransformer,
        inplace: bool = False,
    ) -> Self:
        """Apply fitted transformer to all archives.

        Uses transform_unique on each archive to ensure consistent
        coordinates for individuals across generations.

        Parameters
        ----------
        transformer : FitTransformer
            A fitted transformer.
        inplace : bool, optional
            If True, modifies archives in place. Default is False.

        Returns
        -------
        SimilarityAtlas
            Atlas with transformed (embedded) archives.
        """
        return self.map(
            "transform_unique", transformer=transformer, inplace=inplace,
        )

    # --- Fit / Transform (Grid) ---

    def fit_transformer_grid(
        self,
        transformer_grid: TransformerGrid,
    ) -> TransformerGrid:
        """Fit transformer grid on combined data from all archives.

        For each (space, radius) cell, fits the transformer on combined
        deduplicated data from all archives at that cell.

        Parameters
        ----------
        transformer_grid : TransformerGrid
            Grid of unfitted transformers matching (n_spaces, n_radii).

        Returns
        -------
        TransformerGrid
            The same grid, now with fitted transformers.

        Example
        -------
        >>> grid = TransformerGrid((6, 5))
        >>> grid[:, :] = UMAP(metric="cosine")
        >>> fitted_grid = atlas.fit_transformer_grid(grid)
        >>> embedded = atlas.transform_grid(fitted_grid)
        """
        # Validate shape against first archive
        first_archive = next(iter(self._archives.values()))
        _, n_spaces, n_radii = first_archive.shape

        if transformer_grid.shape != (n_spaces, n_radii):
            msg = (
                f"TransformerGrid shape {transformer_grid.shape} != "
                f"archive (spaces, radii) ({n_spaces}, {n_radii})"
            )
            raise ValueError(
                msg,
            )

        if not transformer_grid.is_filled:
            msg = "TransformerGrid is not fully filled"
            raise ValueError(msg)

        # Validate all archives have same shape
        for name, archive in self._archives.items():
            if archive.shape[1:] != (n_spaces, n_radii):
                msg = (
                    f"Archive '{name}' shape {archive.shape} doesn't match "
                    f"expected (*, {n_spaces}, {n_radii})"
                )
                raise ValueError(
                    msg,
                )

        # Get space/radii labels from first archive
        spaces = first_archive.spaces
        radii = first_archive.radii

        # Fit each cell
        for s_idx, space in enumerate(spaces):
            for r_idx, radius in enumerate(radii):
                # Slice all archives to this (space, radius)
                sliced = self[radius, space]

                # Fit using fit_combined (handles deduplication)
                transformer = transformer_grid[s_idx, r_idx]
                sliced.fit_combined(transformer)

        return transformer_grid

    def transform_grid(
        self,
        transformer_grid: TransformerGrid,
        inplace: bool = False,
    ) -> Self:
        """Apply fitted transformer grid to all archives.

        Stacks ALL data from ALL archives together, performs ONE transform,
        then splits results back. This ensures consistent coordinates across
        archives (avoids UMAP's stochastic transform inconsistency).

        Parameters
        ----------
        transformer_grid : TransformerGrid
            Grid of fitted transformers.
        inplace : bool, optional
            If True, modifies archives in place. Default is False.

        Returns
        -------
        SimilarityAtlas
            Atlas with transformed (embedded) archives.

        Example
        -------
        >>> fitted_grid = atlas.fit_transformer_grid(grid)
        >>> embedded = atlas.transform_grid(fitted_grid)
        """
        first_archive = next(iter(self._archives.values()))
        _n_radii, _n_spaces, _gens = first_archive.shape
        spaces = first_archive.spaces
        radii = first_archive.radii

        # Build structure to hold transformed archives
        # Start with deep copies if not inplace
        if inplace:
            new_archives = self._archives
        else:
            new_archives = {name: deepcopy(arch) for name, arch in self._archives.items()}

        # For each (space, radius) cell
        for s_idx, _space in enumerate(spaces):
            for r_idx, _radius in enumerate(radii):
                transformer = transformer_grid[s_idx, r_idx]

                # --- Stack ALL data from ALL archives ---
                all_rows = []
                row_map: list[tuple[str, int, int]] = []  # (archive_name, frame_idx, local_idx)

                for name, archive in self._archives.items():
                    for frame_idx, frame in enumerate(archive.frames):
                        matrix = frame.matrices[s_idx][r_idx].matrix
                        n_rows = matrix.shape[0]
                        for local_idx in range(n_rows):
                            all_rows.append(matrix[local_idx])
                            row_map.append((name, frame_idx, local_idx))

                # Stack into single array
                if all_rows and sp.issparse(all_rows[0]):
                    stacked = sp.vstack(all_rows, format="csr")
                else:
                    stacked = np.vstack(all_rows)

                # --- ONE transform ---
                embeddings = transformer.transform(stacked)
                if embeddings.ndim == 1:
                    embeddings = embeddings.reshape(-1, 1)

                # --- Split back into archives ---
                # Group by (archive_name, frame_idx)
                from collections import defaultdict
                frame_embeddings: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = defaultdict(list)

                for row_idx, (name, frame_idx, local_idx) in enumerate(row_map):
                    frame_embeddings[name, frame_idx].append((local_idx, embeddings[row_idx]))

                # Reconstruct matrices for each archive/frame
                for (name, frame_idx), rows in frame_embeddings.items():
                    # Sort by local_idx to maintain order
                    rows.sort(key=operator.itemgetter(0))
                    gen_embedding = np.vstack([r[1] for r in rows])

                    # Get original matrix for metadata
                    original_matrix = new_archives[name].frames[frame_idx].matrices[s_idx][r_idx]
                    tags = SimilarityMatrixTags(
                        domain=MatrixDomain.EMBEDDING,
                        radius=original_matrix.radius,
                        is_gap=False,
                    )
                    new_matrix = SimilarityMatrix(
                        gen_embedding,
                        original_matrix.label,
                        tags,
                    )

                    # Replace in archive
                    new_archives[name]._frames[frame_idx]._series[s_idx]._matrices[r_idx] = new_matrix  # TODO bad

        return self.replace(archives=new_archives)

    # --- Representation ---

    @property
    def shape(self) -> dict[str, tuple[int, int, int]]:
        """Shape of each archive: {name: (n_gens, n_spaces, n_radii)}."""
        return {name: arch.shape for name, arch in self._archives.items()}

    def __repr__(self) -> str:
        if not self._archives:
            return "SimilarityAtlas (empty)"

        lines = [f"SimilarityAtlas ({len(self)} archives)"]
        items = list(self._archives.items())
        for i, (name, archive) in enumerate(items):
            prefix = "└─" if i == len(items) - 1 else "├─"
            lines.append(f"{prefix} {name}: {archive.shape}")
        return "\n".join(lines)

    # --- Helpers ---

    def replace(
        self,
        archives: dict[str, SimilarityArchive] | None = None,
    ) -> Self:
        """Create new Atlas with replaced attributes."""
        return self.__class__(
            _archives=archives
            if archives is not None
            else dict(self._archives),
        )

    def add(self, name: str, archive: SimilarityArchive) -> Self:
        """Add an archive to the atlas. Returns new Atlas."""
        new_archives = dict(self._archives)
        new_archives[name] = archive
        return self.replace(archives=new_archives)

    def remove(self, name: str) -> Self:
        """Remove an archive from the atlas. Returns new Atlas."""
        new_archives = {k: v for k, v in self._archives.items() if k != name}
        return self.replace(archives=new_archives)


if __name__ == "__main__":
    from sklearn.decomposition import PCA

    from ..options import MatrixDomain
    from .s1_matrix import SimilarityMatrix, SimilarityMatrixTags
    from .s2_series import SimilaritySeries
    from .s3_frame import SimilarityFrame
    from .s4_archive import SimilarityArchive
    from .transformers import TransformerGrid

    # --- Mock Data Factory ---
    def make_mock_matrix(
        space: str,
        radius: int,
        n_individuals: int = 10,
    ) -> SimilarityMatrix:
        n_features = (radius + 1) * 5
        matrix = sp.random(n_individuals, n_features, density=0.3, format="csr")
        tags = SimilarityMatrixTags(
            domain=MatrixDomain.FEATURES,
            radius=radius,
            is_gap=False,
        )
        return SimilarityMatrix(matrix=matrix, label=space, tags=tags)

    def make_mock_series(
        space: str,
        max_radius: int = 2,
        n_individuals: int = 10,
    ) -> SimilaritySeries:
        matrices = [
            make_mock_matrix(space, r, n_individuals)
            for r in range(max_radius + 1)
        ]
        return SimilaritySeries(matrices=matrices, label=space)

    def make_mock_frame(
        spaces: list[str],
        max_radius: int = 2,
        n_individuals: int = 10,
    ) -> SimilarityFrame:
        series_list = [
            make_mock_series(s, max_radius, n_individuals) for s in spaces
        ]
        return SimilarityFrame(series=series_list)

    def make_mock_archive(
        n_generations: int = 5,
        spaces: list[str] | None = None,
        max_radius: int = 2,
        pop_size: int = 10,
        id_offset: int = 0,
    ) -> SimilarityArchive:
        if spaces is None:
            spaces = ["FRONT", "BACK", "LEFT"]

        frames = [
            make_mock_frame(spaces, max_radius, pop_size)
            for _ in range(n_generations)
        ]

        # Mock ID mapper with offset for different archives
        id_mapper: dict[int, dict[int, int]] = {}
        alive_mapper: dict[int, dict[int, bool]] = {}
        global_id = id_offset
        for gen in range(n_generations):
            id_mapper[gen] = {}
            alive_mapper[gen] = {}
            for row_idx in range(pop_size):
                id_mapper[gen][row_idx] = global_id
                alive_mapper[gen][global_id] = True
                global_id += 1

        return SimilarityArchive(
            frames=frames,
            id_mapper=id_mapper,
            alive_mapper=alive_mapper,
        )

    archive = make_mock_archive()
    sliced = archive.select(radius=0, space="FRONT")

    # --- Test Atlas ---
    archive_a = make_mock_archive(n_generations=5, pop_size=10, id_offset=0)
    archive_b = make_mock_archive(n_generations=4, pop_size=8, id_offset=1000)

    # Create Atlas
    atlas = SimilarityAtlas({"run_a": archive_a, "run_b": archive_b})

    # Test slicing

    # Test single transformer fit/transform
    sliced = atlas[0, "FRONT"]  # radius=0, space=FRONT

    pca = PCA(n_components=2)
    fitted_pca = sliced.fit_combined(pca)

    embedded = sliced.transform_unique(fitted_pca)

    # Test TransformerGrid
    n_spaces = len(archive_a.spaces)
    n_radii = len(archive_a.radii)

    grid = TransformerGrid((n_spaces, n_radii))
    grid[:, :] = PCA(n_components=2)

    # Fit grid
    fitted_grid = atlas.fit_transformer_grid(grid)

    # Check a fitted transformer
    sample_transformer = fitted_grid[0, 0]

    # Transform with grid
    embedded_atlas = atlas.transform_grid(fitted_grid)

    # Verify embedding shapes
    for archive in embedded_atlas.values():
        frame = archive.frames[0]

from _typeshed import Incomplete

__all__ = ['CompImageSection']

class CompImageSection:
    """
    Class enabling subsets of CompImageHDU data to be loaded lazily via slicing.

    Slices of this object load the corresponding section of an image array from
    the underlying FITS file, and applies any BSCALE/BZERO factors.

    Section slices cannot be assigned to, and modifications to a section are
    not saved back to the underlying file.

    See the :ref:`astropy:data-sections` section of the Astropy documentation
    for more details.
    """
    hdu: Incomplete
    _data_shape: Incomplete
    _tile_shape: Incomplete
    _n_dim: Incomplete
    _n_tiles: Incomplete
    def __init__(self, hdu) -> None: ...
    @property
    def shape(self): ...
    @property
    def ndim(self): ...
    @property
    def dtype(self): ...
    def __getitem__(self, index): ...

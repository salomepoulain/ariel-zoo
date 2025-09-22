from _typeshed import Incomplete
from astropy.utils.data_info import MixinInfo
from astropy.utils.shapes import ShapedLikeNDArray
from typing import NamedTuple

__all__ = ['StokesCoord', 'custom_stokes_symbol_mapping', 'StokesSymbol']

class StokesSymbol(NamedTuple):
    """Symbol for a Stokes coordinate."""
    symbol: str = ...
    description: str = ...

def custom_stokes_symbol_mapping(mapping: dict[int, StokesSymbol], replace: bool = False) -> None:
    """
    Add a custom set of mappings from values to Stokes symbols.

    Parameters
    ----------
    mappings
        A list of dictionaries with custom mappings between values (integers)
        and `.StokesSymbol` classes.
    replace
        Replace all mappings with this one.
    """

class StokesCoordInfo(MixinInfo):
    _represent_as_dict_attrs: Incomplete
    _represent_as_dict_primary_data: str
    _construct_from_dict_args: Incomplete
    @property
    def unit(self) -> None: ...
    @property
    def dtype(self): ...
    @staticmethod
    def default_format(val): ...
    def new_like(self, cols, length, metadata_conflicts: str = 'warn', name: Incomplete | None = None):
        """
        Return a new StokesCoord instance which is consistent with the
        input ``cols`` and has ``length`` rows.

        This is intended for creating an empty column object whose elements can
        be set in-place for table operations like join or vstack.

        Parameters
        ----------
        cols : list
            List of input columns
        length : int
            Length of the output column object
        metadata_conflicts : str ('warn'|'error'|'silent')
            How to handle metadata conflicts
        name : str
            Output column name

        Returns
        -------
        col : `~astropy.coordinates.StokesCoord` (or subclass)
            Empty instance of this class consistent with ``cols``

        """
    def get_sortable_arrays(self):
        """
        Return a list of arrays which can be lexically sorted to represent
        the order of the parent column.

        For StokesCoord this is just the underlying values.

        Returns
        -------
        arrays : list of ndarray
        """

class StokesCoord(ShapedLikeNDArray):
    """
    A representation of stokes coordinates with helpers for converting to profile names.

    Parameters
    ----------
    stokes : array-like
        The numeric values representing stokes coordinates.
    """
    info: Incomplete
    _data: Incomplete
    def __init__(self, stokes, copy: bool = False) -> None: ...
    @property
    def shape(self): ...
    @property
    def value(self): ...
    @property
    def dtype(self): ...
    def __array__(self, dtype: Incomplete | None = None, copy=...): ...
    def _apply(self, method, *args, **kwargs): ...
    @staticmethod
    def _from_symbols(symbols):
        """
        Convert an array of symbols to an array of values
        """
    @property
    def symbol(self):
        """The coordinate represented as strings."""
    def __setitem__(self, item, value) -> None: ...
    def __eq__(self, other): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

from .table import Column as Column, Table as Table
from _typeshed import Incomplete
from astropy.utils.data_info import ParentDtypeInfo as ParentDtypeInfo

def simple_table(size: int = 3, cols: Incomplete | None = None, kinds: str = 'ifS', masked: bool = False):
    """
    Return a simple table for testing.

    Example
    --------
    ::

      >>> from astropy.table.table_helpers import simple_table
      >>> print(simple_table(3, 6, masked=True, kinds='ifOS'))
       a   b     c      d   e   f
      --- --- -------- --- --- ---
       -- 1.0 {'c': 2}  --   5 5.0
        2 2.0       --   e   6  --
        3  -- {'e': 4}   f  -- 7.0

    Parameters
    ----------
    size : int
        Number of table rows
    cols : int, optional
        Number of table columns. Defaults to number of kinds.
    kinds : str
        String consisting of the column dtype.kinds.  This string
        will be cycled through to generate the column dtype.
        The allowed values are 'i', 'f', 'S', 'O'.

    Returns
    -------
    out : `Table`
        New table with appropriate characteristics
    """
def complex_table():
    """
    Return a masked table from the io.votable test set that has a wide variety
    of stressing types.
    """

class ArrayWrapperInfo(ParentDtypeInfo):
    _represent_as_dict_primary_data: str
    def _represent_as_dict(self):
        """Represent Column as a dict that can be serialized."""
    def _construct_from_dict(self, map):
        """Construct Column from ``map``."""

class ArrayWrapper:
    """
    Minimal mixin using a simple wrapper around a numpy array.

    TODO: think about the future of this class as it is mostly for demonstration
    purposes (of the mixin protocol). Consider taking it out of core and putting
    it into a tutorial. One advantage of having this in core is that it is
    getting tested in the mixin testing though it doesn't work for multidim
    data.
    """
    info: Incomplete
    data: Incomplete
    def __init__(self, data, copy: bool = True) -> None: ...
    def __getitem__(self, item): ...
    def __setitem__(self, item, value) -> None: ...
    def __len__(self) -> int: ...
    def __eq__(self, other):
        """Minimal equality testing, mostly for mixin unit tests."""
    @property
    def dtype(self): ...
    @property
    def shape(self): ...
    def __repr__(self) -> str: ...

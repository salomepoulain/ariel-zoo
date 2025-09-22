from _typeshed import Incomplete
from astropy.utils.compat import COPY_IF_NEEDED as COPY_IF_NEEDED

class Row:
    """A class to represent one row of a Table object.

    A Row object is returned when a Table object is indexed with an integer
    or when iterating over a table::

      >>> from astropy.table import Table
      >>> table = Table([(1, 2), (3, 4)], names=('a', 'b'),
      ...               dtype=('int32', 'int32'))
      >>> row = table[1]
      >>> row
      <Row index=1>
        a     b
      int32 int32
      ----- -----
          2     4
      >>> row['a']
      np.int32(2)
      >>> row[1]
      np.int32(4)
    """
    _index: Incomplete
    _table: Incomplete
    def __init__(self, table, index) -> None: ...
    def __getitem__(self, item): ...
    def __setitem__(self, item, val) -> None: ...
    def _ipython_key_completions_(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def __array__(self, dtype: Incomplete | None = None, copy=...):
        """Support converting Row to np.array via np.array(table).

        Coercion to a different dtype via np.array(table, dtype) is not
        supported and will raise a ValueError.

        If the parent table is masked then the mask information is dropped.
        """
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def get(self, key, default: Incomplete | None = None, /):
        '''Return the value for key if key is in the columns, else default.

        Parameters
        ----------
        key : `str`, positional-only
            The name of the column to look for.
        default : `object`, optional, positional-only
            The value to return if the ``key`` is not among the columns.

        Returns
        -------
        `object`
            The value in the ``key`` column of the row if present,
            ``default`` otherwise.

        Examples
        --------
        >>> from astropy.table import Table
        >>> t = Table({"a": [2., 3., 5.], "b": [7., 11., 13.]})
        >>> t[0].get("a")
        np.float64(2.0)
        >>> t[1].get("b", 0.)
        np.float64(11.0)
        >>> t[2].get("c", 0.)
        0.0
        '''
    def keys(self): ...
    def values(self): ...
    @property
    def table(self): ...
    @property
    def index(self): ...
    def as_void(self):
        """
        Returns a *read-only* copy of the row values in the form of np.void or
        np.ma.mvoid objects.  This corresponds to the object types returned for
        row indexing of a pure numpy structured array or masked array. This
        method is slow and its use is discouraged when possible.

        Returns
        -------
        void_row : ``numpy.void`` or ``numpy.ma.mvoid``
            Copy of row values.
            ``numpy.void`` if unmasked, ``numpy.ma.mvoid`` else.
        """
    @property
    def meta(self): ...
    @property
    def columns(self): ...
    @property
    def colnames(self): ...
    @property
    def dtype(self): ...
    def _base_repr_(self, html: bool = False):
        """
        Display row as a single-line table but with appropriate header line.
        """
    def _repr_html_(self): ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __bytes__(self) -> bytes: ...

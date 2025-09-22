import numpy as np
from . import groups as groups, pprint as pprint
from ._column_mixins import _ColumnGetitemShim as _ColumnGetitemShim, _MaskedColumnGetitemShim as _MaskedColumnGetitemShim
from _typeshed import Incomplete
from astropy.units import Quantity as Quantity, StructuredUnit as StructuredUnit, Unit as Unit
from astropy.utils.compat import COPY_IF_NEEDED as COPY_IF_NEEDED, NUMPY_LT_2_0 as NUMPY_LT_2_0
from astropy.utils.console import color_print as color_print
from astropy.utils.data_info import BaseColumnInfo as BaseColumnInfo, dtype_info_name as dtype_info_name
from astropy.utils.metadata import MetaData as MetaData
from astropy.utils.misc import dtype_bytes_or_chars as dtype_bytes_or_chars
from collections.abc import Generator
from numpy import ma

FORMATTER: Incomplete

class StringTruncateWarning(UserWarning):
    """
    Warning class for when a string column is assigned a value
    that gets truncated because the base (numpy) string length
    is too short.

    This does not inherit from AstropyWarning because we want to use
    stacklevel=2 to show the user where the issue occurred in their code.
    """

def _auto_names(n_cols): ...

_comparison_functions: Incomplete

def col_copy(col, copy_indices: bool = True):
    """
    Mixin-safe version of Column.copy() (with copy_data=True).

    Parameters
    ----------
    col : Column or mixin column
        Input column
    copy_indices : bool
        Copy the column ``indices`` attribute

    Returns
    -------
    col : Copy of input column
    """

class FalseArray(np.ndarray):
    """
    Boolean mask array that is always False.

    This is used to create a stub ``mask`` property which is a boolean array of
    ``False`` used by default for mixin columns and corresponding to the mixin
    column data shape.  The ``mask`` looks like a normal numpy array but an
    exception will be raised if ``True`` is assigned to any element.  The
    consequences of the limitation are most obvious in the high-level table
    operations.

    Parameters
    ----------
    shape : tuple
        Data shape
    """
    def __new__(cls, shape): ...
    def __setitem__(self, item, val) -> None: ...

def _expand_string_array_for_values(arr, values):
    """
    For string-dtype return a version of ``arr`` that is wide enough for ``values``.
    If ``arr`` is not string-dtype or does not need expansion then return ``arr``.

    Parameters
    ----------
    arr : np.ndarray
        Input array
    values : scalar or array-like
        Values for width comparison for string arrays

    Returns
    -------
    arr_expanded : np.ndarray

    """
def _convert_sequence_data_to_array(data, dtype: Incomplete | None = None):
    '''Convert N-d sequence-like data to ndarray or MaskedArray.

    This is the core function for converting Python lists or list of lists to a
    numpy array. This handles embedded np.ma.masked constants in ``data`` along
    with the special case of an homogeneous list of MaskedArray elements.

    Considerations:

    - np.ma.array is about 50 times slower than np.array for list input. This
      function avoids using np.ma.array on list input.
    - np.array emits a UserWarning for embedded np.ma.masked, but only for int
      or float inputs. For those it converts to np.nan and forces float dtype.
      For other types np.array is inconsistent, for instance converting
      np.ma.masked to "0.0" for str types.
    - Searching in pure Python for np.ma.masked in ``data`` is comparable in
      speed to calling ``np.array(data)``.
    - This function may end up making two additional copies of input ``data``.

    Parameters
    ----------
    data : N-d sequence
        Input data, typically list or list of lists
    dtype : None or dtype-like
        Output datatype (None lets np.array choose)

    Returns
    -------
    np_data : np.ndarray or np.ma.MaskedArray

    '''
def _make_compare(oper):
    """
    Make Column comparison methods which encode the ``other`` object to utf-8
    in the case of a bytestring dtype for Py3+.

    Parameters
    ----------
    oper : str
        Operator name
    """

class ColumnInfo(BaseColumnInfo):
    """
    Container for meta information like name, description, format.

    This is required when the object is used as a mixin column within a table,
    but can be used as a general way to store meta information.
    """
    attr_names: Incomplete
    _attrs_no_copy: Incomplete
    attrs_from_parent = attr_names
    _supports_indexing: bool
    _represent_as_dict_primary_data: str
    def _represent_as_dict(self): ...
    def _construct_from_dict(self, map): ...
    def new_like(self, cols, length, metadata_conflicts: str = 'warn', name: Incomplete | None = None):
        """
        Return a new Column instance which is consistent with the
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
        col : Column (or subclass)
            New instance of this class consistent with ``cols``

        """
    def get_sortable_arrays(self):
        """
        Return a list of arrays which can be lexically sorted to represent
        the order of the parent column.

        For Column this is just the column itself.

        Returns
        -------
        arrays : list of ndarray
        """

class BaseColumn(_ColumnGetitemShim, np.ndarray):
    meta: Incomplete
    _name: Incomplete
    _parent_table: Incomplete
    _format: Incomplete
    description: Incomplete
    indices: Incomplete
    def __new__(cls, data: Incomplete | None = None, name: Incomplete | None = None, dtype: Incomplete | None = None, shape=(), length: int = 0, description: Incomplete | None = None, unit: Incomplete | None = None, format: Incomplete | None = None, meta: Incomplete | None = None, copy=..., copy_indices: bool = True): ...
    @property
    def data(self): ...
    @property
    def value(self):
        """
        An alias for the existing ``data`` attribute.
        """
    @property
    def parent_table(self): ...
    @parent_table.setter
    def parent_table(self, table) -> None: ...
    info: Incomplete
    def copy(self, order: str = 'C', data: Incomplete | None = None, copy_data: bool = True):
        """
        Return a copy of the current instance.

        If ``data`` is supplied then a view (reference) of ``data`` is used,
        and ``copy_data`` is ignored.

        Parameters
        ----------
        order : {'C', 'F', 'A', 'K'}, optional
            Controls the memory layout of the copy. 'C' means C-order,
            'F' means F-order, 'A' means 'F' if ``a`` is Fortran contiguous,
            'C' otherwise. 'K' means match the layout of ``a`` as closely
            as possible. (Note that this function and :func:numpy.copy are very
            similar, but have different default values for their order=
            arguments.)  Default is 'C'.
        data : array, optional
            If supplied then use a view of ``data`` instead of the instance
            data.  This allows copying the instance attributes and meta.
        copy_data : bool, optional
            Make a copy of the internal numpy array instead of using a
            reference.  Default is True.

        Returns
        -------
        col : Column or MaskedColumn
            Copy of the current column (same type as original)
        """
    def __setstate__(self, state) -> None:
        """
        Restore the internal state of the Column/MaskedColumn for pickling
        purposes.  This requires that the last element of ``state`` is a
        5-tuple that has Column-specific state values.
        """
    def __reduce__(self):
        """
        Return a 3-tuple for pickling a Column.  Use the super-class
        functionality but then add in a 5-tuple of Column-specific values
        that get used in __setstate__.
        """
    def __array_finalize__(self, obj) -> None: ...
    def __array_wrap__(self, out_arr, context: Incomplete | None = None, return_scalar: bool = False):
        '''
        __array_wrap__ is called at the end of every ufunc.

        Normally, we want a Column object back and do not have to do anything
        special. But there are two exceptions:

        1) If the output shape is different (e.g. for reduction ufuncs
           like sum() or mean()), a Column still linking to a parent_table
           makes little sense, so we return the output viewed as the
           column content (ndarray or MaskedArray).
           For this case, if numpy tells us to ``return_scalar`` (for numpy
           >= 2.0, otherwise assume to be true), we use "[()]" to ensure we
           convert a zero rank array to a scalar. (For some reason np.sum()
           returns a zero rank scalar array while np.mean() returns a scalar;
           So the [()] is needed for this case.)

        2) When the output is created by any function that returns a boolean
           we also want to consistently return an array rather than a column
           (see #1446 and #1685)
        '''
    @property
    def name(self):
        """
        The name of this column.
        """
    @name.setter
    def name(self, val) -> None: ...
    @property
    def format(self):
        """
        Format string for displaying values in this column.
        """
    @format.setter
    def format(self, format_string) -> None: ...
    @property
    def descr(self):
        """Array-interface compliant full description of the column.

        This returns a 3-tuple (name, type, shape) that can always be
        used in a structured array dtype definition.
        """
    def iter_str_vals(self) -> Generator[Incomplete, Incomplete]:
        """
        Return an iterator that yields the string-formatted values of this
        column.

        Returns
        -------
        str_vals : iterator
            Column values formatted as strings
        """
    def attrs_equal(self, col):
        """Compare the column attributes of ``col`` to this object.

        The comparison attributes are: ``name``, ``unit``, ``dtype``,
        ``format``, ``description``, and ``meta``.

        Parameters
        ----------
        col : Column
            Comparison column

        Returns
        -------
        equal : bool
            True if all attributes are equal
        """
    @property
    def _formatter(self): ...
    def pformat(self, max_lines: int = -1, show_name: bool = True, show_unit: bool = False, show_dtype: bool = False, html: bool = False):
        """Return a list of formatted string representation of column values.

        If ``max_lines=None`` is supplied then the height of the
        screen terminal is used to set ``max_lines``.  If the terminal
        height cannot be determined then the default will be
        determined using the ``astropy.conf.max_lines`` configuration
        item. If a negative value of ``max_lines`` is supplied then
        there is no line limit applied (default).

        Parameters
        ----------
        max_lines : int or None
            Maximum lines of output (header + data rows).
            -1 (default) implies no limit, ``None`` implies using the
            height of the current terminal.

        show_name : bool
            Include column name. Default is True.

        show_unit : bool
            Include a header row for unit. Default is False.

        show_dtype : bool
            Include column dtype. Default is False.

        html : bool
            Format the output as an HTML table. Default is False.

        Returns
        -------
        lines : list
            List of lines with header and formatted column values

        """
    def pprint(self, max_lines: Incomplete | None = None, show_name: bool = True, show_unit: bool = False, show_dtype: bool = False) -> None:
        """Print a formatted string representation of column values.

        If ``max_lines=None`` (default) then the height of the
        screen terminal is used to set ``max_lines``.  If the terminal
        height cannot be determined then the default will be
        determined using the ``astropy.conf.max_lines`` configuration
        item. If a negative value of ``max_lines`` is supplied then
        there is no line limit applied.

        Parameters
        ----------
        max_lines : int
            Maximum number of values in output

        show_name : bool
            Include column name. Default is True.

        show_unit : bool
            Include a header row for unit. Default is False.

        show_dtype : bool
            Include column dtype. Default is True.
        """
    def more(self, max_lines: Incomplete | None = None, show_name: bool = True, show_unit: bool = False) -> None:
        """Interactively browse column with a paging interface.

        Supported keys::

          f, <space> : forward one page
          b : back one page
          r : refresh same page
          n : next row
          p : previous row
          < : go to beginning
          > : go to end
          q : quit browsing
          h : print this help

        Parameters
        ----------
        max_lines : int
            Maximum number of lines in table output.

        show_name : bool
            Include a header row for column names. Default is True.

        show_unit : bool
            Include a header row for unit. Default is False.

        """
    @property
    def unit(self):
        """
        The unit associated with this column.  May be a string or a
        `astropy.units.UnitBase` instance.

        Setting the ``unit`` property does not change the values of the
        data.  To perform a unit conversion, use ``convert_unit_to``.
        """
    _unit: Incomplete
    @unit.setter
    def unit(self, unit) -> None: ...
    @unit.deleter
    def unit(self) -> None: ...
    def searchsorted(self, v, side: str = 'left', sorter: Incomplete | None = None): ...
    def convert_unit_to(self, new_unit, equivalencies=[]) -> None:
        """
        Converts the values of the column in-place from the current
        unit to the given unit.

        To change the unit associated with this column without
        actually changing the data values, simply set the ``unit``
        property.

        Parameters
        ----------
        new_unit : str or `astropy.units.UnitBase` instance
            The unit to convert to.

        equivalencies : list of tuple
           A list of equivalence pairs to try if the unit are not
           directly convertible.  See :ref:`astropy:unit_equivalencies`.

        Raises
        ------
        astropy.units.UnitsError
            If units are inconsistent
        """
    _groups: Incomplete
    @property
    def groups(self): ...
    def group_by(self, keys):
        """
        Group this column by the specified ``keys``.

        This effectively splits the column into groups which correspond to
        unique values of the ``keys`` grouping object.  The output is a new
        `Column` or `MaskedColumn` which contains a copy of this column but
        sorted by row according to ``keys``.

        The ``keys`` input to ``group_by`` must be a numpy array with the
        same length as this column.

        Parameters
        ----------
        keys : numpy array
            Key grouping object

        Returns
        -------
        out : Column
            New column with groups attribute set accordingly
        """
    def _copy_groups(self, out) -> None:
        """
        Copy current groups into a copy of self ``out``.
        """
    def __repr__(self) -> str: ...
    @property
    def quantity(self):
        """
        A view of this table column as a `~astropy.units.Quantity` object with
        units given by the Column's `unit` parameter.
        """
    def to(self, unit, equivalencies=[], **kwargs):
        """
        Converts this table column to a `~astropy.units.Quantity` object with
        the requested units.

        Parameters
        ----------
        unit : unit-like
            The unit to convert to (i.e., a valid argument to the
            :meth:`astropy.units.Quantity.to` method).
        equivalencies : list of tuple
            Equivalencies to use for this conversion.  See
            :meth:`astropy.units.Quantity.to` for more details.

        Returns
        -------
        quantity : `~astropy.units.Quantity`
            A quantity object with the contents of this column in the units
            ``unit``.
        """
    def _copy_attrs(self, obj) -> None:
        """
        Copy key column attributes from ``obj`` to self.
        """
    @staticmethod
    def _encode_str(value):
        """
        Encode anything that is unicode-ish as utf-8.  This method is only
        called for Py3+.
        """
    def tolist(self): ...

class Column(BaseColumn):
    '''Define a data column for use in a Table object.

    Parameters
    ----------
    data : list, ndarray, or None
        Column data values
    name : str
        Column name and key for reference within Table
    dtype : `~numpy.dtype`-like
        Data type for column
    shape : tuple or ()
        Dimensions of a single row element in the column data
    length : int or 0
        Number of row elements in column data
    description : str or None
        Full description of column
    unit : str or None
        Physical unit
    format : str, None, or callable
        Format string for outputting column values.  This can be an
        "old-style" (``format % value``) or "new-style" (`str.format`)
        format specification string or a function or any callable object that
        accepts a single value and returns a string.
    meta : dict-like or None
        Meta-data associated with the column

    Examples
    --------
    A Column can be created in two different ways:

    - Provide a ``data`` value but not ``shape`` or ``length`` (which are
      inferred from the data).

      Examples::

        col = Column(data=[1, 2], name=\'name\')  # shape=(2,)
        col = Column(data=[[1, 2], [3, 4]], name=\'name\')  # shape=(2, 2)
        col = Column(data=[1, 2], name=\'name\', dtype=float)
        col = Column(data=np.array([1, 2]), name=\'name\')
        col = Column(data=[\'hello\', \'world\'], name=\'name\')

      The ``dtype`` argument can be any value which is an acceptable
      fixed-size data-type initializer for the numpy.dtype() method.  See
      `<https://numpy.org/doc/stable/reference/arrays.dtypes.html>`_.
      Examples include:

      - Python non-string type (float, int, bool)
      - Numpy non-string type (e.g. np.float32, np.int64, np.bool\\_)
      - Numpy.dtype array-protocol type strings (e.g. \'i4\', \'f8\', \'S15\')

      If no ``dtype`` value is provide then the type is inferred using
      ``np.array(data)``.

    - Provide ``length`` and optionally ``shape``, but not ``data``

      Examples::

        col = Column(name=\'name\', length=5)
        col = Column(name=\'name\', dtype=int, length=10, shape=(3,4))

      The default ``dtype`` is ``np.float64``.  The ``shape`` argument is the
      array shape of a single cell in the column.

    To access the ``Column`` data as a raw `numpy.ndarray` object, you can use
    one of the ``data`` or ``value`` attributes (which are equivalent)::

        col.data
        col.value
    '''
    def __new__(cls, data: Incomplete | None = None, name: Incomplete | None = None, dtype: Incomplete | None = None, shape=(), length: int = 0, description: Incomplete | None = None, unit: Incomplete | None = None, format: Incomplete | None = None, meta: Incomplete | None = None, copy=..., copy_indices: bool = True): ...
    def __setattr__(self, item, value) -> None: ...
    def _base_repr_(self, html: bool = False): ...
    def _repr_html_(self): ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __bytes__(self) -> bytes: ...
    def _check_string_truncate(self, value) -> None:
        """
        Emit a warning if any elements of ``value`` will be truncated when
        ``value`` is assigned to self.
        """
    def __setitem__(self, index, value) -> None: ...
    __eq__: Incomplete
    __ne__: Incomplete
    __gt__: Incomplete
    __lt__: Incomplete
    __ge__: Incomplete
    __le__: Incomplete
    def insert(self, obj, values, axis: int = 0):
        """
        Insert values before the given indices in the column and return
        a new `~astropy.table.Column` object.

        Parameters
        ----------
        obj : int, slice or sequence of int
            Object that defines the index or indices before which ``values`` is
            inserted.
        values : array-like
            Value(s) to insert.  If the type of ``values`` is different from
            that of the column, ``values`` is converted to the matching type.
            ``values`` should be shaped so that it can be broadcast appropriately.
        axis : int, optional
            Axis along which to insert ``values``.  If ``axis`` is None then
            the column array is flattened before insertion.  Default is 0,
            which will insert a row.

        Returns
        -------
        out : `~astropy.table.Column`
            A copy of column with ``values`` and ``mask`` inserted.  Note that the
            insertion does not occur in-place: a new column is returned.
        """
    name: Incomplete
    unit: Incomplete
    copy: Incomplete
    more: Incomplete
    pprint: Incomplete
    pformat: Incomplete
    convert_unit_to: Incomplete
    quantity: Incomplete
    to: Incomplete

class MaskedColumnInfo(ColumnInfo):
    """
    Container for meta information like name, description, format.

    This is required when the object is used as a mixin column within a table,
    but can be used as a general way to store meta information.  In this case
    it just adds the ``mask_val`` attribute.
    """
    attr_names: Incomplete
    _represent_as_dict_primary_data: str
    mask_val: Incomplete
    serialize_method: Incomplete
    def __init__(self, bound: bool = False) -> None: ...
    def _represent_as_dict(self): ...

class MaskedColumn(Column, _MaskedColumnGetitemShim, ma.MaskedArray):
    '''Define a masked data column for use in a Table object.

    Parameters
    ----------
    data : list, ndarray, or None
        Column data values
    name : str
        Column name and key for reference within Table
    mask : list, ndarray or None
        Boolean mask for which True indicates missing or invalid data
    fill_value : float, int, str, or None
        Value used when filling masked column elements
    dtype : `~numpy.dtype`-like
        Data type for column
    shape : tuple or ()
        Dimensions of a single row element in the column data
    length : int or 0
        Number of row elements in column data
    description : str or None
        Full description of column
    unit : str or None
        Physical unit
    format : str, None, or callable
        Format string for outputting column values.  This can be an
        "old-style" (``format % value``) or "new-style" (`str.format`)
        format specification string or a function or any callable object that
        accepts a single value and returns a string.
    meta : dict-like or None
        Meta-data associated with the column

    Examples
    --------
    A MaskedColumn is similar to a Column except that it includes ``mask`` and
    ``fill_value`` attributes.  It can be created in two different ways:

    - Provide a ``data`` value but not ``shape`` or ``length`` (which are
      inferred from the data).

      Examples::

        col = MaskedColumn(data=[1, 2], name=\'name\')
        col = MaskedColumn(data=[1, 2], name=\'name\', mask=[True, False])
        col = MaskedColumn(data=[1, 2], name=\'name\', dtype=float, fill_value=99)

      The ``mask`` argument will be cast as a boolean array and specifies
      which elements are considered to be missing or invalid.

      The ``dtype`` argument can be any value which is an acceptable
      fixed-size data-type initializer for the numpy.dtype() method.  See
      `<https://numpy.org/doc/stable/reference/arrays.dtypes.html>`_.
      Examples include:

      - Python non-string type (float, int, bool)
      - Numpy non-string type (e.g. np.float32, np.int64, np.bool\\_)
      - Numpy.dtype array-protocol type strings (e.g. \'i4\', \'f8\', \'S15\')

      If no ``dtype`` value is provide then the type is inferred using
      ``np.array(data)``.  When ``data`` is provided then the ``shape``
      and ``length`` arguments are ignored.

    - Provide ``length`` and optionally ``shape``, but not ``data``

      Examples::

        col = MaskedColumn(name=\'name\', length=5)
        col = MaskedColumn(name=\'name\', dtype=int, length=10, shape=(3,4))

      The default ``dtype`` is ``np.float64``.  The ``shape`` argument is the
      array shape of a single cell in the column.

    To access the ``Column`` data as a raw `numpy.ma.MaskedArray` object, you can
    use one of the ``data`` or ``value`` attributes (which are equivalent)::

        col.data
        col.value
    '''
    info: Incomplete
    parent_table: Incomplete
    def __new__(cls, data: Incomplete | None = None, name: Incomplete | None = None, mask: Incomplete | None = None, fill_value: Incomplete | None = None, dtype: Incomplete | None = None, shape=(), length: int = 0, description: Incomplete | None = None, unit: Incomplete | None = None, format: Incomplete | None = None, meta: Incomplete | None = None, copy=..., copy_indices: bool = True): ...
    @property
    def fill_value(self): ...
    _fill_value: Incomplete
    @fill_value.setter
    def fill_value(self, val) -> None:
        """Set fill value both in the masked column view and in the parent table
        if it exists.  Setting one or the other alone doesn't work.
        """
    @property
    def data(self):
        """The plain MaskedArray data held by this column."""
    def filled(self, fill_value: Incomplete | None = None):
        """Return a copy of self, with masked values filled with a given value.

        Parameters
        ----------
        fill_value : scalar; optional
            The value to use for invalid entries (`None` by default).  If
            `None`, the ``fill_value`` attribute of the array is used
            instead.

        Returns
        -------
        filled_column : Column
            A copy of ``self`` with masked entries replaced by `fill_value`
            (be it the function argument or the attribute of ``self``).
        """
    def insert(self, obj, values, mask: Incomplete | None = None, axis: int = 0):
        """
        Insert values along the given axis before the given indices and return
        a new `~astropy.table.MaskedColumn` object.

        Parameters
        ----------
        obj : int, slice or sequence of int
            Object that defines the index or indices before which ``values`` is
            inserted.
        values : array-like
            Value(s) to insert.  If the type of ``values`` is different from
            that of the column, ``values`` is converted to the matching type.
            ``values`` should be shaped so that it can be broadcast appropriately.
        mask : bool or array-like
            Mask value(s) to insert.  If not supplied, and values does not have
            a mask either, then False is used.
        axis : int, optional
            Axis along which to insert ``values``.  If ``axis`` is None then
            the column array is flattened before insertion.  Default is 0,
            which will insert a row.

        Returns
        -------
        out : `~astropy.table.MaskedColumn`
            A copy of column with ``values`` and ``mask`` inserted.  Note that the
            insertion does not occur in-place: a new masked column is returned.
        """
    def convert_unit_to(self, new_unit, equivalencies=[]) -> None: ...
    def _copy_attrs_slice(self, out): ...
    def __setitem__(self, index, value) -> None: ...
    name: Incomplete
    copy: Incomplete
    more: Incomplete
    pprint: Incomplete
    pformat: Incomplete

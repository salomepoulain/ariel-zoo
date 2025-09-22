from . import conf as conf, groups as groups
from .column import BaseColumn as BaseColumn, Column as Column, FalseArray as FalseArray, MaskedColumn as MaskedColumn, _auto_names as _auto_names, _convert_sequence_data_to_array as _convert_sequence_data_to_array, col_copy as col_copy
from .connect import TableRead as TableRead, TableWrite as TableWrite
from .index import Index as Index, SlicedIndex as SlicedIndex, TableILoc as TableILoc, TableIndices as TableIndices, TableLoc as TableLoc, TableLocIndices as TableLocIndices, _IndexModeContext as _IndexModeContext, get_index as get_index
from .info import TableInfo as TableInfo
from .mixins.registry import get_mixin_handler as get_mixin_handler
from .ndarray_mixin import NdarrayMixin as NdarrayMixin
from .pprint import TableFormatter as TableFormatter
from .row import Row as Row
from _typeshed import Incomplete
from astropy import log as log
from astropy.io.registry import UnifiedReadWriteMethod as UnifiedReadWriteMethod
from astropy.units import Quantity as Quantity, QuantityInfo as QuantityInfo
from astropy.utils import ShapedLikeNDArray as ShapedLikeNDArray, deprecated as deprecated, isiterable as isiterable
from astropy.utils.compat import COPY_IF_NEEDED as COPY_IF_NEEDED, NUMPY_LT_1_25 as NUMPY_LT_1_25
from astropy.utils.console import color_print as color_print
from astropy.utils.data_info import BaseColumnInfo as BaseColumnInfo, DataInfo as DataInfo, MixinInfo as MixinInfo
from astropy.utils.decorators import format_doc as format_doc
from astropy.utils.exceptions import AstropyDeprecationWarning as AstropyDeprecationWarning, AstropyUserWarning as AstropyUserWarning
from astropy.utils.masked import Masked as Masked
from astropy.utils.metadata import MetaAttribute as MetaAttribute, MetaData as MetaData
from collections import OrderedDict
from collections.abc import Generator

_implementation_notes: str
__doctest_skip__: Incomplete
_pprint_docs: str
_pformat_docs: str

class TableReplaceWarning(UserWarning):
    """
    Warning class for cases when a table column is replaced via the
    Table.__setitem__ syntax e.g. t['a'] = val.

    This does not inherit from AstropyWarning because we want to use
    stacklevel=3 to show the user where the issue occurred in their code.
    """

def descr(col):
    """Array-interface compliant full description of a column.

    This returns a 3-tuple (name, type, shape) that can always be
    used in a structured array dtype definition.
    """
def has_info_class(obj, cls):
    """Check if the object's info is an instance of cls."""
def _get_names_from_list_of_dict(rows):
    """Return list of column names if ``rows`` is a list of dict that
    defines table data.

    If rows is not a list of dict then return None.
    """

class TableColumns(OrderedDict):
    """OrderedDict subclass for a set of columns.

    This class enhances item access to provide convenient access to columns
    by name or index, including slice access.  It also handles renaming
    of columns.

    The initialization argument ``cols`` can be a list of ``Column`` objects
    or any structure that is valid for initializing a Python dict.  This
    includes a dict, list of (key, val) tuples or [key, val] lists, etc.

    Parameters
    ----------
    cols : dict, list, tuple; optional
        Column objects as data structure that can init dict (see above)
    """
    def __init__(self, cols={}) -> None: ...
    def __getitem__(self, item):
        """Get items from a TableColumns object.

        ::

          tc = TableColumns(cols=[Column(name='a'), Column(name='b'), Column(name='c')])
          tc['a']  # Column('a')
          tc[1] # Column('b')
          tc['a', 'b'] # <TableColumns names=('a', 'b')>
          tc[1:3] # <TableColumns names=('b', 'c')>
        """
    def __setitem__(self, item, value, validated: bool = False) -> None:
        """
        Set item in this dict instance, but do not allow directly replacing an
        existing column unless it is already validated (and thus is certain to
        not corrupt the table).

        NOTE: it is easily possible to corrupt a table by directly *adding* a new
        key to the TableColumns attribute of a Table, e.g.
        ``t.columns['jane'] = 'doe'``.

        """
    def __repr__(self) -> str: ...
    def _rename_column(self, name, new_name) -> None: ...
    def __delitem__(self, name) -> None: ...
    def isinstance(self, cls):
        """
        Return a list of columns which are instances of the specified classes.

        Parameters
        ----------
        cls : class or tuple thereof
            Column class (including mixin) or tuple of Column classes.

        Returns
        -------
        col_list : list of `Column`
            List of Column objects which are instances of given classes.
        """
    def not_isinstance(self, cls):
        """
        Return a list of columns which are not instances of the specified classes.

        Parameters
        ----------
        cls : class or tuple thereof
            Column class (including mixin) or tuple of Column classes.

        Returns
        -------
        col_list : list of `Column`
            List of Column objects which are not instances of given classes.
        """
    def setdefault(self, key, default): ...
    def update(self, *args, **kwargs): ...

class TableAttribute(MetaAttribute):
    """
    Descriptor to define a custom attribute for a Table subclass.

    The value of the ``TableAttribute`` will be stored in a dict named
    ``__attributes__`` that is stored in the table ``meta``.  The attribute
    can be accessed and set in the usual way, and it can be provided when
    creating the object.

    Defining an attribute by this mechanism ensures that it will persist if
    the table is sliced or serialized, for example as a pickle or ECSV file.

    See the `~astropy.utils.metadata.MetaAttribute` documentation for additional
    details.

    Parameters
    ----------
    default : object
        Default value for attribute

    Examples
    --------
      >>> from astropy.table import Table, TableAttribute
      >>> class MyTable(Table):
      ...     identifier = TableAttribute(default=1)
      >>> t = MyTable(identifier=10)
      >>> t.identifier
      10
      >>> t.meta
      OrderedDict([('__attributes__', {'identifier': 10})])
    """

class PprintIncludeExclude(TableAttribute):
    """Maintain tuple that controls table column visibility for print output.

    This is a descriptor that inherits from MetaAttribute so that the attribute
    value is stored in the table meta['__attributes__'].

    This gets used for the ``pprint_include_names`` and ``pprint_exclude_names`` Table
    attributes.
    """
    def __get__(self, instance, owner_cls):
        """Get the attribute.

        This normally returns an instance of this class which is stored on the
        owner object.
        """
    def __set__(self, instance, names):
        """Set value of ``instance`` attribute to ``names``.

        Parameters
        ----------
        instance : object
            Instance that owns the attribute
        names : None, str, list, tuple
            Column name(s) to store, or None to clear
        """
    def __call__(self):
        """Get the value of the attribute.

        Returns
        -------
        names : None, tuple
            Include/exclude names
        """
    def __repr__(self) -> str: ...
    def _add_remove_setup(self, names):
        """Common setup for add and remove.

        - Coerce attribute value to a list
        - Coerce names into a list
        - Get the parent table instance
        """
    def add(self, names) -> None:
        """Add ``names`` to the include/exclude attribute.

        Parameters
        ----------
        names : str, list, tuple
            Column name(s) to add
        """
    def remove(self, names) -> None:
        """Remove ``names`` from the include/exclude attribute.

        Parameters
        ----------
        names : str, list, tuple
            Column name(s) to remove
        """
    def _remove(self, names, raise_exc: bool = False) -> None:
        """Remove ``names`` with optional checking if they exist."""
    def _rename(self, name, new_name) -> None:
        """Rename ``name`` to ``new_name`` if ``name`` is in the list."""
    descriptor_self: Incomplete
    names_orig: Incomplete
    def set(self, names):
        """Set value of include/exclude attribute to ``names``.

        Parameters
        ----------
        names : None, str, list, tuple
            Column name(s) to store, or None to clear
        """

class Table:
    """A class to represent tables of heterogeneous data.

    `~astropy.table.Table` provides a class for heterogeneous tabular data.
    A key enhancement provided by the `~astropy.table.Table` class over
    e.g. a `numpy` structured array is the ability to easily modify the
    structure of the table by adding or removing columns, or adding new
    rows of data.  In addition table and column metadata are fully supported.

    `~astropy.table.Table` differs from `~astropy.nddata.NDData` by the
    assumption that the input data consists of columns of homogeneous data,
    where each column has a unique identifier and may contain additional
    metadata such as the data unit, format, and description.

    See also: https://docs.astropy.org/en/stable/table/

    Parameters
    ----------
    data : numpy ndarray, dict, list, table-like object, optional
        Data to initialize table.
    masked : bool, optional
        Specify whether the table is masked.
    names : list, optional
        Specify column names.
    dtype : list, optional
        Specify column data types.
    meta : dict, optional
        Metadata associated with the table.
    copy : bool, optional
        Copy the input column data and make a deep copy of the input meta.
        Default is True.
    rows : numpy ndarray, list of list, optional
        Row-oriented data for table instead of ``data`` argument.
    copy_indices : bool, optional
        Copy any indices in the input data. Default is True.
    units : list, dict, optional
        List or dict of units to apply to columns.
    descriptions : list, dict, optional
        List or dict of descriptions to apply to columns.
    **kwargs : dict, optional
        Additional keyword args when converting table-like object.
    """
    meta: Incomplete
    Row = Row
    Column = Column
    MaskedColumn = MaskedColumn
    TableColumns = TableColumns
    TableFormatter = TableFormatter
    read: Incomplete
    write: Incomplete
    pprint_exclude_names: Incomplete
    pprint_include_names: Incomplete
    def as_array(self, keep_byteorder: bool = False, names: Incomplete | None = None):
        """
        Return a new copy of the table in the form of a structured np.ndarray or
        np.ma.MaskedArray object (as appropriate).

        Parameters
        ----------
        keep_byteorder : bool, optional
            By default the returned array has all columns in native byte
            order.  However, if this option is `True` this preserves the
            byte order of all columns (if any are non-native).

        names : list, optional:
            List of column names to include for returned structured array.
            Default is to include all table columns.

        Returns
        -------
        table_array : array or `~numpy.ma.MaskedArray`
            Copy of table as a numpy structured array.
            ndarray for unmasked or `~numpy.ma.MaskedArray` for masked.
        """
    columns: Incomplete
    formatter: Incomplete
    _copy_indices: bool
    _init_indices: Incomplete
    primary_key: Incomplete
    def __init__(self, data: Incomplete | None = None, masked: bool = False, names: Incomplete | None = None, dtype: Incomplete | None = None, meta: Incomplete | None = None, copy: bool = True, rows: Incomplete | None = None, copy_indices: bool = True, units: Incomplete | None = None, descriptions: Incomplete | None = None, **kwargs) -> None: ...
    def _set_column_attribute(self, attr, values) -> None:
        """Set ``attr`` for columns to ``values``, which can be either a dict (keyed by column
        name) or a dict of name: value pairs.  This is used for handling the ``units`` and
        ``descriptions`` kwargs to ``__init__``.
        """
    def __getstate__(self): ...
    def __setstate__(self, state) -> None: ...
    @property
    def mask(self): ...
    @mask.setter
    def mask(self, val) -> None: ...
    @property
    def _mask(self):
        """This is needed so that comparison of a masked Table and a
        MaskedArray works.  The requirement comes from numpy.ma.core
        so don't remove this property.
        """
    def filled(self, fill_value: Incomplete | None = None):
        """Return copy of self, with masked values filled.

        If input ``fill_value`` supplied then that value is used for all
        masked entries in the table.  Otherwise the individual
        ``fill_value`` defined for each table column is used.

        Parameters
        ----------
        fill_value : str
            If supplied, this ``fill_value`` is used for all masked entries
            in the entire table.

        Returns
        -------
        filled_table : `~astropy.table.Table`
            New table with masked values filled
        """
    @property
    def indices(self):
        """
        Return the indices associated with columns of the table
        as a TableIndices object.
        """
    @property
    def loc(self):
        """
        Return a TableLoc object that can be used for retrieving
        rows by index in a given data range. Note that both loc
        and iloc work only with single-column indices.
        """
    @property
    def loc_indices(self):
        """
        Return a TableLocIndices object that can be used for retrieving
        the row indices corresponding to given table index key value or values.
        """
    @property
    def iloc(self):
        """
        Return a TableILoc object that can be used for retrieving
        indexed rows in the order they appear in the index.
        """
    def add_index(self, colnames, engine: Incomplete | None = None, unique: bool = False) -> None:
        """
        Insert a new index among one or more columns.
        If there are no indices, make this index the
        primary table index.

        Parameters
        ----------
        colnames : str or list
            List of column names (or a single column name) to index
        engine : type or None
            Indexing engine class to use, either `~astropy.table.SortedArray`,
            `~astropy.table.BST`, or `~astropy.table.SCEngine`. If the supplied
            argument is None (by default), use `~astropy.table.SortedArray`.
        unique : bool (default: False)
            If set to True, an exception will be raised if duplicate rows exist.

        Raises
        ------
        ValueError
            If any selected column does not support indexing, or has more than
            one dimension.
        ValueError
            If unique=True and duplicate rows are found.

        """
    def remove_indices(self, colname) -> None:
        """
        Remove all indices involving the given column.
        If the primary index is removed, the new primary
        index will be the most recently added remaining
        index.

        Parameters
        ----------
        colname : str
            Name of column
        """
    def index_mode(self, mode):
        """
        Return a context manager for an indexing mode.

        Parameters
        ----------
        mode : str
            Either 'freeze', 'copy_on_getitem', or 'discard_on_copy'.
            In 'discard_on_copy' mode,
            indices are not copied whenever columns or tables are copied.
            In 'freeze' mode, indices are not modified whenever columns are
            modified; at the exit of the context, indices refresh themselves
            based on column values. This mode is intended for scenarios in
            which one intends to make many additions or modifications in an
            indexed column.
            In 'copy_on_getitem' mode, indices are copied when taking column
            slices as well as table slices, so col[i0:i1] will preserve
            indices.
        """
    def __array__(self, dtype: Incomplete | None = None, copy=...):
        """Support converting Table to np.array via np.array(table).

        Coercion to a different dtype via np.array(table, dtype) is not
        supported and will raise a ValueError.
        """
    def _check_names_dtype(self, names, dtype, n_cols) -> None:
        """Make sure that names and dtype are both iterable and have
        the same length as data.
        """
    def _init_from_list_of_dicts(self, data, names, dtype, n_cols, copy) -> None:
        """Initialize table from a list of dictionaries representing rows."""
    def _init_from_list(self, data, names, dtype, n_cols, copy) -> None:
        """Initialize table from a list of column data.  A column can be a
        Column object, np.ndarray, mixin, or any other iterable object.
        """
    def _convert_data_to_col(self, data, copy: bool = True, default_name: Incomplete | None = None, dtype: Incomplete | None = None, name: Incomplete | None = None):
        """
        Convert any allowed sequence data ``col`` to a column object that can be used
        directly in the self.columns dict.  This could be a Column, MaskedColumn,
        or mixin column.

        The final column name is determined by::

            name or data.info.name or def_name

        If ``data`` has no ``info`` then ``name = name or def_name``.

        The behavior of ``copy`` for Column objects is:
        - copy=True: new class instance with a copy of data and deep copy of meta
        - copy=False: new class instance with same data and a key-only copy of meta

        For mixin columns:
        - copy=True: new class instance with copy of data and deep copy of meta
        - copy=False: original instance (no copy at all)

        Parameters
        ----------
        data : object (column-like sequence)
            Input column data
        copy : bool
            Make a copy
        default_name : str
            Default name
        dtype : np.dtype or None
            Data dtype
        name : str or None
            Column name

        Returns
        -------
        col : Column, MaskedColumn, mixin-column type
            Object that can be used as a column in self
        """
    def _init_from_ndarray(self, data, names, dtype, n_cols, copy) -> None:
        """Initialize table from an ndarray structured array."""
    def _init_from_dict(self, data, names, dtype, n_cols, copy) -> None:
        """Initialize table from a dictionary of columns."""
    def _get_col_cls_for_table(self, col):
        """Get the correct column class to use for upgrading any Column-like object.

        For a masked table, ensure any Column-like object is a subclass
        of the table MaskedColumn.

        For unmasked table, ensure any MaskedColumn-like object is a subclass
        of the table MaskedColumn.  If not a MaskedColumn, then ensure that any
        Column-like object is a subclass of the table Column.
        """
    def _convert_col_for_table(self, col):
        """
        Make sure that all Column objects have correct base class for this type of
        Table.  For a base Table this most commonly means setting to
        MaskedColumn if the table is masked.  Table subclasses like QTable
        override this method.
        """
    def _init_from_cols(self, cols) -> None:
        """Initialize table from a list of Column or mixin objects."""
    def _new_from_slice(self, slice_):
        """Create a new table as a referenced slice from self."""
    @staticmethod
    def _make_table_from_cols(table, cols, verify: bool = True, names: Incomplete | None = None) -> None:
        """
        Make ``table`` in-place so that it represents the given list of ``cols``.
        """
    def _set_col_parent_table_and_mask(self, col) -> None:
        """
        Set ``col.parent_table = self`` and force ``col`` to have ``mask``
        attribute if the table is masked and ``col.mask`` does not exist.
        """
    def itercols(self) -> Generator[Incomplete]:
        """
        Iterate over the columns of this table.

        Examples
        --------
        To iterate over the columns of a table::

            >>> t = Table([[1], [2]])
            >>> for col in t.itercols():
            ...     print(col)
            col0
            ----
               1
            col1
            ----
               2

        Using ``itercols()`` is similar to  ``for col in t.columns.values()``
        but is syntactically preferred.
        """
    def _base_repr_(self, html: bool = False, descr_vals: Incomplete | None = None, max_width: Incomplete | None = None, tableid: Incomplete | None = None, show_dtype: bool = True, max_lines: Incomplete | None = None, tableclass: Incomplete | None = None): ...
    def _repr_html_(self): ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __bytes__(self) -> bytes: ...
    @property
    def has_mixin_columns(self):
        """
        True if table has any mixin columns (defined as columns that are not Column
        subclasses).
        """
    @property
    def has_masked_columns(self):
        """True if table has any ``MaskedColumn`` columns.

        This does not check for mixin columns that may have masked values, use the
        ``has_masked_values`` property in that case.

        """
    @property
    def has_masked_values(self):
        """True if column in the table has values which are masked.

        This may be relatively slow for large tables as it requires checking the mask
        values of each column.
        """
    def _is_mixin_for_table(self, col):
        """
        Determine if ``col`` should be added to the table directly as
        a mixin column.
        """
    def pprint(self, max_lines: Incomplete | None = None, max_width: Incomplete | None = None, show_name: bool = True, show_unit: Incomplete | None = None, show_dtype: bool = False, align: Incomplete | None = None) -> None:
        """Print a formatted string representation of the table.

        If no value of ``max_lines`` is supplied then the height of the
        screen terminal is used to set ``max_lines``.  If the terminal
        height cannot be determined then the default is taken from the
        configuration item ``astropy.conf.max_lines``.  If a negative
        value of ``max_lines`` is supplied then there is no line limit
        applied.

        The same applies for max_width except the configuration item is
        ``astropy.conf.max_width``.

        """
    def pprint_all(self, max_lines: int = -1, max_width: int = -1, show_name: bool = True, show_unit: Incomplete | None = None, show_dtype: bool = False, align: Incomplete | None = None):
        """Print a formatted string representation of the entire table.

        This method is the same as `astropy.table.Table.pprint` except that
        the default ``max_lines`` and ``max_width`` are both -1 so that by
        default the entire table is printed instead of restricting to the size
        of the screen terminal.

        """
    def _make_index_row_display_table(self, index_row_name): ...
    def show_in_notebook(self, *, backend: str = 'ipydatagrid', **kwargs):
        '''Render the table in HTML and show it in the Jupyter notebook.

        .. note:: The method API was modified in v7.0 to include a ``backend``
           argument and require only keyword arguments.

        Parameters
        ----------
        backend : {"ipydatagrid", "classic"}
            Backend to use for rendering (default="ipydatagrid"). The "classic" backend
            is deprecated since v6.1.

        **kwargs : dict, optional
            Keyword arguments as accepted by desired backend. See `astropy.table.notebook_backends`
            for the available backends and their respective keyword arguments.

        Raises
        ------
        NotImplementedError
            Requested backend is not supported.

        See Also
        --------
        astropy.table.notebook_backends

        '''
    def show_in_browser(self, max_lines: int = 5000, jsviewer: bool = False, browser: str = 'default', jskwargs={'use_local_files': True}, tableid: Incomplete | None = None, table_class: str = 'display compact', css: Incomplete | None = None, show_row_index: str = 'idx') -> None:
        '''Render the table in HTML and show it in a web browser.

        Parameters
        ----------
        max_lines : int
            Maximum number of rows to export to the table (set low by default
            to avoid memory issues, since the browser view requires duplicating
            the table in memory).  A negative value of ``max_lines`` indicates
            no row limit.
        jsviewer : bool
            If `True`, prepends some javascript headers so that the table is
            rendered as a `DataTables <https://datatables.net>`_ data table.
            This allows in-browser searching & sorting.
        browser : str
            Any legal browser name, e.g. ``\'firefox\'``, ``\'chrome\'``,
            ``\'safari\'`` (for mac, you may need to use ``\'open -a
            "/Applications/Google Chrome.app" {}\'`` for Chrome).  If
            ``\'default\'``, will use the system default browser.
        jskwargs : dict
            Passed to the `astropy.table.JSViewer` init. Defaults to
            ``{\'use_local_files\': True}`` which means that the JavaScript
            libraries will be served from local copies.
        tableid : str or None
            An html ID tag for the table.  Default is ``table{id}``, where id
            is the unique integer id of the table object, id(self).
        table_class : str or None
            A string with a list of HTML classes used to style the table.
            Default is "display compact", and other possible values can be
            found in https://www.datatables.net/manual/styling/classes
        css : str
            A valid CSS string declaring the formatting for the table. Defaults
            to ``astropy.table.jsviewer.DEFAULT_CSS``.
        show_row_index : str or False
            If this does not evaluate to False, a column with the given name
            will be added to the version of the table that gets displayed.
            This new column shows the index of the row in the table itself,
            even when the displayed table is re-sorted by another column. Note
            that if a column with this name already exists, this option will be
            ignored. Defaults to "idx".
        '''
    def pformat(self, max_lines: int = -1, max_width: int = -1, show_name: bool = True, show_unit: Incomplete | None = None, show_dtype: bool = False, html: bool = False, tableid: Incomplete | None = None, align: Incomplete | None = None, tableclass: Incomplete | None = None):
        """Return a list of lines for the formatted string representation of
        the table.

        If ``max_lines=None`` is supplied then the height of the
        screen terminal is used to set ``max_lines``.  If the terminal
        height cannot be determined then the default will be
        determined using the ``astropy.conf.max_lines`` configuration
        item. If a negative value of ``max_lines`` is supplied then
        there is no line limit applied (default).

        The same applies for ``max_width`` except the configuration item  is
        ``astropy.conf.max_width``.

        """
    def pformat_all(self, max_lines: int = -1, max_width: int = -1, show_name: bool = True, show_unit: Incomplete | None = None, show_dtype: bool = False, html: bool = False, tableid: Incomplete | None = None, align: Incomplete | None = None, tableclass: Incomplete | None = None):
        """Return a list of lines for the formatted string representation of
        the entire table.

        If ``max_lines=None`` is supplied then the height of the
        screen terminal is used to set ``max_lines``.  If the terminal
        height cannot be determined then the default will be
        determined using the ``astropy.conf.max_lines`` configuration
        item. If a negative value of ``max_lines`` is supplied then
        there is no line limit applied (default).

        The same applies for ``max_width`` except the configuration item  is
        ``astropy.conf.max_width``.

        """
    def more(self, max_lines: Incomplete | None = None, max_width: Incomplete | None = None, show_name: bool = True, show_unit: Incomplete | None = None, show_dtype: bool = False) -> None:
        """Interactively browse table with a paging interface.

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
            Maximum number of lines in table output

        max_width : int or None
            Maximum character width of output

        show_name : bool
            Include a header row for column names. Default is True.

        show_unit : bool
            Include a header row for unit.  Default is to show a row
            for units only if one or more columns has a defined value
            for the unit.

        show_dtype : bool
            Include a header row for column dtypes. Default is False.
        """
    def __getitem__(self, item): ...
    def __setitem__(self, item, value) -> None: ...
    def __delitem__(self, item) -> None: ...
    def _ipython_key_completions_(self): ...
    def field(self, item):
        """Return column[item] for recarray compatibility."""
    @property
    def masked(self): ...
    @masked.setter
    def masked(self, masked) -> None: ...
    _masked: Incomplete
    _column_class: Incomplete
    def _set_masked(self, masked) -> None:
        """
        Set the table masked property.

        Parameters
        ----------
        masked : bool
            State of table masking (`True` or `False`)
        """
    @property
    def ColumnClass(self): ...
    @property
    def dtype(self): ...
    @property
    def colnames(self): ...
    @staticmethod
    def _is_list_or_tuple_of_str(names):
        """Check that ``names`` is a tuple or list of strings."""
    def keys(self): ...
    def values(self): ...
    def items(self): ...
    _first_colname: Incomplete
    def __len__(self) -> int: ...
    def __or__(self, other): ...
    def __ior__(self, other): ...
    def index_column(self, name):
        """
        Return the positional index of column ``name``.

        Parameters
        ----------
        name : str
            column name

        Returns
        -------
        index : int
            Positional index of column ``name``.

        Examples
        --------
        Create a table with three columns 'a', 'b' and 'c'::

            >>> t = Table([[1, 2, 3], [0.1, 0.2, 0.3], ['x', 'y', 'z']],
            ...           names=('a', 'b', 'c'))
            >>> print(t)
             a   b   c
            --- --- ---
              1 0.1   x
              2 0.2   y
              3 0.3   z

        Get index of column 'b' of the table::

            >>> t.index_column('b')
            1
        """
    def add_column(self, col, index: Incomplete | None = None, name: Incomplete | None = None, rename_duplicate: bool = False, copy: bool = True, default_name: Incomplete | None = None) -> None:
        """
        Add a new column to the table using ``col`` as input.  If ``index``
        is supplied then insert column before ``index`` position
        in the list of columns, otherwise append column to the end
        of the list.

        The ``col`` input can be any data object which is acceptable as a
        `~astropy.table.Table` column object or can be converted.  This includes
        mixin columns and scalar or length=1 objects which get broadcast to match
        the table length.

        To add several columns at once use ``add_columns()`` or simply call
        ``add_column()`` for each one.  There is very little performance difference
        in the two approaches.

        Parameters
        ----------
        col : object
            Data object for the new column
        index : int or None
            Insert column before this position or at end (default).
        name : str
            Column name
        rename_duplicate : bool
            Uniquify column name if it already exist. Default is False.
        copy : bool
            Make a copy of the new column. Default is True.
        default_name : str or None
            Name to use if both ``name`` and ``col.info.name`` are not available.
            Defaults to ``col{number_of_columns}``.

        Examples
        --------
        Create a table with two columns 'a' and 'b', then create a third column 'c'
        and append it to the end of the table::

            >>> t = Table([[1, 2], [0.1, 0.2]], names=('a', 'b'))
            >>> col_c = Column(name='c', data=['x', 'y'])
            >>> t.add_column(col_c)
            >>> print(t)
             a   b   c
            --- --- ---
              1 0.1   x
              2 0.2   y

        Add column 'd' at position 1. Note that the column is inserted
        before the given index::

            >>> t.add_column(['a', 'b'], name='d', index=1)
            >>> print(t)
             a   d   b   c
            --- --- --- ---
              1   a 0.1   x
              2   b 0.2   y

        Add second column named 'b' with rename_duplicate::

            >>> t = Table([[1, 2], [0.1, 0.2]], names=('a', 'b'))
            >>> t.add_column(1.1, name='b', rename_duplicate=True)
            >>> print(t)
             a   b  b_1
            --- --- ---
              1 0.1 1.1
              2 0.2 1.1

        Add an unnamed column or mixin object in the table using a default name
        or by specifying an explicit name with ``name``. Name can also be overridden::

            >>> t = Table([[1, 2], [0.1, 0.2]], names=('a', 'b'))
            >>> t.add_column(['a', 'b'])
            >>> t.add_column(col_c, name='d')
            >>> print(t)
             a   b  col2  d
            --- --- ---- ---
              1 0.1    a   x
              2 0.2    b   y
        """
    def add_columns(self, cols, indexes: Incomplete | None = None, names: Incomplete | None = None, copy: bool = True, rename_duplicate: bool = False) -> None:
        """
        Add a list of new columns the table using ``cols`` data objects.  If a
        corresponding list of ``indexes`` is supplied then insert column
        before each ``index`` position in the *original* list of columns,
        otherwise append columns to the end of the list.

        The ``cols`` input can include any data objects which are acceptable as
        `~astropy.table.Table` column objects or can be converted.  This includes
        mixin columns and scalar or length=1 objects which get broadcast to match
        the table length.

        From a performance perspective there is little difference between calling
        this method once or looping over the new columns and calling ``add_column()``
        for each column.

        Parameters
        ----------
        cols : list of object
            List of data objects for the new columns
        indexes : list of int or None
            Insert column before this position or at end (default).
        names : list of str
            Column names
        copy : bool
            Make a copy of the new columns. Default is True.
        rename_duplicate : bool
            Uniquify new column names if they duplicate the existing ones.
            Default is False.

        See Also
        --------
        astropy.table.hstack, update, replace_column

        Examples
        --------
        Create a table with two columns 'a' and 'b', then create columns 'c' and 'd'
        and append them to the end of the table::

            >>> t = Table([[1, 2], [0.1, 0.2]], names=('a', 'b'))
            >>> col_c = Column(name='c', data=['x', 'y'])
            >>> col_d = Column(name='d', data=['u', 'v'])
            >>> t.add_columns([col_c, col_d])
            >>> print(t)
             a   b   c   d
            --- --- --- ---
              1 0.1   x   u
              2 0.2   y   v

        Add column 'c' at position 0 and column 'd' at position 1. Note that
        the columns are inserted before the given position::

            >>> t = Table([[1, 2], [0.1, 0.2]], names=('a', 'b'))
            >>> t.add_columns([['x', 'y'], ['u', 'v']], names=['c', 'd'],
            ...               indexes=[0, 1])
            >>> print(t)
             c   a   d   b
            --- --- --- ---
              x   1   u 0.1
              y   2   v 0.2

        Add second column 'b' and column 'c' with ``rename_duplicate``::

            >>> t = Table([[1, 2], [0.1, 0.2]], names=('a', 'b'))
            >>> t.add_columns([[1.1, 1.2], ['x', 'y']], names=('b', 'c'),
            ...               rename_duplicate=True)
            >>> print(t)
             a   b  b_1  c
            --- --- --- ---
              1 0.1 1.1  x
              2 0.2 1.2  y

        Add unnamed columns or mixin objects in the table using default names
        or by specifying explicit names with ``names``. Names can also be overridden::

            >>> t = Table()
            >>> col_b = Column(name='b', data=['u', 'v'])
            >>> t.add_columns([[1, 2], col_b])
            >>> t.add_columns([[3, 4], col_b], names=['c', 'd'])
            >>> print(t)
            col0  b   c   d
            ---- --- --- ---
               1   u   3   u
               2   v   4   v
        """
    def _replace_column_warnings(self, name, col) -> None:
        """
        Same as replace_column but issues warnings under various circumstances.
        """
    def replace_column(self, name, col, copy: bool = True) -> None:
        """
        Replace column ``name`` with the new ``col`` object.

        The behavior of ``copy`` for Column objects is:
        - copy=True: new class instance with a copy of data and deep copy of meta
        - copy=False: new class instance with same data and a key-only copy of meta

        For mixin columns:
        - copy=True: new class instance with copy of data and deep copy of meta
        - copy=False: original instance (no copy at all)

        Parameters
        ----------
        name : str
            Name of column to replace
        col : `~astropy.table.Column` or `~numpy.ndarray` or sequence
            New column object to replace the existing column.
        copy : bool
            Make copy of the input ``col``, default=True

        See Also
        --------
        add_columns, astropy.table.hstack, update

        Examples
        --------
        Replace column 'a' with a float version of itself::

            >>> t = Table([[1, 2, 3], [0.1, 0.2, 0.3]], names=('a', 'b'))
            >>> float_a = t['a'].astype(float)
            >>> t.replace_column('a', float_a)
        """
    def remove_row(self, index) -> None:
        """
        Remove a row from the table.

        Parameters
        ----------
        index : int
            Index of row to remove

        Examples
        --------
        Create a table with three columns 'a', 'b' and 'c'::

            >>> t = Table([[1, 2, 3], [0.1, 0.2, 0.3], ['x', 'y', 'z']],
            ...           names=('a', 'b', 'c'))
            >>> print(t)
             a   b   c
            --- --- ---
              1 0.1   x
              2 0.2   y
              3 0.3   z

        Remove row 1 from the table::

            >>> t.remove_row(1)
            >>> print(t)
             a   b   c
            --- --- ---
              1 0.1   x
              3 0.3   z

        To remove several rows at the same time use remove_rows.
        """
    def remove_rows(self, row_specifier) -> None:
        """
        Remove rows from the table.

        Parameters
        ----------
        row_specifier : slice or int or array of int
            Specification for rows to remove

        Examples
        --------
        Create a table with three columns 'a', 'b' and 'c'::

            >>> t = Table([[1, 2, 3], [0.1, 0.2, 0.3], ['x', 'y', 'z']],
            ...           names=('a', 'b', 'c'))
            >>> print(t)
             a   b   c
            --- --- ---
              1 0.1   x
              2 0.2   y
              3 0.3   z

        Remove rows 0 and 2 from the table::

            >>> t.remove_rows([0, 2])
            >>> print(t)
             a   b   c
            --- --- ---
              2 0.2   y


        Note that there are no warnings if the slice operator extends
        outside the data::

            >>> t = Table([[1, 2, 3], [0.1, 0.2, 0.3], ['x', 'y', 'z']],
            ...           names=('a', 'b', 'c'))
            >>> t.remove_rows(slice(10, 20, 1))
            >>> print(t)
             a   b   c
            --- --- ---
              1 0.1   x
              2 0.2   y
              3 0.3   z
        """
    def iterrows(self, *names):
        """
        Iterate over rows of table returning a tuple of values for each row.

        This method is especially useful when only a subset of columns are needed.

        The ``iterrows`` method can be substantially faster than using the standard
        Table row iteration (e.g. ``for row in tbl:``), since that returns a new
        ``~astropy.table.Row`` object for each row and accessing a column in that
        row (e.g. ``row['col0']``) is slower than tuple access.

        Parameters
        ----------
        names : list
            List of column names (default to all columns if no names provided)

        Returns
        -------
        rows : iterable
            Iterator returns tuples of row values

        Examples
        --------
        Create a table with three columns 'a', 'b' and 'c'::

            >>> t = Table({'a': [1, 2, 3],
            ...            'b': [1.0, 2.5, 3.0],
            ...            'c': ['x', 'y', 'z']})

        To iterate row-wise using column names::

            >>> for a, c in t.iterrows('a', 'c'):
            ...     print(a, c)
            1 x
            2 y
            3 z

        """
    def _set_of_names_in_colnames(self, names):
        """Return ``names`` as a set if valid, or raise a `KeyError`.

        ``names`` is valid if all elements in it are in ``self.colnames``.
        If ``names`` is a string then it is interpreted as a single column
        name.
        """
    def remove_column(self, name) -> None:
        """
        Remove a column from the table.

        This can also be done with::

          del table[name]

        Parameters
        ----------
        name : str
            Name of column to remove

        Examples
        --------
        Create a table with three columns 'a', 'b' and 'c'::

            >>> t = Table([[1, 2, 3], [0.1, 0.2, 0.3], ['x', 'y', 'z']],
            ...           names=('a', 'b', 'c'))
            >>> print(t)
             a   b   c
            --- --- ---
              1 0.1   x
              2 0.2   y
              3 0.3   z

        Remove column 'b' from the table::

            >>> t.remove_column('b')
            >>> print(t)
             a   c
            --- ---
              1   x
              2   y
              3   z

        To remove several columns at the same time use remove_columns.
        """
    def remove_columns(self, names) -> None:
        """
        Remove several columns from the table.

        Parameters
        ----------
        names : str or iterable of str
            Names of the columns to remove

        Examples
        --------
        Create a table with three columns 'a', 'b' and 'c'::

            >>> t = Table([[1, 2, 3], [0.1, 0.2, 0.3], ['x', 'y', 'z']],
            ...     names=('a', 'b', 'c'))
            >>> print(t)
             a   b   c
            --- --- ---
              1 0.1   x
              2 0.2   y
              3 0.3   z

        Remove columns 'b' and 'c' from the table::

            >>> t.remove_columns(['b', 'c'])
            >>> print(t)
             a
            ---
              1
              2
              3

        Specifying only a single column also works. Remove column 'b' from the table::

            >>> t = Table([[1, 2, 3], [0.1, 0.2, 0.3], ['x', 'y', 'z']],
            ...     names=('a', 'b', 'c'))
            >>> t.remove_columns('b')
            >>> print(t)
             a   c
            --- ---
              1   x
              2   y
              3   z

        This gives the same as using remove_column.
        """
    def _convert_string_dtype(self, in_kind, out_kind, encode_decode_func) -> None:
        """
        Convert string-like columns to/from bytestring and unicode (internal only).

        Parameters
        ----------
        in_kind : str
            Input dtype.kind
        out_kind : str
            Output dtype.kind
        """
    def convert_bytestring_to_unicode(self) -> None:
        """
        Convert bytestring columns (dtype.kind='S') to unicode (dtype.kind='U')
        using UTF-8 encoding.

        Internally this changes string columns to represent each character
        in the string with a 4-byte UCS-4 equivalent, so it is inefficient
        for memory but allows scripts to manipulate string arrays with
        natural syntax.
        """
    def convert_unicode_to_bytestring(self) -> None:
        """
        Convert unicode columns (dtype.kind='U') to bytestring (dtype.kind='S')
        using UTF-8 encoding.

        When exporting a unicode string array to a file, it may be desirable
        to encode unicode columns as bytestrings.
        """
    def keep_columns(self, names) -> None:
        """
        Keep only the columns specified (remove the others).

        Parameters
        ----------
        names : str or iterable of str
            The columns to keep. All other columns will be removed.

        Examples
        --------
        Create a table with three columns 'a', 'b' and 'c'::

            >>> t = Table([[1, 2, 3],[0.1, 0.2, 0.3],['x', 'y', 'z']],
            ...           names=('a', 'b', 'c'))
            >>> print(t)
             a   b   c
            --- --- ---
              1 0.1   x
              2 0.2   y
              3 0.3   z

        Keep only column 'a' of the table::

            >>> t.keep_columns('a')
            >>> print(t)
             a
            ---
              1
              2
              3

        Keep columns 'a' and 'c' of the table::

            >>> t = Table([[1, 2, 3],[0.1, 0.2, 0.3],['x', 'y', 'z']],
            ...           names=('a', 'b', 'c'))
            >>> t.keep_columns(['a', 'c'])
            >>> print(t)
             a   c
            --- ---
              1   x
              2   y
              3   z
        """
    def rename_column(self, name, new_name) -> None:
        """
        Rename a column.

        This can also be done directly by setting the ``name`` attribute
        of the ``info`` property of the column::

          table[name].info.name = new_name

        Parameters
        ----------
        name : str
            The current name of the column.
        new_name : str
            The new name for the column

        Examples
        --------
        Create a table with three columns 'a', 'b' and 'c'::

            >>> t = Table([[1,2],[3,4],[5,6]], names=('a','b','c'))
            >>> print(t)
             a   b   c
            --- --- ---
              1   3   5
              2   4   6

        Renaming column 'a' to 'aa'::

            >>> t.rename_column('a' , 'aa')
            >>> print(t)
             aa  b   c
            --- --- ---
              1   3   5
              2   4   6
        """
    def rename_columns(self, names, new_names) -> None:
        """
        Rename multiple columns.

        Parameters
        ----------
        names : list, tuple
            A list or tuple of existing column names.
        new_names : list, tuple
            A list or tuple of new column names.

        Examples
        --------
        Create a table with three columns 'a', 'b', 'c'::

            >>> t = Table([[1,2],[3,4],[5,6]], names=('a','b','c'))
            >>> print(t)
              a   b   c
             --- --- ---
              1   3   5
              2   4   6

        Renaming columns 'a' to 'aa' and 'b' to 'bb'::

            >>> names = ('a','b')
            >>> new_names = ('aa','bb')
            >>> t.rename_columns(names, new_names)
            >>> print(t)
             aa  bb   c
            --- --- ---
              1   3   5
              2   4   6
        """
    def _set_row(self, idx, colnames, vals) -> None: ...
    def add_row(self, vals: Incomplete | None = None, mask: Incomplete | None = None) -> None:
        '''Add a new row to the end of the table.

        The ``vals`` argument can be:

        sequence (e.g. tuple or list)
            Column values in the same order as table columns.
        mapping (e.g. dict)
            Keys corresponding to column names.  Missing values will be
            filled with np.zeros for the column dtype.
        `None`
            All values filled with np.zeros for the column dtype.

        This method requires that the Table object "owns" the underlying array
        data.  In particular one cannot add a row to a Table that was
        initialized with copy=False from an existing array.

        The ``mask`` attribute should give (if desired) the mask for the
        values. The type of the mask should match that of the values, i.e. if
        ``vals`` is an iterable, then ``mask`` should also be an iterable
        with the same length, and if ``vals`` is a mapping, then ``mask``
        should be a dictionary.

        Parameters
        ----------
        vals : tuple, list, dict or None
            Use the specified values in the new row
        mask : tuple, list, dict or None
            Use the specified mask values in the new row

        Examples
        --------
        Create a table with three columns \'a\', \'b\' and \'c\'::

           >>> t = Table([[1,2],[4,5],[7,8]], names=(\'a\',\'b\',\'c\'))
           >>> print(t)
            a   b   c
           --- --- ---
             1   4   7
             2   5   8

        Adding a new row with entries \'3\' in \'a\', \'6\' in \'b\' and \'9\' in \'c\'::

           >>> t.add_row([3,6,9])
           >>> print(t)
             a   b   c
             --- --- ---
             1   4   7
             2   5   8
             3   6   9
        '''
    def insert_row(self, index, vals: Incomplete | None = None, mask: Incomplete | None = None) -> None:
        """Add a new row before the given ``index`` position in the table.

        The ``vals`` argument can be:

        sequence (e.g. tuple or list)
            Column values in the same order as table columns.
        mapping (e.g. dict)
            Keys corresponding to column names.  Missing values will be
            filled with np.zeros for the column dtype.
        `None`
            All values filled with np.zeros for the column dtype.

        The ``mask`` attribute should give (if desired) the mask for the
        values. The type of the mask should match that of the values, i.e. if
        ``vals`` is an iterable, then ``mask`` should also be an iterable
        with the same length, and if ``vals`` is a mapping, then ``mask``
        should be a dictionary.

        Parameters
        ----------
        vals : tuple, list, dict or None
            Use the specified values in the new row
        mask : tuple, list, dict or None
            Use the specified mask values in the new row
        """
    def _replace_cols(self, columns) -> None: ...
    def setdefault(self, name, default):
        '''Ensure a column named ``name`` exists.

        If ``name`` is already present then ``default`` is ignored.
        Otherwise ``default`` can be any data object which is acceptable as
        a `~astropy.table.Table` column object or can be converted.  This
        includes mixin columns and scalar or length=1 objects which get
        broadcast to match the table length.

        Parameters
        ----------
        name : str
            Name of the column.
        default : object
            Data object for the new column.

        Returns
        -------
        `~astropy.table.Column`, `~astropy.table.MaskedColumn` or mixin-column type
            The column named ``name`` if it is present already, or the
            validated ``default`` converted to a column otherwise.

        Raises
        ------
        TypeError
            If the table is empty and ``default`` is a scalar object.

        Examples
        --------
        Start with a simple table::

          >>> t0 = Table({"a": ["Ham", "Spam"]})
          >>> t0
          <Table length=2>
           a
          str4
          ----
           Ham
          Spam

        Trying to add a column that already exists does not modify it::

          >>> t0.setdefault("a", ["Breakfast"])
          <Column name=\'a\' dtype=\'str4\' length=2>
           Ham
          Spam
          >>> t0
          <Table length=2>
           a
          str4
          ----
           Ham
          Spam

        But if the column does not exist it will be created with the
        default value::

          >>> t0.setdefault("approved", False)
          <Column name=\'approved\' dtype=\'bool\' length=2>
          False
          False
          >>> t0
          <Table length=2>
           a   approved
          str4   bool
          ---- --------
           Ham    False
          Spam    False
        '''
    def update(self, other, copy: bool = True) -> None:
        """
        Perform a dictionary-style update and merge metadata.

        The argument ``other`` must be a |Table|, or something that can be used
        to initialize a table. Columns from (possibly converted) ``other`` are
        added to this table. In case of matching column names the column from
        this table is replaced with the one from ``other``. If ``other`` is a
        |Table| instance then ``|=`` is available as alternate syntax for in-place
        update and ``|`` can be used merge data to a new table.

        Parameters
        ----------
        other : table-like
            Data to update this table with.
        copy : bool
            Whether the updated columns should be copies of or references to
            the originals.

        See Also
        --------
        add_columns, astropy.table.hstack, replace_column

        Examples
        --------
        Update a table with another table::

            >>> t1 = Table({'a': ['foo', 'bar'], 'b': [0., 0.]}, meta={'i': 0})
            >>> t2 = Table({'b': [1., 2.], 'c': [7., 11.]}, meta={'n': 2})
            >>> t1.update(t2)
            >>> t1
            <Table length=2>
             a      b       c
            str3 float64 float64
            ---- ------- -------
             foo     1.0     7.0
             bar     2.0    11.0
            >>> t1.meta
            {'i': 0, 'n': 2}

        Update a table with a dictionary::

            >>> t = Table({'a': ['foo', 'bar'], 'b': [0., 0.]})
            >>> t.update({'b': [1., 2.]})
            >>> t
            <Table length=2>
             a      b
            str3 float64
            ---- -------
             foo     1.0
             bar     2.0
        """
    def argsort(self, keys: Incomplete | None = None, kind: Incomplete | None = None, reverse: bool = False):
        """
        Return the indices which would sort the table according to one or
        more key columns.  This simply calls the `numpy.argsort` function on
        the table with the ``order`` parameter set to ``keys``.

        Parameters
        ----------
        keys : str or list of str
            The column name(s) to order the table by
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            Sorting algorithm used by ``numpy.argsort``.
        reverse : bool
            Sort in reverse order (default=False)

        Returns
        -------
        index_array : ndarray, int
            Array of indices that sorts the table by the specified key
            column(s).
        """
    def sort(self, keys: Incomplete | None = None, *, kind: Incomplete | None = None, reverse: bool = False) -> None:
        """
        Sort the table according to one or more keys. This operates
        on the existing table and does not return a new table.

        Parameters
        ----------
        keys : str or list of str
            The key(s) to order the table by. If None, use the
            primary index of the Table.
        kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
            Sorting algorithm used by ``numpy.argsort``.
        reverse : bool
            Sort in reverse order (default=False)

        Examples
        --------
        Create a table with 3 columns::

            >>> t = Table([['Max', 'Jo', 'John'], ['Miller', 'Miller', 'Jackson'],
            ...            [12, 15, 18]], names=('firstname', 'name', 'tel'))
            >>> print(t)
            firstname   name  tel
            --------- ------- ---
                  Max  Miller  12
                   Jo  Miller  15
                 John Jackson  18

        Sorting according to standard sorting rules, first 'name' then 'firstname'::

            >>> t.sort(['name', 'firstname'])
            >>> print(t)
            firstname   name  tel
            --------- ------- ---
                 John Jackson  18
                   Jo  Miller  15
                  Max  Miller  12

        Sorting according to standard sorting rules, first 'firstname' then 'tel',
        in reverse order::

            >>> t.sort(['firstname', 'tel'], reverse=True)
            >>> print(t)
            firstname   name  tel
            --------- ------- ---
                  Max  Miller  12
                 John Jackson  18
                   Jo  Miller  15
        """
    def reverse(self) -> None:
        """
        Reverse the row order of table rows.  The table is reversed
        in place and there are no function arguments.

        Examples
        --------
        Create a table with three columns::

            >>> t = Table([['Max', 'Jo', 'John'], ['Miller','Miller','Jackson'],
            ...         [12,15,18]], names=('firstname','name','tel'))
            >>> print(t)
            firstname   name  tel
            --------- ------- ---
                  Max  Miller  12
                   Jo  Miller  15
                 John Jackson  18

        Reversing order::

            >>> t.reverse()
            >>> print(t)
            firstname   name  tel
            --------- ------- ---
                 John Jackson  18
                   Jo  Miller  15
                  Max  Miller  12
        """
    def round(self, decimals: int = 0) -> None:
        """
        Round numeric columns in-place to the specified number of decimals.
        Non-numeric columns will be ignored.

        Examples
        --------
        Create three columns with different types:

            >>> t = Table([[1, 4, 5], [-25.55, 12.123, 85],
            ...     ['a', 'b', 'c']], names=('a', 'b', 'c'))
            >>> print(t)
             a    b     c
            --- ------ ---
              1 -25.55   a
              4 12.123   b
              5   85.0   c

        Round them all to 0:

            >>> t.round(0)
            >>> print(t)
             a    b    c
            --- ----- ---
              1 -26.0   a
              4  12.0   b
              5  85.0   c

        Round column 'a' to -1 decimal:

            >>> t.round({'a':-1})
            >>> print(t)
             a    b    c
            --- ----- ---
              0 -26.0   a
              0  12.0   b
              0  85.0   c

        Parameters
        ----------
        decimals: int, dict
            Number of decimals to round the columns to. If a dict is given,
            the columns will be rounded to the number specified as the value.
            If a certain column is not in the dict given, it will remain the
            same.
        """
    def copy(self, copy_data: bool = True):
        """
        Return a copy of the table.

        Parameters
        ----------
        copy_data : bool
            If `True` (the default), copy the underlying data array and make
            a deep copy of the ``meta`` attribute. Otherwise, use the same
            data array and make a shallow (key-only) copy of ``meta``.
        """
    def __deepcopy__(self, memo: Incomplete | None = None): ...
    def __copy__(self): ...
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    def _rows_equal(self, other):
        """
        Row-wise comparison of table with any other object.

        This is actual implementation for __eq__.

        Returns a 1-D boolean numpy array showing result of row-wise comparison,
        or a bool (False) in cases where comparison isn't possible (uncomparable dtypes
        or unbroadcastable shapes). Intended to follow legacy numpy's elementwise
        comparison rules.

        This is the same as the ``==`` comparison for tables.

        Parameters
        ----------
        other : Table or DataFrame or ndarray
             An object to compare with table

        Examples
        --------
        Comparing one Table with other::

            >>> t1 = Table([[1,2],[4,5],[7,8]], names=('a','b','c'))
            >>> t2 = Table([[1,2],[4,5],[7,8]], names=('a','b','c'))
            >>> t1._rows_equal(t2)
            array([ True,  True])

        """
    def values_equal(self, other):
        """
        Element-wise comparison of table with another table, list, or scalar.

        Returns a ``Table`` with the same columns containing boolean values
        showing result of comparison.

        Parameters
        ----------
        other : table-like object or list or scalar
             Object to compare with table

        Examples
        --------
        Compare one Table with other::

          >>> t1 = Table([[1, 2], [4, 5], [-7, 8]], names=('a', 'b', 'c'))
          >>> t2 = Table([[1, 2], [-4, 5], [7, 8]], names=('a', 'b', 'c'))
          >>> t1.values_equal(t2)
          <Table length=2>
           a     b     c
          bool  bool  bool
          ---- ----- -----
          True False False
          True  True  True

        """
    _groups: Incomplete
    @property
    def groups(self): ...
    def group_by(self, keys):
        """
        Group this table by the specified ``keys``.

        This effectively splits the table into groups which correspond to unique
        values of the ``keys`` grouping object.  The output is a new
        `~astropy.table.TableGroups` which contains a copy of this table but
        sorted by row according to ``keys``.

        The ``keys`` input to `group_by` can be specified in different ways:

          - String or list of strings corresponding to table column name(s)
          - Numpy array (homogeneous or structured) with same length as this table
          - `~astropy.table.Table` with same length as this table

        Parameters
        ----------
        keys : str, list of str, numpy array, or `~astropy.table.Table`
            Key grouping object

        Returns
        -------
        out : `~astropy.table.Table`
            New table with groups set
        """
    def to_pandas(self, index: Incomplete | None = None, use_nullable_int: bool = True):
        '''
        Return a :class:`pandas.DataFrame` instance.

        The index of the created DataFrame is controlled by the ``index``
        argument.  For ``index=True`` or the default ``None``, an index will be
        specified for the DataFrame if there is a primary key index on the
        Table *and* if it corresponds to a single column.  If ``index=False``
        then no DataFrame index will be specified.  If ``index`` is the name of
        a column in the table then that will be the DataFrame index.

        In addition to vanilla columns or masked columns, this supports Table
        mixin columns like Quantity, Time, or SkyCoord.  In many cases these
        objects have no analog in pandas and will be converted to a "encoded"
        representation using only Column or MaskedColumn.  The exception is
        Time or TimeDelta columns, which will be converted to the corresponding
        representation in pandas using ``np.datetime64`` or ``np.timedelta64``.
        See the example below.

        Parameters
        ----------
        index : None, bool, str
            Specify DataFrame index mode
        use_nullable_int : bool, default=True
            Convert integer MaskedColumn to pandas nullable integer type.  If
            ``use_nullable_int=False`` then the column is converted to float
            with NaN.

        Returns
        -------
        dataframe : :class:`pandas.DataFrame`
            A pandas :class:`pandas.DataFrame` instance

        Raises
        ------
        ImportError
            If pandas is not installed
        ValueError
            If the Table has multi-dimensional columns

        Examples
        --------
        Here we convert a table with a few mixins to a
        :class:`pandas.DataFrame` instance.

          >>> import pandas as pd
          >>> from astropy.table import QTable
          >>> import astropy.units as u
          >>> from astropy.time import Time, TimeDelta
          >>> from astropy.coordinates import SkyCoord

          >>> q = [1, 2] * u.m
          >>> tm = Time([1998, 2002], format=\'jyear\')
          >>> sc = SkyCoord([5, 6], [7, 8], unit=\'deg\')
          >>> dt = TimeDelta([3, 200] * u.s)

          >>> t = QTable([q, tm, sc, dt], names=[\'q\', \'tm\', \'sc\', \'dt\'])

          >>> df = t.to_pandas(index=\'tm\')
          >>> with pd.option_context(\'display.max_columns\', 20):
          ...     print(df)
                        q  sc.ra  sc.dec              dt
          tm
          1998-01-01  1.0    5.0     7.0 0 days 00:00:03
          2002-01-01  2.0    6.0     8.0 0 days 00:03:20

        '''
    @classmethod
    def from_pandas(cls, dataframe, index: bool = False, units: Incomplete | None = None):
        """
        Create a `~astropy.table.Table` from a :class:`pandas.DataFrame` instance.

        In addition to converting generic numeric or string columns, this supports
        conversion of pandas Date and Time delta columns to `~astropy.time.Time`
        and `~astropy.time.TimeDelta` columns, respectively.

        Parameters
        ----------
        dataframe : :class:`pandas.DataFrame`
            A pandas :class:`pandas.DataFrame` instance
        index : bool
            Include the index column in the returned table (default=False)
        units: dict
            A dict mapping column names to a `~astropy.units.Unit`.
            The columns will have the specified unit in the Table.

        Returns
        -------
        table : `~astropy.table.Table`
            A `~astropy.table.Table` (or subclass) instance

        Raises
        ------
        ImportError
            If pandas is not installed

        Examples
        --------
        Here we convert a :class:`pandas.DataFrame` instance
        to a `~astropy.table.QTable`.

          >>> import numpy as np
          >>> import pandas as pd
          >>> from astropy.table import QTable

          >>> time = pd.Series(['1998-01-01', '2002-01-01'], dtype='datetime64[ns]')
          >>> dt = pd.Series(np.array([1, 300], dtype='timedelta64[s]'))
          >>> df = pd.DataFrame({'time': time})
          >>> df['dt'] = dt
          >>> df['x'] = [3., 4.]
          >>> with pd.option_context('display.max_columns', 20):
          ...     print(df)
                  time              dt    x
          0 1998-01-01 0 days 00:00:01  3.0
          1 2002-01-01 0 days 00:05:00  4.0

          >>> QTable.from_pandas(df)
          <QTable length=2>
                    time              dt       x
                    Time          TimeDelta float64
          ----------------------- --------- -------
          1998-01-01T00:00:00.000       1.0     3.0
          2002-01-01T00:00:00.000     300.0     4.0

        """
    info: Incomplete

class QTable(Table):
    """A class to represent tables of heterogeneous data.

    `~astropy.table.QTable` provides a class for heterogeneous tabular data
    which can be easily modified, for instance adding columns or new rows.

    The `~astropy.table.QTable` class is identical to `~astropy.table.Table`
    except that columns with an associated ``unit`` attribute are converted to
    `~astropy.units.Quantity` objects.

    For more information see:

    - https://docs.astropy.org/en/stable/table/
    - https://docs.astropy.org/en/stable/table/mixin_columns.html

    Parameters
    ----------
    data : numpy ndarray, dict, list, table-like object, optional
        Data to initialize table.
    masked : bool, optional
        Specify whether the table is masked.
    names : list, optional
        Specify column names.
    dtype : list, optional
        Specify column data types.
    meta : dict, optional
        Metadata associated with the table.
    copy : bool, optional
        Copy the input data. If the input is a (Q)Table the ``meta`` is always
        copied regardless of the ``copy`` parameter.
        Default is True.
    rows : numpy ndarray, list of list, optional
        Row-oriented data for table instead of ``data`` argument.
    copy_indices : bool, optional
        Copy any indices in the input data. Default is True.
    units : list, dict, optional
        List or dict of units to apply to columns.
    descriptions : list, dict, optional
        List or dict of descriptions to apply to columns.
    **kwargs : dict, optional
        Additional keyword args when converting table-like object.

    """
    def _is_mixin_for_table(self, col):
        """
        Determine if ``col`` should be added to the table directly as
        a mixin column.
        """
    def _convert_col_for_table(self, col): ...
    def _convert_data_to_col(self, data, copy: bool = True, default_name: Incomplete | None = None, dtype: Incomplete | None = None, name: Incomplete | None = None): ...

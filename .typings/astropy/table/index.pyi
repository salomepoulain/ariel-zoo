from .bst import MaxValue as MaxValue, MinValue as MinValue
from .sorted_array import SortedArray as SortedArray
from _typeshed import Incomplete

class QueryError(ValueError):
    """
    Indicates that a given index cannot handle the supplied query.
    """

class Index:
    """
    The Index class makes it possible to maintain indices
    on columns of a Table, so that column values can be queried
    quickly and efficiently. Column values are stored in lexicographic
    sorted order, which allows for binary searching in O(log n).

    Parameters
    ----------
    columns : list or None
        List of columns on which to create an index. If None,
        create an empty index for purposes of deep copying.
    engine : type, instance, or None
        Indexing engine class to use (from among SortedArray, BST,
        and SCEngine) or actual engine instance.
        If the supplied argument is None (by default), use SortedArray.
    unique : bool (defaults to False)
        Whether the values of the index must be unique
    """
    engine: Incomplete
    data: Incomplete
    columns: Incomplete
    def __init__(self, columns, engine: Incomplete | None = None, unique: bool = False) -> None: ...
    def __len__(self) -> int:
        """
        Number of rows in index.
        """
    def replace_col(self, prev_col, new_col) -> None:
        """
        Replace an indexed column with an updated reference.

        Parameters
        ----------
        prev_col : Column
            Column reference to replace
        new_col : Column
            New column reference
        """
    def reload(self) -> None:
        """
        Recreate the index based on data in self.columns.
        """
    def col_position(self, col_name):
        """
        Return the position of col_name in self.columns.

        Parameters
        ----------
        col_name : str
            Name of column to look up
        """
    def insert_row(self, pos, vals, columns) -> None:
        """
        Insert a new row from the given values.

        Parameters
        ----------
        pos : int
            Position at which to insert row
        vals : list or tuple
            List of values to insert into a new row
        columns : list
            Table column references
        """
    def get_row_specifier(self, row_specifier):
        """
        Return an iterable corresponding to the
        input row specifier.

        Parameters
        ----------
        row_specifier : int, list, ndarray, or slice
        """
    def remove_rows(self, row_specifier) -> None:
        """
        Remove the given rows from the index.

        Parameters
        ----------
        row_specifier : int, list, ndarray, or slice
            Indicates which row(s) to remove
        """
    def remove_row(self, row, reorder: bool = True) -> None:
        """
        Remove the given row from the index.

        Parameters
        ----------
        row : int
            Position of row to remove
        reorder : bool
            Whether to reorder indices after removal
        """
    def find(self, key):
        """
        Return the row values corresponding to key, in sorted order.

        Parameters
        ----------
        key : tuple
            Values to search for in each column
        """
    def same_prefix(self, key):
        """
        Return rows whose keys contain the supplied key as a prefix.

        Parameters
        ----------
        key : tuple
            Prefix for which to search
        """
    def same_prefix_range(self, lower, upper, bounds=(True, True)):
        """
        Return rows whose keys have a prefix in the given range.

        Parameters
        ----------
        lower : tuple
            Lower prefix bound
        upper : tuple
            Upper prefix bound
        bounds : tuple (x, y) of bools
            Indicates whether the search should be inclusive or
            exclusive with respect to the endpoints. The first
            argument x corresponds to an inclusive lower bound,
            and the second argument y to an inclusive upper bound.
        """
    def range(self, lower, upper, bounds=(True, True)):
        """
        Return rows within the given range.

        Parameters
        ----------
        lower : tuple
            Lower prefix bound
        upper : tuple
            Upper prefix bound
        bounds : tuple (x, y) of bools
            Indicates whether the search should be inclusive or
            exclusive with respect to the endpoints. The first
            argument x corresponds to an inclusive lower bound,
            and the second argument y to an inclusive upper bound.
        """
    def replace(self, row, col_name, val) -> None:
        """
        Replace the value of a column at a given position.

        Parameters
        ----------
        row : int
            Row number to modify
        col_name : str
            Name of the Column to modify
        val : col.info.dtype
            Value to insert at specified row of col
        """
    def replace_rows(self, col_slice) -> None:
        """
        Modify rows in this index to agree with the specified
        slice. For example, given an index
        {'5': 1, '2': 0, '3': 2} on a column ['2', '5', '3'],
        an input col_slice of [2, 0] will result in the relabeling
        {'3': 0, '2': 1} on the sliced column ['3', '2'].

        Parameters
        ----------
        col_slice : list
            Indices to slice
        """
    def sort(self) -> None:
        """
        Make row numbers follow the same sort order as the keys
        of the index.
        """
    def sorted_data(self):
        """
        Returns a list of rows in sorted order based on keys;
        essentially acts as an argsort() on columns.
        """
    def __getitem__(self, item):
        """
        Returns a sliced version of this index.

        Parameters
        ----------
        item : slice
            Input slice

        Returns
        -------
        SlicedIndex
            A sliced reference to this index.
        """
    def __repr__(self) -> str: ...
    def __deepcopy__(self, memo):
        """
        Return a deep copy of this index.

        Notes
        -----
        The default deep copy must be overridden to perform
        a shallow copy of the index columns, avoiding infinite recursion.

        Parameters
        ----------
        memo : dict
        """

class SlicedIndex:
    """
    This class provides a wrapper around an actual Index object
    to make index slicing function correctly. Since numpy expects
    array slices to provide an actual data view, a SlicedIndex should
    retrieve data directly from the original index and then adapt
    it to the sliced coordinate system as appropriate.

    Parameters
    ----------
    index : Index
        The original Index reference
    index_slice : tuple, slice
        The slice to which this SlicedIndex corresponds
    original : bool
        Whether this SlicedIndex represents the original index itself.
        For the most part this is similar to index[:] but certain
        copying operations are avoided, and the slice retains the
        length of the actual index despite modification.
    """
    index: Incomplete
    original: Incomplete
    _frozen: bool
    def __init__(self, index, index_slice, original: bool = False) -> None: ...
    @property
    def length(self): ...
    @property
    def stop(self):
        """
        The stopping position of the slice, or the end of the
        index if this is an original slice.
        """
    def __getitem__(self, item):
        """
        Returns another slice of this Index slice.

        Parameters
        ----------
        item : slice
            Index slice
        """
    def sliced_coords(self, rows):
        """
        Convert the input rows to the sliced coordinate system.

        Parameters
        ----------
        rows : list
            Rows in the original coordinate system

        Returns
        -------
        sliced_rows : list
            Rows in the sliced coordinate system
        """
    def orig_coords(self, row):
        """
        Convert the input row from sliced coordinates back
        to original coordinates.

        Parameters
        ----------
        row : int
            Row in the sliced coordinate system

        Returns
        -------
        orig_row : int
            Row in the original coordinate system
        """
    def find(self, key): ...
    def where(self, col_map): ...
    def range(self, lower, upper): ...
    def same_prefix(self, key): ...
    def sorted_data(self): ...
    def replace(self, row, col, val) -> None: ...
    def get_index_or_copy(self): ...
    def insert_row(self, pos, vals, columns) -> None: ...
    def get_row_specifier(self, row_specifier): ...
    def remove_rows(self, row_specifier) -> None: ...
    def replace_rows(self, col_slice) -> None: ...
    def sort(self) -> None: ...
    def __repr__(self) -> str: ...
    def replace_col(self, prev_col, new_col) -> None: ...
    def reload(self) -> None: ...
    def col_position(self, col_name): ...
    def get_slice(self, col_slice, item):
        """
        Return a newly created index from the given slice.

        Parameters
        ----------
        col_slice : Column object
            Already existing slice of a single column
        item : list or ndarray
            Slice for retrieval
        """
    @property
    def columns(self): ...
    @property
    def data(self): ...

def get_index(table, table_copy: Incomplete | None = None, names: Incomplete | None = None):
    """
    Inputs a table and some subset of its columns as table_copy.
    List or tuple containing names of columns as names,and returns an index
    corresponding to this subset or list or None if no such index exists.

    Parameters
    ----------
    table : `Table`
        Input table
    table_copy : `Table`, optional
        Subset of the columns in the ``table`` argument
    names : list, tuple, optional
        Subset of column names in the ``table`` argument

    Returns
    -------
    Index of columns or None

    """
def get_index_by_names(table, names):
    """
    Returns an index in ``table`` corresponding to the ``names`` columns or None
    if no such index exists.

    Parameters
    ----------
    table : `Table`
        Input table
    nmaes : tuple, list
        Column names
    """

class _IndexModeContext:
    '''
    A context manager that allows for special indexing modes, which
    are intended to improve performance. Currently the allowed modes
    are "freeze", in which indices are not modified upon column modification,
    "copy_on_getitem", in which indices are copied upon column slicing,
    and "discard_on_copy", in which indices are discarded upon table
    copying/slicing.
    '''
    _col_subclasses: Incomplete
    table: Incomplete
    mode: Incomplete
    _orig_classes: Incomplete
    def __init__(self, table, mode) -> None:
        """
        Parameters
        ----------
        table : Table
            The table to which the mode should be applied
        mode : str
            Either 'freeze', 'copy_on_getitem', or 'discard_on_copy'.
            In 'discard_on_copy' mode,
            indices are not copied whenever columns or tables are copied.
            In 'freeze' mode, indices are not modified whenever columns are
            modified; at the exit of the context, indices refresh themselves
            based on column values. This mode is intended for scenarios in
            which one intends to make many additions or modifications on an
            indexed column.
            In 'copy_on_getitem' mode, indices are copied when taking column
            slices as well as table slices, so col[i0:i1] will preserve
            indices.
        """
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...
    def _get_copy_on_getitem_shim(self, cls):
        """
        This creates a subclass of the column's class which overrides that
        class's ``__getitem__``, such that when returning a slice of the
        column, the relevant indices are also copied over to the slice.

        Ideally, rather than shimming in a new ``__class__`` we would be able
        to just flip a flag that is checked by the base class's
        ``__getitem__``.  Unfortunately, since the flag needs to be a Python
        variable, this slows down ``__getitem__`` too much in the more common
        case where a copy of the indices is not needed.  See the docstring for
        ``astropy.table._column_mixins`` for more information on that.
        """

class TableIndices(list):
    """
    A special list of table indices allowing
    for retrieval by column name(s).

    Parameters
    ----------
    lst : list
        List of indices
    """
    def __init__(self, lst) -> None: ...
    def __getitem__(self, item):
        """
        Retrieve an item from the list of indices.

        Parameters
        ----------
        item : int, str, tuple, or list
            Position in list or name(s) of indexed column(s)
        """

class TableLoc:
    """
    A pseudo-list of Table rows allowing for retrieval
    of rows by indexed column values.

    Parameters
    ----------
    table : Table
        Indexed table to use
    """
    table: Incomplete
    indices: Incomplete
    def __init__(self, table) -> None: ...
    def _get_rows(self, item):
        """
        Retrieve Table rows indexes by value slice.
        """
    def __getitem__(self, item):
        """
        Retrieve Table rows by value slice.

        Parameters
        ----------
        item : column element, list, ndarray, slice or tuple
            Can be a value of the table primary index, a list/ndarray
            of such values, or a value slice (both endpoints are included).
            If a tuple is provided, the first element must be
            an index to use instead of the primary key, and the
            second element must be as above.
        """
    def __setitem__(self, key, value) -> None:
        """
        Assign Table row's by value slice.

        Parameters
        ----------
        key : column element, list, ndarray, slice or tuple
              Can be a value of the table primary index, a list/ndarray
              of such values, or a value slice (both endpoints are included).
              If a tuple is provided, the first element must be
              an index to use instead of the primary key, and the
              second element must be as above.

        value : New values of the row elements.
                Can be a list of tuples/lists to update the row.
        """

class TableLocIndices(TableLoc):
    def __getitem__(self, item):
        """
        Retrieve Table row's indices by value slice.

        Parameters
        ----------
        item : column element, list, ndarray, slice or tuple
               Can be a value of the table primary index, a list/ndarray
               of such values, or a value slice (both endpoints are included).
               If a tuple is provided, the first element must be
               an index to use instead of the primary key, and the
               second element must be as above.
        """

class TableILoc(TableLoc):
    """
    A variant of TableLoc allowing for row retrieval by
    indexed order rather than data values.

    Parameters
    ----------
    table : Table
        Indexed table to use
    """
    def __init__(self, table) -> None: ...
    def __getitem__(self, item): ...

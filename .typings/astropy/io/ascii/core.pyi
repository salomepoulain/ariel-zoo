import numpy as np
from . import connect as connect
from .docs import READ_DOCSTRING as READ_DOCSTRING, WRITE_DOCSTRING as WRITE_DOCSTRING
from _typeshed import Incomplete
from astropy.table import Table as Table
from astropy.utils.data import get_readable_fileobj as get_readable_fileobj
from astropy.utils.exceptions import AstropyWarning as AstropyWarning
from collections.abc import Generator

FORMAT_CLASSES: Incomplete
FAST_CLASSES: Incomplete

def _check_multidim_table(table, max_ndim) -> None:
    """Check that ``table`` has only columns with ndim <= ``max_ndim``.

    Currently ECSV is the only built-in format that supports output of arbitrary
    N-d columns, but HTML supports 2-d.
    """

class CsvWriter:
    '''
    Internal class to replace the csv writer ``writerow`` and ``writerows``
    functions so that in the case of ``delimiter=\' \'`` and
    ``quoting=csv.QUOTE_MINIMAL``, the output field value is quoted for empty
    fields (when value == \'\').

    This changes the API slightly in that the writerow() and writerows()
    methods return the output written string instead of the length of
    that string.

    Examples
    --------
    >>> from astropy.io.ascii.core import CsvWriter
    >>> writer = CsvWriter(delimiter=\' \')
    >>> print(writer.writerow([\'hello\', \'\', \'world\']))
    hello "" world
    '''
    replace_sentinel: str
    csvfile: Incomplete
    temp_out: Incomplete
    writer: Incomplete
    quotechar2: Incomplete
    quote_empty: Incomplete
    def __init__(self, csvfile: Incomplete | None = None, **kwargs) -> None: ...
    def writerow(self, values):
        """
        Similar to csv.writer.writerow but with the custom quoting behavior.
        Returns the written string instead of the length of that string.
        """
    def writerows(self, values_list):
        """
        Similar to csv.writer.writerows but with the custom quoting behavior.
        Returns the written string instead of the length of that string.
        """
    def _writerow(self, writerow_func, values, has_empty):
        '''
        Call ``writerow_func`` (either writerow or writerows) with ``values``.
        If it has empty fields that have been replaced then change those
        sentinel strings back to quoted empty strings, e.g. ``""``.
        '''

class MaskedConstant(np.ma.core.MaskedConstant):
    """A trivial extension of numpy.ma.masked.

    We want to be able to put the generic term ``masked`` into a dictionary.
    The constant ``numpy.ma.masked`` is not hashable (see
    https://github.com/numpy/numpy/issues/4660), so we need to extend it
    here with a hash value.

    See https://github.com/numpy/numpy/issues/11021 for rationale for
    __copy__ and __deepcopy__ methods.
    """
    def __hash__(self):
        """All instances of this class shall have the same hash."""
    def __copy__(self):
        """This is a singleton so just return self."""
    def __deepcopy__(self, memo): ...

masked: Incomplete

class InconsistentTableError(ValueError):
    """
    Indicates that an input table is inconsistent in some way.

    The default behavior of ``BaseReader`` is to throw an instance of
    this class if a data row doesn't match the header.
    """
class OptionalTableImportError(ImportError):
    """
    Indicates that a dependency for table reading is not present.

    An instance of this class is raised whenever an optional reader
    with certain required dependencies cannot operate because of
    an ImportError.
    """
class ParameterError(NotImplementedError):
    """
    Indicates that a reader cannot handle a passed parameter.

    The C-based fast readers in ``io.ascii`` raise an instance of
    this error class upon encountering a parameter that the
    C engine cannot handle.
    """
class FastOptionsError(NotImplementedError):
    """
    Indicates that one of the specified options for fast
    reading is invalid.
    """
class NoType:
    """
    Superclass for ``StrType`` and ``NumType`` classes.

    This class is the default type of ``Column`` and provides a base
    class for other data types.
    """
class StrType(NoType):
    """
    Indicates that a column consists of text data.
    """
class NumType(NoType):
    """
    Indicates that a column consists of numerical data.
    """
class FloatType(NumType):
    """
    Describes floating-point data.
    """
class BoolType(NoType):
    """
    Describes boolean data.
    """
class IntType(NumType):
    """
    Describes integer data.
    """
class AllType(StrType, FloatType, IntType):
    """
    Subclass of all other data types.

    This type is returned by ``convert_numpy`` if the given numpy
    type does not match ``StrType``, ``FloatType``, or ``IntType``.
    """

class Column:
    """Table column.

    The key attributes of a Column object are:

    * **name** : column name
    * **type** : column type (NoType, StrType, NumType, FloatType, IntType)
    * **dtype** : numpy dtype (optional, overrides **type** if set)
    * **str_vals** : list of column values as strings
    * **fill_values** : dict of fill values
    * **shape** : list of element shape (default [] => scalar)
    * **data** : list of converted column values
    * **subtype** : actual datatype for columns serialized with JSON
    """
    name: Incomplete
    type: Incomplete
    dtype: Incomplete
    str_vals: Incomplete
    fill_values: Incomplete
    shape: Incomplete
    subtype: Incomplete
    def __init__(self, name) -> None: ...

class BaseInputter:
    """
    Get the lines from the table input and return a list of lines.

    """
    encoding: Incomplete
    def get_lines(self, table, newline: Incomplete | None = None):
        """Get the lines from the ``table`` input.

        The input table can be one of:

        * File name (str or pathlike)
        * String (newline separated) with all header and data lines (must have at least 2 lines)
        * File-like object with read() method
        * List of strings

        Parameters
        ----------
        table : str, file-like, list
            Can be either a file name, string (newline separated) with all header and data
            lines (must have at least 2 lines), a file-like object with a
            ``read()`` method, or a list of strings.
        newline :
            Line separator. If `None` use OS default from ``splitlines()``.

        Returns
        -------
        lines : list
            List of lines
        """
    def process_lines(self, lines):
        """Process lines for subsequent use.  In the default case do nothing.
        This routine is not generally intended for removing comment lines or
        stripping whitespace.  These are done (if needed) in the header and
        data line processing.

        Override this method if something more has to be done to convert raw
        input lines to the table rows.  For example the
        ContinuationLinesInputter derived class accounts for continuation
        characters if a row is split into lines.
        """

class BaseSplitter:
    """
    Base splitter that uses python's split method to do the work.

    This does not handle quoted values.  A key feature is the formulation of
    __call__ as a generator that returns a list of the split line values at
    each iteration.

    There are two methods that are intended to be overridden, first
    ``process_line()`` to do pre-processing on each input line before splitting
    and ``process_val()`` to do post-processing on each split string value.  By
    default these apply the string ``strip()`` function.  These can be set to
    another function via the instance attribute or be disabled entirely, for
    example::

      reader.header.splitter.process_val = lambda x: x.lstrip()
      reader.data.splitter.process_val = None

    """
    delimiter: Incomplete
    def process_line(self, line):
        """Remove whitespace at the beginning or end of line.  This is especially useful for
        whitespace-delimited files to prevent spurious columns at the beginning or end.
        """
    def process_val(self, val):
        """Remove whitespace at the beginning or end of value."""
    def __call__(self, lines) -> Generator[Incomplete]: ...
    def join(self, vals): ...

class DefaultSplitter(BaseSplitter):
    """Default class to split strings into columns using python csv.  The class
    attributes are taken from the csv Dialect class.

    Typical usage::

      # lines = ..
      splitter = ascii.DefaultSplitter()
      for col_vals in splitter(lines):
          for col_val in col_vals:
               ...

    """
    delimiter: str
    quotechar: str
    doublequote: bool
    escapechar: Incomplete
    quoting: Incomplete
    skipinitialspace: bool
    csv_writer: Incomplete
    csv_writer_out: Incomplete
    def process_line(self, line):
        """Remove whitespace at the beginning or end of line.  This is especially useful for
        whitespace-delimited files to prevent spurious columns at the beginning or end.
        If splitting on whitespace then replace unquoted tabs with space first.
        """
    def process_val(self, val):
        """Remove whitespace at the beginning or end of value."""
    def __call__(self, lines) -> Generator[Incomplete]:
        """Return an iterator over the table ``lines``, where each iterator output
        is a list of the split line values.

        Parameters
        ----------
        lines : list
            List of table lines

        Yields
        ------
        line : list of str
            Each line's split values.

        """
    def join(self, vals): ...

def _replace_tab_with_space(line, escapechar, quotechar):
    """Replace tabs with spaces in given string, preserving quoted substrings.

    Parameters
    ----------
    line : str
        String containing tabs to be replaced with spaces.
    escapechar : str
        Character in ``line`` used to escape special characters.
    quotechar : str
        Character in ``line`` indicating the start/end of a substring.

    Returns
    -------
    line : str
        A copy of ``line`` with tabs replaced by spaces, preserving quoted substrings.
    """
def _get_line_index(line_or_func, lines):
    """Return the appropriate line index, depending on ``line_or_func`` which
    can be either a function, a positive or negative int, or None.
    """

class BaseHeader:
    """
    Base table header reader.
    """
    auto_format: str
    start_line: Incomplete
    comment: Incomplete
    splitter_class = DefaultSplitter
    names: Incomplete
    write_comment: bool
    write_spacer_lines: Incomplete
    splitter: Incomplete
    def __init__(self) -> None: ...
    cols: Incomplete
    def _set_cols_from_names(self) -> None: ...
    def update_meta(self, lines, meta) -> None:
        """
        Extract any table-level metadata, e.g. keywords, comments, column metadata, from
        the table ``lines`` and update the OrderedDict ``meta`` in place.  This base
        method extracts comment lines and stores them in ``meta`` for output.
        """
    def get_cols(self, lines) -> None:
        """Initialize the header Column objects from the table ``lines``.

        Based on the previously set Header attributes find or create the column names.
        Sets ``self.cols`` with the list of Columns.

        Parameters
        ----------
        lines : list
            List of table lines

        """
    def process_lines(self, lines) -> Generator[Incomplete]:
        """Generator to yield non-blank and non-comment lines."""
    def write_comments(self, lines, meta) -> None: ...
    def write(self, lines) -> None: ...
    @property
    def colnames(self):
        """Return the column names of the table."""
    def remove_columns(self, names) -> None:
        """
        Remove several columns from the table.

        Parameters
        ----------
        names : list
            A list containing the names of the columns to remove
        """
    def rename_column(self, name, new_name) -> None:
        """
        Rename a column.

        Parameters
        ----------
        name : str
            The current name of the column.
        new_name : str
            The new name for the column
        """
    def get_type_map_key(self, col): ...
    def get_col_type(self, col): ...
    def check_column_names(self, names, strict_names, guessing) -> None:
        """
        Check column names.

        This must be done before applying the names transformation
        so that guessing will fail appropriately if ``names`` is supplied.
        For instance if the basic reader is given a table with no column header
        row.

        Parameters
        ----------
        names : list
            User-supplied list of column names
        strict_names : bool
            Whether to impose extra requirements on names
        guessing : bool
            True if this method is being called while guessing the table format
        """

class BaseData:
    """
    Base table data reader.
    """
    start_line: Incomplete
    end_line: Incomplete
    comment: Incomplete
    splitter_class = DefaultSplitter
    write_spacer_lines: Incomplete
    fill_include_names: Incomplete
    fill_exclude_names: Incomplete
    fill_values: Incomplete
    formats: Incomplete
    splitter: Incomplete
    def __init__(self) -> None: ...
    def process_lines(self, lines):
        """
        READ: Strip out comment lines and blank lines from list of ``lines``.

        Parameters
        ----------
        lines : list
            All lines in table

        Returns
        -------
        lines : list
            List of lines

        """
    data_lines: Incomplete
    def get_data_lines(self, lines) -> None:
        """
        READ: Set ``data_lines`` attribute to lines slice comprising table data values.
        """
    def get_str_vals(self):
        """Return a generator that returns a list of column values (as strings)
        for each data line.
        """
    def masks(self, cols) -> None:
        """READ: Set fill value for each column and then apply that fill value.

        In the first step it is evaluated with value from ``fill_values`` applies to
        which column using ``fill_include_names`` and ``fill_exclude_names``.
        In the second step all replacements are done for the appropriate columns.
        """
    def _set_fill_values(self, cols) -> None:
        """READ, WRITE: Set fill values of individual cols based on fill_values of BaseData.

        fill values has the following form:
        <fill_spec> = (<bad_value>, <fill_value>, <optional col_name>...)
        fill_values = <fill_spec> or list of <fill_spec>'s

        """
    def _set_masks(self, cols) -> None:
        """READ: Replace string values in col.str_vals and set masks."""
    def _replace_vals(self, cols) -> None:
        """WRITE: replace string values in col.str_vals."""
    def str_vals(self):
        """WRITE: convert all values in table to a list of lists of strings.

        This sets the fill values and possibly column formats from the input
        formats={} keyword, then ends up calling table.pprint._pformat_col_iter()
        by a circuitous path. That function does the real work of formatting.
        Finally replace anything matching the fill_values.

        Returns
        -------
        values : list of list of str
        """
    def write(self, lines) -> None:
        """Write ``self.cols`` in place to ``lines``.

        Parameters
        ----------
        lines : list
            List for collecting output of writing self.cols.
        """
    def _set_col_formats(self) -> None:
        """WRITE: set column formats."""

def convert_numpy(numpy_type):
    """Return a tuple containing a function which converts a list into a numpy
    array and the type produced by the converter function.

    Parameters
    ----------
    numpy_type : numpy data-type
        The numpy type required of an array returned by ``converter``. Must be a
        valid `numpy type <https://numpy.org/doc/stable/user/basics.types.html>`_
        (e.g., numpy.uint, numpy.int8, numpy.int64, numpy.float64) or a python
        type covered by a numpy type (e.g., int, float, str, bool).

    Returns
    -------
    converter : callable
        ``converter`` is a function which accepts a list and converts it to a
        numpy array of type ``numpy_type``.
    converter_type : type
        ``converter_type`` tracks the generic data type produced by the
        converter function.

    Raises
    ------
    ValueError
        Raised by ``converter`` if the list elements could not be converted to
        the required type.
    """

class BaseOutputter:
    """Output table as a dict of column objects keyed on column name.  The
    table data are stored as plain python lists within the column objects.
    """
    converters: Incomplete
    @staticmethod
    def _validate_and_copy(col, converters):
        """Validate the format for the type converters and then copy those
        which are valid converters for this column (i.e. converter type is
        a subclass of col.type).
        """
    def _convert_vals(self, cols) -> None: ...

def _deduplicate_names(names):
    """Ensure there are no duplicates in ``names``.

    This is done by iteratively adding ``_<N>`` to the name for increasing N
    until the name is unique.
    """

class TableOutputter(BaseOutputter):
    """
    Output the table as an astropy.table.Table object.
    """
    default_converters: Incomplete
    def __call__(self, cols, meta): ...

class MetaBaseReader(type):
    def __init__(cls, name, bases, dct) -> None: ...

def _is_number(x): ...
def _apply_include_exclude_names(table, names, include_names, exclude_names) -> None:
    """
    Apply names, include_names and exclude_names to a table or BaseHeader.

    For the latter this relies on BaseHeader implementing ``colnames``,
    ``rename_column``, and ``remove_columns``.

    Parameters
    ----------
    table : `~astropy.table.Table`, `~astropy.io.ascii.BaseHeader`
        Input table or BaseHeader subclass instance
    names : list
        List of names to override those in table (set to None to use existing names)
    include_names : list
        List of names to include in output
    exclude_names : list
        List of names to exclude from output (applied after ``include_names``)

    """

class BaseReader(metaclass=MetaBaseReader):
    """Class providing methods to read and write an ASCII table using the specified
    header, data, inputter, and outputter instances.

    Typical usage is to instantiate a Reader() object and customize the
    ``header``, ``data``, ``inputter``, and ``outputter`` attributes.  Each
    of these is an object of the corresponding class.

    There is one method ``inconsistent_handler`` that can be used to customize the
    behavior of ``read()`` in the event that a data row doesn't match the header.
    The default behavior is to raise an InconsistentTableError.

    """
    names: Incomplete
    include_names: Incomplete
    exclude_names: Incomplete
    strict_names: bool
    guessing: bool
    encoding: Incomplete
    header_class = BaseHeader
    data_class = BaseData
    inputter_class = BaseInputter
    outputter_class = TableOutputter
    max_ndim: int
    header: Incomplete
    data: Incomplete
    inputter: Incomplete
    outputter: Incomplete
    meta: Incomplete
    def __init__(self) -> None: ...
    def _check_multidim_table(self, table) -> None:
        """Check that the dimensions of columns in ``table`` are acceptable.

        The reader class attribute ``max_ndim`` defines the maximum dimension of
        columns that can be written using this format. The base value is ``1``,
        corresponding to normal scalar columns with just a length.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Input table.

        Raises
        ------
        ValueError
            If any column exceeds the number of allowed dimensions
        """
    lines: Incomplete
    cols: Incomplete
    def read(self, table):
        """Read the ``table`` and return the results in a format determined by
        the ``outputter`` attribute.

        The ``table`` parameter is any string or object that can be processed
        by the instance ``inputter``.  For the base Inputter class ``table`` can be
        one of:

        * File name
        * File-like object
        * String (newline separated) with all header and data lines (must have at least 2 lines)
        * List of strings

        Parameters
        ----------
        table : str, file-like, list
            Input table.

        Returns
        -------
        table : `~astropy.table.Table`
            Output table

        """
    def inconsistent_handler(self, str_vals, ncols):
        """
        Adjust or skip data entries if a row is inconsistent with the header.

        The default implementation does no adjustment, and hence will always trigger
        an exception in read() any time the number of data entries does not match
        the header.

        Note that this will *not* be called if the row already matches the header.

        Parameters
        ----------
        str_vals : list
            A list of value strings from the current row of the table.
        ncols : int
            The expected number of entries from the table header.

        Returns
        -------
        str_vals : list
            List of strings to be parsed into data entries in the output table. If
            the length of this list does not match ``ncols``, an exception will be
            raised in read().  Can also be None, in which case the row will be
            skipped.
        """
    @property
    def comment_lines(self):
        """Return lines in the table that match header.comment regexp."""
    def update_table_data(self, table):
        """
        Update table columns in place if needed.

        This is a hook to allow updating the table columns after name
        filtering but before setting up to write the data.  This is currently
        only used by ECSV and is otherwise just a pass-through.

        Parameters
        ----------
        table : `astropy.table.Table`
            Input table for writing

        Returns
        -------
        table : `astropy.table.Table`
            Output table for writing
        """
    def write_header(self, lines, meta) -> None: ...
    def write(self, table):
        """
        Write ``table`` as list of strings.

        Parameters
        ----------
        table : `~astropy.table.Table`
            Input table data.

        Returns
        -------
        lines : list
            List of strings corresponding to ASCII table

        """

class ContinuationLinesInputter(BaseInputter):
    """Inputter where lines ending in ``continuation_char`` are joined with the subsequent line.

    Example::

      col1 col2 col3
      1       2 3
      4 5       6
    """
    continuation_char: str
    replace_char: str
    no_continue: Incomplete
    def process_lines(self, lines): ...

class WhitespaceSplitter(DefaultSplitter):
    def process_line(self, line):
        """Replace tab with space within ``line`` while respecting quoted substrings."""

extra_reader_pars: Incomplete

def _get_reader(reader_cls, inputter_cls: Incomplete | None = None, outputter_cls: Incomplete | None = None, **kwargs):
    '''Initialize a table reader allowing for common customizations.  See ui.get_reader()
    for param docs.  This routine is for internal (package) use only and is useful
    because it depends only on the "core" module.
    '''

extra_writer_pars: Incomplete

def _get_writer(writer_cls, fast_writer, **kwargs):
    '''Initialize a table writer allowing for common customizations. This
    routine is for internal (package) use only and is useful because it depends
    only on the "core" module.
    '''

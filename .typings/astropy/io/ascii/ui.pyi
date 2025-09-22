import typing
from . import basic as basic, cds as cds, core as core, cparser as cparser, daophot as daophot, ecsv as ecsv, fastbasic as fastbasic, fixedwidth as fixedwidth, html as html, ipac as ipac, latex as latex, mrt as mrt, rst as rst, sextractor as sextractor
from .docs import READ_KWARG_TYPES as READ_KWARG_TYPES, WRITE_KWARG_TYPES as WRITE_KWARG_TYPES
from _typeshed import Incomplete
from astropy.table import Table as Table
from astropy.utils.data import get_readable_fileobj as get_readable_fileobj
from astropy.utils.decorators import format_doc as format_doc
from astropy.utils.exceptions import AstropyWarning as AstropyWarning
from astropy.utils.misc import NOT_OVERWRITING_MSG as NOT_OVERWRITING_MSG
from collections.abc import Generator

_read_trace: Incomplete
_GUESS: bool

def _read_write_help(read_write: str, format: str | None = None, out: typing.IO | None = None) -> None:
    """Helper function to output help documentation for read() or write().

    This uses the ``Table.read/write.help()`` functionality and modifies the output
    to look like the ``ascii.read/write()`` syntax.
    """

READ_WRITE_HELP: str

def read_help(format: str | None = None, out: typing.IO | None = None) -> None: ...
def write_help(format: str | None = None, out: typing.IO | None = None) -> None: ...
def _probably_html(table, maxchars: int = 100000):
    """
    Determine if ``table`` probably contains HTML content.  See PR #3693 and issue
    #3691 for context.
    """
def set_guess(guess) -> None:
    """
    Set the default value of the ``guess`` parameter for read().

    Parameters
    ----------
    guess : bool
        New default ``guess`` value (e.g., True or False)

    """
def get_reader(reader_cls: Incomplete | None = None, inputter_cls: Incomplete | None = None, outputter_cls: Incomplete | None = None, **kwargs):
    """
    Initialize a table reader allowing for common customizations.

    Most of the default behavior for various parameters is determined by the Reader
    class specified by ``reader_cls``.

    Parameters
    ----------
    reader_cls : `~astropy.io.ascii.BaseReader`
        Reader class. Default is :class:`Basic`.
    inputter_cls : `~astropy.io.ascii.BaseInputter`
        Inputter class
    outputter_cls : `~astropy.io.ascii.BaseOutputter`
        Outputter class
    delimiter : str
        Column delimiter string
    comment : str
        Regular expression defining a comment line in table
    quotechar : str
        One-character string to quote fields containing special characters
    header_start : int
        Line index for the header line not counting comment or blank lines. A line with
        only whitespace is considered blank.
    data_start : int
        Line index for the start of data not counting comment or blank lines. A line
        with only whitespace is considered blank.
    data_end : int
        Line index for the end of data not counting comment or blank lines. This value
        can be negative to count from the end.
    converters : dict
        Dict of converters.
    data_splitter_cls : `~astropy.io.ascii.BaseSplitter`
        Splitter class to split data columns.
    header_splitter_cls : `~astropy.io.ascii.BaseSplitter`
        Splitter class to split header columns.
    names : list
        List of names corresponding to each data column.
    include_names : list, optional
        List of names to include in output.
    exclude_names : list
        List of names to exclude from output (applied after ``include_names``).
    fill_values : tuple, list of tuple
        Specification of fill values for bad or missing table values.
    fill_include_names : list
        List of names to include in fill_values.
    fill_exclude_names : list
        List of names to exclude from fill_values (applied after
        ``fill_include_names``).

    Returns
    -------
    reader : `~astropy.io.ascii.BaseReader` subclass
        ASCII format reader instance
    """
def _get_format_class(format): ...
def _get_fast_reader_dict(kwargs):
    """Convert 'fast_reader' key in kwargs into a dict if not already and make sure
    'enable' key is available.
    """
def _validate_read_write_kwargs(read_write, **kwargs):
    """Validate types of keyword arg inputs to read() or write()."""
def _expand_user_if_path(argument): ...
def read(table, guess: Incomplete | None = None, **kwargs): ...
def _guess(table, read_kwargs, format, fast_reader):
    """
    Try to read the table using various sets of keyword args.  Start with the
    standard guess list and filter to make it unique and consistent with
    user-supplied read keyword args.  Finally, if none of those work then
    try the original user-supplied keyword args.

    Parameters
    ----------
    table : str, file-like, list
        Input table as a file name, file-like object, list of strings, or
        single newline-separated string.
    read_kwargs : dict
        Keyword arguments from user to be supplied to reader
    format : str
        Table format
    fast_reader : dict
        Options for the C engine fast reader.  See read() function for details.

    Returns
    -------
    dat : `~astropy.table.Table` or None
        Output table or None if only one guess format was available
    """
def _get_guess_kwargs_list(read_kwargs):
    """Get the full list of reader keyword argument dicts.

    These are the basis for the format guessing process.
    The returned full list will then be:

    - Filtered to be consistent with user-supplied kwargs
    - Cleaned to have only unique entries
    - Used one by one to try reading the input table

    Note that the order of the guess list has been tuned over years of usage.
    Maintainers need to be very careful about any adjustments as the
    reasoning may not be immediately evident in all cases.

    This list can (and usually does) include duplicates.  This is a result
    of the order tuning, but these duplicates get removed later.

    Parameters
    ----------
    read_kwargs : dict
        User-supplied read keyword args

    Returns
    -------
    guess_kwargs_list : list
        List of read format keyword arg dicts
    """
def _read_in_chunks(table, **kwargs):
    """
    For fast_reader read the ``table`` in chunks and vstack to create
    a single table, OR return a generator of chunk tables.
    """
def _read_in_chunks_generator(table, chunk_size, **kwargs) -> Generator[Incomplete]:
    """
    For fast_reader read the ``table`` in chunks and return a generator
    of tables for each chunk.
    """

extra_writer_pars: Incomplete

def get_writer(writer_cls: Incomplete | None = None, fast_writer: bool = True, **kwargs):
    """
    Initialize a table writer allowing for common customizations.

    Most of the default behavior for various parameters is determined by the Writer
    class.

    Parameters
    ----------
    writer_cls : ``writer_cls``
        Writer class. Defaults to :class:`Basic`.
    delimiter : str
        Column delimiter string
    comment : str
        String defining a comment line in table
    quotechar : str
        One-character string to quote fields containing special characters
    formats : dict
        Dictionary of format specifiers or formatting functions
    strip_whitespace : bool
        Strip surrounding whitespace from column values.
    names : list
        List of names corresponding to each data column
    include_names : list
        List of names to include in output.
    exclude_names : list
        List of names to exclude from output (applied after ``include_names``)
    fast_writer : bool
        Whether to use the fast Cython writer.

    Returns
    -------
    writer : `~astropy.io.ascii.BaseReader` subclass
        ASCII format writer instance
    """
def write(table, output: Incomplete | None = None, format: Incomplete | None = None, fast_writer: bool = True, *, overwrite: bool = False, **kwargs) -> None: ...
def get_read_trace():
    """
    Return a traceback of the attempted read formats for the last call to
    `~astropy.io.ascii.read` where guessing was enabled.  This is primarily for
    debugging.

    The return value is a list of dicts, where each dict includes the keyword
    args ``kwargs`` used in the read call and the returned ``status``.

    Returns
    -------
    trace : list of dict
        Ordered list of format guesses and status
    """

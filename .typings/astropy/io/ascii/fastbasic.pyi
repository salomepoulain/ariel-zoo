from . import core as core, cparser as cparser
from _typeshed import Incomplete
from astropy.table import Table as Table
from astropy.utils.misc import _set_locale as _set_locale

class FastBasic(metaclass=core.MetaBaseReader):
    """
    This class is intended to handle the same format addressed by the
    ordinary :class:`Basic` writer, but it acts as a wrapper for underlying C
    code and is therefore much faster. Unlike the other ASCII readers and
    writers, this class is not very extensible and is restricted
    by optimization requirements.
    """
    _format_name: str
    _description: str
    _fast: bool
    fill_extra_cols: bool
    guessing: bool
    strict_names: bool
    delimiter: Incomplete
    write_comment: Incomplete
    comment: Incomplete
    quotechar: Incomplete
    header_start: Incomplete
    data_start: Incomplete
    kwargs: Incomplete
    strip_whitespace_lines: bool
    strip_whitespace_fields: bool
    def __init__(self, default_kwargs={}, **user_kwargs) -> None: ...
    def _read_header(self) -> None: ...
    return_header_chars: Incomplete
    engine: Incomplete
    def read(self, table):
        """
        Read input data (file-like object, filename, list of strings, or
        single string) into a Table and return the result.
        """
    def make_table(self, data, comments):
        """Actually make the output table give the data and comments."""
    def check_header(self) -> None: ...
    def write(self, table, output) -> None:
        """
        Use a fast Cython method to write table data to output,
        where output is a filename or file-like object.
        """
    def _write(self, table, output, default_kwargs, header_output: bool = True, output_types: bool = False) -> None: ...

class FastCsv(FastBasic):
    """
    A faster version of the ordinary :class:`Csv` writer that uses the
    optimized C parsing engine. Note that this reader will append empty
    field values to the end of any row with not enough columns, while
    :class:`FastBasic` simply raises an error.
    """
    _format_name: str
    _description: str
    _fast: bool
    fill_extra_cols: bool
    def __init__(self, **kwargs) -> None: ...
    def write(self, table, output) -> None:
        """
        Override the default write method of `FastBasic` to
        output masked values as empty fields.
        """

class FastTab(FastBasic):
    """
    A faster version of the ordinary :class:`Tab` reader that uses
    the optimized C parsing engine.
    """
    _format_name: str
    _description: str
    _fast: bool
    strip_whitespace_lines: bool
    strip_whitespace_fields: bool
    def __init__(self, **kwargs) -> None: ...

class FastNoHeader(FastBasic):
    '''
    This class uses the fast C engine to read tables with no header line. If
    the names parameter is unspecified, the columns will be autonamed with
    "col{}".
    '''
    _format_name: str
    _description: str
    _fast: bool
    def __init__(self, **kwargs) -> None: ...
    def write(self, table, output) -> None:
        """
        Override the default writing behavior in `FastBasic` so
        that columns names are not included in output.
        """

class FastCommentedHeader(FastBasic):
    """
    A faster version of the :class:`CommentedHeader` reader, which looks for
    column names in a commented line. ``header_start`` denotes the index of
    the header line among all commented lines and is 0 by default.
    """
    _format_name: str
    _description: str
    _fast: bool
    data_start: int
    def __init__(self, **kwargs) -> None: ...
    def make_table(self, data, comments):
        """
        Actually make the output table give the data and comments.  This is
        slightly different from the base FastBasic method in the way comments
        are handled.
        """
    def _read_header(self) -> None: ...
    def write(self, table, output) -> None:
        """
        Override the default writing behavior in `FastBasic` so
        that column names are commented.
        """

class FastRdb(FastBasic):
    """
    A faster version of the :class:`Rdb` reader. This format is similar to
    tab-delimited, but it also contains a header line after the column
    name line denoting the type of each column (N for numeric, S for string).
    """
    _format_name: str
    _description: str
    _fast: bool
    strip_whitespace_lines: bool
    strip_whitespace_fields: bool
    def __init__(self, **kwargs) -> None: ...
    def _read_header(self): ...
    def write(self, table, output) -> None:
        """
        Override the default writing behavior in `FastBasic` to
        output a line with column types after the column name line.
        """

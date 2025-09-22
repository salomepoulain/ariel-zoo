from _typeshed import Incomplete
from astropy import log as log
from astropy.utils.console import Getch as Getch, color_print as color_print, conf as conf
from astropy.utils.data_info import dtype_info_name as dtype_info_name
from collections.abc import Generator

__all__: Incomplete

def default_format_func(format_, val): ...
def _use_str_for_masked_values(format_func):
    """Wrap format function to trap masked values.

    String format functions and most user functions will not be able to deal
    with masked values, so we wrap them to ensure they are passed to str().
    """
def _possible_string_format_functions(format_) -> Generator[Incomplete, None, Incomplete]:
    """Iterate through possible string-derived format functions.

    A string can either be a format specifier for the format built-in,
    a new-style format string, or an old-style format string.
    """
def get_auto_format_func(col: Incomplete | None = None, possible_string_format_functions=...):
    """
    Return a wrapped ``auto_format_func`` function which is used in
    formatting table columns.  This is primarily an internal function but
    gets used directly in other parts of astropy, e.g. `astropy.io.ascii`.

    Parameters
    ----------
    col_name : object, optional
        Hashable object to identify column like id or name. Default is None.

    possible_string_format_functions : func, optional
        Function that yields possible string formatting functions
        (defaults to internal function to do this).

    Returns
    -------
    Wrapped ``auto_format_func`` function
    """
def _get_pprint_include_names(table):
    """Get the set of names to show in pprint from the table pprint_include_names
    and pprint_exclude_names attributes.

    These may be fnmatch unix-style globs.
    """

class TableFormatter:
    @staticmethod
    def _get_pprint_size(max_lines: Incomplete | None = None, max_width: Incomplete | None = None):
        """Get the output size (number of lines and character width) for Column and
        Table pformat/pprint methods.

        If no value of ``max_lines`` is supplied then the height of the
        screen terminal is used to set ``max_lines``.  If the terminal
        height cannot be determined then the default will be determined
        using the ``astropy.table.conf.max_lines`` configuration item. If a
        negative value of ``max_lines`` is supplied then there is no line
        limit applied.

        The same applies for max_width except the configuration item is
        ``astropy.table.conf.max_width``.

        Parameters
        ----------
        max_lines : int or None
            Maximum lines of output (header + data rows)

        max_width : int or None
            Maximum width (characters) output

        Returns
        -------
        max_lines, max_width : int

        """
    def _pformat_col(self, col, max_lines: Incomplete | None = None, show_name: bool = True, show_unit: Incomplete | None = None, show_dtype: bool = False, show_length: Incomplete | None = None, html: bool = False, align: Incomplete | None = None):
        """Return a list of formatted string representation of column values.

        Parameters
        ----------
        max_lines : int
            Maximum lines of output (header + data rows)

        show_name : bool
            Include column name. Default is True.

        show_unit : bool
            Include a header row for unit.  Default is to show a row
            for units only if one or more columns has a defined value
            for the unit.

        show_dtype : bool
            Include column dtype. Default is False.

        show_length : bool
            Include column length at end.  Default is to show this only
            if the column is not shown completely.

        html : bool
            Output column as HTML

        align : str
            Left/right alignment of columns. Default is '>' (right) for all
            columns. Other allowed values are '<', '^', and '0=' for left,
            centered, and 0-padded, respectively.

        Returns
        -------
        lines : list
            List of lines with formatted column values

        outs : dict
            Dict which is used to pass back additional values
            defined within the iterator.

        """
    def _name_and_structure(self, name, dtype, sep: str = ' '):
        '''Format a column name, including a possible structure.

        Normally, just returns the name, but if it has a structured dtype,
        will add the parts in between square brackets.  E.g.,
        "name [f0, f1]" or "name [f0[sf0, sf1], f1]".
        '''
    def _pformat_col_iter(self, col, max_lines, show_name, show_unit, outs, show_dtype: bool = False, show_length: Incomplete | None = None) -> Generator[Incomplete, None, Incomplete]:
        """Iterator which yields formatted string representation of column values.

        Parameters
        ----------
        max_lines : int
            Maximum lines of output (header + data rows)

        show_name : bool
            Include column name. Default is True.

        show_unit : bool
            Include a header row for unit.  Default is to show a row
            for units only if one or more columns has a defined value
            for the unit.

        outs : dict
            Must be a dict which is used to pass back additional values
            defined within the iterator.

        show_dtype : bool
            Include column dtype. Default is False.

        show_length : bool
            Include column length at end.  Default is to show this only
            if the column is not shown completely.
        """
    def _pformat_table(self, table, max_lines: int = -1, max_width: int = -1, show_name: bool = True, show_unit: Incomplete | None = None, show_dtype: bool = False, html: bool = False, tableid: Incomplete | None = None, tableclass: Incomplete | None = None, align: Incomplete | None = None):
        '''Return a list of lines for the formatted string representation of
        the table.

        Parameters
        ----------
        max_lines : int or None
            Maximum number of rows to output
            -1 (default) implies no limit, ``None`` implies using the
            height of the current terminal.

        max_width : int or None
            Maximum character width of output
            -1 (default) implies no limit, ``None`` implies using the
            width of the current terminal.

        show_name : bool
            Include a header row for column names. Default is True.

        show_unit : bool
            Include a header row for unit.  Default is to show a row
            for units only if one or more columns has a defined value
            for the unit.

        show_dtype : bool
            Include a header row for column dtypes. Default is to False.

        html : bool
            Format the output as an HTML table. Default is False.

        tableid : str or None
            An ID tag for the table; only used if html is set.  Default is
            "table{id}", where id is the unique integer id of the table object,
            id(table)

        tableclass : str or list of str or None
            CSS classes for the table; only used if html is set.  Default is
            none

        align : str or list or tuple
            Left/right alignment of columns. Default is \'>\' (right) for all
            columns. Other allowed values are \'<\', \'^\', and \'0=\' for left,
            centered, and 0-padded, respectively. A list of strings can be
            provided for alignment of tables with multiple columns.

        Returns
        -------
        rows : list
            Formatted table as a list of strings

        outs : dict
            Dict which is used to pass back additional values
            defined within the iterator.

        '''
    def _more_tabcol(self, tabcol, max_lines: Incomplete | None = None, max_width: Incomplete | None = None, show_name: bool = True, show_unit: Incomplete | None = None, show_dtype: bool = False) -> None:
        '''Interactive "more" of a table or column.

        Parameters
        ----------
        max_lines : int or None
            Maximum number of rows to output

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
        '''

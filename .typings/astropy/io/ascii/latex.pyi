from . import core as core
from _typeshed import Incomplete

latexdicts: Incomplete
RE_COMMENT: Incomplete

def add_dictval_to_list(adict, key, alist) -> None:
    """
    Add a value from a dictionary to a list.

    Parameters
    ----------
    adict : dictionary
    key : hashable
    alist : list
        List where value should be added
    """
def find_latex_line(lines, latex):
    """
    Find the first line which matches a pattern.

    Parameters
    ----------
    lines : list
        List of strings
    latex : str
        Search pattern

    Returns
    -------
    line_num : int, None
        Line number. Returns None, if no match was found

    """

class LatexInputter(core.BaseInputter):
    def process_lines(self, lines): ...

class LatexSplitter(core.BaseSplitter):
    """Split LaTeX table data. Default delimiter is `&`."""
    delimiter: str
    def __call__(self, lines): ...
    def process_line(self, line):
        """Remove whitespace at the beginning or end of line. Also remove
        \\ at end of line.
        """
    def process_val(self, val):
        """Remove whitespace and {} at the beginning or end of value."""
    def join(self, vals):
        """Join values together and add a few extra spaces for readability."""

class LatexHeader(core.BaseHeader):
    """Class to read the header of Latex Tables."""
    header_start: str
    splitter_class = LatexSplitter
    def start_line(self, lines): ...
    def _get_units(self): ...
    def write(self, lines) -> None: ...

class LatexData(core.BaseData):
    """Class to read the data in LaTeX tables."""
    data_start: Incomplete
    data_end: str
    splitter_class = LatexSplitter
    def start_line(self, lines): ...
    def end_line(self, lines): ...
    def write(self, lines) -> None: ...

class Latex(core.BaseReader):
    '''LaTeX format table.

    This class implements some LaTeX specific commands.  Its main
    purpose is to write out a table in a form that LaTeX can compile. It
    is beyond the scope of this class to implement every possible LaTeX
    command, instead the focus is to generate a syntactically valid
    LaTeX tables.

    This class can also read simple LaTeX tables (one line per table
    row, no ``\\multicolumn`` or similar constructs), specifically, it
    can read the tables that it writes.
    When reading, it will look for the Latex commands to start and end tabular
    data (``\\begin{tabular}`` and ``\\end{tabular}``). That means that
    those lines have to be present in the input file; the benefit is that this
    reader can be used on a LaTeX file with text, tables, and figures and it
    will read the first valid table.

    .. note:: **Units in LaTeX tables**

        The LaTeX writer will output units in the table if they are present in the
        column info::

            >>> import io
            >>> out = io.StringIO()
            >>> import sys
            >>> import astropy.units as u
            >>> from astropy.table import Table
            >>> t = Table({\'v\': [1, 2] * u.km/u.s, \'class\': [\'star\', \'jet\']})
            >>> t.write(out, format=\'ascii.latex\')
            >>> print(out.getvalue())
            \\begin{table}
            \\begin{tabular}{cc}
            v & class \\\\\n            $\\mathrm{km\\,s^{-1}}$ &  \\\\\n            1.0 & star \\\\\n            2.0 & jet \\\\\n            \\end{tabular}
            \\end{table}

        However, it will fail to read a table with units. There are so
        many ways to write units in LaTeX (enclosed in parenthesis or square brackets,
        as a separate row are as part of the column headers, using plain text, LaTeX
        symbols etc. ) that it is not feasible to implement a
        general reader for this. If you need to read a table with units, you can
        skip reading the lines with units to just read the numerical values using the
        ``data_start`` parameter to set the first line where numerical data values appear::

            >>> Table.read(out.getvalue(), format=\'ascii.latex\', data_start=4)
            <Table length=2>
               v    class
            float64  str4
            ------- -----
                1.0  star
                2.0   jet

        Alternatively, you can write a custom reader using your knowledge of the exact
        format of the units in that case, by extending this class.

    Reading a LaTeX table, the following keywords are accepted:

    **ignore_latex_commands** :
        Lines starting with these LaTeX commands will be treated as comments (i.e. ignored).

    When writing a LaTeX table, the some keywords can customize the
    format.  Care has to be taken here, because python interprets ``\\\\``
    in a string as an escape character.  In order to pass this to the
    output either format your strings as raw strings with the ``r``
    specifier or use a double ``\\\\\\\\``.

    Examples::

        caption = r\'My table \\label{mytable}\'
        caption = \'My table \\\\\\\\label{mytable}\'

    **latexdict** : Dictionary of extra parameters for the LaTeX output

        * tabletype : used for first and last line of table.
            The default is ``\\\\begin{table}``.  The following would generate a table,
            which spans the whole page in a two-column document::

                ascii.write(data, sys.stdout, format="latex",
                            latexdict={\'tabletype\': \'table*\'})

            If ``None``, the table environment will be dropped, keeping only
            the ``tabular`` environment.

        * tablealign : positioning of table in text.
            The default is not to specify a position preference in the text.
            If, e.g. the alignment is ``ht``, then the LaTeX will be ``\\\\begin{table}[ht]``.

        * col_align : Alignment of columns
            If not present all columns will be centered.

        * caption : Table caption (string or list of strings)
            This will appear above the table as it is the standard in
            many scientific publications.  If you prefer a caption below
            the table, just write the full LaTeX command as
            ``latexdict[\'tablefoot\'] = r\'\\caption{My table}\'``

        * preamble, header_start, header_end, data_start, data_end, tablefoot: Pure LaTeX
            Each one can be a string or a list of strings. These strings
            will be inserted into the table without any further
            processing. See the examples below.

        * units : dictionary of strings
            Keys in this dictionary should be names of columns. If
            present, a line in the LaTeX table directly below the column
            names is added, which contains the values of the
            dictionary. Example::

              from astropy.io import ascii
              data = {\'name\': [\'bike\', \'car\'], \'mass\': [75,1200], \'speed\': [10, 130]}
              ascii.write(data, format="latex",
                          latexdict={\'units\': {\'mass\': \'kg\', \'speed\': \'km/h\'}})

            If the column has no entry in the ``units`` dictionary, it defaults
            to the **unit** attribute of the column. If this attribute is not
            specified (i.e. it is None), the unit will be written as ``\' \'``.

        Run the following code to see where each element of the
        dictionary is inserted in the LaTeX table::

            from astropy.io import ascii
            data = {\'cola\': [1,2], \'colb\': [3,4]}
            ascii.write(data, format="latex", latexdict=ascii.latex.latexdicts[\'template\'])

        Some table styles are predefined in the dictionary
        ``ascii.latex.latexdicts``. The following generates in table in
        style preferred by A&A and some other journals::

            ascii.write(data, format="latex", latexdict=ascii.latex.latexdicts[\'AA\'])

        As an example, this generates a table, which spans all columns
        and is centered on the page::

            ascii.write(data, format="latex", col_align=\'|lr|\',
                        latexdict={\'preamble\': r\'\\begin{center}\',
                                   \'tablefoot\': r\'\\end{center}\',
                                   \'tabletype\': \'table*\'})

    **caption** : Set table caption
        Shorthand for::

            latexdict[\'caption\'] = caption

    **col_align** : Set the column alignment.
        If not present this will be auto-generated for centered
        columns. Shorthand for::

            latexdict[\'col_align\'] = col_align

    '''
    _format_name: str
    _io_registry_format_aliases: Incomplete
    _io_registry_suffix: str
    _description: str
    header_class = LatexHeader
    data_class = LatexData
    inputter_class = LatexInputter
    max_ndim: Incomplete
    latex: Incomplete
    ignore_latex_commands: Incomplete
    def __init__(self, ignore_latex_commands=['hline', 'vspace', 'tableline', 'toprule', 'midrule', 'bottomrule'], latexdict={}, caption: str = '', col_align: Incomplete | None = None) -> None: ...
    def write(self, table: Incomplete | None = None): ...

class AASTexHeaderSplitter(LatexSplitter):
    """Extract column names from a `deluxetable`_.

    This splitter expects the following LaTeX code **in a single line**:

        \\tablehead{\\colhead{col1} & ... & \\colhead{coln}}
    """
    def __call__(self, lines): ...
    def process_line(self, line):
        """extract column names from tablehead."""
    def join(self, vals): ...

class AASTexHeader(LatexHeader):
    """In a `deluxetable
    <http://fits.gsfc.nasa.gov/standard30/deluxetable.sty>`_ some header
    keywords differ from standard LaTeX.

    This header is modified to take that into account.
    """
    header_start: str
    splitter_class = AASTexHeaderSplitter
    def start_line(self, lines): ...
    def write(self, lines) -> None: ...

class AASTexData(LatexData):
    """In a `deluxetable`_ the data is enclosed in `\\startdata` and `\\enddata`."""
    data_start: str
    data_end: str
    def start_line(self, lines): ...
    def write(self, lines) -> None: ...

class AASTex(Latex):
    """AASTeX format table.

    This class implements some AASTeX specific commands.
    AASTeX is used for the AAS (American Astronomical Society)
    publications like ApJ, ApJL and AJ.

    It derives from the ``Latex`` reader and accepts the same
    keywords.  However, the keywords ``header_start``, ``header_end``,
    ``data_start`` and ``data_end`` in ``latexdict`` have no effect.
    """
    _format_name: str
    _io_registry_format_aliases: Incomplete
    _io_registry_suffix: str
    _description: str
    header_class = AASTexHeader
    data_class = AASTexData
    def __init__(self, **kwargs) -> None: ...

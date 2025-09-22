from . import core as core, fixedwidth as fixedwidth
from .misc import first_false_index as first_false_index, first_true_index as first_true_index, groupmore as groupmore
from _typeshed import Incomplete

class DaophotHeader(core.BaseHeader):
    """
    Read the header from a file produced by the IRAF DAOphot routine.
    """
    comment: str
    re_format: Incomplete
    re_header_keyword: Incomplete
    aperture_values: Incomplete
    def __init__(self) -> None: ...
    col_widths: Incomplete
    def parse_col_defs(self, grouped_lines_dict):
        """Parse a series of column definition lines.

        Examples
        --------
        When parsing, there may be several such blocks in a single file
        (where continuation characters have already been stripped).
        #N ID    XCENTER   YCENTER   MAG         MERR          MSKY           NITER
        #U ##    pixels    pixels    magnitudes  magnitudes    counts         ##
        #F %-9d  %-10.3f   %-10.3f   %-12.3f     %-14.3f       %-15.7g        %-6d
        """
    meta: Incomplete
    names: Incomplete
    def update_meta(self, lines, meta):
        """
        Extract table-level keywords for DAOphot table.  These are indicated by
        a leading '#K ' prefix.
        """
    def extract_keyword_line(self, line):
        """
        Extract info from a header keyword line (#K).
        """
    def get_cols(self, lines) -> None:
        """
        Initialize the header Column objects from the table ``lines`` for a DAOphot
        header.  The DAOphot header is specialized so that we just copy the entire BaseHeader
        get_cols routine and modify as needed.

        Parameters
        ----------
        lines : list
            List of table lines

        Returns
        -------
        col : list
            List of table Columns
        """

class DaophotData(core.BaseData):
    splitter_class = fixedwidth.FixedWidthSplitter
    start_line: int
    comment: str
    is_multiline: bool
    def __init__(self) -> None: ...
    def get_data_lines(self, lines) -> None: ...

class DaophotInputter(core.ContinuationLinesInputter):
    continuation_char: str
    multiline_char: str
    replace_char: str
    re_multiline: Incomplete
    def search_multiline(self, lines, depth: int = 150):
        """
        Search lines for special continuation character to determine number of
        continued rows in a datablock.  For efficiency, depth gives the upper
        limit of lines to search.
        """
    def process_lines(self, lines): ...

class Daophot(core.BaseReader):
    """
    DAOphot format table.

    Example::

      #K MERGERAD   = INDEF                   scaleunit  %-23.7g
      #K IRAF = NOAO/IRAFV2.10EXPORT version %-23s
      #K USER = davis name %-23s
      #K HOST = tucana computer %-23s
      #
      #N ID    XCENTER   YCENTER   MAG         MERR          MSKY           NITER    \\\n      #U ##    pixels    pixels    magnitudes  magnitudes    counts         ##       \\\n      #F %-9d  %-10.3f   %-10.3f   %-12.3f     %-14.3f       %-15.7g        %-6d
      #
      #N         SHARPNESS   CHI         PIER  PERROR                                \\\n      #U         ##          ##          ##    perrors                               \\\n      #F         %-23.3f     %-12.3f     %-6d  %-13s
      #
      14       138.538     INDEF   15.461      0.003         34.85955       4        \\\n                  -0.032      0.802       0     No_error

    The keywords defined in the #K records are available via the output table
    ``meta`` attribute::

      >>> import os
      >>> from astropy.io import ascii
      >>> filename = os.path.join(ascii.__path__[0], 'tests/data/daophot.dat')
      >>> data = ascii.read(filename)
      >>> for name, keyword in data.meta['keywords'].items():
      ...     print(name, keyword['value'], keyword['units'], keyword['format'])
      ...
      MERGERAD INDEF scaleunit %-23.7g
      IRAF NOAO/IRAFV2.10EXPORT version %-23s
      USER  name %-23s
      ...

    The unit and formats are available in the output table columns::

      >>> for colname in data.colnames:
      ...     col = data[colname]
      ...     print(colname, col.unit, col.format)
      ...
      ID None %-9d
      XCENTER pixels %-10.3f
      YCENTER pixels %-10.3f
      ...

    Any column values of INDEF are interpreted as a missing value and will be
    masked out in the resultant table.

    In case of multi-aperture daophot files containing repeated entries for the last
    row of fields, extra unique column names will be created by suffixing
    corresponding field names with numbers starting from 2 to N (where N is the
    total number of apertures).
    For example,
    first aperture radius will be RAPERT and corresponding magnitude will be MAG,
    second aperture radius will be RAPERT2 and corresponding magnitude will be MAG2,
    third aperture radius will be RAPERT3 and corresponding magnitude will be MAG3,
    and so on.

    """
    _format_name: str
    _io_registry_format_aliases: Incomplete
    _io_registry_can_write: bool
    _description: str
    header_class = DaophotHeader
    data_class = DaophotData
    inputter_class = DaophotInputter
    table_width: int
    def __init__(self) -> None: ...
    def write(self, table: Incomplete | None = None) -> None: ...

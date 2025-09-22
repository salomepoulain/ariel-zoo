from .core import DefaultSplitter as DefaultSplitter
from .fixedwidth import FixedWidth as FixedWidth, FixedWidthData as FixedWidthData, FixedWidthHeader as FixedWidthHeader, FixedWidthTwoLineDataSplitter as FixedWidthTwoLineDataSplitter
from _typeshed import Incomplete

class SimpleRSTHeader(FixedWidthHeader):
    position_line: int
    start_line: int
    splitter_class = DefaultSplitter
    position_char: str
    def get_fixedwidth_params(self, line): ...

class SimpleRSTData(FixedWidthData):
    end_line: int
    splitter_class = FixedWidthTwoLineDataSplitter

class RST(FixedWidth):
    '''reStructuredText simple format table.

    See: https://docutils.sourceforge.io/docs/ref/rst/restructuredtext.html#simple-tables

    Example::

      >>> from astropy.table import QTable
      >>> import astropy.units as u
      >>> import sys
      >>> tbl = QTable({"wave": [350, 950] * u.nm, "response": [0.7, 1.2] * u.count})
      >>> tbl.write(sys.stdout,  format="ascii.rst")
      ===== ========
       wave response
      ===== ========
      350.0      0.7
      950.0      1.2
      ===== ========

    Like other fixed-width formats, when writing a table you can provide ``header_rows``
    to specify a list of table rows to output as the header.  For example::

      >>> tbl.write(sys.stdout,  format="ascii.rst", header_rows=[\'name\', \'unit\'])
      ===== ========
       wave response
         nm       ct
      ===== ========
      350.0      0.7
      950.0      1.2
      ===== ========

    Currently there is no support for reading tables which utilize continuation lines,
    or for ones which define column spans through the use of an additional
    line of dashes in the header.

    '''
    _format_name: str
    _description: str
    data_class = SimpleRSTData
    header_class = SimpleRSTHeader
    def __init__(self, header_rows: Incomplete | None = None) -> None: ...
    def write(self, lines): ...
    def read(self, table): ...

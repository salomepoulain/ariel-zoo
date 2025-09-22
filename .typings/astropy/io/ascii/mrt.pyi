from . import cds as cds, core as core, fixedwidth as fixedwidth
from _typeshed import Incomplete
from astropy.table import Column as Column, MaskedColumn as MaskedColumn, Table as Table

MAX_SIZE_README_LINE: int
MAX_COL_INTLIMIT: int
__doctest_skip__: Incomplete
BYTE_BY_BYTE_TEMPLATE: Incomplete
MRT_TEMPLATE: Incomplete

class MrtSplitter(fixedwidth.FixedWidthSplitter):
    """
    Contains the join function to left align the MRT columns
    when writing to a file.
    """
    def join(self, vals, widths): ...

class MrtHeader(cds.CdsHeader):
    _subfmt: str
    def _split_float_format(self, value):
        """
        Splits a Float string into different parts to find number
        of digits after decimal and check if the value is in Scientific
        notation.

        Parameters
        ----------
        value : str
            String containing the float value to split.

        Returns
        -------
        fmt: (int, int, int, bool, bool)
            List of values describing the Float string.
            (size, dec, ent, sign, exp)
            size, length of the given string.
            ent, number of digits before decimal point.
            dec, number of digits after decimal point.
            sign, whether or not given value signed.
            exp, is value in Scientific notation?
        """
    def _set_column_val_limits(self, col) -> None:
        """
        Sets the ``col.min`` and ``col.max`` column attributes,
        taking into account columns with Null values.
        """
    def column_float_formatter(self, col) -> None:
        """
        String formatter function for a column containing Float values.
        Checks if the values in the given column are in Scientific notation,
        by splitting the value string. It is assumed that the column either has
        float values or Scientific notation.

        A ``col.formatted_width`` attribute is added to the column. It is not added
        if such an attribute is already present, say when the ``formats`` argument
        is passed to the writer. A properly formatted format string is also added as
        the ``col.format`` attribute.

        Parameters
        ----------
        col : A ``Table.Column`` object.
        """
    linewidth: Incomplete
    def write_byte_by_byte(self):
        """
        Writes the Byte-By-Byte description of the table.

        Columns that are `astropy.coordinates.SkyCoord` or `astropy.time.TimeSeries`
        objects or columns with values that are such objects are recognized as such,
        and some predefined labels and description is used for them.
        See the Vizier MRT Standard documentation in the link below for more details
        on these. An example Byte-By-Byte table is shown here.

        See: https://vizier.unistra.fr/doc/catstd-3.1.htx

        Example::

        --------------------------------------------------------------------------------
        Byte-by-byte Description of file: table.dat
        --------------------------------------------------------------------------------
        Bytes Format Units  Label     Explanations
        --------------------------------------------------------------------------------
         1- 8  A8     ---    names   Description of names
        10-14  E5.1   ---    e       [-3160000.0/0.01] Description of e
        16-23  F8.5   ---    d       [22.25/27.25] Description of d
        25-31  E7.1   ---    s       [-9e+34/2.0] Description of s
        33-35  I3     ---    i       [-30/67] Description of i
        37-39  F3.1   ---    sameF   [5.0/5.0] Description of sameF
        41-42  I2     ---    sameI   [20] Description of sameI
        44-45  I2     h      RAh     Right Ascension (hour)
        47-48  I2     min    RAm     Right Ascension (minute)
        50-67  F18.15 s      RAs     Right Ascension (second)
           69  A1     ---    DE-     Sign of Declination
        70-71  I2     deg    DEd     Declination (degree)
        73-74  I2     arcmin DEm     Declination (arcmin)
        76-91  F16.13 arcsec DEs     Declination (arcsec)

        --------------------------------------------------------------------------------
        """
    def write(self, lines) -> None:
        """
        Writes the Header of the MRT table, aka ReadMe, which
        also contains the Byte-By-Byte description of the table.
        """

class MrtData(cds.CdsData):
    """MRT table data reader."""
    _subfmt: str
    splitter_class = MrtSplitter
    def write(self, lines) -> None: ...

class Mrt(core.BaseReader):
    """AAS MRT (Machine-Readable Table) format table.

    **Reading**
    ::

      >>> from astropy.io import ascii
      >>> table = ascii.read('data.mrt', format='mrt')

    **Writing**

    Use ``ascii.write(table, 'data.mrt', format='mrt')`` to  write tables to
    Machine Readable Table (MRT) format.

    Note that the metadata of the table, apart from units, column names and
    description, will not be written. These have to be filled in by hand later.

    See also: :ref:`cds_mrt_format`.

    Caveats:

    * The Units and Explanations are available in the column ``unit`` and
      ``description`` attributes, respectively.
    * The other metadata defined by this format is not available in the output table.
    """
    _format_name: str
    _io_registry_format_aliases: Incomplete
    _io_registry_can_write: bool
    _description: str
    data_class = MrtData
    header_class = MrtHeader
    def write(self, table: Incomplete | None = None): ...

from . import basic as basic, core as core
from _typeshed import Incomplete
from astropy.io.ascii.core import convert_numpy as convert_numpy
from astropy.table import meta as meta, serialize as serialize
from astropy.utils.data_info import serialize_context_as as serialize_context_as
from astropy.utils.exceptions import AstropyUserWarning as AstropyUserWarning
from collections.abc import Generator

ECSV_VERSION: str
DELIMITERS: Incomplete
ECSV_DATATYPES: Incomplete

class InvalidEcsvDatatypeWarning(AstropyUserWarning):
    """
    ECSV specific Astropy warning class.
    """

class EcsvHeader(basic.BasicHeader):
    """Header class for which the column definition line starts with the
    comment character.  See the :class:`CommentedHeader` class  for an example.
    """
    def process_lines(self, lines) -> Generator[Incomplete]:
        """Return only non-blank lines that start with the comment regexp.  For these
        lines strip out the matching characters and leading/trailing whitespace.
        """
    def write(self, lines) -> None:
        """
        Write header information in the ECSV ASCII format.

        This function is called at the point when preprocessing has been done to
        convert the input table columns to `self.cols` which is a list of
        `astropy.io.ascii.core.Column` objects. In particular `col.str_vals`
        is available for each column with the string representation of each
        column item for output.

        This format starts with a delimiter separated list of the column names
        in order to make this format readable by humans and simple csv-type
        readers. It then encodes the full table meta and column attributes and
        meta as YAML and pretty-prints this in the header.  Finally the
        delimited column names are repeated again, for humans and readers that
        look for the *last* comment line as defining the column names.
        """
    def write_comments(self, lines, meta) -> None:
        """
        WRITE: Override the default write_comments to do nothing since this is handled
        in the custom write method.
        """
    def update_meta(self, lines, meta) -> None:
        """
        READ: Override the default update_meta to do nothing.  This process is done
        in get_cols() for this reader.
        """
    table_meta: Incomplete
    names: Incomplete
    def get_cols(self, lines) -> None:
        """
        READ: Initialize the header Column objects from the table ``lines``.

        Parameters
        ----------
        lines : list
            List of table lines

        """

def _check_dtype_is_str(col) -> None: ...

class EcsvOutputter(core.TableOutputter):
    '''
    After reading the input lines and processing, convert the Reader columns
    and metadata to an astropy.table.Table object.  This overrides the default
    converters to be an empty list because there is no "guessing" of the
    conversion function.
    '''
    default_converters: Incomplete
    def __call__(self, cols, meta): ...
    def _convert_vals(self, cols) -> None:
        """READ: Convert str_vals in `cols` to final arrays with correct dtypes.

        This is adapted from ``BaseOutputter._convert_vals``. In the case of ECSV
        there is no guessing and all types are known in advance. A big change
        is handling the possibility of JSON-encoded values, both unstructured
        object data and structured values that may contain masked data.
        """

class EcsvData(basic.BasicData):
    def _set_fill_values(self, cols) -> None:
        '''READ: Set the fill values of the individual cols based on fill_values of BaseData.

        For ECSV handle the corner case of data that has been serialized using
        the serialize_method=\'data_mask\' option, which writes the full data and
        mask directly, AND where that table includes a string column with zero-length
        string entries ("") which are valid data.

        Normally the super() method will set col.fill_value=(\'\', \'0\') to replace
        blanks with a \'0\'.  But for that corner case subset, instead do not do
        any filling.
        '''
    def str_vals(self):
        '''WRITE: convert all values in table to a list of lists of strings.

        This version considerably simplifies the base method:
        - No need to set fill values and column formats
        - No per-item formatting, just use repr()
        - Use JSON for object-type or multidim values
        - Only Column or MaskedColumn can end up as cols here.
        - Only replace masked values with "", not the generalized filling
        '''

class Ecsv(basic.Basic):
    """ECSV (Enhanced Character Separated Values) format table.

    Th ECSV format allows for specification of key table and column meta-data, in
    particular the data type and unit.

    See: https://github.com/astropy/astropy-APEs/blob/main/APE6.rst

    Examples
    --------
    >>> from astropy.table import Table
    >>> ecsv_content = '''# %ECSV 0.9
    ... # ---
    ... # datatype:
    ... # - {name: a, unit: m / s, datatype: int64, format: '%03d'}
    ... # - {name: b, unit: km, datatype: int64, description: This is column b}
    ... a b
    ... 001 2
    ... 004 3
    ... '''

    >>> Table.read(ecsv_content, format='ascii.ecsv')
    <Table length=2>
      a     b
    m / s   km
    int64 int64
    ----- -----
      001     2
      004     3

    """
    _format_name: str
    _description: str
    _io_registry_suffix: str
    header_class = EcsvHeader
    data_class = EcsvData
    outputter_class = EcsvOutputter
    max_ndim: Incomplete
    def update_table_data(self, table):
        """
        Update table columns in place if mixin columns are present.

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

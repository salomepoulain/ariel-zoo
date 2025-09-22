from . import BinTableHDU as BinTableHDU, GroupsHDU as GroupsHDU, HDUList as HDUList, TableHDU as TableHDU
from .column import KEYWORD_NAMES as KEYWORD_NAMES, _fortran_to_python_format as _fortran_to_python_format
from .convenience import table_to_hdu as table_to_hdu
from .hdu.hdulist import FITS_SIGNATURE as FITS_SIGNATURE
from .util import first as first
from _typeshed import Incomplete
from astropy.table import Column as Column, MaskedColumn as MaskedColumn, Table as Table, meta as meta, serialize as serialize
from astropy.time import Time as Time
from astropy.utils.data_info import serialize_context_as as serialize_context_as
from astropy.utils.exceptions import AstropyDeprecationWarning as AstropyDeprecationWarning, AstropyUserWarning as AstropyUserWarning
from astropy.utils.misc import NOT_OVERWRITING_MSG as NOT_OVERWRITING_MSG

REMOVE_KEYWORDS: Incomplete
COLUMN_KEYWORD_REGEXP: Incomplete

def is_column_keyword(keyword): ...
def is_fits(origin, filepath, fileobj, *args, **kwargs):
    """
    Determine whether `origin` is a FITS file.

    Parameters
    ----------
    origin : str or readable file-like
        Path or file object containing a potential FITS file.

    Returns
    -------
    is_fits : bool
        Returns `True` if the given file is a FITS file.
    """
def _decode_mixins(tbl):
    """Decode a Table ``tbl`` that has astropy Columns + appropriate meta-data into
    the corresponding table with mixin columns (as appropriate).
    """
def read_table_fits(input, hdu: Incomplete | None = None, astropy_native: bool = False, memmap: bool = False, character_as_bytes: bool = True, unit_parse_strict: str = 'warn', mask_invalid: bool = True):
    '''
    Read a Table object from an FITS file.

    If the ``astropy_native`` argument is ``True``, then input FITS columns
    which are representations of an astropy core object will be converted to
    that class and stored in the ``Table`` as "mixin columns".  Currently this
    is limited to FITS columns which adhere to the FITS Time standard, in which
    case they will be converted to a `~astropy.time.Time` column in the output
    table.

    Parameters
    ----------
    input : str or file-like or compatible `astropy.io.fits` HDU object
        If a string, the filename to read the table from. If a file object, or
        a compatible HDU object, the object to extract the table from. The
        following `astropy.io.fits` HDU objects can be used as input:
        - :class:`~astropy.io.fits.hdu.table.TableHDU`
        - :class:`~astropy.io.fits.hdu.table.BinTableHDU`
        - :class:`~astropy.io.fits.hdu.table.GroupsHDU`
        - :class:`~astropy.io.fits.hdu.hdulist.HDUList`
    hdu : int or str, optional
        The HDU to read the table from.
    astropy_native : bool, optional
        Read in FITS columns as native astropy objects where possible instead
        of standard Table Column objects. Default is False.
    memmap : bool, optional
        Whether to use memory mapping, which accesses data on disk as needed. If
        you are only accessing part of the data, this is often more efficient.
        If you want to access all the values in the table, and you are able to
        fit the table in memory, you may be better off leaving memory mapping
        off. However, if your table would not fit in memory, you should set this
        to `True`.
        When set to `True` then ``mask_invalid`` is set to `False` since the
        masking would cause loading the full data array.
    character_as_bytes : bool, optional
        If `True`, string columns are stored as Numpy byte arrays (dtype ``S``)
        and are converted on-the-fly to unicode strings when accessing
        individual elements. If you need to use Numpy unicode arrays (dtype
        ``U``) internally, you should set this to `False`, but note that this
        will use more memory. If set to `False`, string columns will not be
        memory-mapped even if ``memmap`` is `True`.
    unit_parse_strict : str, optional
        Behaviour when encountering invalid column units in the FITS header.
        Default is "warn", which will emit a ``UnitsWarning`` and create a
        :class:`~astropy.units.core.UnrecognizedUnit`.
        Values are the ones allowed by the ``parse_strict`` argument of
        :class:`~astropy.units.core.Unit`: ``raise``, ``warn`` and ``silent``.
    mask_invalid : bool, optional
        By default the code masks NaNs in float columns and empty strings in
        string columns. Set this parameter to `False` to avoid the performance
        penalty of doing this masking step. The masking is always deactivated
        when using ``memmap=True`` (see above).

    '''
def _encode_mixins(tbl):
    """Encode a Table ``tbl`` that may have mixin columns to a Table with only
    astropy Columns + appropriate meta-data to allow subsequent decoding.
    """
def write_table_fits(input, output, overwrite: bool = False, append: bool = False) -> None:
    """
    Write a Table object to a FITS file.

    Parameters
    ----------
    input : Table
        The table to write out.
    output : str or os.PathLike[str] or file-like
        The filename to write the table to.
    overwrite : bool
        Whether to overwrite any existing file without warning.
    append : bool
        Whether to append the table to an existing file
    """

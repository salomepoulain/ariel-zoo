from . import Card as Card, Header as Header
from _typeshed import Incomplete
from astropy.coordinates import EarthLocation as EarthLocation
from astropy.table import Column as Column, MaskedColumn as MaskedColumn
from astropy.table.column import col_copy as col_copy
from astropy.time import Time as Time, TimeDelta as TimeDelta
from astropy.time.core import BARYCENTRIC_SCALES as BARYCENTRIC_SCALES
from astropy.time.formats import FITS_DEPRECATED_SCALES as FITS_DEPRECATED_SCALES
from astropy.utils.exceptions import AstropyUserWarning as AstropyUserWarning

TCTYP_RE_TYPE: Incomplete
TCTYP_RE_ALGO: Incomplete
FITS_TIME_UNIT: Incomplete
OBSGEO_XYZ: Incomplete
OBSGEO_LBH: Incomplete
TIME_KEYWORDS: Incomplete
COLUMN_TIME_KEYWORDS: Incomplete
COLUMN_TIME_KEYWORD_REGEXP: Incomplete

def is_time_column_keyword(keyword):
    """
    Check if the FITS header keyword is a time column-specific keyword.

    Parameters
    ----------
    keyword : str
        FITS keyword.
    """

GLOBAL_TIME_INFO: Incomplete

def _verify_global_info(global_info) -> None:
    """
    Given the global time reference frame information, verify that
    each global time coordinate attribute will be given a valid value.

    Parameters
    ----------
    global_info : dict
        Global time reference frame information.
    """
def _verify_column_info(column_info, global_info):
    """
    Given the column-specific time reference frame information, verify that
    each column-specific time coordinate attribute has a valid value.
    Return True if the coordinate column is time, or else return False.

    Parameters
    ----------
    global_info : dict
        Global time reference frame information.
    column_info : dict
        Column-specific time reference frame override information.
    """
def _get_info_if_time_column(col, global_info):
    """
    Check if a column without corresponding time column keywords in the
    FITS header represents time or not. If yes, return the time column
    information needed for its conversion to Time.
    This is only applicable to the special-case where a column has the
    name 'TIME' and a time unit.
    """
def _convert_global_time(table, global_info) -> None:
    """
    Convert the table metadata for time informational keywords
    to astropy Time.

    Parameters
    ----------
    table : `~astropy.table.Table`
        The table whose time metadata is to be converted.
    global_info : dict
        Global time reference frame information.
    """
def _convert_time_key(global_info, key):
    """
    Convert a time metadata key to a Time object.

    Parameters
    ----------
    global_info : dict
        Global time reference frame information.
    key : str
        Time key.

    Returns
    -------
    astropy.time.Time

    Raises
    ------
    ValueError
        If key is not a valid global time keyword.
    """
def _convert_time_column(col, column_info):
    """
    Convert time columns to astropy Time columns.

    Parameters
    ----------
    col : `~astropy.table.Column`
        The time coordinate column to be converted to Time.
    column_info : dict
        Column-specific time reference frame override information.
    """
def fits_to_time(hdr, table):
    """
    Read FITS binary table time columns as `~astropy.time.Time`.

    This method reads the metadata associated with time coordinates, as
    stored in a FITS binary table header, converts time columns into
    `~astropy.time.Time` columns and reads global reference times as
    `~astropy.time.Time` instances.

    Parameters
    ----------
    hdr : `~astropy.io.fits.header.Header`
        FITS Header
    table : `~astropy.table.Table`
        The table whose time columns are to be read as Time

    Returns
    -------
    hdr : `~astropy.io.fits.header.Header`
        Modified FITS Header (time metadata removed)
    """
def time_to_fits(table):
    """
    Replace Time columns in a Table with non-mixin columns containing
    each element as a vector of two doubles (jd1, jd2) and return a FITS
    header with appropriate time coordinate keywords.
    jd = jd1 + jd2 represents time in the Julian Date format with
    high-precision.

    Parameters
    ----------
    table : `~astropy.table.Table`
        The table whose Time columns are to be replaced.

    Returns
    -------
    table : `~astropy.table.Table`
        The table with replaced Time columns
    hdr : `~astropy.io.fits.header.Header`
        Header containing global time reference frame FITS keywords
    """

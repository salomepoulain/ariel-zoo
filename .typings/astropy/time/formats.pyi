import datetime
from _typeshed import Incomplete
from astropy.utils.exceptions import AstropyUserWarning
from collections.abc import Generator

__all__ = ['AstropyDatetimeLeapSecondWarning', 'TimeFormat', 'TimeJD', 'TimeMJD', 'TimeFromEpoch', 'TimeUnix', 'TimeUnixTai', 'TimeCxcSec', 'TimeGPS', 'TimeDecimalYear', 'TimePlotDate', 'TimeUnique', 'TimeDatetime', 'TimeString', 'TimeISO', 'TimeISOT', 'TimeFITS', 'TimeYearDayTime', 'TimeEpochDate', 'TimeBesselianEpoch', 'TimeJulianEpoch', 'TimeDeltaFormat', 'TimeDeltaSec', 'TimeDeltaJD', 'TimeDeltaQuantityString', 'TimeEpochDateString', 'TimeBesselianEpochString', 'TimeJulianEpochString', 'TIME_FORMATS', 'TIME_DELTA_FORMATS', 'TimezoneInfo', 'TimeDeltaDatetime', 'TimeDatetime64', 'TimeYMDHMS', 'TimeNumeric', 'TimeDeltaNumeric']

TIME_FORMATS: Incomplete
TIME_DELTA_FORMATS: Incomplete

class AstropyDatetimeLeapSecondWarning(AstropyUserWarning):
    """Warning for leap second when converting to datetime.datetime object."""

class TimeFormat:
    """
    Base class for time representations.

    Parameters
    ----------
    val1 : numpy ndarray, list, number, str, or bytes
        Values to initialize the time or times.  Bytes are decoded as ascii.
        Quantities with time units are allowed for formats where the
        interpretation is unambiguous.
    val2 : numpy ndarray, list, or number; optional
        Value(s) to initialize the time or times.  Only used for numerical
        input, to help preserve precision.
    scale : str
        Time scale of input value(s)
    precision : int
        Precision for seconds as floating point
    in_subfmt : str
        Select subformat for inputting string times
    out_subfmt : str
        Select subformat for outputting string times
    from_jd : bool
        If true then val1, val2 are jd1, jd2
    """
    _default_scale: str
    _default_precision: int
    _min_precision: int
    _max_precision: int
    subfmts: Incomplete
    _registry = TIME_FORMATS
    _check_finite: bool
    def __init__(self, val1, val2, scale, precision, in_subfmt, out_subfmt, from_jd: bool = False) -> None: ...
    def __init_subclass__(cls, **kwargs): ...
    @classmethod
    def _get_allowed_subfmt(cls, subfmt):
        """Get an allowed subfmt for this class, either the input ``subfmt``
        if this is valid or '*' as a default.  This method gets used in situations
        where the format of an existing Time object is changing and so the
        out_ or in_subfmt may need to be coerced to the default '*' if that
        ``subfmt`` is no longer valid.
        """
    @property
    def in_subfmt(self): ...
    _in_subfmt: Incomplete
    @in_subfmt.setter
    def in_subfmt(self, subfmt) -> None: ...
    @property
    def out_subfmt(self): ...
    _out_subfmt: Incomplete
    @out_subfmt.setter
    def out_subfmt(self, subfmt) -> None: ...
    @property
    def jd1(self): ...
    _jd1: Incomplete
    @jd1.setter
    def jd1(self, jd1) -> None: ...
    @property
    def jd2(self): ...
    _jd2: Incomplete
    @jd2.setter
    def jd2(self, jd2) -> None: ...
    @classmethod
    def fill_value(cls, subfmt):
        """
        Return a value corresponding to J2000 (2000-01-01 12:00:00) in this format.

        This is used as a fill value for masked arrays to ensure that any ERFA
        operations on the masked array will not fail due to the masked value.
        """
    def __len__(self) -> int: ...
    _scale: Incomplete
    @property
    def scale(self):
        """Time scale."""
    @scale.setter
    def scale(self, val) -> None: ...
    @property
    def precision(self): ...
    _precision: Incomplete
    @precision.setter
    def precision(self, val) -> None: ...
    def _check_finite_vals(self, val1, val2) -> None:
        """A helper function to TimeFormat._check_val_type that's meant to be
        optionally bypassed in subclasses that have _check_finite=False
        """
    def _check_val_type(self, val1, val2):
        """Input value validation, typically overridden by derived classes."""
    def _check_scale(self, scale):
        """
        Return a validated scale value.

        If there is a class attribute 'scale' then that defines the default /
        required time scale for this format.  In this case if a scale value was
        provided that needs to match the class default, otherwise return
        the class default.

        Otherwise just make sure that scale is in the allowed list of
        scales.  Provide a different error message if `None` (no value) was
        supplied.
        """
    def set_jds(self, val1, val2) -> None:
        """
        Set internal jd1 and jd2 from val1 and val2.  Must be provided
        by derived classes.
        """
    def to_value(self, parent: Incomplete | None = None, out_subfmt: Incomplete | None = None):
        """
        Return time representation from internal jd1 and jd2 in specified
        ``out_subfmt``.

        This is the base method that ignores ``parent`` and uses the ``value``
        property to compute the output. This is done by temporarily setting
        ``self.out_subfmt`` and calling ``self.value``. This is required for
        legacy Format subclasses prior to astropy 4.0  New code should instead
        implement the value functionality in ``to_value()`` and then make the
        ``value`` property be a simple call to ``self.to_value()``.

        Parameters
        ----------
        parent : object
            Parent `~astropy.time.Time` object associated with this
            `~astropy.time.TimeFormat` object
        out_subfmt : str or None
            Output subformt (use existing self.out_subfmt if `None`)

        Returns
        -------
        value : numpy.array, numpy.ma.array
            Array or masked array of formatted time representation values
        """
    @property
    def value(self) -> None: ...
    @classmethod
    def _select_subfmts(cls, pattern):
        """
        Return a list of subformats where name matches ``pattern`` using
        fnmatch.

        If no subformat matches pattern then a ValueError is raised.  A special
        case is a format with no allowed subformats, i.e. subfmts=(), and
        pattern='*'.  This is OK and happens when this method is used for
        validation of an out_subfmt.
        """
    @classmethod
    def _fill_masked_values(cls, val, val2, mask, in_subfmt):
        """Fill masked values with the fill value for this format.

        This also takes care of broadcasting the outputs to the correct shape.

        Parameters
        ----------
        val : ndarray
            Array of values
        val2 : ndarray, None
            Array of second values (or None)
        mask : ndarray
            Mask array
        in_subfmt : str
            Input subformat

        Returns
        -------
        val, val2 : ndarray
            Arrays with masked values filled with the fill value for this format.
            These are copies of the originals.
        """

class TimeNumeric(TimeFormat):
    subfmts: Incomplete
    def _check_val_type(self, val1, val2):
        """Input value validation, typically overridden by derived classes."""
    def to_value(self, jd1: Incomplete | None = None, jd2: Incomplete | None = None, parent: Incomplete | None = None, out_subfmt: Incomplete | None = None):
        """
        Return time representation from internal jd1 and jd2.
        Subclasses that require ``parent`` or to adjust the jds should
        override this method.
        """
    value: Incomplete

class TimeJD(TimeNumeric):
    """
    Julian Date time format.

    This represents the number of days since the beginning of
    the Julian Period.
    For example, 2451544.5 in JD is midnight on January 1, 2000.
    """
    name: str
    def set_jds(self, val1, val2) -> None: ...

class TimeMJD(TimeNumeric):
    """
    Modified Julian Date time format.

    This represents the number of days since midnight on November 17, 1858.
    For example, 51544.0 in MJD is midnight on January 1, 2000.
    """
    name: str
    def set_jds(self, val1, val2) -> None: ...
    def to_value(self, **kwargs): ...
    value: Incomplete

class TimeDecimalYear(TimeNumeric):
    '''
    Time as a decimal year, with integer values corresponding to midnight of the first
    day of each year.

    The fractional part represents the exact fraction of the year, considering the
    precise number of days in the year (365 or 366). The following example shows
    essentially how the decimal year is computed::

      >>> from astropy.time import Time
      >>> tm = Time("2024-04-05T12:34:00")
      >>> tm0 = Time("2024-01-01T00:00:00")
      >>> tm1 = Time("2025-01-01T00:00:00")
      >>> print(2024 + (tm.jd - tm0.jd) / (tm1.jd - tm0.jd))  # doctest: +FLOAT_CMP
      2024.2609934729812
      >>> print(tm.decimalyear)  # doctest: +FLOAT_CMP
      2024.2609934729812

    Since for this format the length of the year varies between 365 and 366 days, it is
    not possible to use Quantity input, in which a year is always 365.25 days.

    This format is convenient for low-precision applications or for plotting data.
    '''
    name: str
    def _check_val_type(self, val1, val2): ...
    def set_jds(self, val1, val2) -> None: ...
    def to_value(self, **kwargs): ...
    value: Incomplete

class TimeFromEpoch(TimeNumeric):
    """
    Base class for times that represent the interval from a particular
    epoch as a numerical multiple of a unit time interval (e.g. seconds
    or days).
    """
    def _epoch(cls): ...
    @property
    def epoch(self):
        """Reference epoch time from which the time interval is measured."""
    def set_jds(self, val1, val2) -> None:
        """
        Initialize the internal jd1 and jd2 attributes given val1 and val2.
        For an TimeFromEpoch subclass like TimeUnix these will be floats giving
        the effective seconds since an epoch time (e.g. 1970-01-01 00:00:00).
        """
    def to_value(self, parent: Incomplete | None = None, **kwargs): ...
    value: Incomplete
    @property
    def _default_scale(self): ...

class TimeUnix(TimeFromEpoch):
    """
    Unix time (UTC): seconds from 1970-01-01 00:00:00 UTC, ignoring leap seconds.

    For example, 946684800.0 in Unix time is midnight on January 1, 2000.

    NOTE: this quantity is not exactly unix time and differs from the strict
    POSIX definition by up to 1 second on days with a leap second.  POSIX
    unix time actually jumps backward by 1 second at midnight on leap second
    days while this class value is monotonically increasing at 86400 seconds
    per UTC day.
    """
    name: str
    unit: Incomplete
    epoch_val: str
    epoch_val2: Incomplete
    epoch_scale: str
    epoch_format: str

class TimeUnixTai(TimeUnix):
    """
    Unix time (TAI): SI seconds elapsed since 1970-01-01 00:00:00 TAI (see caveats).

    This will generally differ from standard (UTC) Unix time by the cumulative
    integral number of leap seconds introduced into UTC since 1972-01-01 UTC
    plus the initial offset of 10 seconds at that date.

    This convention matches the definition of linux CLOCK_TAI
    (https://www.cl.cam.ac.uk/~mgk25/posix-clocks.html),
    and the Precision Time Protocol
    (https://en.wikipedia.org/wiki/Precision_Time_Protocol), which
    is also used by the White Rabbit protocol in High Energy Physics:
    https://white-rabbit.web.cern.ch.

    Caveats:

    - Before 1972, fractional adjustments to UTC were made, so the difference
      between ``unix`` and ``unix_tai`` time is no longer an integer.
    - Because of the fractional adjustments, to be very precise, ``unix_tai``
      is the number of seconds since ``1970-01-01 00:00:00 TAI`` or equivalently
      ``1969-12-31 23:59:51.999918 UTC``.  The difference between TAI and UTC
      at that epoch was 8.000082 sec.
    - On the day of a positive leap second the difference between ``unix`` and
      ``unix_tai`` times increases linearly through the day by 1.0. See also the
      documentation for the `~astropy.time.TimeUnix` class.
    - Negative leap seconds are possible, though none have been needed to date.

    Examples
    --------
      >>> # get the current offset between TAI and UTC
      >>> from astropy.time import Time
      >>> t = Time('2020-01-01', scale='utc')
      >>> t.unix_tai - t.unix
      np.float64(37.0)

      >>> # Before 1972, the offset between TAI and UTC was not integer
      >>> t = Time('1970-01-01', scale='utc')
      >>> t.unix_tai - t.unix  # doctest: +FLOAT_CMP
      np.float64(8.000082)

      >>> # Initial offset of 10 seconds in 1972
      >>> t = Time('1972-01-01', scale='utc')
      >>> t.unix_tai - t.unix
      np.float64(10.0)
    """
    name: str
    epoch_val: str
    epoch_scale: str

class TimeCxcSec(TimeFromEpoch):
    """
    Chandra X-ray Center seconds from 1998-01-01 00:00:00 TT.
    For example, 63072064.184 is midnight on January 1, 2000.
    """
    name: str
    unit: Incomplete
    epoch_val: str
    epoch_val2: Incomplete
    epoch_scale: str
    epoch_format: str

class TimeGPS(TimeFromEpoch):
    """GPS time: seconds from 1980-01-06 00:00:00 UTC
    For example, 630720013.0 is midnight on January 1, 2000.

    Notes
    -----
    This implementation is strictly a representation of the number of seconds
    (including leap seconds) since midnight UTC on 1980-01-06.  GPS can also be
    considered as a time scale which is ahead of TAI by a fixed offset
    (to within about 100 nanoseconds).

    For details, see https://www.usno.navy.mil/USNO/time/gps/usno-gps-time-transfer
    """
    name: str
    unit: Incomplete
    epoch_val: str
    epoch_val2: Incomplete
    epoch_scale: str
    epoch_format: str

class TimePlotDate(TimeFromEpoch):
    """
    Matplotlib `~matplotlib.pyplot.plot_date` input:
    1 + number of days from 0001-01-01 00:00:00 UTC.

    This can be used directly in the matplotlib `~matplotlib.pyplot.plot_date`
    function::

      >>> import matplotlib.pyplot as plt
      >>> jyear = np.linspace(2000, 2001, 20)
      >>> t = Time(jyear, format='jyear', scale='utc')
      >>> plt.plot_date(t.plot_date, jyear)
      >>> plt.gcf().autofmt_xdate()  # orient date labels at a slant
      >>> plt.draw()

    For example, 730120.0003703703 is midnight on January 1, 2000.
    """
    name: str
    unit: float
    epoch_val: float
    epoch_val2: Incomplete
    epoch_scale: str
    epoch_format: str
    def epoch(self):
        """Reference epoch time from which the time interval is measured."""

class TimeStardate(TimeFromEpoch):
    """
    Stardate: date units from 2318-07-05 12:00:00 UTC.
    For example, stardate 41153.7 is 00:52 on April 30, 2363.
    See http://trekguide.com/Stardates.htm#TNG for calculations and reference points.
    """
    name: str
    unit: float
    epoch_val: str
    epoch_val2: Incomplete
    epoch_scale: str
    epoch_format: str

class TimeUnique(TimeFormat):
    """
    Base class for time formats that can uniquely create a time object
    without requiring an explicit format specifier.  This class does
    nothing but provide inheritance to identify a class as unique.
    """

class TimeAstropyTime(TimeUnique):
    """
    Instantiate date from an Astropy Time object (or list thereof).

    This is purely for instantiating from a Time object.  The output
    format is the same as the first time instance.
    """
    name: str
    _location: Incomplete
    def __new__(cls, val1, val2, scale, precision, in_subfmt, out_subfmt, from_jd: bool = False):
        """
        Use __new__ instead of __init__ to output a class instance that
        is the same as the class of the first Time object in the list.
        """

class TimeDatetime(TimeUnique):
    """
    Represent date as Python standard library `~datetime.datetime` object.

    Example::

      >>> from astropy.time import Time
      >>> from datetime import datetime
      >>> t = Time(datetime(2000, 1, 2, 12, 0, 0), scale='utc')
      >>> t.iso
      '2000-01-02 12:00:00.000'
      >>> t.tt.datetime
      datetime.datetime(2000, 1, 2, 12, 1, 4, 184000)
    """
    name: str
    def _check_val_type(self, val1, val2): ...
    def set_jds(self, val1, val2) -> None:
        """Convert datetime object contained in val1 to jd1, jd2."""
    def to_value(self, timezone: Incomplete | None = None, leap_second_strict: str = 'raise', parent: Incomplete | None = None, out_subfmt: Incomplete | None = None):
        '''
        Convert to (potentially timezone-aware) `~datetime.datetime` object.

        If ``timezone`` is not ``None``, return a timezone-aware datetime object.

        Since the `~datetime.datetime` class does not natively handle leap seconds, the
        behavior when converting a time within a leap second is controlled by the
        ``leap_second_strict`` argument. For example::

          >>> from astropy.time import Time
          >>> t = Time("2015-06-30 23:59:60.500")
          >>> print(t.to_datetime(leap_second_strict=\'silent\'))
          2015-07-01 00:00:00.500000

        Parameters
        ----------
        timezone : {`~datetime.tzinfo`, None}, optional
            If not `None`, return timezone-aware datetime.
        leap_second_strict : str, optional
            If ``raise`` (default), raise an exception if the time is within a leap
            second. If ``warn`` then issue a warning. If ``silent`` then silently
            handle the leap second.

        Returns
        -------
        `~datetime.datetime`
            If ``timezone`` is not ``None``, output will be timezone-aware.
        '''
    value: Incomplete

class TimeYMDHMS(TimeUnique):
    '''
    ymdhms: A Time format to represent Time as year, month, day, hour,
    minute, second (thus the name ymdhms).

    Acceptable inputs must have keys or column names in the "YMDHMS" set of
    ``year``, ``month``, ``day`` ``hour``, ``minute``, ``second``:

    - Dict with keys in the YMDHMS set
    - NumPy structured array, record array or astropy Table, or single row
      of those types, with column names in the YMDHMS set

    One can supply a subset of the YMDHMS values, for instance only \'year\',
    \'month\', and \'day\'.  Inputs have the following defaults::

      \'month\': 1, \'day\': 1, \'hour\': 0, \'minute\': 0, \'second\': 0

    When the input is supplied as a ``dict`` then each value can be either a
    scalar value or an array.  The values will be broadcast to a common shape.

    Example::

      >>> from astropy.time import Time
      >>> t = Time({\'year\': 2015, \'month\': 2, \'day\': 3,
      ...           \'hour\': 12, \'minute\': 13, \'second\': 14.567},
      ...           scale=\'utc\')
      >>> t.iso
      \'2015-02-03 12:13:14.567\'
      >>> t.ymdhms.year
      np.int32(2015)
    '''
    name: str
    def _check_val_type(self, val1, val2):
        """
        This checks inputs for the YMDHMS format.

        It is bit more complex than most format checkers because of the flexible
        input that is allowed.  Also, it actually coerces ``val1`` into an appropriate
        dict of ndarrays that can be used easily by ``set_jds()``.  This is useful
        because it makes it easy to get default values in that routine.

        Parameters
        ----------
        val1 : ndarray or None
        val2 : ndarray or None

        Returns
        -------
        val1_as_dict, val2 : val1 as dict or None, val2 is always None

        """
    def set_jds(self, val1, val2) -> None: ...
    @property
    def value(self): ...

class TimezoneInfo(datetime.tzinfo):
    """
    Subclass of the `~datetime.tzinfo` object, used in the
    to_datetime method to specify timezones.

    It may be safer in most cases to use a timezone database package like
    pytz rather than defining your own timezones - this class is mainly
    a workaround for users without pytz.
    """
    _utcoffset: Incomplete
    _tzname: Incomplete
    _dst: Incomplete
    def __init__(self, utc_offset=..., dst=..., tzname: Incomplete | None = None) -> None:
        """
        Parameters
        ----------
        utc_offset : `~astropy.units.Quantity`, optional
            Offset from UTC in days. Defaults to zero.
        dst : `~astropy.units.Quantity`, optional
            Daylight Savings Time offset in days. Defaults to zero
            (no daylight savings).
        tzname : str or None, optional
            Name of timezone

        Examples
        --------
        >>> from datetime import datetime
        >>> from astropy.time import TimezoneInfo  # Specifies a timezone
        >>> import astropy.units as u
        >>> utc = TimezoneInfo()    # Defaults to UTC
        >>> utc_plus_one_hour = TimezoneInfo(utc_offset=1*u.hour)  # UTC+1
        >>> dt_aware = datetime(2000, 1, 1, 0, 0, 0, tzinfo=utc_plus_one_hour)
        >>> print(dt_aware)
        2000-01-01 00:00:00+01:00
        >>> print(dt_aware.astimezone(utc))
        1999-12-31 23:00:00+00:00
        """
    def utcoffset(self, dt): ...
    def tzname(self, dt): ...
    def dst(self, dt): ...

class TimeString(TimeUnique):
    '''
    Base class for string-like time representations.

    This class assumes that anything following the last decimal point to the
    right is a fraction of a second.

    **Fast C-based parser**

    Time format classes can take advantage of a fast C-based parser if the times
    are represented as fixed-format strings with year, month, day-of-month,
    hour, minute, second, OR year, day-of-year, hour, minute, second. This can
    be a factor of 20 or more faster than the pure Python parser.

    Fixed format means that the components always have the same number of
    characters. The Python parser will accept ``2001-9-2`` as a date, but the C
    parser would require ``2001-09-02``.

    A subclass in this case must define a class attribute ``fast_parser_pars``
    which is a `dict` with all of the keys below. An inherited attribute is not
    checked, only an attribute in the class ``__dict__``.

    - ``delims`` (tuple of int): ASCII code for character at corresponding
      ``starts`` position (0 => no character)

    - ``starts`` (tuple of int): position where component starts (including
      delimiter if present). Use -1 for the month component for format that use
      day of year.

    - ``stops`` (tuple of int): position where component ends. Use -1 to
      continue to end of string, or for the month component for formats that use
      day of year.

    - ``break_allowed`` (tuple of int): if true (1) then the time string can
          legally end just before the corresponding component (e.g. "2000-01-01"
          is a valid time but "2000-01-01 12" is not).

    - ``has_day_of_year`` (int): 0 if dates have year, month, day; 1 if year,
      day-of-year
    '''
    def __init_subclass__(cls, **kwargs) -> None: ...
    def _check_val_type(self, val1, val2): ...
    def parse_string(self, timestr, subfmts):
        """Read time from a single string, using a set of possible formats."""
    jd1: Incomplete
    jd2: Incomplete
    def set_jds(self, val1, val2) -> None:
        """Parse the time strings contained in val1 and set jd1, jd2."""
    def get_jds_python(self, val1, val2):
        """Parse the time strings contained in val1 and get jd1, jd2."""
    def get_jds_fast(self, val1, val2):
        """Use fast C parser to parse time strings in val1 and get jd1, jd2."""
    def str_kwargs(self) -> Generator[Incomplete]:
        """
        Generator that yields a dict of values corresponding to the
        calendar date and time for the internal JD values.
        """
    def format_string(self, str_fmt, **kwargs):
        """Write time to a string using a given format.

        By default, just interprets str_fmt as a format string,
        but subclasses can add to this.
        """
    @property
    def value(self): ...

class TimeISO(TimeString):
    '''
    ISO 8601 compliant date-time format "YYYY-MM-DD HH:MM:SS.sss...".
    For example, 2000-01-01 00:00:00.000 is midnight on January 1, 2000.

    The allowed subformats are:

    - \'date_hms\': date + hours, mins, secs (and optional fractional secs)
    - \'date_hm\': date + hours, mins
    - \'date\': date
    '''
    name: str
    subfmts: Incomplete
    fast_parser_pars: Incomplete
    def parse_string(self, timestr, subfmts): ...

class TimeISOT(TimeISO):
    '''
    ISO 8601 compliant date-time format "YYYY-MM-DDTHH:MM:SS.sss...".
    This is the same as TimeISO except for a "T" instead of space between
    the date and time.
    For example, 2000-01-01T00:00:00.000 is midnight on January 1, 2000.

    The allowed subformats are:

    - \'date_hms\': date + hours, mins, secs (and optional fractional secs)
    - \'date_hm\': date + hours, mins
    - \'date\': date
    '''
    name: str
    subfmts: Incomplete
    fast_parser_pars: Incomplete

class TimeYearDayTime(TimeISO):
    '''
    Year, day-of-year and time as "YYYY:DOY:HH:MM:SS.sss...".
    The day-of-year (DOY) goes from 001 to 365 (366 in leap years).
    For example, 2000:001:00:00:00.000 is midnight on January 1, 2000.

    The allowed subformats are:

    - \'date_hms\': date + hours, mins, secs (and optional fractional secs)
    - \'date_hm\': date + hours, mins
    - \'date\': date
    '''
    name: str
    subfmts: Incomplete
    fast_parser_pars: Incomplete

class TimeDatetime64(TimeISOT):
    name: str
    def _check_val_type(self, val1, val2): ...
    jd1: Incomplete
    jd2: Incomplete
    def set_jds(self, val1, val2) -> None: ...
    precision: int
    @property
    def value(self): ...

class TimeFITS(TimeString):
    '''
    FITS format: "[±Y]YYYY-MM-DD[THH:MM:SS[.sss]]".

    ISOT but can give signed five-digit year (mostly for negative years);

    The allowed subformats are:

    - \'date_hms\': date + hours, mins, secs (and optional fractional secs)
    - \'date\': date
    - \'longdate_hms\': as \'date_hms\', but with signed 5-digit year
    - \'longdate\': as \'date\', but with signed 5-digit year

    See Rots et al., 2015, A&A 574:A36 (arXiv:1409.7583).
    '''
    name: str
    subfmts: Incomplete
    _scale: Incomplete
    def parse_string(self, timestr, subfmts):
        """Read time and deprecated scale if present."""
    out_subfmt: Incomplete
    @property
    def value(self):
        """Convert times to strings, using signed 5 digit if necessary."""

class TimeEpochDate(TimeNumeric):
    """
    Base class for support of Besselian and Julian epoch dates.
    """
    _default_scale: str
    def set_jds(self, val1, val2) -> None: ...
    def to_value(self, **kwargs): ...
    value: Incomplete

class TimeBesselianEpoch(TimeEpochDate):
    """Besselian Epoch year as decimal value(s) like 1950.0.

    For information about this epoch format, see:
    `<https://en.wikipedia.org/wiki/Epoch_(astronomy)#Besselian_years>`_.

    The astropy Time class uses the ERFA functions ``epb2jd`` and ``epb`` to convert
    between Besselian epoch years and Julian dates. This is roughly equivalent to the
    following formula (see the wikipedia page for the reference)::

      B = 1900.0 + (Julian date - 2415020.31352) / 365.242198781

    Since for this format the length of the year varies, input needs to be floating
    point; it is not possible to use Quantity input, for which a year always equals
    365.25 days.

    The Besselian epoch year is used for expressing the epoch or equinox in older source
    catalogs, but it has been largely replaced by the Julian epoch year.
    """
    name: str
    epoch_to_jd: str
    jd_to_epoch: str
    def _check_val_type(self, val1, val2): ...

class TimeJulianEpoch(TimeEpochDate):
    '''Julian epoch year as decimal value(s) like 2000.0.

    This format is based the Julian year which is exactly 365.25 days/year and a day is
    exactly 86400 SI seconds.

    The Julian epoch year is defined so that 2000.0 is 12:00 TT on January 1, 2000.
    Using astropy this is expressed as::

      >>> from astropy.time import Time
      >>> import astropy.units as u
      >>> j2000_epoch = Time("2000-01-01T12:00:00", scale="tt")
      >>> print(j2000_epoch.jyear)  # doctest: +FLOAT_CMP
      2000.0
      >>> print((j2000_epoch + 365.25 * u.day).jyear)  # doctest: +FLOAT_CMP
      2001.0

    The Julian year is commonly used in astronomy for expressing the epoch of a source
    catalog or the time of an observation. The Julian epoch year is sometimes written as
    a string like "J2001.5" with a preceding "J". You can initialize a ``Time`` object with
    such a string::

      >>> print(Time("J2001.5").jyear)  # doctest: +FLOAT_CMP
      2001.5

    See also: `<https://en.wikipedia.org/wiki/Julian_year_(astronomy)>`_.
    '''
    name: str
    unit: Incomplete
    epoch_to_jd: str
    jd_to_epoch: str

class TimeEpochDateString(TimeString):
    """
    Base class to support string Besselian and Julian epoch dates
    such as 'B1950.0' or 'J2000.0' respectively.
    """
    _default_scale: str
    def set_jds(self, val1, val2): ...
    @property
    def value(self): ...

class TimeBesselianEpochString(TimeEpochDateString):
    """Besselian Epoch year as string value(s) like 'B1950.0'."""
    name: str
    epoch_to_jd: str
    jd_to_epoch: str
    epoch_prefix: str

class TimeJulianEpochString(TimeEpochDateString):
    """Julian Epoch year as string value(s) like 'J2000.0'."""
    name: str
    epoch_to_jd: str
    jd_to_epoch: str
    epoch_prefix: str

class TimeDeltaFormat(TimeFormat):
    """Base class for time delta representations."""
    _registry = TIME_DELTA_FORMATS
    _default_precision: int
    _min_precision: int
    _max_precision: int
    def _check_scale(self, scale):
        """
        Check that the scale is in the allowed list of scales, or is `None`.
        """

class TimeDeltaNumeric(TimeDeltaFormat, TimeNumeric):
    _check_finite: bool
    def set_jds(self, val1, val2) -> None: ...
    def to_value(self, **kwargs): ...
    value: Incomplete

class TimeDeltaSec(TimeDeltaNumeric):
    """Time delta in SI seconds."""
    name: str
    unit: Incomplete

class TimeDeltaJD(TimeDeltaNumeric, TimeUnique):
    """Time delta in Julian days (86400 SI seconds)."""
    name: str
    unit: float

class TimeDeltaDatetime(TimeDeltaFormat, TimeUnique):
    """Time delta in datetime.timedelta."""
    name: str
    def _check_val_type(self, val1, val2): ...
    def set_jds(self, val1, val2) -> None: ...
    @property
    def value(self): ...

class TimeDeltaQuantityString(TimeDeltaFormat, TimeUnique):
    '''Time delta as a string with one or more Quantity components.

    This format provides a human-readable multi-scale string representation of a time
    delta. It is convenient for applications like a configuration file or a command line
    option.

    The format is specified as follows:

    - The string is a sequence of one or more components.
    - Each component is a number followed by an astropy unit of time.
    - For input, whitespace within the string is allowed but optional.
    - For output, there is a single space between components.
    - The allowed components are listed below.
    - The order (yr, d, hr, min, s) is fixed but individual components are optional.

    The allowed input component units are shown below:

    - "yr": years (365.25 days)
    - "d": days (24 hours)
    - "hr": hours (60 minutes)
    - "min": minutes (60 seconds)
    - "s": seconds

    .. Note:: These definitions correspond to physical units of time and are NOT
       calendar date intervals. Thus adding "1yr" to "2000-01-01 00:00:00" will give
       "2000-12-31 06:00:00" instead of "2001-01-01 00:00:00".

    The ``out_subfmt`` attribute specifies the components to be included in the string
    output.  The default is ``"multi"`` which represents the time delta as
    ``"<days>d <hours>hr <minutes>min <seconds>s"``, where only non-zero components are
    included.

    - "multi": multiple components, e.g. "2d 3hr 15min 5.6s"
    - "yr": years
    - "d": days
    - "hr": hours
    - "min": minutes
    - "s": seconds

    Examples
    --------
    >>> from astropy.time import Time, TimeDelta
    >>> import astropy.units as u

    >>> print(TimeDelta("1yr"))
    365d 6hr

    >>> print(Time("2000-01-01") + TimeDelta("1yr"))
    2000-12-31 06:00:00.000
    >>> print(TimeDelta("+3.6d"))
    3d 14hr 24min
    >>> print(TimeDelta("-3.6d"))
    -3d 14hr 24min
    >>> print(TimeDelta("1yr 3.6d", out_subfmt="d"))
    368.85d

    >>> td = TimeDelta(40 * u.hr)
    >>> print(td.to_value(format="quantity_str"))
    1d 16hr
    >>> print(td.to_value(format="quantity_str", subfmt="d"))
    1.667d
    >>> td.precision = 9
    >>> print(td.to_value(format="quantity_str", subfmt="d"))
    1.666666667d
    '''
    name: str
    subfmts: Incomplete
    re_float: str
    re_ydhms: Incomplete
    def _check_val_type(self, val1, val2): ...
    def parse_string(self, timestr):
        """Read time from a single string"""
    def set_jds(self, val1, val2):
        """Parse the time strings contained in val1 and get jd1, jd2."""
    def to_value(self, parent: Incomplete | None = None, out_subfmt: Incomplete | None = None): ...
    def get_multi_comps(self, jd1, jd2): ...
    @staticmethod
    def fix_comp_vals_overflow(comp_vals) -> None: ...
    @property
    def value(self): ...

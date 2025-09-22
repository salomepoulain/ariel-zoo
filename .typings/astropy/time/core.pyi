import enum
from .formats import TIME_DELTA_FORMATS, TIME_FORMATS
from _typeshed import Incomplete
from astropy.coordinates import EarthLocation
from astropy.utils.data_info import MixinInfo
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.utils.masked import MaskableShapedLikeNDArray

__all__ = ['TimeBase', 'Time', 'TimeDelta', 'TimeInfo', 'TimeInfoBase', 'update_leap_seconds', 'TIME_SCALES', 'STANDARD_TIME_SCALES', 'TIME_DELTA_SCALES', 'ScaleValueError', 'OperandTypeError', 'TimeDeltaMissingUnitWarning']

STANDARD_TIME_SCALES: Incomplete
TIME_SCALES: Incomplete
TIME_DELTA_SCALES: Incomplete

class _LeapSecondsCheck(enum.Enum):
    NOT_STARTED = 0
    RUNNING = 1
    DONE = 2

class TimeInfoBase(MixinInfo):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.

    This base class is common between TimeInfo and TimeDeltaInfo.
    """
    attr_names: Incomplete
    _supports_indexing: bool
    _represent_as_dict_extra_attrs: Incomplete
    _represent_as_dict_primary_data: str
    mask_val: Incomplete
    @property
    def _represent_as_dict_attrs(self): ...
    serialize_method: Incomplete
    def __init__(self, bound: bool = False) -> None: ...
    def get_sortable_arrays(self):
        """
        Return a list of arrays which can be lexically sorted to represent
        the order of the parent column.

        Returns
        -------
        arrays : list of ndarray
        """
    @property
    def unit(self) -> None: ...
    info_summary_stats: Incomplete
    def _construct_from_dict(self, map): ...
    def new_like(self, cols, length, metadata_conflicts: str = 'warn', name: Incomplete | None = None):
        """
        Return a new Time instance which is consistent with the input Time objects
        ``cols`` and has ``length`` rows.

        This is intended for creating an empty Time instance whose elements can
        be set in-place for table operations like join or vstack.  It checks
        that the input locations and attributes are consistent.  This is used
        when a Time object is used as a mixin column in an astropy Table.

        Parameters
        ----------
        cols : list
            List of input columns (Time objects)
        length : int
            Length of the output column object
        metadata_conflicts : str ('warn'|'error'|'silent')
            How to handle metadata conflicts
        name : str
            Output column name

        Returns
        -------
        col : Time (or subclass)
            Empty instance of this class consistent with ``cols``

        """

class TimeInfo(TimeInfoBase):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """
    def _represent_as_dict(self, attrs: Incomplete | None = None):
        """Get the values for the parent ``attrs`` and return as a dict.

        By default, uses '_represent_as_dict_attrs'.
        """
    def _construct_from_dict(self, map): ...

class TimeDeltaInfo(TimeInfoBase):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """
    _represent_as_dict_extra_attrs: Incomplete
    def new_like(self, cols, length, metadata_conflicts: str = 'warn', name: Incomplete | None = None):
        """
        Return a new TimeDelta instance which is consistent with the input Time objects
        ``cols`` and has ``length`` rows.

        This is intended for creating an empty Time instance whose elements can
        be set in-place for table operations like join or vstack.  It checks
        that the input locations and attributes are consistent.  This is used
        when a Time object is used as a mixin column in an astropy Table.

        Parameters
        ----------
        cols : list
            List of input columns (Time objects)
        length : int
            Length of the output column object
        metadata_conflicts : str ('warn'|'error'|'silent')
            How to handle metadata conflicts
        name : str
            Output column name

        Returns
        -------
        col : Time (or subclass)
            Empty instance of this class consistent with ``cols``

        """

class TimeBase(MaskableShapedLikeNDArray):
    """Base time class from which Time and TimeDelta inherit."""
    __array_priority__: int
    _astropy_column_attrs: Incomplete
    def __getnewargs__(self): ...
    def __getstate__(self): ...
    _time: Incomplete
    _format: Incomplete
    _location: Incomplete
    def _init_from_vals(self, val, val2, format, scale, copy, precision: Incomplete | None = None, in_subfmt: Incomplete | None = None, out_subfmt: Incomplete | None = None) -> None:
        """
        Set the internal _format, scale, and _time attrs from user
        inputs.  This handles coercion into the correct shapes and
        some basic input validation.
        """
    def _get_time_fmt(self, val, val2, format, scale, precision, in_subfmt, out_subfmt, mask):
        """
        Given the supplied val, val2, format and scale try to instantiate
        the corresponding TimeFormat class to convert the input values into
        the internal jd1 and jd2.

        If format is `None` and the input is a string-type or object array then
        guess available formats and stop when one matches.
        """
    @property
    def writeable(self): ...
    @writeable.setter
    def writeable(self, value) -> None: ...
    @property
    def format(self):
        """
        Get or set time format.

        The format defines the way times are represented when accessed via the
        ``.value`` attribute.  By default it is the same as the format used for
        initializing the `Time` instance, but it can be set to any other value
        that could be used for initialization.  These can be listed with::

          >>> list(Time.FORMATS)
          ['jd', 'mjd', 'decimalyear', 'unix', 'unix_tai', 'cxcsec', 'gps', 'plot_date',
           'stardate', 'datetime', 'ymdhms', 'iso', 'isot', 'yday', 'datetime64',
           'fits', 'byear', 'jyear', 'byear_str', 'jyear_str']
        """
    @format.setter
    def format(self, format) -> None:
        """Set time format."""
    def to_string(self):
        """Output a string representation of the Time or TimeDelta object.

        Similar to ``str(self.value)`` (which uses numpy array formatting) but
        array values are evaluated only for the items that actually are output.
        For large arrays this can be a substantial performance improvement.

        Returns
        -------
        out : str
            String representation of the time values.

        """
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def __hash__(self): ...
    @property
    def location(self) -> EarthLocation | None: ...
    @location.setter
    def location(self, value) -> None: ...
    @property
    def scale(self):
        """Time scale."""
    def _set_scale(self, scale) -> None:
        """
        This is the key routine that actually does time scale conversions.
        This is not public and not connected to the read-only scale property.
        """
    @property
    def precision(self):
        """
        Decimal precision when outputting seconds as floating point (int
        value between 0 and 9 inclusive).
        """
    @precision.setter
    def precision(self, val) -> None: ...
    @property
    def in_subfmt(self):
        """
        Unix wildcard pattern to select subformats for parsing string input
        times.
        """
    @in_subfmt.setter
    def in_subfmt(self, val) -> None: ...
    @property
    def out_subfmt(self):
        """
        Unix wildcard pattern to select subformats for outputting times.
        """
    @out_subfmt.setter
    def out_subfmt(self, val) -> None: ...
    @property
    def shape(self):
        """The shape of the time instances.

        Like `~numpy.ndarray.shape`, can be set to a new shape by assigning a
        tuple.  Note that if different instances share some but not all
        underlying data, setting the shape of one instance can make the other
        instance unusable.  Hence, it is strongly recommended to get new,
        reshaped instances with the ``reshape`` method.

        Raises
        ------
        ValueError
            If the new shape has the wrong total number of elements.
        AttributeError
            If the shape of the ``jd1``, ``jd2``, ``location``,
            ``delta_ut1_utc``, or ``delta_tdb_tt`` attributes cannot be changed
            without the arrays being copied.  For these cases, use the
            `Time.reshape` method (which copies any arrays that cannot be
            reshaped in-place).
        """
    @shape.setter
    def shape(self, shape) -> None: ...
    def _shaped_like_input(self, value): ...
    @property
    def jd1(self):
        """
        First of the two doubles that internally store time value(s) in JD.
        """
    @property
    def jd2(self):
        """
        Second of the two doubles that internally store time value(s) in JD.
        """
    def to_value(self, format, subfmt: str = '*'):
        """Get time values expressed in specified output format.

        This method allows representing the ``Time`` object in the desired
        output ``format`` and optional sub-format ``subfmt``.  Available
        built-in formats include ``jd``, ``mjd``, ``iso``, and so forth. Each
        format can have its own sub-formats

        For built-in numerical formats like ``jd`` or ``unix``, ``subfmt`` can
        be one of 'float', 'long', 'decimal', 'str', or 'bytes'.  Here, 'long'
        uses ``numpy.longdouble`` for somewhat enhanced precision (with
        the enhancement depending on platform), and 'decimal'
        :class:`decimal.Decimal` for full precision.  For 'str' and 'bytes', the
        number of digits is also chosen such that time values are represented
        accurately.

        For built-in date-like string formats, one of 'date_hms', 'date_hm', or
        'date' (or 'longdate_hms', etc., for 5-digit years in
        `~astropy.time.TimeFITS`).  For sub-formats including seconds, the
        number of digits used for the fractional seconds is as set by
        `~astropy.time.Time.precision`.

        Parameters
        ----------
        format : str
            The format in which one wants the time values. Default: the current
            format.
        subfmt : str or None, optional
            Value or wildcard pattern to select the sub-format in which the
            values should be given.  The default of '*' picks the first
            available for a given format, i.e., 'float' or 'date_hms'.
            If `None`, use the instance's ``out_subfmt``.

        """
    @property
    def value(self):
        """Time value(s) in current format."""
    @property
    def mask(self): ...
    @property
    def masked(self): ...
    def insert(self, obj, values, axis: int = 0):
        """
        Insert values before the given indices in the column and return
        a new `~astropy.time.Time` or  `~astropy.time.TimeDelta` object.

        The values to be inserted must conform to the rules for in-place setting
        of ``Time`` objects (see ``Get and set values`` in the ``Time``
        documentation).

        The API signature matches the ``np.insert`` API, but is more limited.
        The specification of insert index ``obj`` must be a single integer,
        and the ``axis`` must be ``0`` for simple row insertion before the
        index.

        Parameters
        ----------
        obj : int
            Integer index before which ``values`` is inserted.
        values : array-like
            Value(s) to insert.  If the type of ``values`` is different
            from that of quantity, ``values`` is converted to the matching type.
        axis : int, optional
            Axis along which to insert ``values``.  Default is 0, which is the
            only allowed value and will insert a row.

        Returns
        -------
        out : `~astropy.time.Time` subclass
            New time object with inserted value(s)

        """
    def __setitem__(self, item, value) -> None: ...
    def isclose(self, other, atol: Incomplete | None = None):
        """Returns a boolean or boolean array where two Time objects are
        element-wise equal within a time tolerance.

        This evaluates the expression below::

          abs(self - other) <= atol

        Parameters
        ----------
        other : `~astropy.time.Time`
            Time object for comparison.
        atol : `~astropy.units.Quantity` or `~astropy.time.TimeDelta`
            Absolute tolerance for equality with units of time (e.g. ``u.s`` or
            ``u.day``). Default is two bits in the 128-bit JD time representation,
            equivalent to about 40 picosecs.
        """
    def copy(self, format: Incomplete | None = None):
        """
        Return a fully independent copy the Time object, optionally changing
        the format.

        If ``format`` is supplied then the time format of the returned Time
        object will be set accordingly, otherwise it will be unchanged from the
        original.

        In this method a full copy of the internal time arrays will be made.
        The internal time arrays are normally not changeable by the user so in
        most cases the ``replicate()`` method should be used.

        Parameters
        ----------
        format : str, optional
            Time format of the copy.

        Returns
        -------
        tm : Time object
            Copy of this object
        """
    def replicate(self, format: Incomplete | None = None, copy: bool = False, cls: Incomplete | None = None):
        """
        Return a replica of the Time object, optionally changing the format.

        If ``format`` is supplied then the time format of the returned Time
        object will be set accordingly, otherwise it will be unchanged from the
        original.

        If ``copy`` is set to `True` then a full copy of the internal time arrays
        will be made.  By default the replica will use a reference to the
        original arrays when possible to save memory.  The internal time arrays
        are normally not changeable by the user so in most cases it should not
        be necessary to set ``copy`` to `True`.

        The convenience method copy() is available in which ``copy`` is `True`
        by default.

        Parameters
        ----------
        format : str, optional
            Time format of the replica.
        copy : bool, optional
            Return a true copy instead of using references where possible.

        Returns
        -------
        tm : Time object
            Replica of this object
        """
    def _apply(self, method, *args, format: Incomplete | None = None, cls: Incomplete | None = None, **kwargs):
        """Create a new time object, possibly applying a method to the arrays.

        Parameters
        ----------
        method : str or callable
            If string, can be 'replicate'  or the name of a relevant
            `~numpy.ndarray` method. In the former case, a new time instance
            with unchanged internal data is created, while in the latter the
            method is applied to the internal ``jd1`` and ``jd2`` arrays, as
            well as to possible ``location``, ``_delta_ut1_utc``, and
            ``_delta_tdb_tt`` arrays.
            If a callable, it is directly applied to the above arrays.
            Examples: 'copy', '__getitem__', 'reshape', `~numpy.broadcast_to`.
        args : tuple
            Any positional arguments for ``method``.
        kwargs : dict
            Any keyword arguments for ``method``.  If the ``format`` keyword
            argument is present, this will be used as the Time format of the
            replica.

        Examples
        --------
        Some ways this is used internally::

            copy : ``_apply('copy')``
            replicate : ``_apply('replicate')``
            reshape : ``_apply('reshape', new_shape)``
            index or slice : ``_apply('__getitem__', item)``
            broadcast : ``_apply(np.broadcast, shape=new_shape)``
        """
    def __copy__(self):
        """
        Overrides the default behavior of the `copy.copy` function in
        the python stdlib to behave like `Time.copy`. Does *not* make a
        copy of the JD arrays - only copies by reference.
        """
    def __deepcopy__(self, memo):
        """
        Overrides the default behavior of the `copy.deepcopy` function
        in the python stdlib to behave like `Time.copy`. Does make a
        copy of the JD arrays.
        """
    def _advanced_index(self, indices, axis: Incomplete | None = None, keepdims: bool = False):
        """Turn argmin, argmax output into an advanced index.

        Argmin, argmax output contains indices along a given axis in an array
        shaped like the other dimensions.  To use this to get values at the
        correct location, a list is constructed in which the other axes are
        indexed sequentially.  For ``keepdims`` is ``True``, the net result is
        the same as constructing an index grid with ``np.ogrid`` and then
        replacing the ``axis`` item with ``indices`` with its shaped expanded
        at ``axis``. For ``keepdims`` is ``False``, the result is the same but
        with the ``axis`` dimension removed from all list entries.

        For ``axis`` is ``None``, this calls :func:`~numpy.unravel_index`.

        Parameters
        ----------
        indices : array
            Output of argmin or argmax.
        axis : int or None
            axis along which argmin or argmax was used.
        keepdims : bool
            Whether to construct indices that keep or remove the axis along
            which argmin or argmax was used.  Default: ``False``.

        Returns
        -------
        advanced_index : list of arrays
            Suitable for use as an advanced index.
        """
    def argmin(self, axis: Incomplete | None = None, out: Incomplete | None = None):
        """Return indices of the minimum values along the given axis.

        This is similar to :meth:`~numpy.ndarray.argmin`, but adapted to ensure
        that the full precision given by the two doubles ``jd1`` and ``jd2``
        is used.  See :func:`~numpy.argmin` for detailed documentation.
        """
    def argmax(self, axis: Incomplete | None = None, out: Incomplete | None = None):
        """Return indices of the maximum values along the given axis.

        This is similar to :meth:`~numpy.ndarray.argmax`, but adapted to ensure
        that the full precision given by the two doubles ``jd1`` and ``jd2``
        is used.  See :func:`~numpy.argmax` for detailed documentation.
        """
    def argsort(self, axis: int = -1, kind: str = 'stable'):
        """Returns the indices that would sort the time array.

        This is similar to :meth:`~numpy.ndarray.argsort`, but adapted to ensure that
        the full precision given by the two doubles ``jd1`` and ``jd2`` is used, and
        that corresponding attributes are copied.  Internally, it uses
        :func:`~numpy.lexsort`, and hence no sort method can be chosen.

        Parameters
        ----------
        axis : int, optional
            Axis along which to sort. Default is -1, which means sort along the last
            axis.
        kind : 'stable', optional
            Sorting is done with :func:`~numpy.lexsort` so this argument is ignored, but
            kept for compatibility with :func:`~numpy.argsort`. The sorting is stable,
            meaning that the order of equal elements is preserved.

        Returns
        -------
        indices : ndarray
            An array of indices that sort the time array.
        """
    def min(self, axis: Incomplete | None = None, out: Incomplete | None = None, keepdims: bool = False):
        """Minimum along a given axis.

        This is similar to :meth:`~numpy.ndarray.min`, but adapted to ensure
        that the full precision given by the two doubles ``jd1`` and ``jd2``
        is used, and that corresponding attributes are copied.

        Note that the ``out`` argument is present only for compatibility with
        ``np.min``; since `Time` instances are immutable, it is not possible
        to have an actual ``out`` to store the result in.
        """
    def max(self, axis: Incomplete | None = None, out: Incomplete | None = None, keepdims: bool = False):
        """Maximum along a given axis.

        This is similar to :meth:`~numpy.ndarray.max`, but adapted to ensure
        that the full precision given by the two doubles ``jd1`` and ``jd2``
        is used, and that corresponding attributes are copied.

        Note that the ``out`` argument is present only for compatibility with
        ``np.max``; since `Time` instances are immutable, it is not possible
        to have an actual ``out`` to store the result in.
        """
    def _ptp_impl(self, axis: Incomplete | None = None, out: Incomplete | None = None, keepdims: bool = False): ...
    def __array_function__(self, function, types, args, kwargs): ...
    def ptp(self, axis: Incomplete | None = None, out: Incomplete | None = None, keepdims: bool = False):
        """Peak to peak (maximum - minimum) along a given axis.

        This method is similar to the :func:`numpy.ptp` function, but
        adapted to ensure that the full precision given by the two doubles
        ``jd1`` and ``jd2`` is used.

        Note that the ``out`` argument is present only for compatibility with
        `~numpy.ptp`; since `Time` instances are immutable, it is not possible
        to have an actual ``out`` to store the result in.
        """
    def sort(self, axis: int = -1):
        """Return a copy sorted along the specified axis.

        This is similar to :meth:`~numpy.ndarray.sort`, but internally uses
        indexing with :func:`~numpy.lexsort` to ensure that the full precision
        given by the two doubles ``jd1`` and ``jd2`` is kept, and that
        corresponding attributes are properly sorted and copied as well.

        Parameters
        ----------
        axis : int or None
            Axis to be sorted.  If ``None``, the flattened array is sorted.
            By default, sort over the last axis.
        """
    def mean(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None, keepdims: bool = False, *, where: bool = True):
        """Mean along a given axis.

        This is similar to :meth:`~numpy.ndarray.mean`, but adapted to ensure
        that the full precision given by the two doubles ``jd1`` and ``jd2`` is
        used, and that corresponding attributes are copied.

        Note that the ``out`` argument is present only for compatibility with
        ``np.mean``; since `Time` instances are immutable, it is not possible
        to have an actual ``out`` to store the result in.

        Similarly, the ``dtype`` argument is also present for compatibility
        only; it has no meaning for `Time`.

        Parameters
        ----------
        axis : None or int or tuple of ints, optional
            Axis or axes along which the means are computed. The default is to
            compute the mean of the flattened array.
        dtype : None
            Only present for compatibility with :meth:`~numpy.ndarray.mean`,
            must be `None`.
        out : None
            Only present for compatibility with :meth:`~numpy.ndarray.mean`,
            must be `None`.
        keepdims : bool, optional
            If this is set to True, the axes which are reduced are left
            in the result as dimensions with size one. With this option,
            the result will broadcast correctly against the input array.
        where : array_like of bool, optional
            Elements to include in the mean. See `~numpy.ufunc.reduce` for
            details.

        Returns
        -------
        m : Time
            A new Time instance containing the mean values
        """
    def __getattr__(self, attr):
        """
        Get dynamic attributes to output format or do timescale conversion.
        """
    def __dir__(self): ...
    def _match_shape(self, val):
        """
        Ensure that `val` is matched to length of self.  If val has length 1
        then broadcast, otherwise cast to double and make sure shape matches.
        """
    def _time_comparison(self, other, op):
        """If other is of same class as self, compare difference in self.scale.
        Otherwise, return NotImplemented.
        """
    def __lt__(self, other): ...
    def __le__(self, other): ...
    def __eq__(self, other):
        """
        If other is an incompatible object for comparison, return `False`.
        Otherwise, return `True` if the time difference between self and
        other is zero.
        """
    def __ne__(self, other):
        """
        If other is an incompatible object for comparison, return `True`.
        Otherwise, return `False` if the time difference between self and
        other is zero.
        """
    def __gt__(self, other): ...
    def __ge__(self, other): ...

class Time(TimeBase):
    """
    Represent and manipulate times and dates for astronomy.

    A `Time` object is initialized with one or more times in the ``val``
    argument.  The input times in ``val`` must conform to the specified
    ``format`` and must correspond to the specified time ``scale``.  The
    optional ``val2`` time input should be supplied only for numeric input
    formats (e.g. JD) where very high precision (better than 64-bit precision)
    is required.

    The allowed values for ``format`` can be listed with::

      >>> list(Time.FORMATS)
      ['jd', 'mjd', 'decimalyear', 'unix', 'unix_tai', 'cxcsec', 'gps', 'plot_date',
       'stardate', 'datetime', 'ymdhms', 'iso', 'isot', 'yday', 'datetime64',
       'fits', 'byear', 'jyear', 'byear_str', 'jyear_str']

    See also: http://docs.astropy.org/en/stable/time/

    Parameters
    ----------
    val : sequence, ndarray, number, str, bytes, or `~astropy.time.Time` object
        Value(s) to initialize the time or times.  Bytes are decoded as ascii.
    val2 : sequence, ndarray, or number; optional
        Value(s) to initialize the time or times.  Only used for numerical
        input, to help preserve precision.
    format : str, optional
        Format of input value(s), specifying how to interpret them (e.g., ISO, JD, or
        Unix time). By default, the same format will be used for output representation.
    scale : str, optional
        Time scale of input value(s), must be one of the following:
        ('tai', 'tcb', 'tcg', 'tdb', 'tt', 'ut1', 'utc')
    precision : int, optional
        Digits of precision in string representation of time
    in_subfmt : str, optional
        Unix glob to select subformats for parsing input times
    out_subfmt : str, optional
        Unix glob to select subformat for outputting times
    location : `~astropy.coordinates.EarthLocation` or tuple, optional
        If given as an tuple, it should be able to initialize an
        an EarthLocation instance, i.e., either contain 3 items with units of
        length for geocentric coordinates, or contain a longitude, latitude,
        and an optional height for geodetic coordinates.
        Can be a single location, or one for each input time.
        If not given, assumed to be the center of the Earth for time scale
        transformations to and from the solar-system barycenter.
    copy : bool, optional
        Make a copy of the input values
    """
    SCALES = TIME_SCALES
    FORMATS = TIME_FORMATS
    def __new__(cls, val, val2: Incomplete | None = None, format: Incomplete | None = None, scale: Incomplete | None = None, precision: Incomplete | None = None, in_subfmt: Incomplete | None = None, out_subfmt: Incomplete | None = None, location: Incomplete | None = None, copy: bool = False): ...
    _location: Incomplete
    def __init__(self, val, val2: Incomplete | None = None, format: Incomplete | None = None, scale: Incomplete | None = None, precision: Incomplete | None = None, in_subfmt: Incomplete | None = None, out_subfmt: Incomplete | None = None, location: Incomplete | None = None, copy=...) -> None: ...
    def _make_value_equivalent(self, item, value):
        """Coerce setitem value into an equivalent Time object."""
    @classmethod
    def now(cls):
        '''
        Creates a new object corresponding to the instant in time this
        method is called.

        .. note::
            "Now" is determined using the `~datetime.datetime.now`
            function, so its accuracy and precision is determined by that
            function.  Generally that means it is set by the accuracy of
            your system clock. The timezone is set to UTC.

        Returns
        -------
        nowtime : :class:`~astropy.time.Time`
            A new `Time` object (or a subclass of `Time` if this is called from
            such a subclass) at the current time.
        '''
    info: Incomplete
    @classmethod
    def strptime(cls, time_string, format_string, **kwargs):
        """
        Parse a string to a Time according to a format specification.
        See `time.strptime` documentation for format specification.

        >>> Time.strptime('2012-Jun-30 23:59:60', '%Y-%b-%d %H:%M:%S')
        <Time object: scale='utc' format='isot' value=2012-06-30T23:59:60.000>

        Parameters
        ----------
        time_string : str, sequence, or ndarray
            Objects containing time data of type string
        format_string : str
            String specifying format of time_string.
        kwargs : dict
            Any keyword arguments for ``Time``.  If the ``format`` keyword
            argument is present, this will be used as the Time format.

        Returns
        -------
        time_obj : `~astropy.time.Time`
            A new `~astropy.time.Time` object corresponding to the input
            ``time_string``.

        """
    def strftime(self, format_spec):
        """
        Convert Time to a string or a numpy.array of strings according to a
        format specification.
        See `time.strftime` documentation for format specification.

        Parameters
        ----------
        format_spec : str
            Format definition of return string.

        Returns
        -------
        formatted : str or numpy.array
            String or numpy.array of strings formatted according to the given
            format string.

        """
    def light_travel_time(self, skycoord, kind: str = 'barycentric', location: Incomplete | None = None, ephemeris: Incomplete | None = None):
        """Light travel time correction to the barycentre or heliocentre.

        The frame transformations used to calculate the location of the solar
        system barycentre and the heliocentre rely on the erfa routine epv00,
        which is consistent with the JPL DE405 ephemeris to an accuracy of
        11.2 km, corresponding to a light travel time of 4 microseconds.

        The routine assumes the source(s) are at large distance, i.e., neglects
        finite-distance effects.

        Parameters
        ----------
        skycoord : `~astropy.coordinates.SkyCoord`
            The sky location to calculate the correction for.
        kind : str, optional
            ``'barycentric'`` (default) or ``'heliocentric'``
        location : `~astropy.coordinates.EarthLocation`, optional
            The location of the observatory to calculate the correction for.
            If no location is given, the ``location`` attribute of the Time
            object is used
        ephemeris : str, optional
            Solar system ephemeris to use (e.g., 'builtin', 'jpl'). By default,
            use the one set with ``astropy.coordinates.solar_system_ephemeris.set``.
            For more information, see `~astropy.coordinates.solar_system_ephemeris`.

        Returns
        -------
        time_offset : `~astropy.time.TimeDelta`
            The time offset between the barycentre or Heliocentre and Earth,
            in TDB seconds.  Should be added to the original time to get the
            time in the Solar system barycentre or the Heliocentre.
            Also, the time conversion to BJD will then include the relativistic correction as well.
        """
    def earth_rotation_angle(self, longitude: Incomplete | None = None):
        """Calculate local Earth rotation angle.

        Parameters
        ----------
        longitude : `~astropy.units.Quantity`, `~astropy.coordinates.EarthLocation`, str, or None; optional
            The longitude on the Earth at which to compute the Earth rotation
            angle (taken from a location as needed).  If `None` (default), taken
            from the ``location`` attribute of the Time instance. If the special
            string 'tio', the result will be relative to the Terrestrial
            Intermediate Origin (TIO) (i.e., the output of `~erfa.era00`).

        Returns
        -------
        `~astropy.coordinates.Longitude`
            Local Earth rotation angle with units of hourangle.

        See Also
        --------
        astropy.time.Time.sidereal_time

        References
        ----------
        IAU 2006 NFA Glossary
        (currently located at: https://syrte.obspm.fr/iauWGnfa/NFA_Glossary.html)

        Notes
        -----
        The difference between apparent sidereal time and Earth rotation angle
        is the equation of the origins, which is the angle between the Celestial
        Intermediate Origin (CIO) and the equinox. Applying apparent sidereal
        time to the hour angle yields the true apparent Right Ascension with
        respect to the equinox, while applying the Earth rotation angle yields
        the intermediate (CIRS) Right Ascension with respect to the CIO.

        The result includes the TIO locator (s'), which positions the Terrestrial
        Intermediate Origin on the equator of the Celestial Intermediate Pole (CIP)
        and is rigorously corrected for polar motion.
        (except when ``longitude='tio'``).

        """
    def sidereal_time(self, kind, longitude: Incomplete | None = None, model: Incomplete | None = None):
        """Calculate sidereal time.

        Parameters
        ----------
        kind : str
            ``'mean'`` or ``'apparent'``, i.e., accounting for precession
            only, or also for nutation.
        longitude : `~astropy.units.Quantity`, `~astropy.coordinates.EarthLocation`, str, or None; optional
            The longitude on the Earth at which to compute the Earth rotation
            angle (taken from a location as needed).  If `None` (default), taken
            from the ``location`` attribute of the Time instance. If the special
            string  'greenwich' or 'tio', the result will be relative to longitude
            0 for models before 2000, and relative to the Terrestrial Intermediate
            Origin (TIO) for later ones (i.e., the output of the relevant ERFA
            function that calculates greenwich sidereal time).
        model : str or None; optional
            Precession (and nutation) model to use.  The available ones are:
            - {0}: {1}
            - {2}: {3}
            If `None` (default), the last (most recent) one from the appropriate
            list above is used.

        Returns
        -------
        `~astropy.coordinates.Longitude`
            Local sidereal time, with units of hourangle.

        See Also
        --------
        astropy.time.Time.earth_rotation_angle

        References
        ----------
        IAU 2006 NFA Glossary
        (currently located at: https://syrte.obspm.fr/iauWGnfa/NFA_Glossary.html)

        Notes
        -----
        The difference between apparent sidereal time and Earth rotation angle
        is the equation of the origins, which is the angle between the Celestial
        Intermediate Origin (CIO) and the equinox. Applying apparent sidereal
        time to the hour angle yields the true apparent Right Ascension with
        respect to the equinox, while applying the Earth rotation angle yields
        the intermediate (CIRS) Right Ascension with respect to the CIO.

        For the IAU precession models from 2000 onwards, the result includes the
        TIO locator (s'), which positions the Terrestrial Intermediate Origin on
        the equator of the Celestial Intermediate Pole (CIP) and is rigorously
        corrected for polar motion (except when ``longitude='tio'`` or ``'greenwich'``).

        """
    def _sid_time_or_earth_rot_ang(self, longitude, function, scales, include_tio: bool = True):
        """Calculate a local sidereal time or Earth rotation angle.

        Parameters
        ----------
        longitude : `~astropy.units.Quantity`, `~astropy.coordinates.EarthLocation`, str, or None; optional
            The longitude on the Earth at which to compute the Earth rotation
            angle (taken from a location as needed).  If `None` (default), taken
            from the ``location`` attribute of the Time instance.
        function : callable
            The ERFA function to use.
        scales : tuple of str
            The time scales that the function requires on input.
        include_tio : bool, optional
            Whether to includes the TIO locator corrected for polar motion.
            Should be `False` for pre-2000 IAU models.  Default: `True`.

        Returns
        -------
        `~astropy.coordinates.Longitude`
            Local sidereal time or Earth rotation angle, with units of hourangle.

        """
    def _call_erfa(self, function, scales): ...
    def get_delta_ut1_utc(self, iers_table: Incomplete | None = None, return_status: bool = False):
        """Find UT1 - UTC differences by interpolating in IERS Table.

        Parameters
        ----------
        iers_table : `~astropy.utils.iers.IERS`, optional
            Table containing UT1-UTC differences from IERS Bulletins A
            and/or B.  Default: `~astropy.utils.iers.earth_orientation_table`
            (which in turn defaults to the combined version provided by
            `~astropy.utils.iers.IERS_Auto`).
        return_status : bool
            Whether to return status values.  If `False` (default), iers
            raises `IndexError` if any time is out of the range
            covered by the IERS table.

        Returns
        -------
        ut1_utc : float or float array
            UT1-UTC, interpolated in IERS Table
        status : int or int array
            Status values (if ``return_status=`True```)::
            ``astropy.utils.iers.FROM_IERS_B``
            ``astropy.utils.iers.FROM_IERS_A``
            ``astropy.utils.iers.FROM_IERS_A_PREDICTION``
            ``astropy.utils.iers.TIME_BEFORE_IERS_RANGE``
            ``astropy.utils.iers.TIME_BEYOND_IERS_RANGE``

        Notes
        -----
        In normal usage, UT1-UTC differences are calculated automatically
        on the first instance ut1 is needed.

        Examples
        --------
        To check in code whether any times are before the IERS table range::

            >>> from astropy.utils.iers import TIME_BEFORE_IERS_RANGE
            >>> t = Time(['1961-01-01', '2000-01-01'], scale='utc')
            >>> delta, status = t.get_delta_ut1_utc(return_status=True)  # doctest: +REMOTE_DATA
            >>> status == TIME_BEFORE_IERS_RANGE  # doctest: +REMOTE_DATA
            array([ True, False]...)
        """
    def _get_delta_ut1_utc(self, jd1: Incomplete | None = None, jd2: Incomplete | None = None):
        """
        Get ERFA DUT arg = UT1 - UTC.  This getter takes optional jd1 and
        jd2 args because it gets called that way when converting time scales.
        If delta_ut1_utc is not yet set, this will interpolate them from the
        the IERS table.
        """
    _delta_ut1_utc: Incomplete
    def _set_delta_ut1_utc(self, val) -> None: ...
    delta_ut1_utc: Incomplete
    _delta_tdb_tt: Incomplete
    def _get_delta_tdb_tt(self, jd1: Incomplete | None = None, jd2: Incomplete | None = None): ...
    def _set_delta_tdb_tt(self, val) -> None: ...
    delta_tdb_tt: Incomplete
    def __sub__(self, other): ...
    def __add__(self, other): ...
    def __radd__(self, other): ...
    def mean(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None, keepdims: bool = False, *, where: bool = True): ...
    def __array_function__(self, function, types, args, kwargs):
        """
        Wrap numpy functions.

        Parameters
        ----------
        function : callable
            Numpy function to wrap
        types : iterable of classes
            Classes that provide an ``__array_function__`` override. Can
            in principle be used to interact with other classes. Below,
            mostly passed on to `~numpy.ndarray`, which can only interact
            with subclasses.
        args : tuple
            Positional arguments provided in the function call.
        kwargs : dict
            Keyword arguments provided in the function call.
        """
    def to_datetime(self, timezone: Incomplete | None = None, leap_second_strict: str = 'raise'): ...

class TimeDeltaMissingUnitWarning(AstropyDeprecationWarning):
    """Warning for missing unit or format in TimeDelta."""

class TimeDelta(TimeBase):
    '''
    Represent the time difference between two times.

    A TimeDelta object is initialized with one or more times in the ``val``
    argument.  The input times in ``val`` must conform to the specified
    ``format``.  The optional ``val2`` time input should be supplied only for
    numeric input formats (e.g. JD) where very high precision (better than
    64-bit precision) is required.

    The allowed values for ``format`` can be listed with::

      >>> list(TimeDelta.FORMATS)
      [\'sec\', \'jd\', \'datetime\', \'quantity_str\']

    Note that for time differences, the scale can be among three groups:
    geocentric (\'tai\', \'tt\', \'tcg\'), barycentric (\'tcb\', \'tdb\'), and rotational
    (\'ut1\'). Within each of these, the scales for time differences are the
    same. Conversion between geocentric and barycentric is possible, as there
    is only a scale factor change, but one cannot convert to or from \'ut1\', as
    this requires knowledge of the actual times, not just their difference. For
    a similar reason, \'utc\' is not a valid scale for a time difference: a UTC
    day is not always 86400 seconds.

    For more information see:

    - https://docs.astropy.org/en/stable/time/
    - https://docs.astropy.org/en/stable/time/index.html#time-deltas

    Parameters
    ----------
    val : sequence, ndarray, number, `~astropy.units.Quantity` or `~astropy.time.TimeDelta` object
        Value(s) to initialize the time difference(s). Any quantities will
        be converted appropriately (with care taken to avoid rounding
        errors for regular time units).
    val2 : sequence, ndarray, number, or `~astropy.units.Quantity`; optional
        Additional values, as needed to preserve precision.
    format : str, optional
        Format of input value(s). For numerical inputs without units,
        "jd" is assumed and values are interpreted as days.
        A deprecation warning is raised in this case. To avoid the warning,
        either specify the format or add units to the input values.
    scale : str, optional
        Time scale of input value(s), must be one of the following values:
        (\'tdb\', \'tt\', \'ut1\', \'tcg\', \'tcb\', \'tai\'). If not given (or
        ``None``), the scale is arbitrary; when added or subtracted from a
        ``Time`` instance, it will be used without conversion.
    precision : int, optional
        Digits of precision in string representation of time
    in_subfmt : str, optional
        Unix glob to select subformats for parsing input times
    out_subfmt : str, optional
        Unix glob to select subformat for outputting times
    copy : bool, optional
        Make a copy of the input values
    '''
    SCALES = TIME_DELTA_SCALES
    FORMATS = TIME_DELTA_FORMATS
    info: Incomplete
    def __new__(cls, val, val2: Incomplete | None = None, format: Incomplete | None = None, scale: Incomplete | None = None, precision: Incomplete | None = None, in_subfmt: Incomplete | None = None, out_subfmt: Incomplete | None = None, location: Incomplete | None = None, copy: bool = False): ...
    def __init__(self, val, val2: Incomplete | None = None, format: Incomplete | None = None, scale: Incomplete | None = None, *, precision: Incomplete | None = None, in_subfmt: Incomplete | None = None, out_subfmt: Incomplete | None = None, copy=...) -> None: ...
    def _check_numeric_no_unit(self, val, format) -> None: ...
    def replicate(self, *args, **kwargs): ...
    def to_datetime(self):
        """
        Convert to ``datetime.timedelta`` object.
        """
    _time: Incomplete
    def _set_scale(self, scale) -> None:
        """
        This is the key routine that actually does time scale conversions.
        This is not public and not connected to the read-only scale property.
        """
    def _add_sub(self, other, op):
        """Perform common elements of addition / subtraction for two delta times."""
    def __add__(self, other): ...
    def __sub__(self, other): ...
    def __radd__(self, other): ...
    def __rsub__(self, other): ...
    def __neg__(self):
        """Negation of a `TimeDelta` object."""
    def __abs__(self):
        """Absolute value of a `TimeDelta` object."""
    def __mul__(self, other):
        """Multiplication of `TimeDelta` objects by numbers/arrays."""
    def __rmul__(self, other):
        """Multiplication of numbers/arrays with `TimeDelta` objects."""
    def __truediv__(self, other):
        """Division of `TimeDelta` objects by numbers/arrays."""
    def __rtruediv__(self, other):
        """Division by `TimeDelta` objects of numbers/arrays."""
    def to(self, unit, equivalencies=[]):
        """
        Convert to a quantity in the specified unit.

        Parameters
        ----------
        unit : unit-like
            The unit to convert to.
        equivalencies : list of tuple
            A list of equivalence pairs to try if the units are not directly
            convertible (see :ref:`astropy:unit_equivalencies`). If `None`, no
            equivalencies will be applied at all, not even any set globallyq
            or within a context.

        Returns
        -------
        quantity : `~astropy.units.Quantity`
            The quantity in the units specified.

        See Also
        --------
        to_value : get the numerical value in a given unit.
        """
    def to_value(self, *args, **kwargs):
        """Get time delta values expressed in specified output format or unit.

        This method is flexible and handles both conversion to a specified
        ``TimeDelta`` format / sub-format AND conversion to a specified unit.
        If positional argument(s) are provided then the first one is checked
        to see if it is a valid ``TimeDelta`` format, and next it is checked
        to see if it is a valid unit or unit string.

        To convert to a ``TimeDelta`` format and optional sub-format the options
        are::

          tm = TimeDelta(1.0 * u.s)
          tm.to_value('jd')  # equivalent of tm.jd
          tm.to_value('jd', 'decimal')  # convert to 'jd' as a Decimal object
          tm.to_value('jd', subfmt='decimal')
          tm.to_value(format='jd', subfmt='decimal')

        To convert to a unit with optional equivalencies, the options are::

          tm.to_value('hr')  # convert to u.hr (hours)
          tm.to_value('hr', equivalencies=[])
          tm.to_value(unit='hr', equivalencies=[])

        The built-in `~astropy.time.TimeDelta` options for ``format`` are shown below::

          >>> list(TimeDelta.FORMATS)
          ['sec', 'jd', 'datetime', 'quantity_str']

        For the two numerical formats 'jd' and 'sec', the available ``subfmt``
        options are: {'float', 'long', 'decimal', 'str', 'bytes'}. Here, 'long'
        uses ``numpy.longdouble`` for somewhat enhanced precision (with the
        enhancement depending on platform), and 'decimal' instances of
        :class:`decimal.Decimal` for full precision.  For the 'str' and 'bytes'
        sub-formats, the number of digits is also chosen such that time values
        are represented accurately.  Default: as set by ``out_subfmt`` (which by
        default picks the first available for a given format, i.e., 'float').

        Parameters
        ----------
        format : str, optional
            The format in which one wants the `~astropy.time.TimeDelta` values.
            Default: the current format.
        subfmt : str, optional
            Possible sub-format in which the values should be given. Default: as
            set by ``out_subfmt`` (which by default picks the first available
            for a given format, i.e., 'float' or 'date_hms').
        unit : `~astropy.units.UnitBase` instance or str, optional
            The unit in which the value should be given.
        equivalencies : list of tuple
            A list of equivalence pairs to try if the units are not directly
            convertible (see :ref:`astropy:unit_equivalencies`). If `None`, no
            equivalencies will be applied at all, not even any set globally or
            within a context.

        Returns
        -------
        value : ndarray or scalar
            The value in the format or units specified.

        See Also
        --------
        to : Convert to a `~astropy.units.Quantity` instance in a given unit.
        value : The time value in the current format.

        """
    def _make_value_equivalent(self, item, value):
        """Coerce setitem value into an equivalent TimeDelta object."""
    def isclose(self, other, atol: Incomplete | None = None, rtol: float = 0.0):
        """Returns a boolean or boolean array where two TimeDelta objects are
        element-wise equal within a time tolerance.

        This effectively evaluates the expression below::

          abs(self - other) <= atol + rtol * abs(other)

        Parameters
        ----------
        other : `~astropy.units.Quantity` or `~astropy.time.TimeDelta`
            Quantity or TimeDelta object for comparison.
        atol : `~astropy.units.Quantity` or `~astropy.time.TimeDelta`
            Absolute tolerance for equality with units of time (e.g. ``u.s`` or
            ``u.day``). Default is one bit in the 128-bit JD time representation,
            equivalent to about 20 picosecs.
        rtol : float
            Relative tolerance for equality
        """

class ScaleValueError(Exception): ...

class OperandTypeError(TypeError):
    def __init__(self, left, right, op: Incomplete | None = None) -> None: ...

def update_leap_seconds(files: Incomplete | None = None):
    '''If the current ERFA leap second table is out of date, try to update it.

    Uses `astropy.utils.iers.LeapSeconds.auto_open` to try to find an
    up-to-date table.  See that routine for the definition of "out of date".

    In order to make it safe to call this any time, all exceptions are turned
    into warnings,

    Parameters
    ----------
    files : list of path-like, optional
        List of files/URLs to attempt to open.  By default, uses defined by
        `astropy.utils.iers.LeapSeconds.auto_open`, which includes the table
        used by ERFA itself, so if that is up to date, nothing will happen.

    Returns
    -------
    n_update : int
        Number of items updated.

    '''

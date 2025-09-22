import numpy as np
from .util import NotifierMixin
from _typeshed import Incomplete

__all__ = ['Column', 'ColDefs', 'Delayed']

class Delayed:
    """Delayed file-reading data."""
    hdu: Incomplete
    field: Incomplete
    def __init__(self, hdu: Incomplete | None = None, field: Incomplete | None = None) -> None: ...
    def __getitem__(self, key): ...

class _BaseColumnFormat(str):
    """
    Base class for binary table column formats (just called _ColumnFormat)
    and ASCII table column formats (_AsciiColumnFormat).
    """
    def __eq__(self, other): ...
    def __hash__(self): ...
    def dtype(self):
        """
        The Numpy dtype object created from the format's associated recformat.
        """
    @classmethod
    def from_column_format(cls, format):
        """Creates a column format object from another column format object
        regardless of their type.

        That is, this can convert a _ColumnFormat to an _AsciiColumnFormat
        or vice versa at least in cases where a direct translation is possible.
        """

class _ColumnFormat(_BaseColumnFormat):
    """
    Represents a FITS binary table column format.

    This is an enhancement over using a normal string for the format, since the
    repeat count, format code, and option are available as separate attributes,
    and smart comparison is used.  For example 1J == J.
    """
    format: Incomplete
    p_format: Incomplete
    def __new__(cls, format): ...
    @classmethod
    def from_recformat(cls, recformat):
        """Creates a column format from a Numpy record dtype format."""
    def recformat(self):
        """Returns the equivalent Numpy record format string."""
    def canonical(self):
        """
        Returns a 'canonical' string representation of this format.

        This is in the proper form of rTa where T is the single character data
        type code, a is the optional part, and r is the repeat.  If repeat == 1
        (the default) it is left out of this representation.
        """

class _AsciiColumnFormat(_BaseColumnFormat):
    """Similar to _ColumnFormat but specifically for columns in ASCII tables.

    The formats of ASCII table columns and binary table columns are inherently
    incompatible in FITS.  They don't support the same ranges and types of
    values, and even reuse format codes in subtly different ways.  For example
    the format code 'Iw' in ASCII columns refers to any integer whose string
    representation is at most w characters wide, so 'I' can represent
    effectively any integer that will fit in a FITS columns.  Whereas for
    binary tables 'I' very explicitly refers to a 16-bit signed integer.

    Conversions between the two column formats can be performed using the
    ``to/from_binary`` methods on this class, or the ``to/from_ascii``
    methods on the `_ColumnFormat` class.  But again, not all conversions are
    possible and may result in a `ValueError`.
    """
    _pseudo_logical: bool
    def __new__(cls, format, strict: bool = False): ...
    @classmethod
    def from_column_format(cls, format): ...
    @classmethod
    def from_recformat(cls, recformat):
        """Creates a column format from a Numpy record dtype format."""
    def recformat(self):
        """Returns the equivalent Numpy record format string."""
    def canonical(self):
        """
        Returns a 'canonical' string representation of this format.

        This is in the proper form of Tw.d where T is the single character data
        type code, w is the width in characters for this field, and d is the
        number of digits after the decimal place (for format codes 'E', 'F',
        and 'D' only).
        """

class _FormatX(str):
    """For X format in binary tables."""
    def __new__(cls, repeat: int = 1): ...
    def __getnewargs__(self): ...
    @property
    def tform(self): ...

class _FormatP(str):
    """For P format in variable length table."""
    _format_re_template: str
    _format_code: str
    _format_re: Incomplete
    _descriptor_format: str
    def __new__(cls, dtype, repeat: Incomplete | None = None, max: Incomplete | None = None): ...
    def __getnewargs__(self): ...
    @classmethod
    def from_tform(cls, format): ...
    @property
    def tform(self): ...

class _FormatQ(_FormatP):
    """Carries type description of the Q format for variable length arrays.

    The Q format is like the P format but uses 64-bit integers in the array
    descriptors, allowing for heaps stored beyond 2GB into a file.
    """
    _format_code: str
    _format_re: Incomplete
    _descriptor_format: str

class ColumnAttribute:
    """
    Descriptor for attributes of `Column` that are associated with keywords
    in the FITS header and describe properties of the column as specified in
    the FITS standard.

    Each `ColumnAttribute` may have a ``validator`` method defined on it.
    This validates values set on this attribute to ensure that they meet the
    FITS standard.  Invalid values will raise a warning and will not be used in
    formatting the column.  The validator should take two arguments--the
    `Column` it is being assigned to, and the new value for the attribute, and
    it must raise an `AssertionError` if the value is invalid.

    The `ColumnAttribute` itself is a decorator that can be used to define the
    ``validator`` for each column attribute.  For example::

        @ColumnAttribute('TTYPE')
        def name(col, name):
            if not isinstance(name, str):
                raise AssertionError

    The actual object returned by this decorator is the `ColumnAttribute`
    instance though, not the ``name`` function.  As such ``name`` is not a
    method of the class it is defined in.

    The setter for `ColumnAttribute` also updates the header of any table
    HDU this column is attached to in order to reflect the change.  The
    ``validator`` should ensure that the value is valid for inclusion in a FITS
    header.
    """
    _keyword: Incomplete
    _validator: Incomplete
    _attr: Incomplete
    def __init__(self, keyword) -> None: ...
    def __get__(self, obj, objtype: Incomplete | None = None): ...
    def __set__(self, obj, value) -> None: ...
    def __call__(self, func):
        """
        Set the validator for this column attribute.

        Returns ``self`` so that this can be used as a decorator, as described
        in the docs for this class.
        """
    def __repr__(self) -> str: ...

class Column(NotifierMixin):
    """
    Class which contains the definition of one column, e.g.  ``ttype``,
    ``tform``, etc. and the array containing values for the column.
    """
    _dims: Incomplete
    dim: Incomplete
    _pseudo_unsigned_ints: bool
    _physical_values: bool
    _parent_fits_rec: Incomplete
    def __init__(self, name: Incomplete | None = None, format: Incomplete | None = None, unit: Incomplete | None = None, null: Incomplete | None = None, bscale: Incomplete | None = None, bzero: Incomplete | None = None, disp: Incomplete | None = None, start: Incomplete | None = None, dim: Incomplete | None = None, array: Incomplete | None = None, ascii: Incomplete | None = None, coord_type: Incomplete | None = None, coord_unit: Incomplete | None = None, coord_ref_point: Incomplete | None = None, coord_ref_value: Incomplete | None = None, coord_inc: Incomplete | None = None, time_ref_pos: Incomplete | None = None) -> None:
        """
        Construct a `Column` by specifying attributes.  All attributes
        except ``format`` can be optional; see :ref:`astropy:column_creation`
        and :ref:`astropy:creating_ascii_table` for more information regarding
        ``TFORM`` keyword.

        Parameters
        ----------
        name : str, optional
            column name, corresponding to ``TTYPE`` keyword

        format : str
            column format, corresponding to ``TFORM`` keyword

        unit : str, optional
            column unit, corresponding to ``TUNIT`` keyword

        null : str, optional
            null value, corresponding to ``TNULL`` keyword

        bscale : int-like, optional
            bscale value, corresponding to ``TSCAL`` keyword

        bzero : int-like, optional
            bzero value, corresponding to ``TZERO`` keyword

        disp : str, optional
            display format, corresponding to ``TDISP`` keyword

        start : int, optional
            column starting position (ASCII table only), corresponding
            to ``TBCOL`` keyword

        dim : str, optional
            column dimension corresponding to ``TDIM`` keyword

        array : iterable, optional
            a `list`, `numpy.ndarray` (or other iterable that can be used to
            initialize an ndarray) providing initial data for this column.
            The array will be automatically converted, if possible, to the data
            format of the column.  In the case were non-trivial ``bscale``
            and/or ``bzero`` arguments are given, the values in the array must
            be the *physical* values--that is, the values of column as if the
            scaling has already been applied (the array stored on the column
            object will then be converted back to its storage values).

        ascii : bool, optional
            set `True` if this describes a column for an ASCII table; this
            may be required to disambiguate the column format

        coord_type : str, optional
            coordinate/axis type corresponding to ``TCTYP`` keyword

        coord_unit : str, optional
            coordinate/axis unit corresponding to ``TCUNI`` keyword

        coord_ref_point : int-like, optional
            pixel coordinate of the reference point corresponding to ``TCRPX``
            keyword

        coord_ref_value : int-like, optional
            coordinate value at reference point corresponding to ``TCRVL``
            keyword

        coord_inc : int-like, optional
            coordinate increment at reference point corresponding to ``TCDLT``
            keyword

        time_ref_pos : str, optional
            reference position for a time coordinate column corresponding to
            ``TRPOS`` keyword
        """
    def __repr__(self) -> str: ...
    def __eq__(self, other):
        """
        Two columns are equal if their name and format are the same.  Other
        attributes aren't taken into account at this time.
        """
    def __hash__(self):
        """
        Like __eq__, the hash of a column should be based on the unique column
        name and format, and be case-insensitive with respect to the column
        name.
        """
    @property
    def array(self):
        """
        The Numpy `~numpy.ndarray` associated with this `Column`.

        If the column was instantiated with an array passed to the ``array``
        argument, this will return that array.  However, if the column is
        later added to a table, such as via `BinTableHDU.from_columns` as
        is typically the case, this attribute will be updated to reference
        the associated field in the table, which may no longer be the same
        array.
        """
    @array.setter
    def array(self, array) -> None: ...
    @array.deleter
    def array(self) -> None: ...
    def name(col, name) -> None: ...
    def coord_type(col, coord_type) -> None: ...
    def coord_unit(col, coord_unit) -> None: ...
    def coord_ref_point(col, coord_ref_point) -> None: ...
    def coord_ref_value(col, coord_ref_value) -> None: ...
    def coord_inc(col, coord_inc) -> None: ...
    def time_ref_pos(col, time_ref_pos) -> None: ...
    format: Incomplete
    unit: Incomplete
    null: Incomplete
    bscale: Incomplete
    bzero: Incomplete
    disp: Incomplete
    start: Incomplete
    def ascii(self):
        """Whether this `Column` represents a column in an ASCII table."""
    def dtype(self): ...
    def copy(self):
        """
        Return a copy of this `Column`.
        """
    @staticmethod
    def _convert_format(format, cls):
        """The format argument to this class's initializer may come in many
        forms.  This uses the given column format class ``cls`` to convert
        to a format of that type.

        TODO: There should be an abc base class for column format classes
        """
    @classmethod
    def _verify_keywords(cls, name: Incomplete | None = None, format: Incomplete | None = None, unit: Incomplete | None = None, null: Incomplete | None = None, bscale: Incomplete | None = None, bzero: Incomplete | None = None, disp: Incomplete | None = None, start: Incomplete | None = None, dim: Incomplete | None = None, ascii: Incomplete | None = None, coord_type: Incomplete | None = None, coord_unit: Incomplete | None = None, coord_ref_point: Incomplete | None = None, coord_ref_value: Incomplete | None = None, coord_inc: Incomplete | None = None, time_ref_pos: Incomplete | None = None):
        """
        Given the keyword arguments used to initialize a Column, specifically
        those that typically read from a FITS header (so excluding array),
        verify that each keyword has a valid value.

        Returns a 2-tuple of dicts.  The first maps valid keywords to their
        values.  The second maps invalid keywords to a 2-tuple of their value,
        and a message explaining why they were found invalid.
        """
    @classmethod
    def _determine_formats(cls, format, start, dim, ascii):
        """
        Given a format string and whether or not the Column is for an
        ASCII table (ascii=None means unspecified, but lean toward binary table
        where ambiguous) create an appropriate _BaseColumnFormat instance for
        the column's format, and determine the appropriate recarray format.

        The values of the start and dim keyword arguments are also useful, as
        the former is only valid for ASCII tables and the latter only for
        BINARY tables.
        """
    @classmethod
    def _guess_format(cls, format, start, dim): ...
    def _convert_to_valid_data_type(self, array): ...

class ColDefs(NotifierMixin):
    """
    Column definitions class.

    It has attributes corresponding to the `Column` attributes
    (e.g. `ColDefs` has the attribute ``names`` while `Column`
    has ``name``). Each attribute in `ColDefs` is a list of
    corresponding attribute values from all `Column` objects.
    """
    _padding_byte: str
    _col_format_cls = _ColumnFormat
    def __new__(cls, input, ascii: bool = False): ...
    def __getnewargs__(self): ...
    def __init__(self, input, ascii: bool = False) -> None:
        """
        Parameters
        ----------
        input : sequence of `Column` or `ColDefs` or ndarray or `~numpy.recarray`
            An existing table HDU, an existing `ColDefs`, or any multi-field
            Numpy array or `numpy.recarray`.

        ascii : bool
            Use True to ensure that ASCII table columns are used.

        """
    columns: Incomplete
    def _init_from_coldefs(self, coldefs) -> None:
        """Initialize from an existing ColDefs object (just copy the
        columns and convert their formats if necessary).
        """
    def _init_from_sequence(self, columns) -> None: ...
    def _init_from_array(self, array) -> None: ...
    def _init_from_table(self, table) -> None: ...
    def __copy__(self): ...
    def __deepcopy__(self, memo): ...
    def _copy_column(self, column):
        """Utility function used currently only by _init_from_coldefs
        to help convert columns from binary format to ASCII format or vice
        versa if necessary (otherwise performs a straight copy).
        """
    def __getattr__(self, name):
        """
        Automatically returns the values for the given keyword attribute for
        all `Column`s in this list.

        Implements for example self.units, self.formats, etc.
        """
    def dtype(self): ...
    def names(self): ...
    def formats(self): ...
    def _arrays(self): ...
    def _recformats(self): ...
    def _dims(self):
        """Returns the values of the TDIMn keywords parsed into tuples."""
    def __getitem__(self, key): ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str: ...
    def __add__(self, other, option: str = 'left'): ...
    def __radd__(self, other): ...
    def __sub__(self, other): ...
    def _update_column_attribute_changed(self, column, attr, old_value, new_value) -> None:
        """
        Handle column attribute changed notifications from columns that are
        members of this `ColDefs`.

        `ColDefs` itself does not currently do anything with this, and just
        bubbles the notification up to any listening table HDUs that may need
        to update their headers, etc.  However, this also informs the table of
        the numerical index of the column that changed.
        """
    def add_col(self, column):
        """
        Append one `Column` to the column definition.
        """
    def del_col(self, col_name):
        """
        Delete (the definition of) one `Column`.

        col_name : str or int
            The column's name or index
        """
    def change_attrib(self, col_name, attrib, new_value) -> None:
        """
        Change an attribute (in the ``KEYWORD_ATTRIBUTES`` list) of a `Column`.

        Parameters
        ----------
        col_name : str or int
            The column name or index to change

        attrib : str
            The attribute name

        new_value : object
            The new value for the attribute
        """
    def change_name(self, col_name, new_name) -> None:
        """
        Change a `Column`'s name.

        Parameters
        ----------
        col_name : str
            The current name of the column

        new_name : str
            The new name of the column
        """
    def change_unit(self, col_name, new_unit) -> None:
        """
        Change a `Column`'s unit.

        Parameters
        ----------
        col_name : str or int
            The column name or index

        new_unit : str
            The new unit for the column
        """
    def info(self, attrib: str = 'all', output: Incomplete | None = None):
        '''
        Get attribute(s) information of the column definition.

        Parameters
        ----------
        attrib : str
            Can be one or more of the attributes listed in
            ``astropy.io.fits.column.KEYWORD_ATTRIBUTES``.  The default is
            ``"all"`` which will print out all attributes.  It forgives plurals
            and blanks.  If there are two or more attribute names, they must be
            separated by comma(s).

        output : file-like, optional
            File-like object to output to.  Outputs to stdout by default.
            If `False`, returns the attributes as a `dict` instead.

        Notes
        -----
        This function doesn\'t return anything by default; it just prints to
        stdout.
        '''

class _AsciiColDefs(ColDefs):
    """ColDefs implementation for ASCII tables."""
    _padding_byte: str
    _col_format_cls = _AsciiColumnFormat
    _spans: Incomplete
    _width: Incomplete
    def __init__(self, input, ascii: bool = True) -> None: ...
    def dtype(self): ...
    @property
    def spans(self):
        """A list of the widths of each field in the table."""
    def _recformats(self): ...
    def add_col(self, column) -> None: ...
    def del_col(self, col_name) -> None: ...
    def _update_field_metrics(self) -> None:
        """
        Updates the list of the start columns, the list of the widths of each
        field, and the total width of each record in the table.
        """

class _VLF(np.ndarray):
    """Variable length field object."""
    max: int
    element_dtype: Incomplete
    def __new__(cls, input, dtype: str = 'S'):
        """
        Parameters
        ----------
        input
            a sequence of variable-sized elements.
        """
    def __array_finalize__(self, obj) -> None: ...
    def __setitem__(self, key, value) -> None:
        """
        To make sure the new item has consistent data type to avoid
        misalignment.
        """
    def tolist(self): ...

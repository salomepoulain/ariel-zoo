import numpy as np
from .column import ASCII2NUMPY as ASCII2NUMPY, ASCII2STR as ASCII2STR, ASCIITNULL as ASCIITNULL, ColDefs as ColDefs, Delayed as Delayed, FITS2NUMPY as FITS2NUMPY, _AsciiColDefs as _AsciiColDefs, _FormatP as _FormatP, _FormatX as _FormatX, _VLF as _VLF, _get_index as _get_index, _makep as _makep, _unwrapx as _unwrapx, _wrapx as _wrapx
from .util import _rstrip_inplace as _rstrip_inplace, decode_ascii as decode_ascii, encode_ascii as encode_ascii
from _typeshed import Incomplete
from astropy.utils import lazyproperty as lazyproperty

class FITS_record:
    """
    FITS record class.

    `FITS_record` is used to access records of the `FITS_rec` object.
    This will allow us to deal with scaled columns.  It also handles
    conversion/scaling of columns in ASCII tables.  The `FITS_record`
    class expects a `FITS_rec` object as input.
    """
    array: Incomplete
    row: Incomplete
    base: Incomplete
    def __init__(self, input, row: int = 0, start: Incomplete | None = None, end: Incomplete | None = None, step: Incomplete | None = None, base: Incomplete | None = None, **kwargs) -> None:
        """
        Parameters
        ----------
        input : array
            The array to wrap.
        row : int, optional
            The starting logical row of the array.
        start : int, optional
            The starting column in the row associated with this object.
            Used for subsetting the columns of the `FITS_rec` object.
        end : int, optional
            The ending column in the row associated with this object.
            Used for subsetting the columns of the `FITS_rec` object.
        """
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...
    def __len__(self) -> int: ...
    def __repr__(self) -> str:
        """
        Display a single row.
        """
    def field(self, field):
        """
        Get the field data of the record.
        """
    def setfield(self, field, value) -> None:
        """
        Set the field data of the record.
        """
    def _bases(self): ...
    def _get_index(self, indx): ...

class FITS_rec(np.recarray):
    """
    FITS record array class.

    `FITS_rec` is the data part of a table HDU's data part.  This is a layer
    over the `~numpy.recarray`, so we can deal with scaled columns.

    It inherits all of the standard methods from `numpy.ndarray`.
    """
    _record_type = FITS_record
    _character_as_bytes: bool
    _load_variable_length_data: bool
    _nfields: Incomplete
    def __new__(subtype, input):
        """
        Construct a FITS record array from a recarray.
        """
    _col_weakrefs: Incomplete
    def __setstate__(self, state) -> None: ...
    def __reduce__(self):
        """
        Return a 3-tuple for pickling a FITS_rec. Use the super-class
        functionality but then add in a tuple of FITS_rec-specific
        values that get used in __setstate__.
        """
    _converted: Incomplete
    _heapoffset: Incomplete
    _heapsize: Incomplete
    _tbsize: Incomplete
    _gap: Incomplete
    _uint: Incomplete
    def __array_finalize__(self, obj) -> None: ...
    def _init(self) -> None:
        """Initializes internal attributes specific to FITS-isms."""
    @classmethod
    def from_columns(cls, columns, nrows: int = 0, fill: bool = False, character_as_bytes: bool = False):
        """
        Given a `ColDefs` object of unknown origin, initialize a new `FITS_rec`
        object.

        .. note::

            This was originally part of the ``new_table`` function in the table
            module but was moved into a class method since most of its
            functionality always had more to do with initializing a `FITS_rec`
            object than anything else, and much of it also overlapped with
            ``FITS_rec._scale_back``.

        Parameters
        ----------
        columns : sequence of `Column` or a `ColDefs`
            The columns from which to create the table data.  If these
            columns have data arrays attached that data may be used in
            initializing the new table.  Otherwise the input columns
            will be used as a template for a new table with the requested
            number of rows.

        nrows : int
            Number of rows in the new table.  If the input columns have data
            associated with them, the size of the largest input column is used.
            Otherwise the default is 0.

        fill : bool
            If `True`, will fill all cells with zeros or blanks.  If
            `False`, copy the data from input, undefined cells will still
            be filled with zeros/blanks.
        """
    def __repr__(self) -> str: ...
    def __getattribute__(self, attr): ...
    def __getitem__(self, key): ...
    def __setitem__(self, key, value) -> None: ...
    def _ipython_key_completions_(self): ...
    def copy(self, order: str = 'C'):
        """
        The Numpy documentation lies; `numpy.ndarray.copy` is not equivalent to
        `numpy.copy`.  Differences include that it re-views the copied array as
        self's ndarray subclass, as though it were taking a slice; this means
        ``__array_finalize__`` is called and the copy shares all the array
        attributes (including ``._converted``!).  So we need to make a deep
        copy of all those attributes so that the two arrays truly do not share
        any data.
        """
    @property
    def columns(self):
        """A user-visible accessor for the coldefs."""
    @property
    def _coldefs(self): ...
    @_coldefs.setter
    def _coldefs(self, cols) -> None: ...
    @_coldefs.deleter
    def _coldefs(self) -> None: ...
    def __del__(self) -> None: ...
    @property
    def names(self):
        """List of column names."""
    @property
    def formats(self):
        """List of column FITS formats."""
    @property
    def _raw_itemsize(self):
        """
        Returns the size of row items that would be written to the raw FITS
        file, taking into account the possibility of unicode columns being
        compactified.

        Currently for internal use only.
        """
    def field(self, key):
        """
        A view of a `Column`'s data as an array.
        """
    def _cache_field(self, name, field) -> None:
        """
        Do not store fields in _converted if one of its bases is self,
        or if it has a common base with self.

        This results in a reference cycle that cannot be broken since
        ndarrays do not participate in cyclic garbage collection.
        """
    def _update_column_attribute_changed(self, column, idx, attr, old_value, new_value) -> None:
        """
        Update how the data is formatted depending on changes to column
        attributes initiated by the user through the `Column` interface.

        Dispatches column attribute change notifications to individual methods
        for each attribute ``_update_column_<attr>``
        """
    def _update_column_name(self, column, idx, old_name, name) -> None:
        """Update the dtype field names when a column name is changed."""
    def _convert_x(self, field, recformat):
        """Convert a raw table column to a bit array as specified by the
        FITS X format.
        """
    def _convert_p(self, column, field, recformat):
        """Convert a raw table column of FITS P or Q format descriptors
        to a VLA column with the array data returned from the heap.
        """
    def _convert_ascii(self, column, field):
        """
        Special handling for ASCII table columns to convert columns containing
        numeric types to actual numeric arrays from the string representation.
        """
    def _convert_other(self, column, field, recformat):
        """Perform conversions on any other fixed-width column data types.

        This may not perform any conversion at all if it's not necessary, in
        which case the original column array is returned.
        """
    def _get_heap_data(self):
        """
        Returns a pointer into the table's raw data to its heap (if present).

        This is returned as a numpy byte array.
        """
    def _get_raw_data(self):
        '''
        Returns the base array of self that "raw data array" that is the
        array in the format that it was first read from a file before it was
        sliced or viewed as a different type in any way.

        This is determined by walking through the bases until finding one that
        has at least the same number of bytes as self, plus the heapsize.  This
        may be the immediate .base but is not always.  This is used primarily
        for variable-length array support which needs to be able to find the
        heap (the raw data *may* be larger than nbytes + heapsize if it
        contains a gap or padding).

        May return ``None`` if no array resembling the "raw data" according to
        the stated criteria can be found.
        '''
    def _get_scale_factors(self, column):
        """Get all the scaling flags and factors for one column."""
    def _scale_back(self, update_heap_pointers: bool = True) -> None:
        """
        Update the parent array, using the (latest) scaled array.

        If ``update_heap_pointers`` is `False`, this will leave all the heap
        pointers in P/Q columns as they are verbatim--it only makes sense to do
        this if there is already data on the heap and it can be guaranteed that
        that data has not been modified, and there is not new data to add to
        the heap.  Currently this is only used as an optimization for
        CompImageHDU that does its own handling of the heap.
        """
    def _scale_back_strings(self, col_idx, input_field, output_field) -> None: ...
    def _scale_back_ascii(self, col_idx, input_field, output_field) -> None:
        """
        Convert internal array values back to ASCII table representation.

        The ``input_field`` is the internal representation of the values, and
        the ``output_field`` is the character array representing the ASCII
        output that will be written.
        """
    def tolist(self): ...

def _get_recarray_field(array, key):
    """
    Compatibility function for using the recarray base class's field method.
    This incorporates the legacy functionality of returning string arrays as
    Numeric-style chararray objects.
    """

class _UnicodeArrayEncodeError(UnicodeEncodeError):
    index: Incomplete
    def __init__(self, encoding, object_, start, end, reason, index) -> None: ...

def _ascii_encode(inarray, out: Incomplete | None = None):
    """
    Takes a unicode array and fills the output string array with the ASCII
    encodings (if possible) of the elements of the input array.  The two arrays
    must be the same size (though not necessarily the same shape).

    This is like an inplace version of `np.char.encode` though simpler since
    it's only limited to ASCII, and hence the size of each character is
    guaranteed to be 1 byte.

    If any strings are non-ASCII an UnicodeArrayEncodeError is raised--this is
    just a `UnicodeEncodeError` with an additional attribute for the index of
    the item that couldn't be encoded.
    """
def _has_unicode_fields(array):
    """
    Returns True if any fields in a structured array have Unicode dtype.
    """

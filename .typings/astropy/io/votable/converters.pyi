from _typeshed import Incomplete

__all__ = ['get_converter', 'Converter', 'table_column_to_votable_datatype']

class Converter:
    """
    The base class for all converters.  Each subclass handles
    converting a specific VOTABLE data type to/from the TABLEDATA_ and
    BINARY_ on-disk representations.

    Parameters
    ----------
    field : `~astropy.io.votable.tree.Field`
        object describing the datatype

    config : dict
        The parser configuration dictionary

    pos : tuple
        The position in the XML file where the FIELD object was
        found.  Used for error messages.

    """
    def __init__(self, field, config: Incomplete | None = None, pos: Incomplete | None = None) -> None: ...
    @staticmethod
    def _parse_length(read): ...
    @staticmethod
    def _write_length(length): ...
    def supports_empty_values(self, config):
        """
        Returns True when the field can be completely empty.
        """
    def parse(self, value, config: Incomplete | None = None, pos: Incomplete | None = None) -> None:
        """
        Convert the string *value* from the TABLEDATA_ format into an
        object with the correct native in-memory datatype and mask flag.

        Parameters
        ----------
        value : str
            value in TABLEDATA format

        Returns
        -------
        native : tuple
            A two-element tuple of: value, mask.
            The value as a Numpy array or scalar, and *mask* is True
            if the value is missing.
        """
    def parse_scalar(self, value, config: Incomplete | None = None, pos: Incomplete | None = None):
        """
        Parse a single scalar of the underlying type of the converter.
        For non-array converters, this is equivalent to parse.  For
        array converters, this is used to parse a single
        element of the array.

        Parameters
        ----------
        value : str
            value in TABLEDATA format

        Returns
        -------
        native : (2,) tuple
            (value, mask)
            The value as a Numpy array or scalar, and *mask* is True
            if the value is missing.
        """
    def output(self, value, mask) -> None:
        """
        Convert the object *value* (in the native in-memory datatype)
        to a unicode string suitable for serializing in the TABLEDATA_
        format.

        Parameters
        ----------
        value
            The value, the native type corresponding to this converter

        mask : bool
            If `True`, will return the string representation of a
            masked value.

        Returns
        -------
        tabledata_repr : unicode
        """
    def binparse(self, read) -> None:
        """
        Reads some number of bytes from the BINARY_ format
        representation by calling the function *read*, and returns the
        native in-memory object representation for the datatype
        handled by *self*.

        Parameters
        ----------
        read : function
            A function that given a number of bytes, returns a byte
            string.

        Returns
        -------
        native : (2,) tuple
            (value, mask). The value as a Numpy array or scalar, and *mask* is
            True if the value is missing.
        """
    def binoutput(self, value, mask) -> None:
        """
        Convert the object *value* in the native in-memory datatype to
        a string of bytes suitable for serialization in the BINARY_
        format.

        Parameters
        ----------
        value
            The value, the native type corresponding to this converter

        mask : bool
            If `True`, will return the string representation of a
            masked value.

        Returns
        -------
        bytes : bytes
            The binary representation of the value, suitable for
            serialization in the BINARY_ format.
        """

class Char(Converter):
    """
    Handles the char datatype. (7-bit unsigned characters).

    Missing values are not handled for string or unicode types.
    """
    default = _empty_bytes
    field_name: Incomplete
    format: str
    binparse: Incomplete
    binoutput: Incomplete
    arraysize: str
    _struct_format: Incomplete
    def __init__(self, field, config: Incomplete | None = None, pos: Incomplete | None = None) -> None: ...
    def supports_empty_values(self, config): ...
    def parse(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    def output(self, value, mask): ...
    def _binparse_var(self, read): ...
    def _binparse_fixed(self, read): ...
    def _binoutput_var(self, value, mask): ...
    def _binoutput_fixed(self, value, mask): ...

class UnicodeChar(Converter):
    """
    Handles the unicodeChar data type. UTF-16-BE.

    Missing values are not handled for string or unicode types.
    """
    default: str
    format: str
    binparse: Incomplete
    binoutput: Incomplete
    arraysize: str
    _struct_format: Incomplete
    def __init__(self, field, config: Incomplete | None = None, pos: Incomplete | None = None) -> None: ...
    def parse(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    def output(self, value, mask): ...
    def _binparse_var(self, read): ...
    def _binparse_fixed(self, read): ...
    def _binoutput_var(self, value, mask): ...
    def _binoutput_fixed(self, value, mask): ...

class Array(Converter):
    """
    Handles both fixed and variable-lengths arrays.
    """
    _splitter: Incomplete
    def __init__(self, field, config: Incomplete | None = None, pos: Incomplete | None = None) -> None: ...
    def parse_scalar(self, value, config: Incomplete | None = None, pos: int = 0): ...
    @staticmethod
    def _splitter_pedantic(value, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    @staticmethod
    def _splitter_lax(value, config: Incomplete | None = None, pos: Incomplete | None = None): ...

class VarArray(Array):
    """
    Handles variable lengths arrays (i.e. where *arraysize* is '*').
    """
    format: str
    _base: Incomplete
    default: Incomplete
    def __init__(self, field, base, arraysize, config: Incomplete | None = None, pos: Incomplete | None = None) -> None: ...
    def output(self, value, mask): ...
    def binparse(self, read): ...
    def binoutput(self, value, mask): ...

class ArrayVarArray(VarArray):
    """
    Handles an array of variable-length arrays, i.e. where *arraysize*
    ends in '*'.
    """
    def parse(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...

class ScalarVarArray(VarArray):
    """
    Handles a variable-length array of numeric scalars.
    """
    def parse(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...

class NumericArray(Array):
    """
    Handles a fixed-length array of numeric scalars.
    """
    vararray_type = ArrayVarArray
    _base: Incomplete
    _arraysize: Incomplete
    format: Incomplete
    _items: Incomplete
    _memsize: Incomplete
    _bigendian_format: Incomplete
    default: Incomplete
    def __init__(self, field, base, arraysize, config: Incomplete | None = None, pos: Incomplete | None = None) -> None: ...
    def parse(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    def parse_parts(self, parts, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    def output(self, value, mask): ...
    def binparse(self, read): ...
    def binoutput(self, value, mask): ...

class Numeric(Converter):
    """
    The base class for all numeric data types.
    """
    array_type = NumericArray
    vararray_type = ScalarVarArray
    null: Incomplete
    _memsize: Incomplete
    _bigendian_format: Incomplete
    default: Incomplete
    is_null: Incomplete
    def __init__(self, field, config: Incomplete | None = None, pos: Incomplete | None = None) -> None: ...
    def binparse(self, read): ...
    def _is_null(self, value): ...

class FloatingPoint(Numeric):
    """
    The base class for floating-point datatypes.
    """
    default: Incomplete
    _output_format: Incomplete
    nan: Incomplete
    _null_output: str
    _null_binoutput: Incomplete
    filter_array: Incomplete
    parse: Incomplete
    def __init__(self, field, config: Incomplete | None = None, pos: Incomplete | None = None) -> None: ...
    def supports_empty_values(self, config): ...
    def _parse_pedantic(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    def _parse_permissive(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    @property
    def output_format(self): ...
    def output(self, value, mask): ...
    def binoutput(self, value, mask): ...
    def _filter_nan(self, value, mask): ...
    def _filter_null(self, value, mask): ...

class Double(FloatingPoint):
    """
    Handles the double datatype.  Double-precision IEEE
    floating-point.
    """
    format: str

class Float(FloatingPoint):
    """
    Handles the float datatype.  Single-precision IEEE floating-point.
    """
    format: str

class Integer(Numeric):
    """
    The base class for all the integral datatypes.
    """
    default: int
    def __init__(self, field, config: Incomplete | None = None, pos: Incomplete | None = None) -> None: ...
    def parse(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    def output(self, value, mask): ...
    def binoutput(self, value, mask): ...
    def filter_array(self, value, mask): ...

class UnsignedByte(Integer):
    """
    Handles the unsignedByte datatype.  Unsigned 8-bit integer.
    """
    format: str
    val_range: Incomplete
    bit_size: str

class Short(Integer):
    """
    Handles the short datatype.  Signed 16-bit integer.
    """
    format: str
    val_range: Incomplete
    bit_size: str

class Int(Integer):
    """
    Handles the int datatype.  Signed 32-bit integer.
    """
    format: str
    val_range: Incomplete
    bit_size: str

class Long(Integer):
    """
    Handles the long datatype.  Signed 64-bit integer.
    """
    format: str
    val_range: Incomplete
    bit_size: str

class ComplexArrayVarArray(VarArray):
    """
    Handles an array of variable-length arrays of complex numbers.
    """
    def parse(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...

class ComplexVarArray(VarArray):
    """
    Handles a variable-length array of complex numbers.
    """
    def parse(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...

class ComplexArray(NumericArray):
    """
    Handles a fixed-size array of complex numbers.
    """
    vararray_type = ComplexArrayVarArray
    def __init__(self, field, base, arraysize, config: Incomplete | None = None, pos: Incomplete | None = None) -> None: ...
    def parse(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    def parse_parts(self, parts, config: Incomplete | None = None, pos: Incomplete | None = None): ...

class Complex(FloatingPoint, Array):
    """
    The base class for complex numbers.
    """
    array_type = ComplexArray
    vararray_type = ComplexVarArray
    default: Incomplete
    def __init__(self, field, config: Incomplete | None = None, pos: Incomplete | None = None) -> None: ...
    def parse(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    _parse_permissive = parse
    _parse_pedantic = parse
    def parse_parts(self, parts, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    def output(self, value, mask): ...

class FloatComplex(Complex):
    """
    Handle floatComplex datatype.  Pair of single-precision IEEE
    floating-point numbers.
    """
    format: str

class DoubleComplex(Complex):
    """
    Handle doubleComplex datatype.  Pair of double-precision IEEE
    floating-point numbers.
    """
    format: str

class BitArray(NumericArray):
    """
    Handles an array of bits.
    """
    vararray_type = ArrayVarArray
    _bytes: Incomplete
    def __init__(self, field, base, arraysize, config: Incomplete | None = None, pos: Incomplete | None = None) -> None: ...
    @staticmethod
    def _splitter_pedantic(value, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    @staticmethod
    def _splitter_lax(value, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    def output(self, value, mask): ...
    def binparse(self, read): ...
    def binoutput(self, value, mask): ...

class Bit(Converter):
    """
    Handles the bit datatype.
    """
    format: str
    array_type = BitArray
    vararray_type = ScalarVarArray
    default: bool
    binary_one: bytes
    binary_zero: bytes
    def parse(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    def output(self, value, mask): ...
    def binparse(self, read): ...
    def binoutput(self, value, mask): ...

class BooleanArray(NumericArray):
    """
    Handles an array of boolean values.
    """
    vararray_type = ArrayVarArray
    def binparse(self, read): ...
    def binoutput(self, value, mask): ...

class Boolean(Converter):
    """
    Handles the boolean datatype.
    """
    format: str
    array_type = BooleanArray
    vararray_type = ScalarVarArray
    default: bool
    binary_question_mark: bytes
    binary_true: bytes
    binary_false: bytes
    def parse(self, value, config: Incomplete | None = None, pos: Incomplete | None = None): ...
    def output(self, value, mask): ...
    def binparse(self, read): ...
    _binparse_mapping: Incomplete
    def binparse_value(self, value): ...
    def binoutput(self, value, mask): ...

def get_converter(field, config: Incomplete | None = None, pos: Incomplete | None = None):
    """
    Get an appropriate converter instance for a given field.

    Parameters
    ----------
    field : astropy.io.votable.tree.Field

    config : dict, optional
        Parser configuration dictionary

    pos : tuple
        Position in the input XML file.  Used for error messages.

    Returns
    -------
    converter : astropy.io.votable.converters.Converter
    """
def table_column_to_votable_datatype(column):
    '''
    Given a `astropy.table.Column` instance, returns the attributes
    necessary to create a VOTable FIELD element that corresponds to
    the type of the column.

    This necessarily must perform some heuristics to determine the
    type of variable length arrays fields, since they are not directly
    supported by Numpy.

    If the column has dtype of "object", it performs the following
    tests:

       - If all elements are byte or unicode strings, it creates a
         variable-length byte or unicode field, respectively.

       - If all elements are numpy arrays of the same dtype and with a
         consistent shape in all but the first dimension, it creates a
         variable length array of fixed sized arrays.  If the dtypes
         match, but the shapes do not, a variable length array is
         created.

    If the dtype of the input is not understood, it sets the data type
    to the most inclusive: a variable length unicodeChar array.

    Parameters
    ----------
    column : `astropy.table.Column` instance

    Returns
    -------
    attributes : dict
        A dict containing \'datatype\' and \'arraysize\' keys that can be
        set on a VOTable FIELD element.
    '''

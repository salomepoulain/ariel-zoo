from _typeshed import Incomplete

__all__ = ['__version__', 'dottedQuadToNum', 'numToDottedQuad', 'ValidateError', 'VdtUnknownCheckError', 'VdtParamError', 'VdtTypeError', 'VdtValueError', 'VdtValueTooSmallError', 'VdtValueTooBigError', 'VdtValueTooShortError', 'VdtValueTooLongError', 'VdtMissingValue', 'Validator', 'is_integer', 'is_float', 'is_boolean', 'is_list', 'is_tuple', 'is_ip_addr', 'is_string', 'is_int_list', 'is_bool_list', 'is_float_list', 'is_string_list', 'is_ip_addr_list', 'is_mixed_list', 'is_option', '__docformat__']

__version__: str
string_type = str
long = int

def dottedQuadToNum(ip):
    """
    Convert decimal dotted quad string to long integer

    >>> int(dottedQuadToNum('1 '))
    1
    >>> int(dottedQuadToNum(' 1.2'))
    16777218
    >>> int(dottedQuadToNum(' 1.2.3 '))
    16908291
    >>> int(dottedQuadToNum('1.2.3.4'))
    16909060
    >>> dottedQuadToNum('255.255.255.255')
    4294967295
    >>> dottedQuadToNum('255.255.255.256')
    Traceback (most recent call last):
    ValueError: Not a good dotted-quad IP: 255.255.255.256
    """
def numToDottedQuad(num):
    """
    Convert int or long int to dotted quad string

    >>> numToDottedQuad(long(-1))
    Traceback (most recent call last):
    ValueError: Not a good numeric IP: -1
    >>> numToDottedQuad(long(1))
    '0.0.0.1'
    >>> numToDottedQuad(long(16777218))
    '1.0.0.2'
    >>> numToDottedQuad(long(16908291))
    '1.2.0.3'
    >>> numToDottedQuad(long(16909060))
    '1.2.3.4'
    >>> numToDottedQuad(long(4294967295))
    '255.255.255.255'
    >>> numToDottedQuad(long(4294967296))
    Traceback (most recent call last):
    ValueError: Not a good numeric IP: 4294967296
    >>> numToDottedQuad(-1)
    Traceback (most recent call last):
    ValueError: Not a good numeric IP: -1
    >>> numToDottedQuad(1)
    '0.0.0.1'
    >>> numToDottedQuad(16777218)
    '1.0.0.2'
    >>> numToDottedQuad(16908291)
    '1.2.0.3'
    >>> numToDottedQuad(16909060)
    '1.2.3.4'
    >>> numToDottedQuad(4294967295)
    '255.255.255.255'
    >>> numToDottedQuad(4294967296)
    Traceback (most recent call last):
    ValueError: Not a good numeric IP: 4294967296

    """

class ValidateError(Exception):
    """
    This error indicates that the check failed.
    It can be the base class for more specific errors.

    Any check function that fails ought to raise this error.
    (or a subclass)

    >>> raise ValidateError
    Traceback (most recent call last):
    ValidateError
    """
class VdtMissingValue(ValidateError):
    """No value was supplied to a check that needed one."""

class VdtUnknownCheckError(ValidateError):
    """An unknown check function was requested"""
    def __init__(self, value) -> None:
        '''
        >>> raise VdtUnknownCheckError(\'yoda\')
        Traceback (most recent call last):
        VdtUnknownCheckError: the check "yoda" is unknown.
        '''

class VdtParamError(SyntaxError):
    """An incorrect parameter was passed"""
    def __init__(self, name, value) -> None:
        '''
        >>> raise VdtParamError(\'yoda\', \'jedi\')
        Traceback (most recent call last):
        VdtParamError: passed an incorrect value "jedi" for parameter "yoda".
        '''

class VdtTypeError(ValidateError):
    """The value supplied was of the wrong type"""
    def __init__(self, value) -> None:
        '''
        >>> raise VdtTypeError(\'jedi\')
        Traceback (most recent call last):
        VdtTypeError: the value "jedi" is of the wrong type.
        '''

class VdtValueError(ValidateError):
    """The value supplied was of the correct type, but was not an allowed value."""
    def __init__(self, value) -> None:
        '''
        >>> raise VdtValueError(\'jedi\')
        Traceback (most recent call last):
        VdtValueError: the value "jedi" is unacceptable.
        '''

class VdtValueTooSmallError(VdtValueError):
    """The value supplied was of the correct type, but was too small."""
    def __init__(self, value) -> None:
        '''
        >>> raise VdtValueTooSmallError(\'0\')
        Traceback (most recent call last):
        VdtValueTooSmallError: the value "0" is too small.
        '''

class VdtValueTooBigError(VdtValueError):
    """The value supplied was of the correct type, but was too big."""
    def __init__(self, value) -> None:
        '''
        >>> raise VdtValueTooBigError(\'1\')
        Traceback (most recent call last):
        VdtValueTooBigError: the value "1" is too big.
        '''

class VdtValueTooShortError(VdtValueError):
    """The value supplied was of the correct type, but was too short."""
    def __init__(self, value) -> None:
        '''
        >>> raise VdtValueTooShortError(\'jed\')
        Traceback (most recent call last):
        VdtValueTooShortError: the value "jed" is too short.
        '''

class VdtValueTooLongError(VdtValueError):
    """The value supplied was of the correct type, but was too long."""
    def __init__(self, value) -> None:
        '''
        >>> raise VdtValueTooLongError(\'jedie\')
        Traceback (most recent call last):
        VdtValueTooLongError: the value "jedie" is too long.
        '''

class Validator:
    '''
    Validator is an object that allows you to register a set of \'checks\'.
    These checks take input and test that it conforms to the check.

    This can also involve converting the value from a string into
    the correct datatype.

    The ``check`` method takes an input string which configures which
    check is to be used and applies that check to a supplied value.

    An example input string would be:
    \'int_range(param1, param2)\'

    You would then provide something like:

    >>> def int_range_check(value, min, max):
    ...     # turn min and max from strings to integers
    ...     min = int(min)
    ...     max = int(max)
    ...     # check that value is of the correct type.
    ...     # possible valid inputs are integers or strings
    ...     # that represent integers
    ...     if not isinstance(value, (int, long, string_type)):
    ...         raise VdtTypeError(value)
    ...     elif isinstance(value, string_type):
    ...         # if we are given a string
    ...         # attempt to convert to an integer
    ...         try:
    ...             value = int(value)
    ...         except ValueError:
    ...             raise VdtValueError(value)
    ...     # check the value is between our constraints
    ...     if not min <= value:
    ...          raise VdtValueTooSmallError(value)
    ...     if not value <= max:
    ...          raise VdtValueTooBigError(value)
    ...     return value

    >>> fdict = {\'int_range\': int_range_check}
    >>> vtr1 = Validator(fdict)
    >>> vtr1.check(\'int_range(20, 40)\', \'30\')
    30
    >>> vtr1.check(\'int_range(20, 40)\', \'60\')
    Traceback (most recent call last):
    VdtValueTooBigError: the value "60" is too big.

    New functions can be added with : ::

    >>> vtr2 = Validator()
    >>> vtr2.functions[\'int_range\'] = int_range_check

    Or by passing in a dictionary of functions when Validator
    is instantiated.

    Your functions *can* use keyword arguments,
    but the first argument should always be \'value\'.

    If the function doesn\'t take additional arguments,
    the parentheses are optional in the check.
    It can be written with either of : ::

        keyword = function_name
        keyword = function_name()

    The first program to utilise Validator() was Michael Foord\'s
    ConfigObj, an alternative to ConfigParser which supports lists and
    can validate a config file using a config schema.
    For more details on using Validator with ConfigObj see:
    https://configobj.readthedocs.org/en/latest/configobj.html
    '''
    _func_re: Incomplete
    _key_arg: Incomplete
    _list_arg = _list_arg
    _list_members = _list_members
    _paramfinder: Incomplete
    _matchfinder: Incomplete
    functions: Incomplete
    baseErrorClass: Incomplete
    _cache: Incomplete
    def __init__(self, functions: Incomplete | None = None) -> None:
        """
        >>> vtri = Validator()
        """
    def check(self, check, value, missing: bool = False):
        '''
        Usage: check(check, value)

        Arguments:
            check: string representing check to apply (including arguments)
            value: object to be checked
        Returns value, converted to correct type if necessary

        If the check fails, raises a ``ValidateError`` subclass.

        >>> vtor.check(\'yoda\', \'\')
        Traceback (most recent call last):
        VdtUnknownCheckError: the check "yoda" is unknown.
        >>> vtor.check(\'yoda()\', \'\')
        Traceback (most recent call last):
        VdtUnknownCheckError: the check "yoda" is unknown.

        >>> vtor.check(\'string(default="")\', \'\', missing=True)
        \'\'
        '''
    def _handle_none(self, value): ...
    def _parse_with_caching(self, check): ...
    def _check_value(self, value, fun_name, fun_args, fun_kwargs): ...
    def _parse_check(self, check): ...
    def _unquote(self, val):
        """Unquote a value if necessary."""
    def _list_handle(self, listmatch):
        """Take apart a ``keyword=list('val, 'val')`` type string."""
    def _pass(self, value):
        """
        Dummy check that always passes

        >>> vtor.check('', 0)
        0
        >>> vtor.check('', '0')
        '0'
        """
    def get_default_value(self, check):
        """
        Given a check, return the default value for the check
        (converted to the right type).

        If the check doesn't specify a default value then a
        ``KeyError`` will be raised.
        """

def is_integer(value, min: Incomplete | None = None, max: Incomplete | None = None):
    '''
    A check that tests that a given value is an integer (int, or long)
    and optionally, between bounds. A negative value is accepted, while
    a float will fail.

    If the value is a string, then the conversion is done - if possible.
    Otherwise a VdtError is raised.

    >>> vtor.check(\'integer\', \'-1\')
    -1
    >>> vtor.check(\'integer\', \'0\')
    0
    >>> vtor.check(\'integer\', 9)
    9
    >>> vtor.check(\'integer\', \'a\')
    Traceback (most recent call last):
    VdtTypeError: the value "a" is of the wrong type.
    >>> vtor.check(\'integer\', \'2.2\')
    Traceback (most recent call last):
    VdtTypeError: the value "2.2" is of the wrong type.
    >>> vtor.check(\'integer(10)\', \'20\')
    20
    >>> vtor.check(\'integer(max=20)\', \'15\')
    15
    >>> vtor.check(\'integer(10)\', \'9\')
    Traceback (most recent call last):
    VdtValueTooSmallError: the value "9" is too small.
    >>> vtor.check(\'integer(10)\', 9)
    Traceback (most recent call last):
    VdtValueTooSmallError: the value "9" is too small.
    >>> vtor.check(\'integer(max=20)\', \'35\')
    Traceback (most recent call last):
    VdtValueTooBigError: the value "35" is too big.
    >>> vtor.check(\'integer(max=20)\', 35)
    Traceback (most recent call last):
    VdtValueTooBigError: the value "35" is too big.
    >>> vtor.check(\'integer(0, 9)\', False)
    0
    '''
def is_float(value, min: Incomplete | None = None, max: Incomplete | None = None):
    '''
    A check that tests that a given value is a float
    (an integer will be accepted), and optionally - that it is between bounds.

    If the value is a string, then the conversion is done - if possible.
    Otherwise a VdtError is raised.

    This can accept negative values.

    >>> vtor.check(\'float\', \'2\')
    2.0

    From now on we multiply the value to avoid comparing decimals

    >>> vtor.check(\'float\', \'-6.8\') * 10
    -68.0
    >>> vtor.check(\'float\', \'12.2\') * 10
    122.0
    >>> vtor.check(\'float\', 8.4) * 10
    84.0
    >>> vtor.check(\'float\', \'a\')
    Traceback (most recent call last):
    VdtTypeError: the value "a" is of the wrong type.
    >>> vtor.check(\'float(10.1)\', \'10.2\') * 10
    102.0
    >>> vtor.check(\'float(max=20.2)\', \'15.1\') * 10
    151.0
    >>> vtor.check(\'float(10.0)\', \'9.0\')
    Traceback (most recent call last):
    VdtValueTooSmallError: the value "9.0" is too small.
    >>> vtor.check(\'float(max=20.0)\', \'35.0\')
    Traceback (most recent call last):
    VdtValueTooBigError: the value "35.0" is too big.
    '''
def is_boolean(value):
    '''
    Check if the value represents a boolean.

    >>> vtor.check(\'boolean\', 0)
    0
    >>> vtor.check(\'boolean\', False)
    0
    >>> vtor.check(\'boolean\', \'0\')
    0
    >>> vtor.check(\'boolean\', \'off\')
    0
    >>> vtor.check(\'boolean\', \'false\')
    0
    >>> vtor.check(\'boolean\', \'no\')
    0
    >>> vtor.check(\'boolean\', \'nO\')
    0
    >>> vtor.check(\'boolean\', \'NO\')
    0
    >>> vtor.check(\'boolean\', 1)
    1
    >>> vtor.check(\'boolean\', True)
    1
    >>> vtor.check(\'boolean\', \'1\')
    1
    >>> vtor.check(\'boolean\', \'on\')
    1
    >>> vtor.check(\'boolean\', \'true\')
    1
    >>> vtor.check(\'boolean\', \'yes\')
    1
    >>> vtor.check(\'boolean\', \'Yes\')
    1
    >>> vtor.check(\'boolean\', \'YES\')
    1
    >>> vtor.check(\'boolean\', \'\')
    Traceback (most recent call last):
    VdtTypeError: the value "" is of the wrong type.
    >>> vtor.check(\'boolean\', \'up\')
    Traceback (most recent call last):
    VdtTypeError: the value "up" is of the wrong type.

    '''
def is_ip_addr(value):
    '''
    Check that the supplied value is an Internet Protocol address, v.4,
    represented by a dotted-quad string, i.e. \'1.2.3.4\'.

    >>> vtor.check(\'ip_addr\', \'1 \')
    \'1\'
    >>> vtor.check(\'ip_addr\', \' 1.2\')
    \'1.2\'
    >>> vtor.check(\'ip_addr\', \' 1.2.3 \')
    \'1.2.3\'
    >>> vtor.check(\'ip_addr\', \'1.2.3.4\')
    \'1.2.3.4\'
    >>> vtor.check(\'ip_addr\', \'0.0.0.0\')
    \'0.0.0.0\'
    >>> vtor.check(\'ip_addr\', \'255.255.255.255\')
    \'255.255.255.255\'
    >>> vtor.check(\'ip_addr\', \'255.255.255.256\')
    Traceback (most recent call last):
    VdtValueError: the value "255.255.255.256" is unacceptable.
    >>> vtor.check(\'ip_addr\', \'1.2.3.4.5\')
    Traceback (most recent call last):
    VdtValueError: the value "1.2.3.4.5" is unacceptable.
    >>> vtor.check(\'ip_addr\', 0)
    Traceback (most recent call last):
    VdtTypeError: the value "0" is of the wrong type.
    '''
def is_list(value, min: Incomplete | None = None, max: Incomplete | None = None):
    '''
    Check that the value is a list of values.

    You can optionally specify the minimum and maximum number of members.

    It does no check on list members.

    >>> vtor.check(\'list\', ())
    []
    >>> vtor.check(\'list\', [])
    []
    >>> vtor.check(\'list\', (1, 2))
    [1, 2]
    >>> vtor.check(\'list\', [1, 2])
    [1, 2]
    >>> vtor.check(\'list(3)\', (1, 2))
    Traceback (most recent call last):
    VdtValueTooShortError: the value "(1, 2)" is too short.
    >>> vtor.check(\'list(max=5)\', (1, 2, 3, 4, 5, 6))
    Traceback (most recent call last):
    VdtValueTooLongError: the value "(1, 2, 3, 4, 5, 6)" is too long.
    >>> vtor.check(\'list(min=3, max=5)\', (1, 2, 3, 4))
    [1, 2, 3, 4]
    >>> vtor.check(\'list\', 0)
    Traceback (most recent call last):
    VdtTypeError: the value "0" is of the wrong type.
    >>> vtor.check(\'list\', \'12\')
    Traceback (most recent call last):
    VdtTypeError: the value "12" is of the wrong type.
    '''
def is_tuple(value, min: Incomplete | None = None, max: Incomplete | None = None):
    '''
    Check that the value is a tuple of values.

    You can optionally specify the minimum and maximum number of members.

    It does no check on members.

    >>> vtor.check(\'tuple\', ())
    ()
    >>> vtor.check(\'tuple\', [])
    ()
    >>> vtor.check(\'tuple\', (1, 2))
    (1, 2)
    >>> vtor.check(\'tuple\', [1, 2])
    (1, 2)
    >>> vtor.check(\'tuple(3)\', (1, 2))
    Traceback (most recent call last):
    VdtValueTooShortError: the value "(1, 2)" is too short.
    >>> vtor.check(\'tuple(max=5)\', (1, 2, 3, 4, 5, 6))
    Traceback (most recent call last):
    VdtValueTooLongError: the value "(1, 2, 3, 4, 5, 6)" is too long.
    >>> vtor.check(\'tuple(min=3, max=5)\', (1, 2, 3, 4))
    (1, 2, 3, 4)
    >>> vtor.check(\'tuple\', 0)
    Traceback (most recent call last):
    VdtTypeError: the value "0" is of the wrong type.
    >>> vtor.check(\'tuple\', \'12\')
    Traceback (most recent call last):
    VdtTypeError: the value "12" is of the wrong type.
    '''
def is_string(value, min: Incomplete | None = None, max: Incomplete | None = None):
    '''
    Check that the supplied value is a string.

    You can optionally specify the minimum and maximum number of members.

    >>> vtor.check(\'string\', \'0\')
    \'0\'
    >>> vtor.check(\'string\', 0)
    Traceback (most recent call last):
    VdtTypeError: the value "0" is of the wrong type.
    >>> vtor.check(\'string(2)\', \'12\')
    \'12\'
    >>> vtor.check(\'string(2)\', \'1\')
    Traceback (most recent call last):
    VdtValueTooShortError: the value "1" is too short.
    >>> vtor.check(\'string(min=2, max=3)\', \'123\')
    \'123\'
    >>> vtor.check(\'string(min=2, max=3)\', \'1234\')
    Traceback (most recent call last):
    VdtValueTooLongError: the value "1234" is too long.
    '''
def is_int_list(value, min: Incomplete | None = None, max: Incomplete | None = None):
    '''
    Check that the value is a list of integers.

    You can optionally specify the minimum and maximum number of members.

    Each list member is checked that it is an integer.

    >>> vtor.check(\'int_list\', ())
    []
    >>> vtor.check(\'int_list\', [])
    []
    >>> vtor.check(\'int_list\', (1, 2))
    [1, 2]
    >>> vtor.check(\'int_list\', [1, 2])
    [1, 2]
    >>> vtor.check(\'int_list\', [1, \'a\'])
    Traceback (most recent call last):
    VdtTypeError: the value "a" is of the wrong type.
    '''
def is_bool_list(value, min: Incomplete | None = None, max: Incomplete | None = None):
    '''
    Check that the value is a list of booleans.

    You can optionally specify the minimum and maximum number of members.

    Each list member is checked that it is a boolean.

    >>> vtor.check(\'bool_list\', ())
    []
    >>> vtor.check(\'bool_list\', [])
    []
    >>> check_res = vtor.check(\'bool_list\', (True, False))
    >>> check_res == [True, False]
    1
    >>> check_res = vtor.check(\'bool_list\', [True, False])
    >>> check_res == [True, False]
    1
    >>> vtor.check(\'bool_list\', [True, \'a\'])
    Traceback (most recent call last):
    VdtTypeError: the value "a" is of the wrong type.
    '''
def is_float_list(value, min: Incomplete | None = None, max: Incomplete | None = None):
    '''
    Check that the value is a list of floats.

    You can optionally specify the minimum and maximum number of members.

    Each list member is checked that it is a float.

    >>> vtor.check(\'float_list\', ())
    []
    >>> vtor.check(\'float_list\', [])
    []
    >>> vtor.check(\'float_list\', (1, 2.0))
    [1.0, 2.0]
    >>> vtor.check(\'float_list\', [1, 2.0])
    [1.0, 2.0]
    >>> vtor.check(\'float_list\', [1, \'a\'])
    Traceback (most recent call last):
    VdtTypeError: the value "a" is of the wrong type.
    '''
def is_string_list(value, min: Incomplete | None = None, max: Incomplete | None = None):
    '''
    Check that the value is a list of strings.

    You can optionally specify the minimum and maximum number of members.

    Each list member is checked that it is a string.

    >>> vtor.check(\'string_list\', ())
    []
    >>> vtor.check(\'string_list\', [])
    []
    >>> vtor.check(\'string_list\', (\'a\', \'b\'))
    [\'a\', \'b\']
    >>> vtor.check(\'string_list\', [\'a\', 1])
    Traceback (most recent call last):
    VdtTypeError: the value "1" is of the wrong type.
    >>> vtor.check(\'string_list\', \'hello\')
    Traceback (most recent call last):
    VdtTypeError: the value "hello" is of the wrong type.
    '''
def is_ip_addr_list(value, min: Incomplete | None = None, max: Incomplete | None = None):
    '''
    Check that the value is a list of IP addresses.

    You can optionally specify the minimum and maximum number of members.

    Each list member is checked that it is an IP address.

    >>> vtor.check(\'ip_addr_list\', ())
    []
    >>> vtor.check(\'ip_addr_list\', [])
    []
    >>> vtor.check(\'ip_addr_list\', (\'1.2.3.4\', \'5.6.7.8\'))
    [\'1.2.3.4\', \'5.6.7.8\']
    >>> vtor.check(\'ip_addr_list\', [\'a\'])
    Traceback (most recent call last):
    VdtValueError: the value "a" is unacceptable.
    '''
def is_mixed_list(value, *args):
    '''
    Check that the value is a list.
    Allow specifying the type of each member.
    Work on lists of specific lengths.

    You specify each member as a positional argument specifying type

    Each type should be one of the following strings :
      \'integer\', \'float\', \'ip_addr\', \'string\', \'boolean\'

    So you can specify a list of two strings, followed by
    two integers as :

      mixed_list(\'string\', \'string\', \'integer\', \'integer\')

    The length of the list must match the number of positional
    arguments you supply.

    >>> mix_str = "mixed_list(\'integer\', \'float\', \'ip_addr\', \'string\', \'boolean\')"
    >>> check_res = vtor.check(mix_str, (1, 2.0, \'1.2.3.4\', \'a\', True))
    >>> check_res == [1, 2.0, \'1.2.3.4\', \'a\', True]
    1
    >>> check_res = vtor.check(mix_str, (\'1\', \'2.0\', \'1.2.3.4\', \'a\', \'True\'))
    >>> check_res == [1, 2.0, \'1.2.3.4\', \'a\', True]
    1
    >>> vtor.check(mix_str, (\'b\', 2.0, \'1.2.3.4\', \'a\', True))
    Traceback (most recent call last):
    VdtTypeError: the value "b" is of the wrong type.
    >>> vtor.check(mix_str, (1, 2.0, \'1.2.3.4\', \'a\'))
    Traceback (most recent call last):
    VdtValueTooShortError: the value "(1, 2.0, \'1.2.3.4\', \'a\')" is too short.
    >>> vtor.check(mix_str, (1, 2.0, \'1.2.3.4\', \'a\', 1, \'b\'))
    Traceback (most recent call last):
    VdtValueTooLongError: the value "(1, 2.0, \'1.2.3.4\', \'a\', 1, \'b\')" is too long.
    >>> vtor.check(mix_str, 0)
    Traceback (most recent call last):
    VdtTypeError: the value "0" is of the wrong type.

    >>> vtor.check(\'mixed_list("yoda")\', (\'a\'))
    Traceback (most recent call last):
    VdtParamError: passed an incorrect value "KeyError(\'yoda\',)" for parameter "\'mixed_list\'"
    '''
def is_option(value, *options):
    '''
    This check matches the value to any of a set of options.

    >>> vtor.check(\'option("yoda", "jedi")\', \'yoda\')
    \'yoda\'
    >>> vtor.check(\'option("yoda", "jedi")\', \'jed\')
    Traceback (most recent call last):
    VdtValueError: the value "jed" is unacceptable.
    >>> vtor.check(\'option("yoda", "jedi")\', 0)
    Traceback (most recent call last):
    VdtTypeError: the value "0" is of the wrong type.
    '''

# Names in __all__ with no definition:
#   __docformat__

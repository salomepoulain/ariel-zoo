import json
from _typeshed import Incomplete
from collections.abc import Generator

__all__ = ['isiterable', 'silence', 'format_exception', 'NumpyRNGContext', 'find_api_page', 'is_path_hidden', 'walk_skip_hidden', 'JsonCustomEncoder', 'indent', 'dtype_bytes_or_chars']

def isiterable(obj):
    """Returns `True` if the given object is iterable."""
def indent(s, shift: int = 1, width: int = 4):
    """Indent a block of text.  The indentation is applied to each line."""

class _DummyFile:
    """A noop writeable object."""
    def write(self, s) -> None: ...

def silence() -> Generator[None]:
    """A context manager that silences sys.stdout and sys.stderr."""
def format_exception(msg, *args, **kwargs):
    """Fill in information about the exception that occurred.

    Given an exception message string, uses new-style formatting arguments
    ``{filename}``, ``{lineno}``, ``{func}`` and/or ``{text}`` to fill in
    information about the exception that occurred.  For example:

        try:
            1/0
        except:
            raise ZeroDivisionError(
                format_except('A divide by zero occurred in {filename} at '
                              'line {lineno} of function {func}.'))

    Any additional positional or keyword arguments passed to this function are
    also used to format the message.

    .. note::
        This uses `sys.exc_info` to gather up the information needed to fill
        in the formatting arguments. Since `sys.exc_info` is not carried
        outside a handled exception, it's not wise to use this
        outside of an ``except`` clause - if it is, this will substitute
        '<unknown>' for the 4 formatting arguments.
    """

class NumpyRNGContext:
    """
    A context manager (for use with the ``with`` statement) that will seed the
    numpy random number generator (RNG) to a specific value, and then restore
    the RNG state back to whatever it was before.

    This is primarily intended for use in the astropy testing suit, but it
    may be useful in ensuring reproducibility of Monte Carlo simulations in a
    science context.

    Parameters
    ----------
    seed : int
        The value to use to seed the numpy RNG

    Examples
    --------
    A typical use case might be::

        with NumpyRNGContext(<some seed value you pick>):
            from numpy import random

            randarr = random.randn(100)
            ... run your test using `randarr` ...

        #Any code using numpy.random at this indent level will act just as it
        #would have if it had been before the with statement - e.g. whatever
        #the default seed is.


    """
    seed: Incomplete
    def __init__(self, seed) -> None: ...
    startstate: Incomplete
    def __enter__(self) -> None: ...
    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: types.TracebackType | None) -> None: ...

def find_api_page(obj, version: Incomplete | None = None, openinbrowser: bool = True, timeout: Incomplete | None = None):
    """
    Determines the URL of the API page for the specified object, and
    optionally open that page in a web browser.

    .. note::
        You must be connected to the internet for this to function even if
        ``openinbrowser`` is `False`, unless you provide a local version of
        the documentation to ``version`` (e.g., ``file:///path/to/docs``).

    Parameters
    ----------
    obj
        The object to open the docs for or its fully-qualified name
        (as a str).
    version : str
        The doc version - either a version number like '0.1', 'dev' for
        the development/latest docs, or a URL to point to a specific
        location that should be the *base* of the documentation. Defaults to
        latest if you are on aren't on a release, otherwise, the version you
        are on.
    openinbrowser : bool
        If `True`, the `webbrowser` package will be used to open the doc
        page in a new web browser window.
    timeout : number, optional
        The number of seconds to wait before timing-out the query to
        the astropy documentation.  If not given, the default python
        stdlib timeout will be used.

    Returns
    -------
    url : str
        The loaded URL

    Raises
    ------
    ValueError
        If the documentation can't be found

    """
def is_path_hidden(filepath):
    """
    Determines if a given file or directory is hidden.

    Parameters
    ----------
    filepath : str
        The path to a file or directory

    Returns
    -------
    hidden : bool
        Returns `True` if the file is hidden
    """
def walk_skip_hidden(top, onerror: Incomplete | None = None, followlinks: bool = False) -> Generator[Incomplete]:
    """
    A wrapper for `os.walk` that skips hidden files and directories.

    This function does not have the parameter ``topdown`` from
    `os.walk`: the directories must always be recursed top-down when
    using this function.

    See Also
    --------
    os.walk : For a description of the parameters
    """

class JsonCustomEncoder(json.JSONEncoder):
    """Support for data types that JSON default encoder
    does not do.

    This includes:

        * Numpy array or number
        * Complex number
        * Set
        * Bytes
        * astropy.UnitBase
        * astropy.Quantity

    Examples
    --------
    >>> import json
    >>> import numpy as np
    >>> from astropy.utils.misc import JsonCustomEncoder
    >>> json.dumps(np.arange(3), cls=JsonCustomEncoder)
    '[0, 1, 2]'

    """
    def default(self, obj): ...

def dtype_bytes_or_chars(dtype):
    """
    Parse the number out of a dtype.str value like '<U5' or '<f8'.

    See #5819 for discussion on the need for this function for getting
    the number of characters corresponding to a string dtype.

    Parameters
    ----------
    dtype : numpy dtype object
        Input dtype

    Returns
    -------
    bytes_or_chars : int or None
        Bits (for numeric types) or characters (for string types)
    """

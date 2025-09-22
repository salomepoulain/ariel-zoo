from _typeshed import Incomplete
from astropy.utils import data as data
from astropy.utils.compat.optional_deps import HAS_DASK as HAS_DASK
from astropy.utils.exceptions import AstropyUserWarning as AstropyUserWarning
from collections.abc import Generator

path_like: Incomplete
cmp: Incomplete
all_integer_types: Incomplete

class NotifierMixin:
    """
    Mixin class that provides services by which objects can register
    listeners to changes on that object.

    All methods provided by this class are underscored, since this is intended
    for internal use to communicate between classes in a generic way, and is
    not machinery that should be exposed to users of the classes involved.

    Use the ``_add_listener`` method to register a listener on an instance of
    the notifier.  This registers the listener with a weak reference, so if
    no other references to the listener exist it is automatically dropped from
    the list and does not need to be manually removed.

    Call the ``_notify`` method on the notifier to update all listeners
    upon changes.  ``_notify('change_type', *args, **kwargs)`` results
    in calling ``listener._update_change_type(*args, **kwargs)`` on all
    listeners subscribed to that notifier.

    If a particular listener does not have the appropriate update method
    it is ignored.

    Examples
    --------
    >>> class Widget(NotifierMixin):
    ...     state = 1
    ...     def __init__(self, name):
    ...         self.name = name
    ...     def update_state(self):
    ...         self.state += 1
    ...         self._notify('widget_state_changed', self)
    ...
    >>> class WidgetListener:
    ...     def _update_widget_state_changed(self, widget):
    ...         print('Widget {0} changed state to {1}'.format(
    ...             widget.name, widget.state))
    ...
    >>> widget = Widget('fred')
    >>> listener = WidgetListener()
    >>> widget._add_listener(listener)
    >>> widget.update_state()
    Widget fred changed state to 2
    """
    _listeners: Incomplete
    def _add_listener(self, listener) -> None:
        """
        Add an object to the list of listeners to notify of changes to this
        object.  This adds a weakref to the list of listeners that is
        removed from the listeners list when the listener has no other
        references to it.
        """
    def _remove_listener(self, listener) -> None:
        """
        Removes the specified listener from the listeners list.  This relies
        on object identity (i.e. the ``is`` operator).
        """
    def _notify(self, notification, *args, **kwargs) -> None:
        """
        Notify all listeners of some particular state change by calling their
        ``_update_<notification>`` method with the given ``*args`` and
        ``**kwargs``.

        The notification does not by default include the object that actually
        changed (``self``), but it certainly may if required.
        """
    def __getstate__(self):
        """
        Exclude listeners when saving the listener's state, since they may be
        ephemeral.
        """

def first(iterable):
    """
    Returns the first item returned by iterating over an iterable object.

    Examples
    --------
    >>> a = [1, 2, 3]
    >>> first(a)
    1
    """
def itersubclasses(cls, _seen: Incomplete | None = None) -> Generator[Incomplete, Incomplete]:
    """
    Generator over all subclasses of a given class, in depth first order.

    >>> class A: pass
    >>> class B(A): pass
    >>> class C(A): pass
    >>> class D(B,C): pass
    >>> class E(D): pass
    >>>
    >>> for cls in itersubclasses(A):
    ...     print(cls.__name__)
    B
    D
    E
    C
    >>> # get ALL classes currently defined
    >>> [cls.__name__ for cls in itersubclasses(object)]
    [...'tuple', ...'type', ...]

    From http://code.activestate.com/recipes/576949/
    """
def ignore_sigint(func):
    """
    This decorator registers a custom SIGINT handler to catch and ignore SIGINT
    until the wrapped function is completed.
    """
def encode_ascii(s): ...
def decode_ascii(s): ...
def isreadable(f):
    """
    Returns True if the file-like object can be read from.  This is a common-
    sense approximation of io.IOBase.readable.
    """
def iswritable(f):
    """
    Returns True if the file-like object can be written to.  This is a common-
    sense approximation of io.IOBase.writable.
    """
def isfile(f):
    """
    Returns True if the given object represents an OS-level file (that is,
    ``isinstance(f, file)``).

    This also returns True if the given object is higher level
    wrapper on top of a FileIO object, such as a TextIOWrapper.
    """
def fileobj_name(f):
    """
    Returns the 'name' of file-like object *f*, if it has anything that could be
    called its name.  Otherwise f's class or type is returned.  If f is a
    string f itself is returned.
    """
def fileobj_closed(f):
    """
    Returns True if the given file-like object is closed or if *f* is a string
    (and assumed to be a pathname).

    Returns False for all other types of objects, under the assumption that
    they are file-like objects with no sense of a 'closed' state.
    """
def fileobj_mode(f):
    """
    Returns the 'mode' string of a file-like object if such a thing exists.
    Otherwise returns None.
    """
def _fileobj_normalize_mode(f):
    """Takes care of some corner cases in Python where the mode string
    is either oddly formatted or does not truly represent the file mode.
    """
def fileobj_is_binary(f):
    """
    Returns True if the give file or file-like object has a file open in binary
    mode.  When in doubt, returns True by default.
    """
def translate(s, table, deletechars): ...
def fill(text, width, **kwargs):
    """
    Like :func:`textwrap.wrap` but preserves existing paragraphs which
    :func:`textwrap.wrap` does not otherwise handle well.  Also handles section
    headers.
    """

CHUNKED_FROMFILE: Incomplete

def _array_from_file(infile, dtype, count):
    """Create a numpy array from a file or a file-like object."""

_OSX_WRITE_LIMIT: Incomplete
_WIN_WRITE_LIMIT: Incomplete

def _array_to_file(arr, outfile):
    """
    Write a numpy array to a file or a file-like object.

    Parameters
    ----------
    arr : ndarray
        The Numpy array to write.
    outfile : file-like
        A file-like object such as a Python file object, an `io.BytesIO`, or
        anything else with a ``write`` method.  The file object must support
        the buffer interface in its ``write``.

    If writing directly to an on-disk file this delegates directly to
    `ndarray.tofile`.  Otherwise a slower Python implementation is used.
    """
def _array_to_file_like(arr, fileobj) -> None:
    """
    Write a `~numpy.ndarray` to a file-like object (which is not supported by
    `numpy.ndarray.tofile`).
    """
def _write_string(f, s) -> None:
    """
    Write a string to a file, encoding to ASCII if the file is open in binary
    mode, or decoding if the file is open in text mode.
    """
def _convert_array(array, dtype):
    """
    Converts an array to a new dtype--if the itemsize of the new dtype is
    the same as the old dtype and both types are not numeric, a view is
    returned.  Otherwise a new array must be created.
    """
def _pseudo_zero(dtype):
    '''
    Given a numpy dtype, finds its "zero" point, which is exactly in the
    middle of its range.
    '''
def _is_pseudo_integer(dtype): ...
def _is_int(val): ...
def _str_to_num(val):
    """Converts a given string to either an int or a float if necessary."""
def _words_group(s, width):
    """
    Split a long string into parts where each part is no longer than ``strlen``
    and no word is cut into two pieces.  But if there are any single words
    which are longer than ``strlen``, then they will be split in the middle of
    the word.
    """
def _tmp_name(input):
    """
    Create a temporary file name which should not already exist.  Use the
    directory of the input file as the base name of the mkstemp() output.
    """
def _get_array_mmap(array):
    """
    If the array has an mmap.mmap at base of its base chain, return the mmap
    object; otherwise return None.
    """
def _free_space_check(hdulist, dirname: Incomplete | None = None) -> Generator[None]: ...
def _extract_number(value, default):
    """
    Attempts to extract an integer number from the given value. If the
    extraction fails, the value of the 'default' argument is returned.
    """
def get_testdata_filepath(filename):
    """
    Return a string representing the path to the file requested from the
    io.fits test data set.

    .. versionadded:: 2.0.3

    Parameters
    ----------
    filename : str
        The filename of the test data file.

    Returns
    -------
    filepath : str
        The path to the requested file.
    """
def _rstrip_inplace(array):
    """
    Performs an in-place rstrip operation on string arrays. This is necessary
    since the built-in `np.char.rstrip` in Numpy does not perform an in-place
    calculation.
    """
def _is_dask_array(data):
    """Check whether data is a dask array."""

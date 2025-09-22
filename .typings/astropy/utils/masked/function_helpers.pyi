from _typeshed import Incomplete

__all__ = ['MASKED_SAFE_FUNCTIONS', 'APPLY_TO_BOTH_FUNCTIONS', 'DISPATCHED_FUNCTIONS', 'UNSUPPORTED_FUNCTIONS', 'bincount', 'broadcast_arrays', 'broadcast_to', 'choose', 'copyto', 'count_nonzero', 'empty_like', 'full_like', 'insert', 'interp', 'lexsort', 'nanargmax', 'nanargmin', 'nancumprod', 'nancumsum', 'nanmax', 'nanmean', 'nanmedian', 'nanmin', 'nanpercentile', 'nanprod', 'nanquantile', 'nanstd', 'nansum', 'nanvar', 'ones_like', 'piecewise', 'place', 'put', 'putmask', 'select', 'zeros_like']

MASKED_SAFE_FUNCTIONS: Incomplete
APPLY_TO_BOTH_FUNCTIONS: Incomplete
DISPATCHED_FUNCTIONS: Incomplete
UNSUPPORTED_FUNCTIONS: Incomplete

def broadcast_to(array, shape, subok: bool = False):
    """Broadcast array to the given shape.

    Like `numpy.broadcast_to`, and applied to both unmasked data and mask.
    Note that ``subok`` is taken to mean whether or not subclasses of
    the unmasked data and mask are allowed, i.e., for ``subok=False``,
    a `~astropy.utils.masked.MaskedNDArray` will be returned.
    """
def empty_like(prototype, dtype: Incomplete | None = None, order: str = 'K', subok: bool = True, shape: Incomplete | None = None, *, device: Incomplete | None = None):
    """Return a new array with the same shape and type as a given array.

        Like `numpy.empty_like`, but will add an empty mask.
        """
def zeros_like(a, dtype: Incomplete | None = None, order: str = 'K', subok: bool = True, shape: Incomplete | None = None, *, device: Incomplete | None = None):
    """Return an array of zeros with the same shape and type as a given array.

        Like `numpy.zeros_like`, but will add an all-false mask.
        """
def ones_like(a, dtype: Incomplete | None = None, order: str = 'K', subok: bool = True, shape: Incomplete | None = None, *, device: Incomplete | None = None):
    """Return an array of ones with the same shape and type as a given array.

        Like `numpy.ones_like`, but will add an all-false mask.
        """
def full_like(a, fill_value, dtype: Incomplete | None = None, order: str = 'K', subok: bool = True, shape: Incomplete | None = None, *, device: Incomplete | None = None):
    """Return a full array with the same shape and type as a given array.

        Like `numpy.full_like`, but with a mask that is also set.
        If ``fill_value`` is `numpy.ma.masked`, the data will be left unset
        (i.e., as created by `numpy.empty_like`).
        """
def put(a, ind, v, mode: str = 'raise') -> None:
    """Replaces specified elements of an array with given values.

    Like `numpy.put`, but for masked array ``a`` and possibly masked
    value ``v``.  Masked indices ``ind`` are not supported.
    """
def putmask(a, mask, values) -> None:
    """Changes elements of an array based on conditional and input values.

    Like `numpy.putmask`, but for masked array ``a`` and possibly masked
    ``values``.  Masked ``mask`` is not supported.
    """
def place(arr, mask, vals) -> None:
    """Change elements of an array based on conditional and input values.

    Like `numpy.place`, but for masked array ``a`` and possibly masked
    ``values``.  Masked ``mask`` is not supported.
    """
def copyto(dst, src, casting: str = 'same_kind', where: bool = True) -> None:
    """Copies values from one array to another, broadcasting as necessary.

    Like `numpy.copyto`, but for masked destination ``dst`` and possibly
    masked source ``src``.
    """
def bincount(x, weights: Incomplete | None = None, minlength: int = 0):
    """Count number of occurrences of each value in array of non-negative ints.

    Like `numpy.bincount`, but masked entries in ``x`` will be skipped.
    Any masked entries in ``weights`` will lead the corresponding bin to
    be masked.
    """
def broadcast_arrays(*args, subok: bool = False):
    """Broadcast arrays to a common shape.

    Like `numpy.broadcast_arrays`, applied to both unmasked data and masks.
    Note that ``subok`` is taken to mean whether or not subclasses of
    the unmasked data and masks are allowed, i.e., for ``subok=False``,
    `~astropy.utils.masked.MaskedNDArray` instances will be returned.
    """
def insert(arr, obj, values, axis: Incomplete | None = None):
    """Insert values along the given axis before the given indices.

    Like `numpy.insert` but for possibly masked ``arr`` and ``values``.
    Masked ``obj`` is not supported.
    """
def count_nonzero(a, axis: Incomplete | None = None, *, keepdims: bool = False):
    """Counts the number of non-zero values in the array ``a``.

    Like `numpy.count_nonzero`, with masked values counted as 0 or `False`.
    """
def choose(a, choices, out: Incomplete | None = None, mode: str = 'raise'):
    """Construct an array from an index array and a set of arrays to choose from.

    Like `numpy.choose`.  Masked indices in ``a`` will lead to masked output
    values and underlying data values are ignored if out of bounds (for
    ``mode='raise'``).  Any values masked in ``choices`` will be propagated
    if chosen.

    """
def select(condlist, choicelist, default: int = 0):
    """Return an array drawn from elements in choicelist, depending on conditions.

    Like `numpy.select`, with masks in ``choicelist`` are propagated.
    Any masks in ``condlist`` are ignored.

    """
def piecewise(x, condlist, funclist, *args, **kw):
    """Evaluate a piecewise-defined function.

    Like `numpy.piecewise` but for masked input array ``x``.
    Any masks in ``condlist`` are ignored.

    """
def interp(x, xp, fp, *args, **kwargs):
    """One-dimensional linear interpolation.

    Like `numpy.interp`, but any masked points in ``xp`` and ``fp``
    are ignored.  Any masked values in ``x`` will still be evaluated,
    but masked on output.
    """
def lexsort(keys, axis: int = -1):
    """Perform an indirect stable sort using a sequence of keys.

    Like `numpy.lexsort` but for possibly masked ``keys``.  Masked
    values are sorted towards the end for each key.
    """

class MaskedFormat:
    """Formatter for masked array scalars.

    For use in `numpy.array2string`, wrapping the regular formatters such
    that if a value is masked, its formatted string is replaced.

    Typically initialized using the ``from_data`` class method.
    """
    format_function: Incomplete
    def __init__(self, format_function) -> None: ...
    def __call__(self, x): ...
    @classmethod
    def from_data(cls, data, **options): ...

# Names in __all__ with no definition:
#   nanargmax
#   nanargmin
#   nancumprod
#   nancumsum
#   nanmax
#   nanmean
#   nanmedian
#   nanmin
#   nanpercentile
#   nanprod
#   nanquantile
#   nanstd
#   nansum
#   nanvar

import abc
import numpy as np
from _typeshed import Incomplete
from astropy.utils.data_info import ParentDtypeInfo
from astropy.utils.shapes import NDArrayShapeMethods, ShapedLikeNDArray

__all__ = ['Masked', 'MaskableShapedLikeNDArray', 'MaskedNDArray', 'get_data_and_mask', 'combine_masks']

def get_data_and_mask(array):
    """Split possibly masked array into unmasked and mask.

    Parameters
    ----------
    array : array-like
        Possibly masked item, judged by whether it has a ``mask`` attribute.
        If so, checks for having an ``unmasked`` attribute (as expected for
        instances of `~astropy.utils.masked.Masked`), or uses the ``_data``
        attribute if the inpuit is an instance of `~numpy.ma.MaskedArray`.

    Returns
    -------
    unmasked, mask : array-like
        If the input array had no mask, this will be ``array, None``.

    Raises
    ------
    AttributeError
        If ``array`` has a ``mask`` but not an ``unmasked`` attribute, and
        is not an instance of `~numpy.ma.MaskedArray`.
    ValueError
        If ``array`` is ``np.ma.masked`` (since it has no data).

    """
def combine_masks(masks, *, out: Incomplete | None = None, where: bool = True, copy: bool = True):
    """Combine masks, possibly storing it in some output.

    Parameters
    ----------
    masks : tuple of array of bool or False or None
        Input masks.  Any that are `None` or `False` are ignored.
        Should broadcast to each other.  For structured dtype,
        an element is considered masked if any of the fields is.
    out : array, optional
        Possible output array to hold the result.
    where : array of bool, optional
        Which elements of the output array to fill.
    copy : bool optional
        Whether to ensure a copy is made. Only relevant if just a
        single input mask is not `None`, and ``out`` is not given.

    Returns
    -------
    mask : array
        Combined mask.
    """

class Masked(NDArrayShapeMethods):
    """A scalar value or array of values with associated mask.

    The resulting instance will take its exact type from whatever the
    contents are, with the type generated on the fly as needed.

    Parameters
    ----------
    data : array-like
        The data for which a mask is to be added.  The result will be a
        a subclass of the type of ``data``.
    mask : array-like of bool, optional
        The initial mask to assign.  If not given, taken from the data.
        If the data already has a mask, the masks are combined.
    copy : bool
        Whether the data and mask should be copied. Default: `False`.

    """
    _base_classes: Incomplete
    _masked_classes: Incomplete
    def __new__(cls, *args, **kwargs): ...
    def __init_subclass__(cls, base_cls: Incomplete | None = None, data_cls: Incomplete | None = None, **kwargs) -> None:
        """Register a Masked subclass.

        Parameters
        ----------
        base_cls : type, optional
            If given, it is taken to mean that ``cls`` can be used as
            a base for masked versions of all subclasses of ``base_cls``,
            so it is registered as such in ``_base_classes``.
        data_cls : type, optional
            If given, ``cls`` should will be registered as the masked version of
            ``data_cls``.  Will set the private ``cls._data_cls`` attribute,
            and auto-generate a docstring if not present already.
        **kwargs
            Passed on for possible further initialization by superclasses.

        """
    @classmethod
    def from_unmasked(cls, data, mask: Incomplete | None = None, copy=...):
        """Create an instance from unmasked data and a mask."""
    @classmethod
    def _get_masked_instance(cls, data, mask: Incomplete | None = None, copy=...): ...
    @classmethod
    def _get_masked_cls(cls, data_cls):
        """Get the masked wrapper for a given data class.

        If the data class does not exist yet but is a subclass of any of the
        registered base data classes, it is automatically generated
        (except we skip `~numpy.ma.MaskedArray` subclasses, since then the
        masking mechanisms would interfere).
        """
    def _get_mask(self):
        """The mask.

        If set, replace the original mask, with whatever it is set with,
        using a view if no broadcasting or type conversion is required.
        """
    _mask: Incomplete
    def _set_mask(self, mask, copy: bool = False) -> None: ...
    mask: Incomplete
    @property
    def unmasked(self):
        """The unmasked values.

        See Also
        --------
        astropy.utils.masked.Masked.filled
        """
    def filled(self, fill_value):
        """Get a copy of the underlying data, with masked values filled in.

        Parameters
        ----------
        fill_value : object
            Value to replace masked values with.

        See Also
        --------
        astropy.utils.masked.Masked.unmasked
        """
    def _apply(self, method, *args, **kwargs): ...
    def __setitem__(self, item, value) -> None: ...

class MaskableShapedLikeNDArray(ShapedLikeNDArray, metaclass=abc.ABCMeta):
    """Like ShapedLikeNDArray, but for classes that can work with masked data.

    Defines default unmasked property as well as a filled method, and inherits
    private class methods that help deal with masked inputs.

    Any class using this must provide a masked property, which tells whether
    the underlying data are Masked, as well as a mask property, which
    generally should provide a read-only copy of the underlying mask.

    """
    @property
    @abc.abstractmethod
    def masked(self):
        """Whether or not the instance uses masked values."""
    @property
    @abc.abstractmethod
    def mask(self):
        """The mask."""
    @property
    def unmasked(self):
        """Get an instance without the mask.

        Note that while one gets a new instance, the underlying data will be shared.

        See Also
        --------
        filled : get a copy of the underlying data, with masked values filled in.
        """
    def filled(self, fill_value):
        """Get a copy of the underlying data, with masked values filled in.

        Parameters
        ----------
        fill_value : object
            Value to replace masked values with.

        Returns
        -------
        filled : instance
            Copy of ``self`` with masked items replaced by ``fill_value``.

        See Also
        --------
        unmasked : get an instance without the mask.
        """

class MaskedInfoBase:
    mask_val: Incomplete
    serialize_method: Incomplete
    def __init__(self, bound: bool = False) -> None: ...

class MaskedNDArrayInfo(MaskedInfoBase, ParentDtypeInfo):
    """
    Container for meta information like name, description, format.
    """
    attr_names: Incomplete
    _represent_as_dict_primary_data: str
    def _represent_as_dict(self): ...
    def _construct_from_dict(self, map): ...

class MaskedArraySubclassInfo(MaskedInfoBase):
    """Mixin class to create a subclasses such as MaskedQuantityInfo."""
    def _represent_as_dict(self): ...

class MaskedIterator:
    """
    Flat iterator object to iterate over Masked Arrays.

    A `~astropy.utils.masked.MaskedIterator` iterator is returned by ``m.flat``
    for any masked array ``m``.  It allows iterating over the array as if it
    were a 1-D array, either in a for-loop or by calling its `next` method.

    Iteration is done in C-contiguous style, with the last index varying the
    fastest. The iterator can also be indexed using basic slicing or
    advanced indexing.

    Notes
    -----
    The design of `~astropy.utils.masked.MaskedIterator` follows that of
    `~numpy.ma.core.MaskedIterator`.  It is not exported by the
    `~astropy.utils.masked` module.  Instead of instantiating directly,
    use the ``flat`` method in the masked array instance.
    """
    _masked: Incomplete
    _dataiter: Incomplete
    _maskiter: Incomplete
    def __init__(self, m) -> None: ...
    def __iter__(self): ...
    def __getitem__(self, indx): ...
    def __setitem__(self, index, value) -> None: ...
    def __next__(self):
        """
        Return the next value, or raise StopIteration.
        """
    next = __next__

class MaskedNDArray(Masked, np.ndarray, base_cls=np.ndarray, data_cls=np.ndarray):
    _mask: Incomplete
    info: Incomplete
    mask: Incomplete
    def __new__(cls, *args, mask: Incomplete | None = None, **kwargs):
        """Get data class instance from arguments and then set mask."""
    def __init_subclass__(cls, **kwargs): ...
    @classmethod
    def from_unmasked(cls, data, mask: Incomplete | None = None, copy=...): ...
    @property
    def unmasked(self): ...
    @classmethod
    def _get_masked_cls(cls, data_cls): ...
    @property
    def flat(self):
        """A 1-D iterator over the Masked array.

        This returns a ``MaskedIterator`` instance, which behaves the same
        as the `~numpy.flatiter` instance returned by `~numpy.ndarray.flat`,
        and is similar to Python's built-in iterator, except that it also
        allows assignment.
        """
    @property
    def _baseclass(self):
        """Work-around for MaskedArray initialization.

        Allows the base class to be inferred correctly when a masked instance
        is used to initialize (or viewed as) a `~numpy.ma.MaskedArray`.

        """
    def view(self, dtype: Incomplete | None = None, type: Incomplete | None = None):
        """New view of the masked array.

        Like `numpy.ndarray.view`, but always returning a masked array subclass.
        """
    def __array_finalize__(self, obj) -> None: ...
    @property
    def shape(self):
        """The shape of the data and the mask.

        Usually used to get the current shape of an array, but may also be
        used to reshape the array in-place by assigning a tuple of array
        dimensions to it.  As with `numpy.reshape`, one of the new shape
        dimensions can be -1, in which case its value is inferred from the
        size of the array and the remaining dimensions.

        Raises
        ------
        AttributeError
            If a copy is required, of either the data or the mask.

        """
    @shape.setter
    def shape(self, shape) -> None: ...
    _eq_simple: Incomplete
    _ne_simple: Incomplete
    __lt__: Incomplete
    __le__: Incomplete
    __gt__: Incomplete
    __ge__: Incomplete
    def __eq__(self, other): ...
    def __ne__(self, other): ...
    @staticmethod
    def _get_data_and_masks(arrays):
        """Extracts the data and masks from the given arrays.

        Parameters
        ----------
        arrays : iterable of array
            An iterable of arrays, possibly masked.

        Returns
        -------
        datas, masks: tuple of array
            Extracted data and mask arrays. For any input array without
            a mask, the corresponding entry in ``masks`` is `None`.
        """
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs): ...
    def __array_function__(self, function, types, args, kwargs): ...
    def _not_implemented_or_raise(self, function, types): ...
    def _masked_result(self, result, mask, out): ...
    def __array_wrap__(self, obj, context: Incomplete | None = None, return_scalar: bool = False): ...
    def _reduce_defaults(self, kwargs, initial_func: Incomplete | None = None):
        """Get default where and initial for masked reductions.

        Generally, the default should be to skip all masked elements.  For
        reductions such as np.minimum.reduce, we also need an initial value,
        which can be determined using ``initial_func``.

        """
    def trace(self, offset: int = 0, axis1: int = 0, axis2: int = 1, dtype: Incomplete | None = None, out: Incomplete | None = None): ...
    def min(self, axis: Incomplete | None = None, out: Incomplete | None = None, **kwargs): ...
    def max(self, axis: Incomplete | None = None, out: Incomplete | None = None, **kwargs): ...
    def ptp(self, axis: Incomplete | None = None, out: Incomplete | None = None, **kwargs): ...
    def nonzero(self): ...
    def compress(self, condition, axis: Incomplete | None = None, out: Incomplete | None = None): ...
    def repeat(self, repeats, axis: Incomplete | None = None): ...
    def choose(self, choices, out: Incomplete | None = None, mode: str = 'raise'): ...
    def argmin(self, axis: Incomplete | None = None, out: Incomplete | None = None, *, keepdims: bool = False): ...
    def argmax(self, axis: Incomplete | None = None, out: Incomplete | None = None, *, keepdims: bool = False): ...
    def argsort(self, axis: int = -1, kind: Incomplete | None = None, order: Incomplete | None = None, *, stable: Incomplete | None = None):
        """Returns the indices that would sort an array.

        Perform an indirect sort along the given axis on both the array
        and the mask, with masked items being sorted to the end.

        Parameters
        ----------
        axis : int or None, optional
            Axis along which to sort.  The default is -1 (the last axis).
            If None, the flattened array is used.
        kind : str or None, ignored.
            The kind of sort.  Present only to allow subclasses to work.
        order : str or list of str.
            For an array with fields defined, the fields to compare first,
            second, etc.  A single field can be specified as a string, and not
            all fields need be specified, but unspecified fields will still be
            used, in dtype order, to break ties.
        stable: bool, keyword-only, ignored
            Sort stability. Present only to allow subclasses to work.

        Returns
        -------
        index_array : ndarray, int
            Array of indices that sorts along the specified ``axis``.  Use
            ``np.take_along_axis(self, index_array, axis=axis)`` to obtain
            the sorted array.

        """
    def sort(self, axis: int = -1, kind: Incomplete | None = None, order: Incomplete | None = None, *, stable: bool = False) -> None:
        """Sort an array in-place. Refer to `numpy.sort` for full documentation.

        Notes
        -----
        Masked items will be sorted to the end. The implementation
        is via `numpy.lexsort` and thus ignores the ``kind`` and ``stable`` arguments;
        they are present only so that subclasses can pass them on.
        """
    def argpartition(self, kth, axis: int = -1, kind: str = 'introselect', order: Incomplete | None = None): ...
    def partition(self, kth, axis: int = -1, kind: str = 'introselect', order: Incomplete | None = None): ...
    def cumsum(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None): ...
    def cumprod(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None): ...
    def clip(self, min: Incomplete | None = None, max: Incomplete | None = None, out: Incomplete | None = None, **kwargs):
        """Return an array whose values are limited to ``[min, max]``.

        Like `~numpy.clip`, but any masked values in ``min`` and ``max``
        are ignored for clipping.  The mask of the input array is propagated.
        """
    def mean(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None, keepdims: bool = False, *, where: bool = True): ...
    def var(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None, ddof: int = 0, keepdims: bool = False, *, where: bool = True): ...
    def std(self, axis: Incomplete | None = None, dtype: Incomplete | None = None, out: Incomplete | None = None, ddof: int = 0, keepdims: bool = False, *, where: bool = True): ...
    def __bool__(self) -> bool: ...
    def any(self, axis: Incomplete | None = None, out: Incomplete | None = None, keepdims: bool = False, *, where: bool = True): ...
    def all(self, axis: Incomplete | None = None, out: Incomplete | None = None, keepdims: bool = False, *, where: bool = True): ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __format__(self, format_spec) -> str: ...
    def __hash__(self): ...

class MaskedRecarrayInfo(MaskedNDArrayInfo):
    def _represent_as_dict(self): ...

class MaskedRecarray(np.recarray, MaskedNDArray, data_cls=np.recarray):
    info: Incomplete
    def __array_finalize__(self, obj) -> None: ...
    def getfield(self, dtype, offset: int = 0): ...
    def setfield(self, val, dtype, offset: int = 0) -> None: ...
    def __repr__(self) -> str: ...

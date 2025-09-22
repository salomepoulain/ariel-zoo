import abc
import numpy as np
from _typeshed import Incomplete
from collections.abc import Sequence
from numpy.typing import NDArray
from types import EllipsisType
from typing import Self, TypeVar

__all__ = ['NDArrayShapeMethods', 'ShapedLikeNDArray', 'check_broadcast', 'IncompatibleShapeError', 'simplify_basic_index', 'unbroadcast']

DT = TypeVar('DT', bound=np.generic)

class NDArrayShapeMethods:
    """Mixin class to provide shape-changing methods.

    The class proper is assumed to have some underlying data, which are arrays
    or array-like structures. It must define a ``shape`` property, which gives
    the shape of those data, as well as an ``_apply`` method that creates a new
    instance in which a `~numpy.ndarray` method has been applied to those.

    Furthermore, for consistency with `~numpy.ndarray`, it is recommended to
    define a setter for the ``shape`` property, which, like the
    `~numpy.ndarray.shape` property allows in-place reshaping the internal data
    (and, unlike the ``reshape`` method raises an exception if this is not
    possible).

    This class only provides the shape-changing methods and is meant in
    particular for `~numpy.ndarray` subclasses that need to keep track of
    other arrays.  For other classes, `~astropy.utils.shapes.ShapedLikeNDArray`
    is recommended.

    """
    def __getitem__(self, item): ...
    def copy(self, *args, **kwargs):
        """Return an instance containing copies of the internal data.

        Parameters are as for :meth:`~numpy.ndarray.copy`.
        """
    def reshape(self, *args, **kwargs):
        """Returns an instance containing the same data with a new shape.

        Parameters are as for :meth:`~numpy.ndarray.reshape`.  Note that it is
        not always possible to change the shape of an array without copying the
        data (see :func:`~numpy.reshape` documentation). If you want an error
        to be raise if the data is copied, you should assign the new shape to
        the shape attribute (note: this may not be implemented for all classes
        using ``NDArrayShapeMethods``).
        """
    def ravel(self, *args, **kwargs):
        """Return an instance with the array collapsed into one dimension.

        Parameters are as for :meth:`~numpy.ndarray.ravel`. Note that it is
        not always possible to unravel an array without copying the data.
        If you want an error to be raise if the data is copied, you should
        should assign shape ``(-1,)`` to the shape attribute.
        """
    def flatten(self, *args, **kwargs):
        """Return a copy with the array collapsed into one dimension.

        Parameters are as for :meth:`~numpy.ndarray.flatten`.
        """
    def transpose(self, *args, **kwargs):
        """Return an instance with the data transposed.

        Parameters are as for :meth:`~numpy.ndarray.transpose`.  All internal
        data are views of the data of the original.
        """
    @property
    def T(self) -> Self:
        """Return an instance with the data transposed.

        Parameters are as for :attr:`~numpy.ndarray.T`.  All internal
        data are views of the data of the original.
        """
    def swapaxes(self, *args, **kwargs):
        """Return an instance with the given axes interchanged.

        Parameters are as for :meth:`~numpy.ndarray.swapaxes`:
        ``axis1, axis2``.  All internal data are views of the data of the
        original.
        """
    def diagonal(self, *args, **kwargs):
        """Return an instance with the specified diagonals.

        Parameters are as for :meth:`~numpy.ndarray.diagonal`.  All internal
        data are views of the data of the original.
        """
    def squeeze(self, *args, **kwargs):
        """Return an instance with single-dimensional shape entries removed.

        Parameters are as for :meth:`~numpy.ndarray.squeeze`.  All internal
        data are views of the data of the original.
        """
    def take(self, indices, axis: Incomplete | None = None, out: Incomplete | None = None, mode: str = 'raise'):
        """Return a new instance formed from the elements at the given indices.

        Parameters are as for :meth:`~numpy.ndarray.take`, except that,
        obviously, no output array can be given.
        """

class ShapedLikeNDArray(NDArrayShapeMethods, metaclass=abc.ABCMeta):
    """Mixin class to provide shape-changing methods.

    The class proper is assumed to have some underlying data, which are arrays
    or array-like structures. It must define a ``shape`` property, which gives
    the shape of those data, as well as an ``_apply`` method that creates a new
    instance in which a `~numpy.ndarray` method has been applied to those.

    Furthermore, for consistency with `~numpy.ndarray`, it is recommended to
    define a setter for the ``shape`` property, which, like the
    `~numpy.ndarray.shape` property allows in-place reshaping the internal data
    (and, unlike the ``reshape`` method raises an exception if this is not
    possible).

    This class also defines default implementations for ``ndim`` and ``size``
    properties, calculating those from the ``shape``.  These can be overridden
    by subclasses if there are faster ways to obtain those numbers.

    """
    @property
    @abc.abstractmethod
    def shape(self) -> tuple[int, ...]:
        """The shape of the underlying data."""
    @abc.abstractmethod
    def _apply(method, *args, **kwargs):
        """Create a new instance, with ``method`` applied to underlying data.

        The method is any of the shape-changing methods for `~numpy.ndarray`
        (``reshape``, ``swapaxes``, etc.), as well as those picking particular
        elements (``__getitem__``, ``take``, etc.). It will be applied to the
        underlying arrays (e.g., ``jd1`` and ``jd2`` in `~astropy.time.Time`),
        with the results used to create a new instance.

        Parameters
        ----------
        method : str
            Method to be applied to the instance's internal data arrays.
        args : tuple
            Any positional arguments for ``method``.
        kwargs : dict
            Any keyword arguments for ``method``.

        """
    @property
    def ndim(self) -> int:
        """The number of dimensions of the instance and underlying arrays."""
    @property
    def size(self) -> int:
        """The size of the object, as calculated from its shape."""
    @property
    def isscalar(self) -> bool: ...
    def __len__(self) -> int: ...
    def __bool__(self) -> bool:
        """Any instance should evaluate to True, except when it is empty."""
    def __getitem__(self, item): ...
    def __iter__(self): ...
    _APPLICABLE_FUNCTIONS: Incomplete
    _METHOD_FUNCTIONS: Incomplete
    def __array_function__(self, function, types, args, kwargs):
        """Wrap numpy functions that make sense."""

class IncompatibleShapeError(ValueError):
    def __init__(self, shape_a: tuple[int, ...], shape_a_idx: int, shape_b: tuple[int, ...], shape_b_idx: int) -> None: ...

def check_broadcast(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    """
    Determines whether two or more Numpy arrays can be broadcast with each
    other based on their shape tuple alone.

    Parameters
    ----------
    *shapes : tuple
        All shapes to include in the comparison.  If only one shape is given it
        is passed through unmodified.  If no shapes are given returns an empty
        `tuple`.

    Returns
    -------
    broadcast : `tuple`
        If all shapes are mutually broadcastable, returns a tuple of the full
        broadcast shape.
    """
def unbroadcast(array: NDArray[DT]) -> NDArray[DT]:
    """
    Given an array, return a new array that is the smallest subset of the
    original array that can be re-broadcasted back to the original array.

    See https://stackoverflow.com/questions/40845769/un-broadcasting-numpy-arrays
    for more details.
    """
def simplify_basic_index(basic_index: int | slice | Sequence[int | slice | EllipsisType | None], *, shape: Sequence[int]) -> tuple[int | slice, ...]:
    """
    Given a Numpy basic index, return a tuple of integers and slice objects
    with no default values (`None`) if possible.

    If one of the dimensions has a slice and the step is negative and the stop
    value of the slice was originally `None`, the new stop value of the slice
    may still be set to `None`.

    For more information on valid basic indices, see
    https://numpy.org/doc/stable/user/basics.indexing.html#basic-indexing

    Parameters
    ----------
    basic_index
        A valid Numpy basic index
    shape
        The shape of the array being indexed
    """

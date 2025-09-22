from _typeshed import Incomplete
from astropy.stats.funcs import mad_std as mad_std
from astropy.units import Quantity as Quantity
from astropy.utils.compat.optional_deps import HAS_BOTTLENECK as HAS_BOTTLENECK
from collections.abc import Callable as Callable
from numpy.typing import ArrayLike as ArrayLike, NDArray as NDArray

def _move_tuple_axes_last(array: ArrayLike, axis: tuple[int, ...] | None = None) -> ArrayLike:
    """
        Move the specified axes of a NumPy array to the last positions
        and combine them.

        Bottleneck can only take integer axis, not tuple, so this
        function takes all the axes to be operated on and combines them
        into the last dimension of the array so that we can then use
        axis=-1.

        Parameters
        ----------
        array : `~numpy.ndarray`
            The input array.

        axis : tuple of int
            The axes on which to move and combine.

        Returns
        -------
        array_new : `~numpy.ndarray`
            Array with the axes being operated on moved into the last
            dimension.
        """
def _apply_bottleneck(function: Callable, array: ArrayLike, axis: int | tuple[int, ...] | None = None, **kwargs) -> float | NDArray | Quantity:
    """Wrap bottleneck function to handle tuple axis.

        Also takes care to ensure the output is of the expected type,
        i.e., a quantity, numpy array, or numpy scalar.
        """

bn_funcs: Incomplete
np_funcs: Incomplete

def _dtype_dispatch(func_name): ...

nansum: Incomplete
nanmin: Incomplete
nanmax: Incomplete
nanmean: Incomplete
nanmedian: Incomplete
nanstd: Incomplete
nanvar: Incomplete

def nanmadstd(array: ArrayLike, axis: int | tuple[int, ...] | None = None) -> float | NDArray:
    """mad_std function that ignores NaNs by default."""

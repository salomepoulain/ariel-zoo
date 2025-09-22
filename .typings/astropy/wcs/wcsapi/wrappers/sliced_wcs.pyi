from .base import BaseWCSWrapper
from _typeshed import Incomplete

__all__ = ['sanitize_slices', 'SlicedLowLevelWCS']

def sanitize_slices(slices, ndim):
    """
    Given a slice as input sanitise it to an easier to parse format.format.

    This function returns a list ``ndim`` long containing slice objects (or ints).
    """

class SlicedLowLevelWCS(BaseWCSWrapper):
    """
    A Low Level WCS wrapper which applies an array slice to a WCS.

    This class does not modify the underlying WCS object and can therefore drop
    coupled dimensions as it stores which pixel and world dimensions have been
    sliced out (or modified) in the underlying WCS and returns the modified
    results on all the Low Level WCS methods.

    Parameters
    ----------
    wcs : `~astropy.wcs.wcsapi.BaseLowLevelWCS`
        The WCS to slice.
    slices : `slice` or `tuple` or `int`
        A valid array slice to apply to the WCS.

    """
    _wcs: Incomplete
    _slices_array: Incomplete
    _slices_pixel: Incomplete
    _pixel_keep: Incomplete
    _world_keep: Incomplete
    def __init__(self, wcs, slices) -> None: ...
    def dropped_world_dimensions(self):
        """
        Information describing the dropped world dimensions.
        """
    @property
    def pixel_n_dim(self): ...
    @property
    def world_n_dim(self): ...
    @property
    def world_axis_physical_types(self): ...
    @property
    def world_axis_units(self): ...
    @property
    def pixel_axis_names(self): ...
    @property
    def world_axis_names(self): ...
    def _pixel_to_world_values_all(self, *pixel_arrays): ...
    def pixel_to_world_values(self, *pixel_arrays): ...
    def world_to_pixel_values(self, *world_arrays): ...
    @property
    def world_axis_object_components(self): ...
    @property
    def world_axis_object_classes(self): ...
    @property
    def array_shape(self): ...
    @property
    def pixel_shape(self): ...
    @property
    def pixel_bounds(self): ...
    @property
    def axis_correlation_matrix(self): ...

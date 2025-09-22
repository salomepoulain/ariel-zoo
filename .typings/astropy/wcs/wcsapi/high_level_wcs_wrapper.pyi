from .high_level_api import HighLevelWCSMixin
from _typeshed import Incomplete

__all__ = ['HighLevelWCSWrapper']

class HighLevelWCSWrapper(HighLevelWCSMixin):
    """
    Wrapper class that can take any :class:`~astropy.wcs.wcsapi.BaseLowLevelWCS`
    object and expose the high-level WCS API.
    """
    _low_level_wcs: Incomplete
    def __init__(self, low_level_wcs) -> None: ...
    @property
    def low_level_wcs(self): ...
    @property
    def pixel_n_dim(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_n_dim`.
        """
    @property
    def world_n_dim(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_n_dim`.
        """
    @property
    def world_axis_physical_types(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_physical_types`.
        """
    @property
    def world_axis_units(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.world_axis_units`.
        """
    @property
    def array_shape(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.array_shape`.
        """
    @property
    def pixel_bounds(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.pixel_bounds`.
        """
    @property
    def axis_correlation_matrix(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS.axis_correlation_matrix`.
        """
    def _as_mpl_axes(self):
        """
        See `~astropy.wcs.wcsapi.BaseLowLevelWCS._as_mpl_axes`.
        """
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...

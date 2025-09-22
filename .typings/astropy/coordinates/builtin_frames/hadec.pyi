from _typeshed import Incomplete
from astropy.coordinates import representation as r
from astropy.coordinates.baseframe import BaseCoordinateFrame

__all__ = ['HADec']

class HADec(BaseCoordinateFrame):
    """
    A coordinate or frame in the Hour Angle-Declination system (Equatorial
    coordinates) with respect to the WGS84 ellipsoid.  Hour Angle is oriented
    with respect to upper culmination such that the hour angle is negative to
    the East and positive to the West.

    This frame is assumed to *include* refraction effects if the ``pressure``
    frame attribute is non-zero.

    The frame attributes are listed under **Other Parameters**, which are
    necessary for transforming from HADec to some other system.
    """
    frame_specific_representation_info: Incomplete
    default_representation = r.SphericalRepresentation
    default_differential = r.SphericalCosLatDifferential
    obstime: Incomplete
    location: Incomplete
    pressure: Incomplete
    temperature: Incomplete
    relative_humidity: Incomplete
    obswl: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @staticmethod
    def _set_data_lon_wrap_angle(data): ...
    def represent_as(self, base, s: str = 'base', in_frame_units: bool = False):
        """
        Ensure the wrap angle for any spherical
        representations.
        """

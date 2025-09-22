from _typeshed import Incomplete
from astropy.coordinates import representation as r
from astropy.coordinates.baseframe import BaseCoordinateFrame

__all__ = ['AltAz']

class AltAz(BaseCoordinateFrame):
    """
    A coordinate or frame in the Altitude-Azimuth system (Horizontal
    coordinates) with respect to the WGS84 ellipsoid.  Azimuth is oriented
    East of North (i.e., N=0, E=90 degrees).  Altitude is also known as
    elevation angle, so this frame is also in the Azimuth-Elevation system.

    This frame is assumed to *include* refraction effects if the ``pressure``
    frame attribute is non-zero.

    The frame attributes are listed under **Other Parameters**, which are
    necessary for transforming from AltAz to some other system.
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
    @property
    def secz(self):
        """
        Secant of the zenith angle for this coordinate, a common estimate of
        the airmass.
        """
    @property
    def zen(self):
        """
        The zenith angle (or zenith distance / co-altitude) for this coordinate.
        """

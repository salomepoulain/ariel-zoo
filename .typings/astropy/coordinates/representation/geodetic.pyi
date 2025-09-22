from .base import BaseRepresentation as BaseRepresentation
from .cartesian import CartesianRepresentation as CartesianRepresentation
from _typeshed import Incomplete
from astropy.coordinates.angles import Latitude as Latitude, Longitude as Longitude
from astropy.utils.decorators import format_doc as format_doc

ELLIPSOIDS: Incomplete
geodetic_base_doc: str

class BaseGeodeticRepresentation(BaseRepresentation):
    """
    Base class for geodetic representations.

    Subclasses need to set attributes ``_equatorial_radius`` and ``_flattening``
    to quantities holding correct values (with units of length and dimensionless,
    respectively), or alternatively an ``_ellipsoid`` attribute to the relevant ERFA
    index (as passed in to `erfa.eform`). The geodetic latitude is defined by the
    angle between the vertical to the surface at a specific point of the spheroid and
    its projection onto the equatorial plane.
    """
    attr_classes: Incomplete
    def __init_subclass__(cls, **kwargs) -> None: ...
    def __init__(self, lon, lat: Incomplete | None = None, height: Incomplete | None = None, copy: bool = True) -> None: ...
    def to_cartesian(self):
        """
        Converts geodetic coordinates to 3D rectangular (geocentric)
        cartesian coordinates.
        """
    @classmethod
    def from_cartesian(cls, cart):
        """
        Converts 3D rectangular cartesian coordinates (assumed geocentric) to
        geodetic coordinates.
        """

class BaseBodycentricRepresentation(BaseRepresentation):
    """Representation of points in bodycentric 3D coordinates.

    Subclasses need to set attributes ``_equatorial_radius`` and ``_flattening``
    to quantities holding correct values (with units of length and dimensionless,
    respectively). the bodycentric latitude and longitude are spherical latitude
    and longitude relative to the barycenter of the body.
    """
    attr_classes: Incomplete
    def __init_subclass__(cls, **kwargs) -> None: ...
    def __init__(self, lon, lat: Incomplete | None = None, height: Incomplete | None = None, copy: bool = True) -> None: ...
    def to_cartesian(self):
        """
        Converts bodycentric coordinates to 3D rectangular (geocentric)
        cartesian coordinates.
        """
    @classmethod
    def from_cartesian(cls, cart):
        """
        Converts 3D rectangular cartesian coordinates (assumed geocentric) to
        bodycentric coordinates.
        """

class WGS84GeodeticRepresentation(BaseGeodeticRepresentation):
    """Representation of points in WGS84 3D geodetic coordinates."""
    _ellipsoid: str

class WGS72GeodeticRepresentation(BaseGeodeticRepresentation):
    """Representation of points in WGS72 3D geodetic coordinates."""
    _ellipsoid: str

class GRS80GeodeticRepresentation(BaseGeodeticRepresentation):
    """Representation of points in GRS80 3D geodetic coordinates."""
    _ellipsoid: str

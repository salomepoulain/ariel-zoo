from _typeshed import Incomplete
from astropy.coordinates import representation as r
from astropy.coordinates.baseframe import BaseCoordinateFrame

__all__ = ['Galactic']

class Galactic(BaseCoordinateFrame):
    """
    A coordinate or frame in the Galactic coordinate system.

    This frame is used in a variety of Galactic contexts because it has as its
    x-y plane the plane of the Milky Way.  The positive x direction (i.e., the
    l=0, b=0 direction) points to the center of the Milky Way and the z-axis
    points toward the North Galactic Pole (following the IAU's 1958 definition
    [1]_). However, unlike the `~astropy.coordinates.Galactocentric` frame, the
    *origin* of this frame in 3D space is the solar system barycenter, not
    the center of the Milky Way.
    """
    frame_specific_representation_info: Incomplete
    default_representation = r.SphericalRepresentation
    default_differential = r.SphericalCosLatDifferential
    _ngp_B1950: Incomplete
    _lon0_B1950: Incomplete
    _ngp_J2000: Incomplete
    _lon0_J2000: Incomplete

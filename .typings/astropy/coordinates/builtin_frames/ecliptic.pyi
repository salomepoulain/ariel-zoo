from _typeshed import Incomplete
from astropy.coordinates import representation as r
from astropy.coordinates.baseframe import BaseCoordinateFrame

__all__ = ['GeocentricMeanEcliptic', 'BarycentricMeanEcliptic', 'HeliocentricMeanEcliptic', 'BaseEclipticFrame', 'GeocentricTrueEcliptic', 'BarycentricTrueEcliptic', 'HeliocentricTrueEcliptic', 'HeliocentricEclipticIAU76', 'CustomBarycentricEcliptic']

class BaseEclipticFrame(BaseCoordinateFrame):
    '''
    A base class for frames that have names and conventions like that of
    ecliptic frames.

    .. warning::
            In the current version of astropy, the ecliptic frames do not yet have
            stringent accuracy tests.  We recommend you test to "known-good" cases
            to ensure this frames are what you are looking for. (and then ideally
            you would contribute these tests to Astropy!)
    '''
    default_representation = r.SphericalRepresentation
    default_differential = r.SphericalCosLatDifferential

class GeocentricMeanEcliptic(BaseEclipticFrame):
    '''
    Geocentric mean ecliptic coordinates.  These origin of the coordinates are the
    geocenter (Earth), with the x axis pointing to the *mean* (not true) equinox
    at the time specified by the ``equinox`` attribute, and the xy-plane in the
    plane of the ecliptic for that date.

    Be aware that the definition of "geocentric" here means that this frame
    *includes* light deflection from the sun, aberration, etc when transforming
    to/from e.g. ICRS.

    The frame attributes are listed under **Other Parameters**.
    '''
    equinox: Incomplete
    obstime: Incomplete

class GeocentricTrueEcliptic(BaseEclipticFrame):
    '''
    Geocentric true ecliptic coordinates.  These origin of the coordinates are the
    geocenter (Earth), with the x axis pointing to the *true* (not mean) equinox
    at the time specified by the ``equinox`` attribute, and the xy-plane in the
    plane of the ecliptic for that date.

    Be aware that the definition of "geocentric" here means that this frame
    *includes* light deflection from the sun, aberration, etc when transforming
    to/from e.g. ICRS.

    The frame attributes are listed under **Other Parameters**.
    '''
    equinox: Incomplete
    obstime: Incomplete

class BarycentricMeanEcliptic(BaseEclipticFrame):
    """
    Barycentric mean ecliptic coordinates.  These origin of the coordinates are the
    barycenter of the solar system, with the x axis pointing in the direction of
    the *mean* (not true) equinox as at the time specified by the ``equinox``
    attribute (as seen from Earth), and the xy-plane in the plane of the
    ecliptic for that date.

    The frame attributes are listed under **Other Parameters**.
    """
    equinox: Incomplete

class BarycentricTrueEcliptic(BaseEclipticFrame):
    """
    Barycentric true ecliptic coordinates.  These origin of the coordinates are the
    barycenter of the solar system, with the x axis pointing in the direction of
    the *true* (not mean) equinox as at the time specified by the ``equinox``
    attribute (as seen from Earth), and the xy-plane in the plane of the
    ecliptic for that date.

    The frame attributes are listed under **Other Parameters**.
    """
    equinox: Incomplete

class HeliocentricMeanEcliptic(BaseEclipticFrame):
    """
    Heliocentric mean ecliptic coordinates.  These origin of the coordinates are the
    center of the sun, with the x axis pointing in the direction of
    the *mean* (not true) equinox as at the time specified by the ``equinox``
    attribute (as seen from Earth), and the xy-plane in the plane of the
    ecliptic for that date.

    The frame attributes are listed under **Other Parameters**.

    {params}


    """
    equinox: Incomplete
    obstime: Incomplete

class HeliocentricTrueEcliptic(BaseEclipticFrame):
    """
    Heliocentric true ecliptic coordinates.  These origin of the coordinates are the
    center of the sun, with the x axis pointing in the direction of
    the *true* (not mean) equinox as at the time specified by the ``equinox``
    attribute (as seen from Earth), and the xy-plane in the plane of the
    ecliptic for that date.

    The frame attributes are listed under **Other Parameters**.

    {params}


    """
    equinox: Incomplete
    obstime: Incomplete

class HeliocentricEclipticIAU76(BaseEclipticFrame):
    """
    Heliocentric mean (IAU 1976) ecliptic coordinates.  These origin of the coordinates are the
    center of the sun, with the x axis pointing in the direction of
    the *mean* (not true) equinox of J2000, and the xy-plane in the plane of the
    ecliptic of J2000 (according to the IAU 1976/1980 obliquity model).
    It has, therefore, a fixed equinox and an older obliquity value
    than the rest of the frames.

    The frame attributes are listed under **Other Parameters**.

    {params}


    """
    obstime: Incomplete

class CustomBarycentricEcliptic(BaseEclipticFrame):
    """
    Barycentric ecliptic coordinates with custom obliquity.
    These origin of the coordinates are the
    barycenter of the solar system, with the x axis pointing in the direction of
    the *mean* (not true) equinox of J2000, and the xy-plane in the plane of the
    ecliptic tilted a custom obliquity angle.

    The frame attributes are listed under **Other Parameters**.
    """
    obliquity: Incomplete

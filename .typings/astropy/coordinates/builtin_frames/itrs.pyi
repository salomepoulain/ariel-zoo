from _typeshed import Incomplete
from astropy.coordinates.baseframe import BaseCoordinateFrame
from astropy.coordinates.representation import CartesianDifferential, CartesianRepresentation

__all__ = ['ITRS']

class ITRS(BaseCoordinateFrame):
    '''
    A coordinate or frame in the International Terrestrial Reference System
    (ITRS).  This is approximately a geocentric system, although strictly it is
    defined by a series of reference locations near the surface of the Earth (the ITRF).
    For more background on the ITRS, see the references provided in the
    :ref:`astropy:astropy-coordinates-seealso` section of the documentation.

    This frame also includes frames that are defined *relative* to the center of the Earth,
    but that are offset (in both position and velocity) from the center of the Earth. You
    may see such non-geocentric coordinates referred to as "topocentric".

    Topocentric ITRS frames are convenient for observations of near Earth objects where
    stellar aberration is not included. One can merely subtract the observing site\'s
    EarthLocation geocentric ITRS coordinates from the object\'s geocentric ITRS coordinates,
    put the resulting vector into a topocentric ITRS frame and then transform to
    `~astropy.coordinates.AltAz` or `~astropy.coordinates.HADec`. The other way around is
    to transform an observed `~astropy.coordinates.AltAz` or `~astropy.coordinates.HADec`
    position to a topocentric ITRS frame and add the observing site\'s EarthLocation geocentric
    ITRS coordinates to yield the object\'s geocentric ITRS coordinates.

    On the other hand, using ``transform_to`` to transform geocentric ITRS coordinates to
    topocentric ITRS, observed `~astropy.coordinates.AltAz`, or observed
    `~astropy.coordinates.HADec` coordinates includes the difference between stellar aberration
    from the point of view of an observer at the geocenter and stellar aberration from the
    point of view of an observer on the surface of the Earth. If the geocentric ITRS
    coordinates of the object include stellar aberration at the geocenter (e.g. certain ILRS
    ephemerides), then this is the way to go.

    Note to ILRS ephemeris users: Astropy does not currently consider relativistic
    effects of the Earth\'s gravatational field. Nor do the `~astropy.coordinates.AltAz`
    or `~astropy.coordinates.HADec` refraction corrections compute the change in the
    range due to the curved path of light through the atmosphere, so Astropy is no
    substitute for the ILRS software in these respects.

    '''
    default_representation = CartesianRepresentation
    default_differential = CartesianDifferential
    obstime: Incomplete
    location: Incomplete
    @property
    def earth_location(self):
        """
        The data in this frame as an `~astropy.coordinates.EarthLocation` class.
        """

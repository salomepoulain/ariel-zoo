from .baseradec import BaseRADecFrame
from _typeshed import Incomplete
from astropy.coordinates import representation as r
from astropy.coordinates.baseframe import BaseCoordinateFrame

__all__ = ['LSR', 'GalacticLSR', 'LSRK', 'LSRD']

class LSR(BaseRADecFrame):
    '''A coordinate or frame in the Local Standard of Rest (LSR).

    This coordinate frame is axis-aligned and co-spatial with
    `~astropy.coordinates.ICRS`, but has a velocity offset relative to the
    solar system barycenter to remove the peculiar motion of the sun relative
    to the LSR. Roughly, the LSR is the mean velocity of the stars in the solar
    neighborhood, but the precise definition of which depends on the study. As
    defined in Schönrich et al. (2010): "The LSR is the rest frame at the
    location of the Sun of a star that would be on a circular orbit in the
    gravitational potential one would obtain by azimuthally averaging away
    non-axisymmetric features in the actual Galactic potential." No such orbit
    truly exists, but it is still a commonly used velocity frame.

    We use default values from Schönrich et al. (2010) for the barycentric
    velocity relative to the LSR, which is defined in Galactic (right-handed)
    cartesian velocity components
    :math:`(U, V, W) = (11.1, 12.24, 7.25)~{{\\rm km}}~{{\\rm s}}^{{-1}}`. These
    values are customizable via the ``v_bary`` argument which specifies the
    velocity of the solar system barycenter with respect to the LSR.

    The frame attributes are listed under **Other Parameters**.

    '''
    v_bary: Incomplete

class GalacticLSR(BaseCoordinateFrame):
    '''A coordinate or frame in the Local Standard of Rest (LSR), axis-aligned
    to the Galactic frame.

    This coordinate frame is axis-aligned and co-spatial with
    `~astropy.coordinates.ICRS`, but has a velocity offset relative to the
    solar system barycenter to remove the peculiar motion of the sun relative
    to the LSR. Roughly, the LSR is the mean velocity of the stars in the solar
    neighborhood, but the precise definition of which depends on the study. As
    defined in Schönrich et al. (2010): "The LSR is the rest frame at the
    location of the Sun of a star that would be on a circular orbit in the
    gravitational potential one would obtain by azimuthally averaging away
    non-axisymmetric features in the actual Galactic potential." No such orbit
    truly exists, but it is still a commonly used velocity frame.

    We use default values from Schönrich et al. (2010) for the barycentric
    velocity relative to the LSR, which is defined in Galactic (right-handed)
    cartesian velocity components
    :math:`(U, V, W) = (11.1, 12.24, 7.25)~{{\\rm km}}~{{\\rm s}}^{{-1}}`. These
    values are customizable via the ``v_bary`` argument which specifies the
    velocity of the solar system barycenter with respect to the LSR.

    The frame attributes are listed under **Other Parameters**.

    '''
    frame_specific_representation_info: Incomplete
    default_representation = r.SphericalRepresentation
    default_differential = r.SphericalCosLatDifferential
    v_bary: Incomplete

class LSRK(BaseRADecFrame):
    """A coordinate or frame in the Kinematic Local Standard of Rest (LSR).

    This frame is defined as having a velocity of 20 km/s towards RA=270 Dec=30
    (B1900) relative to the solar system Barycenter. This is defined in:

        Gordon 1975, Methods of Experimental Physics: Volume 12:
        Astrophysics, Part C: Radio Observations - Section 6.1.5.

    This coordinate frame is axis-aligned and co-spatial with
    `~astropy.coordinates.ICRS`, but has a velocity offset relative to the
    solar system barycenter to remove the peculiar motion of the sun relative
    to the LSRK.

    """
class LSRD(BaseRADecFrame):
    """A coordinate or frame in the Dynamical Local Standard of Rest (LSRD).

    This frame is defined as a velocity of U=9 km/s, V=12 km/s,
    and W=7 km/s in Galactic coordinates or 16.552945 km/s
    towards l=53.13 b=25.02. This is defined in:

       Delhaye 1965, Solar Motion and Velocity Distribution of
       Common Stars.

    This coordinate frame is axis-aligned and co-spatial with
    `~astropy.coordinates.ICRS`, but has a velocity offset relative to the
    solar system barycenter to remove the peculiar motion of the sun relative
    to the LSRD.

    """

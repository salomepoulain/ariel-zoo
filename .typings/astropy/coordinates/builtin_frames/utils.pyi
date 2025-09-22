from _typeshed import Incomplete
from astropy.coordinates.earth import EarthLocation as EarthLocation
from astropy.coordinates.representation import CartesianDifferential as CartesianDifferential
from astropy.time import Time as Time
from astropy.utils import iers as iers
from astropy.utils.exceptions import AstropyWarning as AstropyWarning

EQUINOX_J2000: Incomplete
EQUINOX_B1950: Incomplete
DEFAULT_OBSTIME: Incomplete
EARTH_CENTER: Incomplete
PIOVER2: Incomplete
_DEFAULT_PM: Incomplete

def get_polar_motion(time):
    """
    gets the two polar motion components in radians for use with apio.
    """
def _warn_iers(ierserr) -> None:
    """
    Generate a warning for an IERSRangeerror.

    Parameters
    ----------
    ierserr : An `~astropy.utils.iers.IERSRangeError`
    """
def get_dut1utc(time):
    """
    This function is used to get UT1-UTC in coordinates because normally it
    gives an error outside the IERS range, but in coordinates we want to allow
    it to go through but with a warning.
    """
def get_jd12(time, scale):
    """
    Gets ``jd1`` and ``jd2`` from a time object in a particular scale.

    Parameters
    ----------
    time : `~astropy.time.Time`
        The time to get the jds for
    scale : str
        The time scale to get the jds for

    Returns
    -------
    jd1 : float
    jd2 : float
    """
def norm(p):
    """
    Normalise a p-vector.
    """
def pav2pv(p, v):
    """
    Combine p- and v- vectors into a pv-vector.
    """
def get_cip(jd1, jd2):
    """
    Find the X, Y coordinates of the CIP and the CIO locator, s.

    Parameters
    ----------
    jd1 : float or `np.ndarray`
        First part of two part Julian date (TDB)
    jd2 : float or `np.ndarray`
        Second part of two part Julian date (TDB)

    Returns
    -------
    x : float or `np.ndarray`
        x coordinate of the CIP
    y : float or `np.ndarray`
        y coordinate of the CIP
    s : float or `np.ndarray`
        CIO locator, s
    """
def aticq(srepr, astrom):
    """
    A slightly modified version of the ERFA function ``eraAticq``.

    ``eraAticq`` performs the transformations between two coordinate systems,
    with the details of the transformation being encoded into the ``astrom`` array.

    There are two issues with the version of aticq in ERFA. Both are associated
    with the handling of light deflection.

    The companion function ``eraAtciqz`` is meant to be its inverse. However, this
    is not true for directions close to the Solar centre, since the light deflection
    calculations are numerically unstable and therefore not reversible.

    This version sidesteps that problem by artificially reducing the light deflection
    for directions which are within 90 arcseconds of the Sun's position. This is the
    same approach used by the ERFA functions above, except that they use a threshold of
    9 arcseconds.

    In addition, ERFA's aticq assumes a distant source, so there is no difference between
    the object-Sun vector and the observer-Sun vector. This can lead to errors of up to a
    few arcseconds in the worst case (e.g a Venus transit).

    Parameters
    ----------
    srepr : `~astropy.coordinates.SphericalRepresentation`
        Astrometric GCRS or CIRS position of object from observer
    astrom : eraASTROM array
        ERFA astrometry context, as produced by, e.g. ``eraApci13`` or ``eraApcs13``

    Returns
    -------
    rc : float or `~numpy.ndarray`
        Right Ascension in radians
    dc : float or `~numpy.ndarray`
        Declination in radians
    """
def atciqz(srepr, astrom):
    """
    A slightly modified version of the ERFA function ``eraAtciqz``.

    ``eraAtciqz`` performs the transformations between two coordinate systems,
    with the details of the transformation being encoded into the ``astrom`` array.

    There are two issues with the version of atciqz in ERFA. Both are associated
    with the handling of light deflection.

    The companion function ``eraAticq`` is meant to be its inverse. However, this
    is not true for directions close to the Solar centre, since the light deflection
    calculations are numerically unstable and therefore not reversible.

    This version sidesteps that problem by artificially reducing the light deflection
    for directions which are within 90 arcseconds of the Sun's position. This is the
    same approach used by the ERFA functions above, except that they use a threshold of
    9 arcseconds.

    In addition, ERFA's atciqz assumes a distant source, so there is no difference between
    the object-Sun vector and the observer-Sun vector. This can lead to errors of up to a
    few arcseconds in the worst case (e.g a Venus transit).

    Parameters
    ----------
    srepr : `~astropy.coordinates.SphericalRepresentation`
        Astrometric ICRS position of object from observer
    astrom : eraASTROM array
        ERFA astrometry context, as produced by, e.g. ``eraApci13`` or ``eraApcs13``

    Returns
    -------
    ri : float or `~numpy.ndarray`
        Right Ascension in radians
    di : float or `~numpy.ndarray`
        Declination in radians
    """
def prepare_earth_position_vel(time):
    """
    Get barycentric position and velocity, and heliocentric position of Earth.

    Parameters
    ----------
    time : `~astropy.time.Time`
        time at which to calculate position and velocity of Earth

    Returns
    -------
    earth_pv : `np.ndarray`
        Barycentric position and velocity of Earth, in au and au/day
    earth_helio : `np.ndarray`
        Heliocentric position of Earth in au
    """
def get_offset_sun_from_barycenter(time, include_velocity: bool = False, reverse: bool = False):
    """
    Returns the offset of the Sun center from the solar-system barycenter (SSB).

    Parameters
    ----------
    time : `~astropy.time.Time`
        Time at which to calculate the offset
    include_velocity : `bool`
        If ``True``, attach the velocity as a differential.  Defaults to ``False``.
    reverse : `bool`
        If ``True``, return the offset of the barycenter from the Sun.  Defaults to ``False``.

    Returns
    -------
    `~astropy.coordinates.CartesianRepresentation`
        The offset
    """

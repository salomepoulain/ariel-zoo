from .builtin_frames.utils import get_jd12 as get_jd12
from .matrix_utilities import matrix_transpose as matrix_transpose, rotation_matrix as rotation_matrix
from _typeshed import Incomplete
from astropy.time import Time as Time

jd1950: Incomplete
jd2000: Incomplete

def eccentricity(jd):
    """
    Eccentricity of the Earth's orbit at the requested Julian Date.

    Parameters
    ----------
    jd : scalar or array-like
        Julian date at which to compute the eccentricity

    Returns
    -------
    eccentricity : scalar or array
        The eccentricity (or array of eccentricities)

    References
    ----------
    * Explanatory Supplement to the Astronomical Almanac: P. Kenneth
      Seidelmann (ed), University Science Books (1992).
    """
def mean_lon_of_perigee(jd):
    """
    Computes the mean longitude of perigee of the Earth's orbit at the
    requested Julian Date.

    Parameters
    ----------
    jd : scalar or array-like
        Julian date at which to compute the mean longitude of perigee

    Returns
    -------
    mean_lon_of_perigee : scalar or array
        Mean longitude of perigee in degrees (or array of mean longitudes)

    References
    ----------
    * Explanatory Supplement to the Astronomical Almanac: P. Kenneth
      Seidelmann (ed), University Science Books (1992).
    """
def obliquity(jd, algorithm: int = 2006):
    """
    Computes the obliquity of the Earth at the requested Julian Date.

    Parameters
    ----------
    jd : scalar or array-like
        Julian date (TT) at which to compute the obliquity
    algorithm : int
        Year of algorithm based on IAU adoption. Can be 2006, 2000 or 1980.
        The IAU 2006 algorithm is based on Hilton et al. 2006.
        The IAU 1980 algorithm is based on the Explanatory Supplement to the
        Astronomical Almanac (1992).
        The IAU 2000 algorithm starts with the IAU 1980 algorithm and applies a
        precession-rate correction from the IAU 2000 precession model.

    Returns
    -------
    obliquity : scalar or array
        Mean obliquity in degrees (or array of obliquities)

    References
    ----------
    * Hilton, J. et al., 2006, Celest.Mech.Dyn.Astron. 94, 351
    * Capitaine, N., et al., 2003, Astron.Astrophys. 400, 1145-1154
    * Explanatory Supplement to the Astronomical Almanac: P. Kenneth
      Seidelmann (ed), University Science Books (1992).
    """
def precession_matrix_Capitaine(fromepoch, toepoch):
    """
    Computes the precession matrix from one Julian epoch to another, per IAU 2006.

    Parameters
    ----------
    fromepoch : `~astropy.time.Time`
        The epoch to precess from.
    toepoch : `~astropy.time.Time`
        The epoch to precess to.

    Returns
    -------
    pmatrix : 3x3 array
        Precession matrix to get from ``fromepoch`` to ``toepoch``

    References
    ----------
    Hilton, J. et al., 2006, Celest.Mech.Dyn.Astron. 94, 351
    """
def _precession_matrix_besselian(epoch1, epoch2):
    """
    Computes the precession matrix from one Besselian epoch to another using
    Newcomb's method.

    ``epoch1`` and ``epoch2`` are in Besselian year numbers.
    """
def nutation_components2000B(jd):
    """
    Computes nutation components following the IAU 2000B specification.

    Parameters
    ----------
    jd : scalar
        Julian date (TT) at which to compute the nutation components

    Returns
    -------
    eps : float
        epsilon in radians
    dpsi : float
        dpsi in radians
    deps : float
        depsilon in raidans
    """
def nutation_matrix(epoch):
    """
    Nutation matrix generated from nutation components, IAU 2000B model.

    Matrix converts from mean coordinate to true coordinate as
    r_true = M * r_mean

    Parameters
    ----------
    epoch : `~astropy.time.Time`
        The epoch at which to compute the nutation matrix

    Returns
    -------
    nmatrix : 3x3 array
        Nutation matrix for the specified epoch

    References
    ----------
    * Explanatory Supplement to the Astronomical Almanac: P. Kenneth
      Seidelmann (ed), University Science Books (1992).
    """

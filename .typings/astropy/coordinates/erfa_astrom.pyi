from .builtin_frames.utils import get_cip as get_cip, get_jd12 as get_jd12, get_polar_motion as get_polar_motion, pav2pv as pav2pv, prepare_earth_position_vel as prepare_earth_position_vel
from .matrix_utilities import rotation_matrix as rotation_matrix
from _typeshed import Incomplete
from astropy.time import Time as Time
from astropy.utils.exceptions import AstropyWarning as AstropyWarning
from astropy.utils.state import ScienceState as ScienceState

__all__: Incomplete

class ErfaAstrom:
    """
    The default provider for astrometry values.
    A utility class to extract the necessary arguments for
    erfa functions from frame attributes, call the corresponding
    erfa functions and return the astrom object.
    """
    @staticmethod
    def apco(frame_or_coord):
        """
        Wrapper for ``erfa.apco``, used in conversions AltAz <-> ICRS and CIRS <-> ICRS.

        Parameters
        ----------
        frame_or_coord : ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``
            Frame or coordinate instance in the corresponding frame
            for which to calculate the calculate the astrom values.
            For this function, an AltAz or CIRS frame is expected.
        """
    @staticmethod
    def apcs(frame_or_coord):
        """
        Wrapper for ``erfa.apcs``, used in conversions GCRS <-> ICRS.

        Parameters
        ----------
        frame_or_coord : ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``
            Frame or coordinate instance in the corresponding frame
            for which to calculate the calculate the astrom values.
            For this function, a GCRS frame is expected.
        """
    @staticmethod
    def apio(frame_or_coord):
        """
        Slightly modified equivalent of ``erfa.apio``, used in conversions AltAz <-> CIRS.

        Since we use a topocentric CIRS frame, we have dropped the steps needed to calculate
        diurnal aberration.

        Parameters
        ----------
        frame_or_coord : ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``
            Frame or coordinate instance in the corresponding frame
            for which to calculate the calculate the astrom values.
            For this function, an AltAz frame is expected.
        """

class ErfaAstromInterpolator(ErfaAstrom):
    """
    A provider for astrometry values that does not call erfa
    for each individual timestamp but interpolates linearly
    between support points.

    For the interpolation, float64 MJD values are used, so time precision
    for the interpolation will be around a microsecond.

    This can dramatically speed up coordinate transformations,
    e.g. between CIRS and ICRS,
    when obstime is an array of many values (factors of 10 to > 100 depending
    on the selected resolution, number of points and the time range of the values).

    The precision of the transformation will still be in the order of microseconds
    for reasonable values of time_resolution, e.g. ``300 * u.s``.

    Users should benchmark performance and accuracy with the default transformation
    for their specific use case and then choose a suitable ``time_resolution``
    from there.

    This class is intended be used together with the ``erfa_astrom`` science state,
    e.g. in a context manager like this

    Example
    -------
    >>> from astropy.coordinates import SkyCoord, CIRS
    >>> from astropy.coordinates.erfa_astrom import erfa_astrom, ErfaAstromInterpolator
    >>> import astropy.units as u
    >>> from astropy.time import Time
    >>> import numpy as np

    >>> obstime = Time('2010-01-01T20:00:00') + np.linspace(0, 4, 1000) * u.hour
    >>> crab = SkyCoord(ra='05h34m31.94s', dec='22d00m52.2s')
    >>> with erfa_astrom.set(ErfaAstromInterpolator(300 * u.s)):
    ...    cirs = crab.transform_to(CIRS(obstime=obstime))
    """
    mjd_resolution: Incomplete
    def __init__(self, time_resolution) -> None: ...
    def _get_support_points(self, obstime):
        """
        Calculate support points for the interpolation.

        We divide the MJD by the time resolution (as single float64 values),
        and calculate ceil and floor.
        Then we take the unique and sorted values and scale back to MJD.
        This will create a sparse support for non-regular input obstimes.
        """
    @staticmethod
    def _prepare_earth_position_vel(support, obstime):
        """
        Calculate Earth's position and velocity.

        Uses the coarser grid ``support`` to do the calculation, and interpolates
        onto the finer grid ``obstime``.
        """
    @staticmethod
    def _get_c2i(support, obstime):
        """
        Calculate the Celestial-to-Intermediate rotation matrix.

        Uses the coarser grid ``support`` to do the calculation, and interpolates
        onto the finer grid ``obstime``.
        """
    @staticmethod
    def _get_cip(support, obstime):
        """
        Find the X, Y coordinates of the CIP and the CIO locator, s.

        Uses the coarser grid ``support`` to do the calculation, and interpolates
        onto the finer grid ``obstime``.
        """
    @staticmethod
    def _get_polar_motion(support, obstime):
        """
        Find the two polar motion components in radians.

        Uses the coarser grid ``support`` to do the calculation, and interpolates
        onto the finer grid ``obstime``.
        """
    def apco(self, frame_or_coord):
        """
        Wrapper for ``erfa.apco``, used in conversions AltAz <-> ICRS and CIRS <-> ICRS.

        Parameters
        ----------
        frame_or_coord : ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``
            Frame or coordinate instance in the corresponding frame
            for which to calculate the calculate the astrom values.
            For this function, an AltAz or CIRS frame is expected.
        """
    def apcs(self, frame_or_coord):
        """
        Wrapper for ``erfa.apci``, used in conversions GCRS <-> ICRS.

        Parameters
        ----------
        frame_or_coord : ``astropy.coordinates.BaseCoordinateFrame`` or ``astropy.coordinates.SkyCoord``
            Frame or coordinate instance in the corresponding frame
            for which to calculate the calculate the astrom values.
            For this function, a GCRS frame is expected.
        """

class erfa_astrom(ScienceState):
    """
    ScienceState to select with astrom provider is used in
    coordinate transformations.
    """
    _value: Incomplete
    @classmethod
    def validate(cls, value): ...

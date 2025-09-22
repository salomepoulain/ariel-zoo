from .baseradec import BaseRADecFrame
from _typeshed import Incomplete

__all__ = ['FK4', 'FK4NoETerms']

class FK4(BaseRADecFrame):
    """
    A coordinate or frame in the FK4 system.

    Note that this is a barycentric version of FK4 - that is, the origin for
    this frame is the Solar System Barycenter, *not* the Earth geocenter.

    The frame attributes are listed under **Other Parameters**.
    """
    equinox: Incomplete
    obstime: Incomplete

class FK4NoETerms(BaseRADecFrame):
    """
    A coordinate or frame in the FK4 system, but with the E-terms of aberration
    removed.

    The frame attributes are listed under **Other Parameters**.
    """
    equinox: Incomplete
    obstime: Incomplete
    @staticmethod
    def _precession_matrix(oldequinox, newequinox):
        """
        Compute and return the precession matrix for FK4 using Newcomb's method.
        Used inside some of the transformation functions.

        Parameters
        ----------
        oldequinox : `~astropy.time.Time`
            The equinox to precess from.
        newequinox : `~astropy.time.Time`
            The equinox to precess to.

        Returns
        -------
        newcoord : array
            The precession matrix to transform to the new equinox
        """

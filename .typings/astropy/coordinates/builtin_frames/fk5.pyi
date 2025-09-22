from .baseradec import BaseRADecFrame
from _typeshed import Incomplete

__all__ = ['FK5']

class FK5(BaseRADecFrame):
    """
    A coordinate or frame in the FK5 system.

    Note that this is a barycentric version of FK5 - that is, the origin for
    this frame is the Solar System Barycenter, *not* the Earth geocenter.

    The frame attributes are listed under **Other Parameters**.
    """
    equinox: Incomplete
    @staticmethod
    def _precession_matrix(oldequinox, newequinox):
        """
        Compute and return the precession matrix for FK5 based on Capitaine et
        al. 2003/IAU2006.  Used inside some of the transformation functions.

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

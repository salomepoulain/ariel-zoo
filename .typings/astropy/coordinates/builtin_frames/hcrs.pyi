from .baseradec import BaseRADecFrame
from _typeshed import Incomplete

__all__ = ['HCRS']

class HCRS(BaseRADecFrame):
    """
    A coordinate or frame in a Heliocentric system, with axes aligned to ICRS.

    The ICRS has an origin at the Barycenter and axes which are fixed with
    respect to space.

    This coordinate system is distinct from ICRS mainly in that it is relative
    to the Sun's center-of-mass rather than the solar system Barycenter.
    In principle, therefore, this frame should include the effects of
    aberration (unlike ICRS), but this is not done, since they are very small,
    of the order of 8 milli-arcseconds.

    For more background on the ICRS and related coordinate transformations, see
    the references provided in the :ref:`astropy:astropy-coordinates-seealso`
    section of the documentation.

    The frame attributes are listed under **Other Parameters**.
    """
    obstime: Incomplete

from .altaz import AltAz as AltAz
from .baseradec import BaseRADecFrame as BaseRADecFrame
from .cirs import CIRS as CIRS
from .ecliptic import BarycentricMeanEcliptic as BarycentricMeanEcliptic, BarycentricTrueEcliptic as BarycentricTrueEcliptic, BaseEclipticFrame as BaseEclipticFrame, CustomBarycentricEcliptic as CustomBarycentricEcliptic, GeocentricMeanEcliptic as GeocentricMeanEcliptic, GeocentricTrueEcliptic as GeocentricTrueEcliptic, HeliocentricEclipticIAU76 as HeliocentricEclipticIAU76, HeliocentricMeanEcliptic as HeliocentricMeanEcliptic, HeliocentricTrueEcliptic as HeliocentricTrueEcliptic
from .equatorial import TEME as TEME, TETE as TETE
from .fk4 import FK4 as FK4, FK4NoETerms as FK4NoETerms
from .fk5 import FK5 as FK5
from .galactic import Galactic as Galactic
from .galactocentric import Galactocentric as Galactocentric, galactocentric_frame_defaults as galactocentric_frame_defaults
from .gcrs import GCRS as GCRS, PrecessedGeocentric as PrecessedGeocentric
from .hadec import HADec as HADec
from .hcrs import HCRS as HCRS
from .icrs import ICRS as ICRS
from .itrs import ITRS as ITRS
from .lsr import GalacticLSR as GalacticLSR, LSR as LSR, LSRD as LSRD, LSRK as LSRK
from .skyoffset import SkyOffsetFrame as SkyOffsetFrame
from .supergalactic import Supergalactic as Supergalactic

__all__ = ['ICRS', 'FK5', 'FK4', 'FK4NoETerms', 'Galactic', 'Galactocentric', 'Supergalactic', 'AltAz', 'HADec', 'GCRS', 'CIRS', 'ITRS', 'HCRS', 'TEME', 'TETE', 'PrecessedGeocentric', 'GeocentricMeanEcliptic', 'BarycentricMeanEcliptic', 'HeliocentricMeanEcliptic', 'GeocentricTrueEcliptic', 'BarycentricTrueEcliptic', 'HeliocentricTrueEcliptic', 'HeliocentricEclipticIAU76', 'CustomBarycentricEcliptic', 'LSR', 'LSRK', 'LSRD', 'GalacticLSR', 'SkyOffsetFrame', 'BaseEclipticFrame', 'BaseRADecFrame', 'galactocentric_frame_defaults', 'make_transform_graph_docs']

def make_transform_graph_docs(transform_graph):
    """
    Generates a string that can be used in other docstrings to include a
    transformation graph, showing the available transforms and
    coordinate systems.

    Parameters
    ----------
    transform_graph : `~astropy.coordinates.TransformGraph`

    Returns
    -------
    docstring : str
        A string that can be added to the end of a docstring to show the
        transform graph.
    """
__doc__ = _transform_graph_docs
